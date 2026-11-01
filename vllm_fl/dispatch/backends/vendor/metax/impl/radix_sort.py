"""Experimental stable FP32 row radix sort for MetaX sampling.

Four byte passes distribute work across row tiles instead of serializing each
large row. Serving integration is controlled by VLLM_FL_METAX_RADIX_SORT.
"""

import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry


@triton.jit
def _digit(values, SHIFT: tl.constexpr):
    bits = values.to(tl.uint32, bitcast=True)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    return ((ordered >> SHIFT) & 255).to(tl.int32)


@libentry()
@triton.jit
def _histogram(X, H, N: tl.constexpr, TILES: tl.constexpr,
               SHIFT: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + row * N + offsets)
    counts = tl.histogram(_digit(values, SHIFT), 256)
    bins = tl.arange(0, 256)
    tl.store(H + (row * 256 + bins) * TILES + tile, counts)


@libentry()
@triton.jit
def _tile_prefix(H, PREFIX, TOTAL, TILES: tl.constexpr, BLOCK: tl.constexpr):
    digit = tl.program_id(0)
    row = tl.program_id(1)
    tiles = tl.arange(0, BLOCK)
    counts = tl.load(H + (row * 256 + digit) * TILES + tiles,
                     tiles < TILES, other=0)
    inclusive = tl.cumsum(counts)
    tl.store(PREFIX + (row * 256 + digit) * TILES + tiles,
             inclusive - counts, tiles < TILES)
    tl.store(TOTAL + row * 256 + digit, tl.sum(counts))


@libentry()
@triton.jit
def _bin_prefix(TOTAL, BASE):
    row = tl.program_id(0)
    bins = tl.arange(0, 256)
    counts = tl.load(TOTAL + row * 256 + bins)
    tl.store(BASE + row * 256 + bins, tl.cumsum(counts) - counts)


@libentry()
@triton.jit
def _scatter(X, IDS, Y, OUT_IDS, PREFIX, BASE,
             N: tl.constexpr, TILES: tl.constexpr, SHIFT: tl.constexpr,
             FIRST: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    row = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    offsets = tile * BLOCK + lane
    values = tl.load(X + row * N + offsets)
    digits = _digit(values, SHIFT)
    # Integer keys retain input order for equal digits.
    packed = tl.sort(digits * BLOCK + lane, descending=False)
    source = packed % BLOCK
    sorted_digits = packed // BLOCK
    previous = tl.gather(sorted_digits, tl.maximum(lane - 1, 0), 0)
    run_start = tl.where((lane == 0) | (previous != sorted_digits), lane, 0)
    run_start = tl.associative_scan(run_start, 0, _maximum)
    before = tl.load(PREFIX + (row * 256 + sorted_digits) * TILES + tile)
    base = tl.load(BASE + row * 256 + sorted_digits)
    destination = base + before + lane - run_start
    sorted_values = tl.gather(values, source, 0)
    if FIRST:
        ids = offsets
    else:
        ids = tl.load(IDS + row * N + offsets)
    sorted_ids = tl.gather(ids, source, 0)
    tl.store(Y + row * N + destination, sorted_values)
    tl.store(OUT_IDS + row * N + destination, sorted_ids)


@triton.jit
def _maximum(a, b):
    return tl.maximum(a, b)


def radix_sort_rows(x: torch.Tensor, block: int = 512):
    if x.dtype != torch.float32 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("experimental sorter requires contiguous FP32 [batch,vocab]")
    rows, n = x.shape
    if block not in (256, 512, 1024) or n % block:
        raise ValueError("vocabulary must be divisible by block (256/512/1024)")
    tiles = n // block
    values = [torch.empty_like(x), torch.empty_like(x)]
    ids = [torch.empty(x.shape, dtype=torch.int32, device=x.device) for _ in range(2)]
    final_ids = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    counts = torch.empty((rows, 256, tiles), dtype=torch.int32, device=x.device)
    prefix = torch.empty_like(counts)
    totals = torch.empty((rows, 256), dtype=torch.int32, device=x.device)
    bases = torch.empty_like(totals)
    source, source_ids = x, ids[0]
    for index in range(4):
        target = values[index % 2]
        target_ids = final_ids if index == 3 else ids[index % 2]
        _histogram[(tiles, rows)](source, counts, n, tiles, index * 8, block)
        _tile_prefix[(256, rows)](counts, prefix, totals, tiles,
                                   triton.next_power_of_2(tiles))
        _bin_prefix[(rows,)](totals, bases)
        _scatter[(tiles, rows)](source, source_ids, target, target_ids,
                                prefix, bases, n, tiles, index * 8,
                                index == 0, block)
        source, source_ids = target, target_ids
    return source, final_ids


def radix_top_p(logits: torch.Tensor, p: torch.Tensor):
    """Keep the existing softmax/cumsum/mask/scatter operations unchanged."""
    values, indices = radix_sort_rows(logits)
    probabilities = values.softmax(dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1, out=probabilities)
    mask = cumulative <= 1 - p.unsqueeze(1)
    mask[:, -1] = False
    values.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)
