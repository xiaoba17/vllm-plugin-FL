"""Sampling glue; stable radix kernels live in FlagGems."""

import torch


def radix_sort_rows(x: torch.Tensor, block: int = 512):
    import flag_gems

    return flag_gems.radix_sort_rows(x, block)


def radix_top_p(logits: torch.Tensor, p: torch.Tensor):
    """Keep the existing softmax/cumsum/mask/scatter operations unchanged."""
    values, indices = radix_sort_rows(logits)
    probabilities = values.softmax(dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1, out=probabilities)
    mask = cumulative <= 1 - p.unsqueeze(1)
    mask[:, -1] = False
    values.masked_fill_(mask, -float("inf"))
    return logits.scatter_(dim=-1, index=indices, src=values)
