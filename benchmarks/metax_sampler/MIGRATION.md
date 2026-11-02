# FlagGems companion migration

Companion FlagGems branch: `feat/metax-stable-radix-sort`, based on the competition
commit `a7620cc191a0b42e040194622c5758b22a7a25dc`.

The Triton kernels have moved out of the plugin. `impl/radix_sort.py` now only
calls `flag_gems.radix_sort_rows` and performs the original filtering operations.
The opt-in remains `VLLM_FL_METAX_RADIX_SORT=1`; the default is still off. The
original dispatcher is used if the matching FlagGems API is missing.

This is a migration-only branch, separate from matrix-multiply tuning fixes.
It does not establish a TTFT improvement or formal acceptance. Keep it unmerged
until paired-commit correctness, performance and precision validation succeeds.
