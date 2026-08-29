# What "matches the reference" means here

Measured on this machine before writing any tolerances, so the numbers below are
the yardstick every milestone is judged against — not guesses.

| quantity | value |
|---|---|
| bf16 round-trip relative error | **2.0e-3** |
| bf16 matmul vs float64 — MLX | 3.97e-3 |
| bf16 matmul vs float64 — torch | 3.97e-3 |
| bf16 matmul, MLX vs torch | 2.1e-3 |
| fp32 matmul vs float64 — MLX | 8.0e-4 |
| fp32 matmul vs float64 — torch | 2.4e-6 |

Two things follow, and they set the whole testing strategy.

**MLX's fp32 matmul is ~300x less accurate than torch's** (8e-4 vs 2.4e-6
against a float64 reference). That looks alarming and is not: MLX evidently uses
a reduced-precision accumulation path for fp32 GEMM on Metal.

**But the model runs in bf16, where MLX and torch are identical** — both land at
3.968e-3 against float64. And MLX's fp32 matmul error (8e-4) is already four
times *better* than the bf16 round-trip floor (2.0e-3) that the real weights
carry anyway.

So the bar for a port is not "matches torch to fp32 precision". It is:

> the difference between mdream and the reference must be **smaller than the
> precision the model actually runs at** — i.e. well under 2.0e-3 relative.

Tolerances in `tests/` are set from that, with the measured floor printed
alongside each result so a regression is visible rather than buried in a
threshold. A check that comes in at 1e-6 is reported as such; one that creeps to
1e-3 is still "passing" but worth looking at, because it means something is
being computed differently even if the output is usable.

Exception: the reference explicitly keeps the final `(x - x_pred) / sigma` in
**fp32**, with a comment that bf16 there noticeably degrades samples. That step
is held to fp32 tolerances, not bf16 ones.
