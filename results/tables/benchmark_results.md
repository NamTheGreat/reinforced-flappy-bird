# Buffer Microbenchmark Results

**Configuration:** Capacity = 65536 ($2^{16}$), Batch Size = 64

| Implementation | Sample (batch 64) / sec | Priority Update (batch 64) / sec | Sample Speedup vs Rust | Update Speedup vs Rust |
|---|---|---|---|---|
| Pure-Python SumTree | 6,585.1 | 11,236.8 | 1.0x (ref) | 1.0x (ref) |
| NumPy `random.choice(p=...)` | 3,645.3 | 61,482.2 | 0.55x | 5.47x |
| **Rust `PerTree` (PyO3)** | **197,869.6** | **1,317,886.0** | **30.05x** | **117.28x** |

### Summary
- Rust `PerTree` achieves **30.0x speedup** in sampling over Pure-Python.
- Rust `PerTree` achieves **117.3x speedup** in priority updates over Pure-Python.
- NumPy `random.choice` requires $O(N)$ full probability vector renormalization on every update, while the SumTree maintains $O(\log N)$ updates.
