# Buffer Microbenchmark Results

**Configuration:** Capacity = 65536 ($2^{16}$), Batch Size = 64

| Implementation | Sample (batch 64) / sec | Priority Update (batch 64) / sec | Sample Speedup vs Rust | Update Speedup vs Rust |
|---|---|---|---|---|
| Pure-Python SumTree | 2,242.0 | 3,802.3 | 1.0x (ref) | 1.0x (ref) |
| NumPy `random.choice(p=...)` | 1,777.3 | 22,472.7 | 0.79x | 5.91x |
| **Rust `PerTree` (PyO3)** | **109,954.7** | **402,763.0** | **49.04x** | **105.93x** |

### Summary
- Rust `PerTree` achieves **49.0x speedup** in sampling over Pure-Python.
- Rust `PerTree` achieves **105.9x speedup** in priority updates over Pure-Python.
- NumPy `random.choice` requires $O(N)$ full probability vector renormalization on every update, while the SumTree maintains $O(\log N)$ updates.
