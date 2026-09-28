"""Microbenchmark: Pure Python vs NumPy vs Rust SumTree."""

import time
import argparse
import os
import numpy as np
import per_rs
from flappy_rl.buffers import PurePythonSumTree


def bench_pure_python(cap: int, batch_size: int, n_samples: int, n_updates: int):
    tree = PurePythonSumTree(cap, alpha=0.6)
    for _ in range(cap):
        tree.add()
    # Random initial priorities
    tds = np.random.uniform(0.1, 5.0, size=cap)
    tree.update(np.arange(cap, dtype=np.int64), tds)

    # Benchmark sampling
    t0 = time.perf_counter()
    for _ in range(n_samples):
        tree.sample(batch_size, beta=0.4)
    sample_time = time.perf_counter() - t0
    samples_per_sec = n_samples / sample_time

    # Benchmark updates
    test_idxs = np.random.randint(0, cap, size=(n_updates, batch_size)).astype(np.int64)
    test_tds = np.random.uniform(0.1, 5.0, size=(n_updates, batch_size)).astype(np.float64)

    t0 = time.perf_counter()
    for k in range(n_updates):
        tree.update(test_idxs[k], test_tds[k])
    update_time = time.perf_counter() - t0
    updates_per_sec = n_updates / update_time

    return samples_per_sec, updates_per_sec


def bench_numpy_choice(cap: int, batch_size: int, n_samples: int, n_updates: int):
    # Priorities array
    priorities = np.random.uniform(0.1, 5.0, size=cap).astype(np.float64)
    probs = priorities / priorities.sum()

    # Benchmark sampling using np.random.choice with probabilities
    t0 = time.perf_counter()
    for _ in range(n_samples):
        _ = np.random.choice(cap, size=batch_size, p=probs)
    sample_time = time.perf_counter() - t0
    samples_per_sec = n_samples / sample_time

    # Benchmark update: recomputing total sum and probabilities
    test_idxs = np.random.randint(0, cap, size=(n_updates, batch_size)).astype(np.int64)
    test_tds = np.random.uniform(0.1, 5.0, size=(n_updates, batch_size)).astype(np.float64)

    t0 = time.perf_counter()
    for k in range(n_updates):
        priorities[test_idxs[k]] = (np.abs(test_tds[k]) + 1e-6) ** 0.6
        probs = priorities / priorities.sum()
    update_time = time.perf_counter() - t0
    updates_per_sec = n_updates / update_time

    return samples_per_sec, updates_per_sec


def bench_rust(cap: int, batch_size: int, n_samples: int, n_updates: int):
    tree = per_rs.PerTree(cap, 0.6)
    for _ in range(cap):
        tree.add()
    tds = np.random.uniform(0.1, 5.0, size=cap).astype(np.float64)
    tree.update(np.arange(cap, dtype=np.int64), tds)

    # Benchmark sampling
    t0 = time.perf_counter()
    for _ in range(n_samples):
        tree.sample(batch_size, 0.4)
    sample_time = time.perf_counter() - t0
    samples_per_sec = n_samples / sample_time

    # Benchmark updates
    test_idxs = np.random.randint(0, cap, size=(n_updates, batch_size)).astype(np.int64)
    test_tds = np.random.uniform(0.1, 5.0, size=(n_updates, batch_size)).astype(np.float64)

    t0 = time.perf_counter()
    for k in range(n_updates):
        tree.update(test_idxs[k], test_tds[k])
    update_time = time.perf_counter() - t0
    updates_per_sec = n_updates / update_time

    return samples_per_sec, updates_per_sec


def main():
    parser = argparse.ArgumentParser(description="Buffer microbenchmark")
    parser.add_argument("--cap", type=int, default=65536, help="Buffer capacity (power of 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--samples", type=int, default=2000, help="Number of sample calls to test")
    parser.add_argument("--updates", type=int, default=2000, help="Number of update calls to test")
    parser.add_argument("--out-dir", type=str, default="results/tables", help="Output directory")
    args = parser.parse_args()

    print(f"Running buffer microbenchmark (Capacity={args.cap}, Batch={args.batch_size})...")

    print("1. Benchmarking NumPy random.choice...")
    np_s_rate, np_u_rate = bench_numpy_choice(args.cap, args.batch_size, args.samples, args.updates)
    print(f"   NumPy: {np_s_rate:.1f} samples/s | {np_u_rate:.1f} updates/s")

    print("2. Benchmarking Pure-Python SumTree...")
    py_s_rate, py_u_rate = bench_pure_python(args.cap, args.batch_size, args.samples, args.updates)
    print(f"   Pure Python: {py_s_rate:.1f} samples/s | {py_u_rate:.1f} updates/s")

    print("3. Benchmarking Rust PerTree...")
    rs_s_rate, rs_u_rate = bench_rust(args.cap, args.batch_size, args.samples, args.updates)
    print(f"   Rust: {rs_s_rate:.1f} samples/s | {rs_u_rate:.1f} updates/s")

    rust_speedup_sample_py = rs_s_rate / py_s_rate if py_s_rate > 0 else 0
    rust_speedup_update_py = rs_u_rate / py_u_rate if py_u_rate > 0 else 0

    rust_speedup_sample_np = rs_s_rate / np_s_rate if np_s_rate > 0 else 0
    rust_speedup_update_np = rs_u_rate / np_u_rate if np_u_rate > 0 else 0

    table_md = f"""# Buffer Microbenchmark Results

**Configuration:** Capacity = {args.cap} ($2^{{16}}$), Batch Size = {args.batch_size}

| Implementation | Sample (batch {args.batch_size}) / sec | Priority Update (batch {args.batch_size}) / sec | Sample Speedup vs Rust | Update Speedup vs Rust |
|---|---|---|---|---|
| Pure-Python SumTree | {py_s_rate:,.1f} | {py_u_rate:,.1f} | 1.0x (ref) | 1.0x (ref) |
| NumPy `random.choice(p=...)` | {np_s_rate:,.1f} | {np_u_rate:,.1f} | {np_s_rate / py_s_rate:.2f}x | {np_u_rate / py_u_rate:.2f}x |
| **Rust `PerTree` (PyO3)** | **{rs_s_rate:,.1f}** | **{rs_u_rate:,.1f}** | **{rust_speedup_sample_py:.2f}x** | **{rust_speedup_update_py:.2f}x** |

### Summary
- Rust `PerTree` achieves **{rust_speedup_sample_py:.1f}x speedup** in sampling over Pure-Python.
- Rust `PerTree` achieves **{rust_speedup_update_py:.1f}x speedup** in priority updates over Pure-Python.
- NumPy `random.choice` requires $O(N)$ full probability vector renormalization on every update, while the SumTree maintains $O(\\log N)$ updates.
"""
    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "benchmark_results.md")
    with open(out_file, "w") as f:
        f.write(table_md)

    csv_file = os.path.join(args.out_dir, "benchmark_results.csv")
    with open(csv_file, "w") as f:
        f.write("implementation,samples_per_sec,updates_per_sec\n")
        f.write(f"pure_python,{py_s_rate:.2f},{py_u_rate:.2f}\n")
        f.write(f"numpy_choice,{np_s_rate:.2f},{np_u_rate:.2f}\n")
        f.write(f"rust_pertree,{rs_s_rate:.2f},{rs_u_rate:.2f}\n")

    print("\nBenchmark Markdown Table:")
    print(table_md)
    print(f"Saved to {out_file} and {csv_file}")


if __name__ == "__main__":
    main()
