use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rand::Rng;

#[pyclass]
struct PerTree {
    cap: usize,
    tree: Vec<f64>, // 1-indexed heap, leaves at cap..2*cap
    write: usize,
    size: usize,
    max_p: f64,     // max stored priority (already raised to alpha)
    alpha: f64,
}

impl PerTree {
    fn set(&mut self, leaf: usize, p: f64) {
        let mut i = leaf + self.cap;
        let delta = p - self.tree[i];
        while i >= 1 {
            self.tree[i] += delta;
            i /= 2;
        }
    }
}

#[pymethods]
impl PerTree {
    #[new]
    fn new(capacity: usize, alpha: f64) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be a power of two");
        Self {
            cap: capacity,
            tree: vec![0.0; 2 * capacity],
            write: 0,
            size: 0,
            max_p: 1.0,
            alpha,
        }
    }

    fn total(&self) -> f64 {
        self.tree[1]
    }

    fn len(&self) -> usize {
        self.size
    }

    fn capacity(&self) -> usize {
        self.cap
    }

    /// Reserve next slot (circular), give it max priority, return its index.
    fn add(&mut self) -> usize {
        let idx = self.write;
        self.set(idx, self.max_p);
        self.write = (self.write + 1) % self.cap;
        self.size = (self.size + 1).min(self.cap);
        idx
    }

    fn update(&mut self, idx: PyReadonlyArray1<i64>, td: PyReadonlyArray1<f64>) {
        let (idx, td) = (idx.as_array(), td.as_array());
        for (&i, &t) in idx.iter().zip(td.iter()) {
            let p = (t.abs() + 1e-6).powf(self.alpha);
            self.max_p = self.max_p.max(p);
            self.set(i as usize, p);
        }
    }

    /// Stratified sampling. Returns (indices, normalized IS weights).
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch: usize,
        beta: f64,
    ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
        assert!(self.size > 0, "Cannot sample from an empty tree");
        let total = self.tree[1];
        assert!(total > 0.0, "Total priority must be greater than zero");

        let seg = total / batch as f64;
        let n = self.size as f64;
        let mut rng = rand::thread_rng();
        let mut idxs = Vec::with_capacity(batch);
        let mut ws = Vec::with_capacity(batch);

        for k in 0..batch {
            let low = seg * k as f64;
            let high = (seg * (k + 1) as f64).min(total);
            let mut s = if high > low {
                rng.gen_range(low..high)
            } else {
                low
            };
            let mut i = 1;
            while i < self.cap {
                let l = 2 * i;
                if s <= self.tree[l] || self.tree[l + 1] == 0.0 {
                    i = l;
                } else {
                    s -= self.tree[l];
                    i = l + 1;
                }
            }
            idxs.push((i - self.cap) as i64);
            let p_i = self.tree[i].max(1e-12);
            let p_sample = p_i / total;
            ws.push((n * p_sample).powf(-beta));
        }

        let max_w = ws.iter().cloned().fold(f64::MIN, f64::max);
        let norm_factor = if max_w > 0.0 && max_w.is_finite() {
            max_w
        } else {
            1.0
        };
        ws.iter_mut().for_each(|w| *w /= norm_factor);

        (idxs.into_pyarray_bound(py), ws.into_pyarray_bound(py))
    }
}

#[pymodule]
fn per_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PerTree>()
}
