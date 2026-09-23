//! Differential tests against the `f64` reference interpreter
//! (`meganeura::reference`): every op's kernels on the GPU, and every
//! autodiff rule against finite differences on the CPU.
mod losses;
mod smoke;
