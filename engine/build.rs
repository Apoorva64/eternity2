//! Compile the CUDA kernel to PTX with `nvcc`.
//!
//! - Output: `$OUT_DIR/eternity_solver.ptx`, embedded by `attack_gpu` via
//!   `include_bytes!`.
//! - On wasm or when `nvcc` is missing, writes a tiny stub PTX containing a
//!   sentinel so the binary build succeeds and the runtime errors with a
//!   readable message.
//! - Re-runs only when the kernel source or env vars change; it does NOT
//!   re-run on every Rust source edit.

use std::env;
use std::path::PathBuf;
use std::process::Command;

const KERNEL_SRC: &str = "kernels/eternity_solver.cu";
const STUB: &[u8] = b"// no nvcc available at build time\n";

fn main() {
    println!("cargo:rerun-if-changed={KERNEL_SRC}");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));
    let ptx_out = out_dir.join("eternity_solver.ptx");

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32") {
        std::fs::write(&ptx_out, STUB).expect("write stub ptx");
        return;
    }
    if !PathBuf::from(KERNEL_SRC).exists() {
        std::fs::write(&ptx_out, STUB).expect("write stub ptx");
        return;
    }

    let nvcc = nvcc_path();
    let mut cmd = Command::new(&nvcc);
    cmd.args([
        "-ptx",
        "-O3",
        // compute_75 PTX runs on every Turing+ card, including the RTX 4070
        // (sm_89). The driver JIT-compiles it on first launch and caches the
        // result. Stays compatible with the system's nvcc 11.5.
        "-arch=compute_75",
        "--use_fast_math",
        "-o",
    ]);
    cmd.arg(&ptx_out);
    cmd.arg(KERNEL_SRC);

    let status = cmd.status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            println!(
                "cargo:warning=nvcc exited with {s} when building {KERNEL_SRC}; attack_gpu will fail at runtime",
            );
            std::fs::write(&ptx_out, STUB).expect("write stub ptx");
        }
        Err(e) => {
            println!(
                "cargo:warning=could not invoke nvcc ({e}); install CUDA toolkit or set NVCC. attack_gpu will fail at runtime",
            );
            std::fs::write(&ptx_out, STUB).expect("write stub ptx");
        }
    }
}

fn nvcc_path() -> PathBuf {
    if let Some(p) = env::var_os("NVCC") {
        return PathBuf::from(p);
    }
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Some(root) = env::var_os(var) {
            let p = PathBuf::from(root).join("bin").join("nvcc");
            if p.exists() {
                return p;
            }
        }
    }
    PathBuf::from("nvcc")
}
