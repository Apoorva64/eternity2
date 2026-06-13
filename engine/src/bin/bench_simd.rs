//! Apples-to-apples benchmark: scalar `Solver` vs `SimdSolver`.
//!
//! The two solvers are run on the **same puzzles, same paths**, in the
//! same single thread. For each (size, path) cell we print a table row
//! with per-solver puzzles/sec and nodes/sec plus the SIMD speedup.
//!
//! A second pass picks a handful of fixed seeds and reports the per-puzzle
//! times so you can see how stable the ratio is across instances (some
//! puzzles solve in a few μs and dominated by setup, others run the
//! search engine to its limits — the ratio shifts).
//!
//! Run with: `cargo run --release --bin bench_simd`.

use eternity2_engine::simd_solver::SimdSolver;
use eternity2_engine::{build_path, generate, Puzzle, Solver, Status};
use std::time::{Duration, Instant};

/// Seconds to spend on each (size, path) cell in the throughput table.
const PER_CELL_SECS: f64 = 1.5;
/// Sizes to sweep in the throughput table.
const SIZES: &[u8] = &[4, 5, 6];
/// Paths to sweep in the throughput table.
const PATHS: &[&str] = &["snake", "spiral-in", "border-first", "diagonal"];

/// Stream the scalar `Solver` for `secs` seconds, return totals.
fn run_scalar(size: u8, path: &[u16], secs: f64, mut seed: u32) -> Totals {
    let mut t = Totals::default();
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        let p = generate(size, size, seed);
        seed = seed.wrapping_add(1);
        let mut s = Solver::new(&p, path, false, false, 0).unwrap();
        loop {
            let r = s.step(5_000_000);
            if r.status != Status::Running {
                t.nodes += r.nodes as u64;
                t.attempts += r.attempts as u64;
                t.solved += u64::from(r.status == Status::Solved);
                t.puzzles += 1;
                break;
            }
        }
    }
    t.secs = t0.elapsed().as_secs_f64();
    t
}

fn run_simd(size: u8, path: &[u16], secs: f64, mut seed: u32) -> Totals {
    let mut t = Totals::default();
    let t0 = Instant::now();
    while t0.elapsed().as_secs_f64() < secs {
        let p = generate(size, size, seed);
        seed = seed.wrapping_add(1);
        let mut s = SimdSolver::new(&p, path, false).unwrap();
        loop {
            let r = s.step(5_000_000);
            if r.status != Status::Running {
                t.nodes += r.nodes as u64;
                t.attempts += r.attempts as u64;
                t.solved += u64::from(r.status == Status::Solved);
                t.puzzles += 1;
                break;
            }
        }
    }
    t.secs = t0.elapsed().as_secs_f64();
    t
}

/// Run a single puzzle once with each solver and return (scalar, simd) time.
fn time_one(puzzle: &Puzzle, path: &[u16]) -> (Duration, Duration, u64, u64) {
    let t0 = Instant::now();
    let mut s = Solver::new(puzzle, path, false, false, 0).unwrap();
    let s_nodes;
    loop {
        let r = s.step(5_000_000);
        if r.status != Status::Running {
            s_nodes = r.nodes as u64;
            break;
        }
    }
    let scalar = t0.elapsed();

    let t0 = Instant::now();
    let mut s = SimdSolver::new(puzzle, path, false).unwrap();
    let v_nodes;
    loop {
        let r = s.step(5_000_000);
        if r.status != Status::Running {
            v_nodes = r.nodes as u64;
            break;
        }
    }
    let simd = t0.elapsed();
    (scalar, simd, s_nodes, v_nodes)
}

#[derive(Default, Clone, Copy)]
struct Totals {
    secs: f64,
    puzzles: u64,
    solved: u64,
    nodes: u64,
    attempts: u64,
}

impl Totals {
    fn pps(self) -> f64 {
        self.puzzles as f64 / self.secs
    }
    fn nps(self) -> f64 {
        self.nodes as f64 / self.secs
    }
}

fn main() {
    // Warm up — let allocators settle, branch predictors prime.
    let path = build_path("snake", 4, 4, 0).unwrap();
    let _ = run_scalar(4, &path, 0.2, 1);
    let _ = run_simd(4, &path, 0.2, 1);

    // ---- Throughput table ----------------------------------------------
    println!("Throughput: scalar Solver vs SimdSolver, same puzzles + path");
    println!("Each cell run for {PER_CELL_SECS:.1}s on identical seed streams.\n");

    println!(
        "  {:<5} {:<14} | {:>10} {:>12} | {:>10} {:>12} | {:>6} {:>7}",
        "size", "path", "scalar P/s", "scalar Mn/s", "simd P/s", "simd Mn/s", "x P/s", "x n/s",
    );
    println!("  {}", "-".repeat(95));

    for &size in SIZES {
        for &kind in PATHS {
            let path = build_path(kind, size, size, 0).unwrap();
            // Same starting seed for both runs ⇒ they solve the same puzzles
            // in the same order. Even with different totals per-second, the
            // *per-puzzle* work is comparable.
            let s = run_scalar(size, &path, PER_CELL_SECS, 1);
            let v = run_simd(size, &path, PER_CELL_SECS, 1);
            println!(
                "  {:<5} {:<14} | {:>10.1} {:>12.2} | {:>10.1} {:>12.2} | x{:>5.2} x{:>5.2}",
                format!("{size}x{size}"),
                kind,
                s.pps(),
                s.nps() / 1e6,
                v.pps(),
                v.nps() / 1e6,
                v.pps() / s.pps(),
                v.nps() / s.nps(),
            );
        }
    }

    // ---- Per-puzzle detail ---------------------------------------------
    println!("\nPer-puzzle wall-time on fixed seeds (6x6, snake)");
    println!(
        "  {:>5}   {:>10}   {:>10}   {:>6}   {:>10}   {:>6}",
        "seed", "scalar μs", "simd μs", "x time", "nodes", "match?"
    );
    println!("  {}", "-".repeat(64));
    let path = build_path("snake", 6, 6, 0).unwrap();
    for seed in 1u32..=8 {
        let p = generate(6, 6, seed);
        let (sd, vd, sn, vn) = time_one(&p, &path);
        // Nodes must agree for the same puzzle/path with hint-less DFS,
        // both solvers follow the exact same search order.
        let match_str = if sn == vn { "yes" } else { "NO" };
        println!(
            "  {:>5}   {:>10.0}   {:>10.0}   x{:>4.2}   {:>10}   {:>6}",
            seed,
            sd.as_secs_f64() * 1e6,
            vd.as_secs_f64() * 1e6,
            sd.as_secs_f64() / vd.as_secs_f64(),
            sn,
            match_str,
        );
    }

    // ---- One bigger puzzle, all paths -----------------------------------
    println!("\nSingle 7x7 puzzle (seed 1), every path: scalar vs simd");
    println!(
        "  {:<14}   {:>10}   {:>10}   {:>6}   {:>10}",
        "path", "scalar ms", "simd ms", "x time", "nodes"
    );
    println!("  {}", "-".repeat(60));
    let p = generate(7, 7, 1);
    for &kind in PATHS {
        let path = build_path(kind, 7, 7, 0).unwrap();
        let (sd, vd, sn, vn) = time_one(&p, &path);
        assert_eq!(sn, vn, "node count diverged for path {kind}");
        println!(
            "  {:<14}   {:>10.2}   {:>10.2}   x{:>4.2}   {:>10}",
            kind,
            sd.as_secs_f64() * 1e3,
            vd.as_secs_f64() * 1e3,
            sd.as_secs_f64() / vd.as_secs_f64(),
            sn,
        );
    }
}
