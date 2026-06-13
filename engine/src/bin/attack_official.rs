//! Parallel SIMD attack on the official Eternity II puzzle.
//!
//! All available cores, one `SimdSolver` per thread. Each thread runs an
//! endless stream of attempts: pick a path (cycling through PATH_KINDS, with
//! "random" shuffled per attempt), start a fresh solver from the puzzle's
//! 5 official clue pieces, run it to `Exhausted` or until a configurable
//! node cap, then start again with a different path / seed combination.
//!
//! A shared `Mutex<Best>` records the deepest the search has ever been on
//! any thread. When a thread sets a new record we persist:
//!   - `out/best_official.json`  (board + metadata, pretty-printed)
//!   - `out/best_official.txt`   (human-readable diagram)
//! Snapshots are also flushed every `SNAPSHOT_SECS` regardless.
//!
//! The official puzzle is not going to fall to a depth-first attack; this
//! binary exists to (a) demonstrate end-to-end SIMD throughput on real
//! piece counts, and (b) track the best partial board you've ever seen.
//!
//! Run with: `cargo run --release --bin attack_official`
//! Stop with: Ctrl-C (Atomic flag, threads exit cleanly).

use eternity2_engine::simd_solver::SimdSolver;
use eternity2_engine::{build_path, official_puzzle, Puzzle, Status, PATH_KINDS};
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const OUT_DIR: &str = "out";
const JSON_PATH: &str = "out/best_official.json";
const TXT_PATH: &str = "out/best_official.txt";
const SNAPSHOT_SECS: u64 = 30;
/// Seconds between throughput trace lines (current + lifetime nodes/s).
const TRACE_SECS: f64 = 2.0;
/// Nodes before abandoning an attempt and resampling a path/seed.
/// Tuned so each attempt lasts a few seconds on a 16x16 puzzle.
const PER_ATTEMPT_NODE_CAP: u64 = 50_000_000;
/// Step budget per `step()` call (we re-enter to check the stop flag and
/// to keep responsiveness on Ctrl-C).
const STEP_BUDGET: u32 = 2_000_000;

#[derive(Clone, Serialize)]
struct BestSnapshot {
    /// Number of pieces on the board, hints included.
    placed: u32,
    /// Total cells on the puzzle (256 for the official 16x16).
    cells: u32,
    /// Matched-edge score for `board`.
    score: u32,
    /// Maximum achievable score (480 for 16x16).
    max_score: u32,
    /// Total attempts (solver instances spun up across all threads).
    attempts: u64,
    /// Total successful placements across all threads.
    nodes: u64,
    /// Seconds since the attack started.
    elapsed_secs: f64,
    /// Which thread set this record.
    thread: usize,
    /// Path kind that produced this record.
    path_kind: String,
    /// Seed used for that path (only meaningful for "random").
    path_seed: u32,
    /// `board[i]` = piece*4 + rot if cell i is filled, else -1.
    board: Vec<i32>,
}

struct Shared {
    puzzle: Puzzle,
    best: Mutex<BestSnapshot>,
    nodes: AtomicU64,
    attempts: AtomicU64,
    stop: AtomicBool,
    started: Instant,
}

fn main() {
    fs::create_dir_all(OUT_DIR).expect("create out/");

    let puzzle = official_puzzle();
    let cells = puzzle.cell_count() as u32;
    let max_score = puzzle.max_score();

    println!(
        "Official Eternity II: {}x{}, {} pieces, {} hints, max score = {max_score}",
        puzzle.width,
        puzzle.height,
        puzzle.pieces.len(),
        puzzle.hints.len()
    );

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("threads = {threads}");

    let initial = BestSnapshot {
        placed: puzzle.hints.len() as u32,
        cells,
        score: 0,
        max_score,
        attempts: 0,
        nodes: 0,
        elapsed_secs: 0.0,
        thread: 0,
        path_kind: String::new(),
        path_seed: 0,
        board: vec![-1; cells as usize],
    };
    let shared = Shared {
        puzzle,
        best: Mutex::new(initial),
        nodes: AtomicU64::new(0),
        attempts: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        started: Instant::now(),
    };

    // Ctrl-C → ask all threads to stop. Best-effort; ignore on systems
    // without the `ctrlc`-equivalent. Std doesn't ship one, so we install
    // a SIGINT handler via libc only if available.
    install_sigint_handler(&shared);

    std::thread::scope(|scope| {
        for t in 0..threads {
            let shared = &shared;
            scope.spawn(move || worker(t, shared));
        }
        // Throughput tracer: current + lifetime nodes/s every TRACE_SECS.
        let shared_for_tracer = &shared;
        scope.spawn(move || tracer(shared_for_tracer));
        // Periodic snapshot: best-so-far flushed to disk every SNAPSHOT_SECS.
        let shared_for_snap = &shared;
        scope.spawn(move || snapshotter(shared_for_snap));
    });

    println!("\nstopped. final best:");
    let best = shared.best.lock().unwrap().clone();
    println!(
        "  placed = {}/{}  score = {}/{}  attempts = {}  nodes = {}",
        best.placed,
        best.cells,
        best.score,
        best.max_score,
        shared.attempts.load(Ordering::Relaxed),
        shared.nodes.load(Ordering::Relaxed),
    );
}

/// Worker loop: spin attempts with different (path_kind, seed) tuples
/// forever until `shared.stop` is set. Reports nodes/attempts back via the
/// shared atomic counters.
fn worker(tid: usize, shared: &Shared) {
    let kinds: Vec<&'static str> = PATH_KINDS.iter().copied().collect();
    let mut local_attempt: u64 = 0;
    // Thread-local seed stream so paths/seeds diverge across threads.
    let mut seed: u32 = (tid as u32).wrapping_mul(0x9E37_79B9).wrapping_add(1);

    while !shared.stop.load(Ordering::Relaxed) {
        // Cycle through path kinds; "random" gets a fresh seed each time.
        let kind = kinds[(local_attempt as usize) % kinds.len()];
        let path_seed = seed;
        seed = seed.wrapping_add(0x9E37_79B9);
        let path = match build_path(
            kind,
            shared.puzzle.width,
            shared.puzzle.height,
            path_seed,
        ) {
            Some(p) => p,
            None => continue,
        };

        let mut solver = match SimdSolver::new(&shared.puzzle, &path, true) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let mut local_nodes: u64 = 0;
        let mut local_best_placed: u32 = 0;
        loop {
            if shared.stop.load(Ordering::Relaxed) {
                return;
            }
            let r = solver.step(STEP_BUDGET);
            // Flush incremental node count.
            let n = r.nodes as u64;
            if n > local_nodes {
                shared.nodes.fetch_add(n - local_nodes, Ordering::Relaxed);
                local_nodes = n;
            }
            // Track best of this attempt; only take the shared lock when we
            // beat our own previous local best (cheap filter to avoid
            // hammering the mutex on every step).
            if r.best_placed > local_best_placed {
                local_best_placed = r.best_placed;
                maybe_publish(shared, tid, kind, path_seed, &solver, r.best_placed);
            }
            if r.status != Status::Running {
                break;
            }
            if n >= PER_ATTEMPT_NODE_CAP {
                break;
            }
        }
        shared.attempts.fetch_add(1, Ordering::Relaxed);
        local_attempt += 1;
    }
}

fn maybe_publish(
    shared: &Shared,
    tid: usize,
    path_kind: &str,
    path_seed: u32,
    solver: &SimdSolver,
    placed: u32,
) {
    let board = solver.best_board().to_vec();
    let score = score_board(&shared.puzzle, &board);
    let mut best = shared.best.lock().unwrap();
    if placed > best.placed || (placed == best.placed && score > best.score) {
        best.placed = placed;
        best.score = score;
        best.attempts = shared.attempts.load(Ordering::Relaxed);
        best.nodes = shared.nodes.load(Ordering::Relaxed);
        best.elapsed_secs = shared.started.elapsed().as_secs_f64();
        best.thread = tid;
        best.path_kind = path_kind.to_string();
        best.path_seed = path_seed;
        best.board = board;
        println!(
            "[{:>6.1}s] new best: placed={}/{} score={}/{}  (thread {tid}, path={path_kind}, seed={path_seed})",
            best.elapsed_secs, best.placed, best.cells, best.score, best.max_score
        );
        if let Err(e) = write_snapshot(&shared.puzzle, &best) {
            eprintln!("snapshot write failed: {e}");
        }
    }
}

fn snapshotter(shared: &Shared) {
    let mut last = Instant::now();
    while !shared.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        if last.elapsed() >= Duration::from_secs(SNAPSHOT_SECS) {
            last = Instant::now();
            let best = shared.best.lock().unwrap().clone();
            let _ = write_snapshot(&shared.puzzle, &best);
        }
    }
}

/// Print one line every `TRACE_SECS` with the *current* window nodes/sec
/// and the *lifetime* nodes/sec, plus running totals. Cheap: reads the
/// shared atomics only, no locks on the hot path.
fn tracer(shared: &Shared) {
    let mut last_t = Instant::now();
    let mut last_nodes = shared.nodes.load(Ordering::Relaxed);
    let mut last_attempts = shared.attempts.load(Ordering::Relaxed);
    while !shared.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
        let dt = last_t.elapsed().as_secs_f64();
        if dt < TRACE_SECS {
            continue;
        }
        let now_nodes = shared.nodes.load(Ordering::Relaxed);
        let now_attempts = shared.attempts.load(Ordering::Relaxed);
        let dn = now_nodes - last_nodes;
        let da = now_attempts - last_attempts;
        let elapsed = shared.started.elapsed().as_secs_f64().max(1e-9);
        let cur = dn as f64 / dt / 1e6;
        let life = now_nodes as f64 / elapsed / 1e6;
        let best_placed = shared.best.lock().unwrap().placed;
        println!(
            "[{:>6.1}s] {:>6.1}M nodes/s (cur)  {:>6.1}M nodes/s (avg)  attempts +{da} (total {now_attempts})  best placed={best_placed}",
            elapsed, cur, life,
        );
        last_t = Instant::now();
        last_nodes = now_nodes;
        last_attempts = now_attempts;
    }
}

fn write_snapshot(puzzle: &Puzzle, best: &BestSnapshot) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(best).expect("serialize");
    let tmp_json = format!("{JSON_PATH}.tmp");
    let mut f = fs::File::create(&tmp_json)?;
    f.write_all(json.as_bytes())?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp_json, JSON_PATH)?;

    let txt = render_board(puzzle, best);
    let tmp_txt = format!("{TXT_PATH}.tmp");
    let mut f = fs::File::create(&tmp_txt)?;
    f.write_all(txt.as_bytes())?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp_txt, TXT_PATH)?;
    Ok(())
}

/// Pieces matched against neighbors on `board`; same metric as
/// `solver::score_board` but inlined to avoid a public-API dep on it.
fn score_board(puzzle: &Puzzle, board: &[i32]) -> u32 {
    use eternity2_engine::rotated;
    let (w, h) = (puzzle.width as usize, puzzle.height as usize);
    let mut score = 0u32;
    for y in 0..h {
        for x in 0..w {
            let here = board[y * w + x];
            if here < 0 {
                continue;
            }
            let pid = (here as usize) / 4;
            let rot = (here as u8) & 3;
            let edges = rotated(puzzle.pieces[pid], rot);
            // Right.
            if x + 1 < w {
                let n = board[y * w + (x + 1)];
                if n >= 0 {
                    let npid = (n as usize) / 4;
                    let nrot = (n as u8) & 3;
                    let nedges = rotated(puzzle.pieces[npid], nrot);
                    if edges[1] == nedges[3] {
                        score += 1;
                    }
                }
            }
            // Bottom.
            if y + 1 < h {
                let n = board[(y + 1) * w + x];
                if n >= 0 {
                    let npid = (n as usize) / 4;
                    let nrot = (n as u8) & 3;
                    let nedges = rotated(puzzle.pieces[npid], nrot);
                    if edges[2] == nedges[0] {
                        score += 1;
                    }
                }
            }
        }
    }
    score
}

fn render_board(puzzle: &Puzzle, best: &BestSnapshot) -> String {
    let (w, h) = (puzzle.width as usize, puzzle.height as usize);
    let mut out = String::new();
    out.push_str(&format!(
        "Eternity II — best partial\nplaced {}/{}  score {}/{}  elapsed {:.1}s\nthread {}  path {} (seed {})  attempts {}  nodes {}\n\n",
        best.placed, best.cells, best.score, best.max_score, best.elapsed_secs,
        best.thread, best.path_kind, best.path_seed, best.attempts, best.nodes,
    ));
    for y in 0..h {
        for x in 0..w {
            let v = best.board[y * w + x];
            if v < 0 {
                out.push_str("  . ");
            } else {
                let pid = (v as usize) / 4;
                let rot = (v as u32) & 3;
                out.push_str(&format!("{pid:>3}{}", rot_char(rot)));
            }
        }
        out.push('\n');
    }
    out
}

fn rot_char(r: u32) -> char {
    match r {
        0 => 'u',
        1 => 'r',
        2 => 'd',
        3 => 'l',
        _ => '?',
    }
}

#[cfg(unix)]
fn install_sigint_handler(shared: &Shared) {
    // Safe one-shot installer using libc::signal. The handler only flips an
    // atomic bool; safe even though it executes in a signal context.
    use std::sync::OnceLock;
    static STOP_PTR: OnceLock<usize> = OnceLock::new();
    STOP_PTR
        .set(&shared.stop as *const AtomicBool as usize)
        .ok();
    extern "C" fn handler(_sig: i32) {
        if let Some(p) = STOP_PTR.get() {
            // Safety: pointer originates from a borrow of shared.stop that
            // outlives the threads (it's owned by main()'s `shared` value).
            let stop = unsafe { &*(*p as *const AtomicBool) };
            stop.store(true, Ordering::SeqCst);
        }
    }
    // SAFETY: registering a process-wide handler. `signal` is async-signal-safe.
    unsafe {
        libc::signal(libc::SIGINT, handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handler as *const () as libc::sighandler_t);
    }
    let _ = Path::new(OUT_DIR); // keep Path import in scope on non-unix
}

#[cfg(not(unix))]
fn install_sigint_handler(_shared: &Shared) {
    // No-op fallback. Stop only via process kill.
    let _ = Path::new(OUT_DIR);
}
