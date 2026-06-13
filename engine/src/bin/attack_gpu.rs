//! GPU attack on the official Eternity II puzzle via CUDA + cuda-oxide.
//!
//! One CUDA thread = one beam (greedy forward walk, no backtracking).
//! The kernel lives in `kernels/eternity_solver.cu`; build.rs compiles it
//! to PTX with nvcc and we embed the PTX bytes via `include_bytes!`.
//!
//! Each launch fires `GRID` independent beams. Beams differ by their
//! per-beam tiebreak seed (a hash of `tid` + `run_seed`), so distinct
//! beams take distinct candidate orderings and end on distinct boards.
//! The host loop uses a fresh `run_seed` and a different path from a
//! pool every launch.
//!
//! Run with: `cargo run --release --bin attack_gpu`
//! Stop with: Ctrl-C.

use cuda_oxide::*;
use eternity2_engine::{build_path, official_puzzle, rotated, Puzzle, PATH_KINDS};
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const OUT_DIR: &str = "out";
const JSON_PATH: &str = "out/best_official_gpu.json";
const TXT_PATH: &str = "out/best_official_gpu.txt";
const LOG_PATH: &str = "out/best_official_gpu_launches.csv";
const SNAPSHOT_SECS: u64 = 30;
const TRACE_SECS: f64 = 2.0;

/// Total threads = BLOCKS * THREADS_PER_BLOCK. One thread = one beam.
/// Tune for your card; on a 4070 Laptop, 16x256 = 4096 beams runs at
/// well under a second per launch.
const BLOCKS: u32 = 16;
const THREADS_PER_BLOCK: u32 = 256;
const GRID: usize = (BLOCKS as usize) * (THREADS_PER_BLOCK as usize);
/// Pool of precomputed search paths. The host cycles through it,
/// uploading one path per launch so beams diverge across launches
/// without the kernel paying for path indirection.
const NUM_PATHS: usize = 64;

/// PTX produced by build.rs. May be a small stub if nvcc was missing at
/// build time; we detect that at runtime and bail.
static PTX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/eternity_solver.ptx"));

#[derive(Clone, Serialize)]
struct BestSnapshot {
    placed: u32,
    cells: u32,
    score: u32,
    max_score: u32,
    attempts: u64,
    nodes: u64,
    elapsed_secs: f64,
    /// CUDA thread index that recorded this best.
    thread: usize,
    path_kind: String,
    path_seed: u32,
    board: Vec<i32>,
}

fn main() {
    fs::create_dir_all(OUT_DIR).expect("create out/");

    if !ptx_looks_real(PTX_BYTES) {
        eprintln!(
            "PTX kernel not built (build.rs wrote a stub). Install the CUDA \
             toolkit so `nvcc` is on PATH (or set NVCC=/path/to/nvcc) and \
             rebuild with `cargo clean -p eternity2-engine && cargo build \
             --release --bin attack_gpu`."
        );
        std::process::exit(2);
    }

    let puzzle = official_puzzle();
    let cells = puzzle.cell_count();
    let n_pieces = puzzle.pieces.len();
    let n_candidates = n_pieces * 4;
    let used_words = n_pieces.div_ceil(64);
    let hint_count = puzzle.hints.len();
    let search_len = cells - hint_count;
    let max_score = puzzle.max_score();

    println!(
        "Official Eternity II: {}x{}, {} pieces, {} hints, max score = {max_score}",
        puzzle.width,
        puzzle.height,
        n_pieces,
        hint_count
    );
    println!(
        "GPU launch shape: {BLOCKS} blocks x {THREADS_PER_BLOCK} threads = {GRID} beams/launch"
    );
    println!("path pool = {NUM_PATHS} (one path per launch, beam-search greedy walk, no backtracking)");

    // Per-candidate (piece, rotation) edge table: 5 bytes packed
    // [top, right, bottom, left, distinct]. distinct=0 marks a rotation
    // identical to a lower-numbered rotation of the same piece.
    let mut tables: Vec<u8> = vec![0u8; n_candidates * 5];
    for pid in 0..n_pieces {
        for r in 0..4u8 {
            let row = pid * 4 + r as usize;
            let e = rotated(puzzle.pieces[pid], r);
            tables[row * 5] = e[0];
            tables[row * 5 + 1] = e[1];
            tables[row * 5 + 2] = e[2];
            tables[row * 5 + 3] = e[3];
            let mut distinct = 1u8;
            for prev in 0..r {
                if rotated(puzzle.pieces[pid], prev) == e {
                    distinct = 0;
                    break;
                }
            }
            tables[row * 5 + 4] = distinct;
        }
    }

    // Hint-prefilled board. -1 = empty. Pieces written as piece*4+rot.
    let mut init_board = vec![-1i32; cells];
    for h in &puzzle.hints {
        init_board[h.pos as usize] = (h.piece as i32) * 4 + i32::from(h.rot & 3);
    }

    // Path pool. For each kind/seed we build a path and filter out hint
    // cells, mirroring `Solver::new`'s `frames` construction. Lengths must
    // all be equal so the kernel can index `paths[idx*search_len + depth]`.
    let kinds: Vec<&'static str> = PATH_KINDS.iter().copied().collect();
    let mut paths_flat: Vec<u16> = Vec::with_capacity(NUM_PATHS * search_len);
    for i in 0..NUM_PATHS {
        let kind = kinds[i % kinds.len()];
        let seed = (i as u32).wrapping_mul(0x9E37_79B9).wrapping_add(1);
        let path = build_path(kind, puzzle.width, puzzle.height, seed)
            .or_else(|| build_path("row-major", puzzle.width, puzzle.height, 0))
            .expect("path build");
        let filtered: Vec<u16> = path
            .iter()
            .copied()
            .filter(|&c| init_board[c as usize] == -1)
            .collect();
        assert_eq!(
            filtered.len(),
            search_len,
            "path '{kind}' filtered length mismatch"
        );
        paths_flat.extend_from_slice(&filtered);
    }

    // Pack scalar args into two u32s so we stay under the 16-tuple limit
    // for `Stream::launch` parameters.
    assert!(used_words <= 0xFF, "scalar pack ranges");
    assert!(cells <= 0xFFFF && search_len <= 0xFFFF, "len pack ranges");
    let packed_dims =
        (u32::from(puzzle.width) << 16) | (u32::from(puzzle.height) << 8) | (used_words as u32);
    let packed_lens = ((cells as u32) << 16) | (search_len as u32);

    // CUDA setup. The Handle is !Send, so the driver loop runs on this
    // (main) thread; tracer/snapshotter just read shared atomics.
    Cuda::init().expect("Cuda::init");
    let devices = Cuda::list_devices().expect("list_devices");
    let device = devices.first().expect("no CUDA device found");
    println!("device: {}", device.name().unwrap_or_default());
    let mut context = Context::new(device).expect("Context::new");
    let handle = context.enter().expect("context.enter");

    // PTX needs a trailing NUL byte for cuModuleLoadData.
    let mut ptx_with_nul = Vec::with_capacity(PTX_BYTES.len() + 1);
    ptx_with_nul.extend_from_slice(PTX_BYTES);
    ptx_with_nul.push(0);
    let module = Module::load(&handle, &ptx_with_nul).expect("load PTX");
    let function = module.get_function("beam_run").expect("kernel symbol 'beam_run'");

    let d_tables = DeviceBox::new(&handle, &tables[..]).expect("upload tables");
    let d_init_board =
        DeviceBox::new(&handle, as_bytes_i32(&init_board)).expect("upload init_board");
    let d_paths = DeviceBox::new(&handle, as_bytes_u16(&paths_flat)).expect("upload paths");

    let d_beam_board = DeviceBox::alloc(&handle, (GRID * cells * 4) as u64).expect("alloc board");
    let d_beam_used =
        DeviceBox::alloc(&handle, (GRID * used_words * 8) as u64).expect("alloc beam_used");
    let d_beam_placed = DeviceBox::alloc(&handle, (GRID * 4) as u64).expect("alloc beam_placed");
    let d_beam_score = DeviceBox::alloc(&handle, (GRID * 4) as u64).expect("alloc beam_score");

    let mut stream = Stream::new(&handle).expect("Stream::new");

    let initial = BestSnapshot {
        placed: hint_count as u32,
        cells: cells as u32,
        score: 0,
        max_score,
        attempts: 0,
        nodes: 0,
        elapsed_secs: 0.0,
        thread: 0,
        path_kind: String::new(),
        path_seed: 0,
        board: init_board.clone(),
    };
    let nodes_total = AtomicU64::new(0);
    let attempts_total = AtomicU64::new(0);
    let stop = AtomicBool::new(false);
    let started = Instant::now();
    let best = Mutex::new(initial);
    // Per-launch max best_placed (total, hints included). Used for the
    // end-of-run distribution summary.
    let launch_maxes: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    // Per-launch CSV log (one row per kernel launch). Header on creation.
    let mut log_file = fs::File::create(LOG_PATH).expect("create launch log");
    writeln!(
        log_file,
        "launch_idx,elapsed_secs,launch_secs,grid,placed_min,placed_med,placed_max,nodes,nodes_per_sec,best_placed"
    )
    .expect("write log header");
    log_file.sync_all().ok();
    let log_mutex = Mutex::new(log_file);

    install_sigint_handler(&stop);

    std::thread::scope(|scope| {
        let stop_ref = &stop;
        let nodes_ref = &nodes_total;
        let attempts_ref = &attempts_total;
        let best_ref = &best;
        let puzzle_ref = &puzzle;

        scope.spawn(move || tracer(stop_ref, nodes_ref, attempts_ref, best_ref, started));
        scope.spawn(move || snapshotter(stop_ref, best_ref, puzzle_ref));

        // Driver loop on the main thread.
        let mut launch_idx: u32 = 0;
        while !stop.load(Ordering::Relaxed) {
            let launch_t0 = Instant::now();
            let path_index = (launch_idx as usize) % NUM_PATHS;
            let run_seed = launch_idx
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(0x1234_5678);
            unsafe {
                stream
                    .launch(
                        &function,
                        BLOCKS,
                        THREADS_PER_BLOCK,
                        0u32,
                        (
                            &d_tables,
                            &d_init_board,
                            &d_paths,
                            &d_beam_board,
                            &d_beam_used,
                            &d_beam_placed,
                            &d_beam_score,
                            packed_dims,
                            packed_lens,
                            n_candidates as i32,
                            path_index as u32,
                            run_seed,
                        ),
                    )
                    .expect("launch");
            }
            stream.sync().expect("stream.sync");
            let launch_secs = launch_t0.elapsed().as_secs_f64();

            let bp_bytes = d_beam_placed.load().expect("load beam_placed");
            let bs_bytes = d_beam_score.load().expect("load beam_score");
            let bp: &[u32] =
                unsafe { std::slice::from_raw_parts(bp_bytes.as_ptr() as *const u32, GRID) };
            let bs: &[u32] =
                unsafe { std::slice::from_raw_parts(bs_bytes.as_ptr() as *const u32, GRID) };

            // For beam-search, "nodes" is just the work the kernel did:
            // each beam evaluated `placed * n_candidates` candidate rows.
            let mut sum_nodes: u64 = 0;
            let mut max_idx: usize = 0;
            let mut max_val: u32 = 0;
            let mut max_score: u32 = 0;
            let mut sum_placed: u64 = 0;
            for i in 0..GRID {
                sum_nodes += (bp[i] as u64) * (n_candidates as u64);
                sum_placed += bp[i] as u64;
                // Pick winner by (placed, score) lex order so we publish a
                // beam that filled the most cells with the most matches.
                let better = bp[i] > max_val
                    || (bp[i] == max_val && bs[i] > max_score);
                if better {
                    max_val = bp[i];
                    max_score = bs[i];
                    max_idx = i;
                }
            }
            nodes_total.fetch_add(sum_nodes, Ordering::Relaxed);
            attempts_total.fetch_add(GRID as u64, Ordering::Relaxed);

            // Per-launch stats line, stats.rs-style: distribution of
            // beam placed counts across the GRID + total work this launch.
            let mut bp_sorted: Vec<u32> = bp.to_vec();
            bp_sorted.sort_unstable();
            let bp_min = bp_sorted[0];
            let bp_med = bp_sorted[GRID / 2];
            let bp_max = max_val;
            let mut bs_sorted: Vec<u32> = bs.to_vec();
            bs_sorted.sort_unstable();
            let bs_max = bs_sorted[GRID - 1];
            eprintln!(
                "  launch {launch_idx}: {GRID} beams in {launch_secs:.2}s, \
                 path={path_index} \
                 placed min/med/max = {bp_min}/{bp_med}/{bp_max} (of {cells}), \
                 score max = {bs_max}/{max_score_total}, \
                 mean placed {:.1} ({:.1}M cand-evals/s)",
                sum_placed as f64 / GRID as f64,
                sum_nodes as f64 / launch_secs.max(1e-9) / 1e6,
                max_score_total = max_score
            );
            launch_maxes.lock().unwrap().push(bp_max);

            // Persist per-launch row to out/best_official_gpu_launches.csv
            // so a second process can tail it without holding any locks.
            let elapsed = started.elapsed().as_secs_f64();
            let nps = sum_nodes as f64 / launch_secs.max(1e-9);
            let cur_best = best.lock().unwrap().placed;
            if let Ok(mut f) = log_mutex.lock() {
                let _ = writeln!(
                    f,
                    "{launch_idx},{elapsed:.3},{launch_secs:.3},{GRID},{bp_min},{bp_med},{bp_max},{sum_nodes},{nps:.1},{cur_best}"
                );
                let _ = f.flush();
            }

            // Beam kernel reports `beam_placed` as total pieces on the
            // board (hints already counted), matching attack_official's
            // BestSnapshot.placed semantics.
            let new_placed = max_val;
            let cur_placed = best.lock().unwrap().placed;
            if new_placed > cur_placed {
                let bb_bytes = d_beam_board.load().expect("load beam_board");
                let bb: &[i32] = unsafe {
                    std::slice::from_raw_parts(bb_bytes.as_ptr() as *const i32, GRID * cells)
                };
                let board: Vec<i32> = bb[max_idx * cells..(max_idx + 1) * cells].to_vec();
                let score = score_board(&puzzle, &board);

                let mut b = best.lock().unwrap();
                if new_placed > b.placed || (new_placed == b.placed && score > b.score) {
                    b.placed = new_placed;
                    b.score = score;
                    b.attempts = attempts_total.load(Ordering::Relaxed);
                    b.nodes = nodes_total.load(Ordering::Relaxed);
                    b.elapsed_secs = started.elapsed().as_secs_f64();
                    b.thread = max_idx;
                    b.path_kind = format!("path{path_index}");
                    b.path_seed = run_seed;
                    b.board = board;
                    println!(
                        "[{:>6.1}s] new best: placed={}/{} score={}/{}  (gpu beam {}, path {})",
                        b.elapsed_secs,
                        b.placed,
                        b.cells,
                        b.score,
                        b.max_score,
                        max_idx,
                        path_index,
                    );
                    if let Err(e) = write_snapshot(&puzzle, &b) {
                        eprintln!("snapshot write failed: {e}");
                    }
                }
            }

            launch_idx = launch_idx.wrapping_add(1);
        }
    });

    println!("\nstopped. final best:");
    let b = best.lock().unwrap();
    println!(
        "  placed = {}/{}  score = {}/{}  attempts = {}  nodes = {}",
        b.placed,
        b.cells,
        b.score,
        b.max_score,
        attempts_total.load(Ordering::Relaxed),
        nodes_total.load(Ordering::Relaxed),
    );

    // Per-launch best_placed distribution, stats.rs-style.
    let maxes = launch_maxes.lock().unwrap();
    if !maxes.is_empty() {
        let mut sorted = maxes.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let med = sorted[n / 2];
        let mn = sorted[0];
        let mx = sorted[n - 1];
        let mean: f64 = sorted.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        println!(
            "  launches = {n}  per-launch best_placed min/median/mean/max = {mn}/{med}/{mean:.1}/{mx} (of {})",
            b.cells
        );
    }
}

fn ptx_looks_real(bytes: &[u8]) -> bool {
    // nvcc-emitted PTX always contains a `.version` directive near the top.
    bytes.windows(8).any(|w| w == b".version")
}

fn as_bytes_i32(v: &[i32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn as_bytes_u16(v: &[u16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn tracer(
    stop: &AtomicBool,
    nodes_total: &AtomicU64,
    attempts_total: &AtomicU64,
    best: &Mutex<BestSnapshot>,
    started: Instant,
) {
    let mut last_t = Instant::now();
    let mut last_nodes = 0u64;
    let mut last_attempts = 0u64;
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(200));
        let dt = last_t.elapsed().as_secs_f64();
        if dt < TRACE_SECS {
            continue;
        }
        let now_nodes = nodes_total.load(Ordering::Relaxed);
        let now_attempts = attempts_total.load(Ordering::Relaxed);
        let dn = now_nodes - last_nodes;
        let da = now_attempts - last_attempts;
        let elapsed = started.elapsed().as_secs_f64().max(1e-9);
        let cur = dn as f64 / dt / 1e6;
        let life = now_nodes as f64 / elapsed / 1e6;
        let best_placed = best.lock().unwrap().placed;
        println!(
            "[{:>6.1}s] {:>6.1}M nodes/s (cur)  {:>6.1}M nodes/s (avg)  attempts +{da} (total {now_attempts})  best placed={best_placed}",
            elapsed, cur, life,
        );
        last_t = Instant::now();
        last_nodes = now_nodes;
        last_attempts = now_attempts;
    }
}

fn snapshotter(stop: &AtomicBool, best: &Mutex<BestSnapshot>, puzzle: &Puzzle) {
    let mut last = Instant::now();
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(500));
        if last.elapsed() >= Duration::from_secs(SNAPSHOT_SECS) {
            last = Instant::now();
            let b = best.lock().unwrap().clone();
            let _ = write_snapshot(puzzle, &b);
        }
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

fn score_board(puzzle: &Puzzle, board: &[i32]) -> u32 {
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
        "Eternity II GPU — best partial\nplaced {}/{}  score {}/{}  elapsed {:.1}s\nthread {}  path {} (seed {})  attempts {}  nodes {}\n\n",
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
fn install_sigint_handler(stop: &AtomicBool) {
    use std::sync::OnceLock;
    static STOP_PTR: OnceLock<usize> = OnceLock::new();
    STOP_PTR.set(stop as *const AtomicBool as usize).ok();
    extern "C" fn handler(_sig: i32) {
        if let Some(p) = STOP_PTR.get() {
            let stop = unsafe { &*(*p as *const AtomicBool) };
            stop.store(true, Ordering::SeqCst);
        }
    }
    unsafe {
        libc::signal(libc::SIGINT, handler as *const () as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handler as *const () as libc::sighandler_t);
    }
    let _ = Path::new(OUT_DIR);
}

#[cfg(not(unix))]
fn install_sigint_handler(_stop: &AtomicBool) {
    let _ = Path::new(OUT_DIR);
}
