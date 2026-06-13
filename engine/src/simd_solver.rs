//! SIMD-accelerated variant of `solver::Solver`.
//!
//! Same explicit step machine, same `Report`, but the per-position candidate
//! scan tests 16 (piece, rotation) rows at once with `wide::u8x16`: four
//! edge-byte SIMD compares + bitwise AND, then `to_bitmask().trailing_zeros()`
//! to pick the first matching lane.
//!
//! Native only (depends on `wide`, which won't compile to wasm32-unknown).
//! Does not support `shuffle_pieces`; iterates pieces in id order so the
//! 4-piece-per-chunk layout stays valid.

use crate::solver::{Report, Status};
use crate::types::{rotated, Color, Puzzle, BORDER};
use wide::u8x16;

const LANES: usize = 16;
/// Padding byte for edge tables past `n_pieces*4`. Never matches a real
/// color and the `valid` mask zeroes the lane anyway.
const PAD: u8 = 0xFF;

struct Frame {
    pos: u16,
    /// Next candidate row index (0..n_chunks*16) to try on re-entry.
    cursor: u32,
    placed: u32,
}

pub struct SimdSolver {
    width: usize,
    height: usize,
    n_pieces: usize,
    n_chunks: usize,
    /// Edge bytes packed per chunk of 16 rows = 4 pieces × 4 rotations.
    tops: Vec<u8x16>,
    rights: Vec<u8x16>,
    bottoms: Vec<u8x16>,
    lefts: Vec<u8x16>,
    /// 0xFF if row is a real, distinct rotation; 0x00 for duplicates / padding.
    valid: Vec<u8x16>,
    /// Per-piece availability bits (1 = used). Pieces live in id order so a
    /// chunk maps to 4 consecutive bits.
    used: Vec<u64>,
    board: Vec<i32>,
    frames: Vec<Frame>,
    depth: usize,
    hint_count: u32,
    status: Status,
    nodes: u64,
    attempts: u64,
    backtracks: u64,
    best_placed: u32,
    best_board: Vec<i32>,
}

impl SimdSolver {
    pub fn new(puzzle: &Puzzle, path: &[u16], use_hints: bool) -> Result<Self, String> {
        let n_cells = puzzle.cell_count();
        let n_pieces = puzzle.pieces.len();
        if n_pieces != n_cells {
            return Err(format!("puzzle has {n_pieces} pieces for {n_cells} cells"));
        }
        if path.len() != n_cells {
            return Err(format!("path covers {} of {n_cells} cells", path.len()));
        }
        let mut seen = vec![false; n_cells];
        for &c in path {
            if c as usize >= n_cells || seen[c as usize] {
                return Err("path is not a permutation of the cells".into());
            }
            seen[c as usize] = true;
        }

        // Pad row count to a multiple of 16 (= multiple of 4 pieces).
        let n_chunks = n_pieces.div_ceil(4);
        let padded_rows = n_chunks * LANES;

        let mut tops_flat = vec![PAD; padded_rows];
        let mut rights_flat = vec![PAD; padded_rows];
        let mut bottoms_flat = vec![PAD; padded_rows];
        let mut lefts_flat = vec![PAD; padded_rows];
        let mut valid_flat = vec![0u8; padded_rows];

        for (id, &e) in puzzle.pieces.iter().enumerate() {
            let mut rots: [[Color; 4]; 4] = [[0; 4]; 4];
            for r in 0..4u8 {
                rots[r as usize] = rotated(e, r);
            }
            for r in 0..4 {
                let row = id * 4 + r;
                let edges = rots[r];
                tops_flat[row] = edges[0];
                rights_flat[row] = edges[1];
                bottoms_flat[row] = edges[2];
                lefts_flat[row] = edges[3];
                // Mark as invalid if this rotation duplicates a lower one.
                let dup = (0..r).any(|prev| rots[prev] == edges);
                valid_flat[row] = if dup { 0 } else { 0xFF };
            }
        }

        let pack = |flat: &[u8]| -> Vec<u8x16> {
            flat.chunks_exact(LANES)
                .map(|c| u8x16::from(<[u8; LANES]>::try_from(c).unwrap()))
                .collect()
        };
        let tops = pack(&tops_flat);
        let rights = pack(&rights_flat);
        let bottoms = pack(&bottoms_flat);
        let lefts = pack(&lefts_flat);
        let valid = pack(&valid_flat);

        let mut board = vec![-1i32; n_cells];
        let mut used = vec![0u64; n_pieces.div_ceil(64)];
        let mut hint_count = 0u32;
        if use_hints {
            for h in &puzzle.hints {
                let pos = h.pos as usize;
                let pid = h.piece as usize;
                if pos >= n_cells || pid >= n_pieces || (used[pid >> 6] >> (pid & 63)) & 1 == 1 {
                    return Err(format!("invalid hint at position {pos}"));
                }
                board[pos] = (pid as i32) * 4 + i32::from(h.rot & 3);
                used[pid >> 6] |= 1u64 << (pid & 63);
                hint_count += 1;
            }
        }

        let frames = path
            .iter()
            .filter(|&&c| board[c as usize] == -1)
            .map(|&c| Frame {
                pos: c,
                cursor: 0,
                placed: u32::MAX,
            })
            .collect();

        let best_board = board.clone();
        Ok(SimdSolver {
            width: puzzle.width as usize,
            height: puzzle.height as usize,
            n_pieces,
            n_chunks,
            tops,
            rights,
            bottoms,
            lefts,
            valid,
            used,
            board,
            frames,
            depth: 0,
            hint_count,
            status: Status::Running,
            nodes: 0,
            attempts: 0,
            backtracks: 0,
            best_placed: hint_count,
            best_board,
        })
    }

    #[inline]
    fn used_get(&self, pid: usize) -> bool {
        (self.used[pid >> 6] >> (pid & 63)) & 1 == 1
    }

    /// Constraints at `pos`: returns `(byte, specific)` for top/right/bottom/left.
    /// `specific = true`  → lane matches when edge == byte.
    /// `specific = false` → lane matches when edge != BORDER (interior, neighbor empty).
    #[inline]
    fn constraints(&self, pos: usize) -> [(u8, bool); 4] {
        let (w, h) = (self.width, self.height);
        let (x, y) = (pos % w, pos / w);

        let top = if y == 0 {
            (BORDER, true)
        } else {
            let n = self.board[pos - w];
            if n >= 0 {
                (self.edge_of(n as usize, 2), true)
            } else {
                (0, false)
            }
        };
        let right = if x == w - 1 {
            (BORDER, true)
        } else {
            let n = self.board[pos + 1];
            if n >= 0 {
                (self.edge_of(n as usize, 3), true)
            } else {
                (0, false)
            }
        };
        let bottom = if y == h - 1 {
            (BORDER, true)
        } else {
            let n = self.board[pos + w];
            if n >= 0 {
                (self.edge_of(n as usize, 0), true)
            } else {
                (0, false)
            }
        };
        let left = if x == 0 {
            (BORDER, true)
        } else {
            let n = self.board[pos - 1];
            if n >= 0 {
                (self.edge_of(n as usize, 1), true)
            } else {
                (0, false)
            }
        };
        [top, right, bottom, left]
    }

    #[inline]
    fn edge_of(&self, row: usize, side: usize) -> u8 {
        // Tables are chunk-major; recover the byte directly.
        let chunk = row >> 4;
        let lane = row & 15;
        let arr = match side {
            0 => self.tops[chunk].as_array_ref(),
            1 => self.rights[chunk].as_array_ref(),
            2 => self.bottoms[chunk].as_array_ref(),
            3 => self.lefts[chunk].as_array_ref(),
            _ => unreachable!(),
        };
        arr[lane]
    }

    /// Build the 16-byte availability mask for chunk `c`: 0xFF where the
    /// corresponding piece is still free, 0 where it is already placed or
    /// out of range.
    #[inline]
    fn avail_chunk(&self, c: usize) -> u8x16 {
        let base = c * 4;
        let mut a = [0u8; LANES];
        for p in 0..4 {
            let pid = base + p;
            let free = pid < self.n_pieces && !self.used_get(pid);
            let byte = if free { 0xFF } else { 0 };
            a[p * 4..p * 4 + 4].fill(byte);
        }
        u8x16::from(a)
    }

    /// Find the next candidate row >= `cursor` that fits at `pos`.
    /// Returns the matching row (and chunks scanned for the attempts counter).
    #[inline]
    fn find_next(&self, pos: usize, cursor: usize) -> (Option<u32>, u64) {
        let [(tn, ts), (rn, rs), (bn, bs), (ln, ls)] = self.constraints(pos);
        let zero = u8x16::splat(BORDER);
        let mut chunks_scanned = 0u64;
        let mut row = cursor;
        while row < self.n_chunks * LANES {
            let c = row >> 4;
            chunks_scanned += 1;

            let tm = if ts {
                self.tops[c].cmp_eq(u8x16::splat(tn))
            } else {
                !self.tops[c].cmp_eq(zero)
            };
            let rm = if rs {
                self.rights[c].cmp_eq(u8x16::splat(rn))
            } else {
                !self.rights[c].cmp_eq(zero)
            };
            let bm = if bs {
                self.bottoms[c].cmp_eq(u8x16::splat(bn))
            } else {
                !self.bottoms[c].cmp_eq(zero)
            };
            let lm = if ls {
                self.lefts[c].cmp_eq(u8x16::splat(ln))
            } else {
                !self.lefts[c].cmp_eq(zero)
            };

            let combined = tm & rm & bm & lm & self.valid[c] & self.avail_chunk(c);
            let mut bits = combined.move_mask() as u32;

            // Mask off lanes below the cursor within this chunk.
            let offset = row & 15;
            if offset != 0 {
                bits &= !((1u32 << offset) - 1);
            }
            if bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                return (Some((c * LANES + lane) as u32), chunks_scanned);
            }
            row = (c + 1) * LANES;
        }
        (None, chunks_scanned)
    }

    /// Run up to `budget` placements/backtracks.
    pub fn step(&mut self, budget: u32) -> Report {
        let mut remaining = budget;
        while remaining > 0 && self.status == Status::Running {
            remaining -= 1;

            if self.depth == self.frames.len() {
                self.status = Status::Solved;
                self.best_placed = self.placed();
                self.best_board.copy_from_slice(&self.board);
                break;
            }

            let pos = self.frames[self.depth].pos as usize;
            let cursor = self.frames[self.depth].cursor as usize;
            let (found, chunks) = self.find_next(pos, cursor);
            // Count one "attempt" per SIMD lane evaluated, comparable to the
            // scalar solver's per-candidate counter.
            self.attempts += chunks * LANES as u64;

            if let Some(row) = found {
                let pid = (row / 4) as usize;
                self.board[pos] = row as i32;
                self.used[pid >> 6] |= 1u64 << (pid & 63);
                let f = &mut self.frames[self.depth];
                f.cursor = row + 1;
                f.placed = row;
                self.depth += 1;
                self.nodes += 1;
                let placed = self.placed();
                if placed > self.best_placed {
                    self.best_placed = placed;
                    self.best_board.copy_from_slice(&self.board);
                }
            } else {
                let f = &mut self.frames[self.depth];
                f.cursor = 0;
                if self.depth == 0 {
                    self.status = Status::Exhausted;
                    break;
                }
                self.depth -= 1;
                let prev = &mut self.frames[self.depth];
                let row = prev.placed;
                prev.placed = u32::MAX;
                self.board[prev.pos as usize] = -1;
                let pid = (row / 4) as usize;
                self.used[pid >> 6] &= !(1u64 << (pid & 63));
                self.backtracks += 1;
            }
        }
        self.report()
    }

    pub fn placed(&self) -> u32 {
        self.hint_count + self.depth as u32
    }

    pub fn report(&self) -> Report {
        Report {
            status: self.status,
            nodes: self.nodes as f64,
            attempts: self.attempts as f64,
            backtracks: self.backtracks as f64,
            placed: self.placed(),
            best_placed: self.best_placed,
        }
    }

    pub fn board(&self) -> &[i32] {
        &self.board
    }

    pub fn best_board(&self) -> &[i32] {
        &self.best_board
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::generate;
    use crate::paths::{build_path, PATH_KINDS};
    use crate::solver::score_board;

    fn solve_full(puzzle: &Puzzle, path_kind: &str) -> Report {
        let path = build_path(path_kind, puzzle.width, puzzle.height, 0).unwrap();
        let mut s = SimdSolver::new(puzzle, &path, true).unwrap();
        loop {
            let r = s.step(5_000_000);
            if r.status != Status::Running {
                return r;
            }
            assert!(r.attempts < 5e10, "runaway search in test");
        }
    }

    #[test]
    fn solves_generated_puzzles_on_all_paths() {
        for kind in PATH_KINDS {
            let p = generate(4, 4, 11);
            let r = solve_full(&p, kind);
            assert_eq!(r.status, Status::Solved, "path {kind} failed");
            assert_eq!(r.placed, 16);
        }
    }

    #[test]
    fn solved_board_has_max_score() {
        let p = generate(5, 5, 3);
        let path = build_path("snake", 5, 5, 0).unwrap();
        let mut s = SimdSolver::new(&p, &path, true).unwrap();
        while s.step(1_000_000).status == Status::Running {}
        assert_eq!(score_board(&p, s.board()), p.max_score());
    }

    #[test]
    fn matches_scalar_solver_on_small_puzzles() {
        use crate::solver::Solver;
        for seed in 0..8u32 {
            let p = generate(4, 4, seed);
            let path = build_path("row-major", 4, 4, 0).unwrap();
            let mut a = Solver::new(&p, &path, true, false, 0).unwrap();
            let mut b = SimdSolver::new(&p, &path, true).unwrap();
            while a.step(1_000_000).status == Status::Running {}
            while b.step(1_000_000).status == Status::Running {}
            // Same scalar solver and SIMD solver should both solve.
            assert_eq!(a.report().status, Status::Solved);
            assert_eq!(b.report().status, Status::Solved);
            assert_eq!(score_board(&p, a.board()), p.max_score());
            assert_eq!(score_board(&p, b.board()), p.max_score());
        }
    }
}
