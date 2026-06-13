# Scan-vectorized SIMD solver

A drop-in faster variant of the scalar [`Solver`](../src/solver.rs) that
vectorizes the candidate scan inside a single search using `wide::u8x16` —
16 (piece, rotation) rows tested per inner-loop step instead of one.

Lives in [src/simd_solver.rs](../src/simd_solver.rs). Native-only because the
`wide` crate doesn't build for `wasm32-unknown-unknown`. Benchmark binary:
[src/bin/bench_simd.rs](../src/bin/bench_simd.rs).

## Same shape as the scalar solver

`SimdSolver` is structurally identical to `Solver`:

- Explicit step machine driven by `step(budget)`, not recursion.
- One board, one `used` bitset, one frame stack, one search.
- Same `Report` / `Status` types reused from the scalar module.
- Same DFS semantics — placing on success, advancing the cursor on
  failure, backtracking when the cursor runs out at a depth.
- Same hint handling, same path validation, same `best_board` tracking.

What changes is the inner loop: instead of testing one candidate row per
iteration, it tests 16, fused into one SIMD AND chain.

## Memory layout

Pieces are iterated in **id order** (no `shuffle_pieces` — the chunk
layout would break otherwise). The edge tables are padded up to a multiple
of 16 rows and stored as `Vec<u8x16>`, one entry per 16-row chunk = exactly
**4 pieces × 4 rotations**:

```
tops    : Vec<u8x16>    // tops[c]    = top edges of rows 16c .. 16c+15
rights  : Vec<u8x16>
bottoms : Vec<u8x16>
lefts   : Vec<u8x16>
valid   : Vec<u8x16>    // 0xFF where the row is a distinct rotation, 0 elsewhere
```

A row's piece id is `row / 4` and its rotation is `row & 3`. The four
piece ids in chunk `c` are `4c, 4c+1, 4c+2, 4c+3`. Padding past
`n_pieces*4` is filled with `PAD = 0xFF`; the `valid` mask is 0 in those
lanes so they never produce a hit.

The `valid` mask also zeros out *duplicate* rotations of the same piece
(e.g. a piece with two equal opposite edges has two identical rotations;
we only want to explore one of them). Same trick the scalar solver uses,
just baked into the SIMD mask instead of a separate `distinct` array.

## Constraint encoding

For a position with neighbors already placed (or borders), each side has a
**specific** color requirement — "must equal X". For a position whose
neighbor is still empty (but interior), the only constraint is "this side
cannot be `BORDER`" (no inward-facing border edges in the interior). The
encoding handles both cases as `(byte, specific: bool)`:

```rust
let mask = if specific {
    chunk.cmp_eq(splat(byte))           // edge == byte
} else {
    !chunk.cmp_eq(splat(BORDER))        // edge != BORDER
};
```

The `if` is hoisted outside the SIMD work, so the inner loop is
branch-free per side. With four sides this is four `cmp_eq`s, four `splat`s,
and the side-select branch fires four times per chunk — cheap.

## Availability mask

`avail_chunk(c)` is the only piece-state read on the hot path. Four piece
ids per chunk × 4 lanes each (one lane per rotation) = a `u8x16` where each
4-lane block is either all-`0xFF` (piece free) or all-`0` (piece placed).
Construction is a 4-iteration scalar loop over the bitset, then
`u8x16::from(arr)`. Cheap compared to the SIMD compares.

## The inner loop

```rust
let combined = tops_cmp & rights_cmp & bottoms_cmp & lefts_cmp
             & valid[c] & avail_chunk(c);
let mut bits = combined.move_mask() as u32;     // 16 bits, one per lane

if cursor lies inside this chunk {
    bits &= !((1u32 << offset) - 1);            // mask off lanes already tried
}
if bits != 0 {
    let lane = bits.trailing_zeros() as usize;
    return Some(c * 16 + lane);                 // first hit in scan order
}
```

`move_mask` is the standard SSE2 `pmovmskb`-style reduction. Combined with
`trailing_zeros`, we get the first matching candidate row in two
instructions after the AND chain — no per-lane branches.

The cursor offset trick is what makes the scan resumable after a backtrack:
when we re-enter a frame, `cursor` is the row we want to start from, and
the low bits below `cursor & 15` get masked out so we don't re-place
earlier rows in the same chunk.

## Where the speedup comes from

The scalar solver does roughly one branch + four byte loads + four
compares + four ANDs **per candidate**. The SIMD solver does the same work
**per 16 candidates**: same number of vector instructions, ~16x less
loop overhead, far fewer mispredicts (the inner loop is essentially
straight-line). On the bench (6×6 stream, snake path) the measured
single-thread speedup is roughly **x4** end-to-end, which is what you'd
expect once the cost of `find_next`'s setup (constraint extraction,
avail-mask build) is added back.

The win shrinks on puzzles whose first-match-per-position is consistently
in the first chunk anyway (then SIMD overhead is bigger than the scan
savings), and grows on puzzles where the scan walks many chunks before a
hit — which is exactly the harder cases you care about.

## Counter accounting

`attempts` is incremented by `chunks_scanned * LANES` (= 16) per call to
`find_next`. That's the SIMD-accurate analog of the scalar solver's
"per-candidate attempt" — one attempt = one byte-compare against a piece
rotation, scaled so the numbers compare apples-to-apples across solvers in
the bench output.

## API summary

```rust
let mut s = SimdSolver::new(&puzzle, &path, use_hints)?;
loop {
    let r = s.step(1_000_000);
    match r.status {
        Status::Running   => continue,
        Status::Solved    => break,
        Status::Exhausted => break,
    }
}
let solution: &[i32] = s.board();      // -1 where empty, else (piece*4 + rot)
```

Constraints on inputs:

- `path` must be a permutation of `0..puzzle.cell_count()` (verified).
- `puzzle.pieces.len()` must equal `puzzle.cell_count()` (verified).
- Hints are honored when `use_hints == true`; they pre-place pieces and
  are skipped during the DFS.

`SimdSolver` does **not** implement `shuffle_pieces` or a seed parameter —
the chunk layout depends on iterating pieces in id order. Shuffle support
would require reordering the packed edge tables per shuffle, which defeats
the precomputation; if you need it, run the scalar `Solver` instead.

## What was tried and dropped

- **`u8x32` / AVX2 chunks of 32.** `wide` exposes it but only AVX2
  microarchitectures see the gain, and the bench machine isn't always one.
  Kept `u8x16` for portability; the loop generalizes trivially if you want
  to swap it.
- **Caching the constraints across cursor resumes.** The scalar solver
  already amortizes constraint extraction within a frame entry, and so
  does `SimdSolver` via `find_next` being called once per re-entry to a
  frame. Adding an explicit cache like `LockstepSolver` does was measured
  as net-neutral here — single-search lockstep doesn't pay a divergence
  tax, so the SIMD inner loop is already the bottleneck.
