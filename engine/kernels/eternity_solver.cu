// Eternity II GPU beam-search kernel.
//
// One CUDA thread = one beam. Each beam runs a forward greedy walk along
// `path` with NO backtracking: at every depth it places the candidate
// whose 4-edge constraint set (a) is satisfied and (b) maximizes the
// matched-edge count with already-placed neighbors. Ties are broken by
// a per-beam hash so distinct beams diverge.
//
// Why beam (greedy-forward) on GPU: backtracking DFS is the wrong shape
// for SIMT. Warps run at the speed of their slowest lane and any beam
// that bails out into a deep backtrack stalls 31 healthy lanes. A forward
// walk is uniform: every active lane evaluates exactly `n_candidates`
// rows per step, so the warp rolls forward at full width.
//
// Width-1 means each beam is just an aggressive greedy walk. The
// "search" budget is the diversity across `GRID` beams (`run_seed`
// permutes tie-break order per beam). Width > 1 (true beam pruning)
// would require a separate selection kernel that keeps the global top-B
// across all `B*K` extensions; not implemented here.
//
// Edge convention matches types.rs: Color 0 == BORDER. For each side:
//   * `must == 1` : edge byte must equal `pinned`
//   * `must == 0` : edge byte must NOT equal 0 (interior, neighbor empty)
//
// Compile: nvcc -ptx -arch=compute_75 (build.rs handles this).

#include <stdint.h>

extern "C" __global__ void beam_run(
    const unsigned char*  tables,        // 5 * n_candidates: [t,r,b,l,distinct]
    const int*            init_board,    // [cells]   hint board, -1 = empty
    const unsigned short* paths,         // [num_paths * search_len]
    int*                  beam_board,    // [GRID * cells]   final per-beam board
    unsigned long long*   beam_used,     // [GRID * used_words] per-beam scratch
    unsigned int*         beam_placed,   // [GRID]   final piece count per beam
    unsigned int*         beam_score,    // [GRID]   final matched-edge score per beam
    unsigned int          packed_dims,   // (width<<16)|(height<<8)|used_words
    unsigned int          packed_lens,   // (cells<<16)|search_len
    int                   n_candidates,
    unsigned int          path_index,    // which row of `paths` this launch uses
    unsigned int          run_seed
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int width      = (int)((packed_dims >> 16) & 0xFFu);
    const int height     = (int)((packed_dims >>  8) & 0xFFu);
    const int used_words = (int)( packed_dims        & 0xFFu);
    const int cells      = (int)((packed_lens >> 16) & 0xFFFFu);
    const int search_len = (int)( packed_lens        & 0xFFFFu);

    int*                my_board = beam_board + (size_t)tid * cells;
    unsigned long long* my_used  = beam_used  + (size_t)tid * used_words;

    const unsigned short* path = paths + (size_t)path_index * search_len;

    // Init from hints. used bitset is rebuilt from init_board so the host
    // does not have to maintain a parallel buffer.
    unsigned int placed_count = 0;
    for (int i = 0; i < used_words; i++) my_used[i] = 0ULL;
    for (int i = 0; i < cells; i++) {
        int v = init_board[i];
        my_board[i] = v;
        if (v >= 0) {
            int pid = v >> 2;
            my_used[pid >> 6] |= (1ULL << (pid & 63));
            placed_count++;
        }
    }

    // Per-beam tiebreak seed. Diverges across `tid` and across runs.
    unsigned int beam_seed = run_seed ^ ((unsigned int)tid * 0x9E3779B9u);

    unsigned int total_score = 0;

    for (int d = 0; d < search_len; d++) {
        int pos = path[d];
        int x = pos % width;
        int y = pos / width;

        // Resolve the 4 neighbor constraints exactly as Solver::fits does.
        unsigned char tn = 0, rn = 0, bn = 0, ln = 0;
        int           tm = 0, rm = 0, bm = 0, lm = 0;

        if (y == 0) { tm = 1; }
        else {
            int n = my_board[pos - width];
            if (n >= 0) { tn = tables[5 * n + 2]; tm = 1; }
        }
        if (x == width - 1) { rm = 1; }
        else {
            int n = my_board[pos + 1];
            if (n >= 0) { rn = tables[5 * n + 3]; rm = 1; }
        }
        if (y == height - 1) { bm = 1; }
        else {
            int n = my_board[pos + width];
            if (n >= 0) { bn = tables[5 * n + 0]; bm = 1; }
        }
        if (x == 0) { lm = 1; }
        else {
            int n = my_board[pos - 1];
            if (n >= 0) { ln = tables[5 * n + 1]; lm = 1; }
        }

        // One linear pass over candidates, tracking (best_score, best_row).
        // Score = number of matched non-border neighbor edges (max 4).
        int          best_score = -1;
        unsigned int best_row   = 0xFFFFFFFFu;
        unsigned int best_tag   = 0u;

        for (int row = 0; row < n_candidates; row++) {
            if (tables[5 * row + 4] == 0) continue;            // duplicate rotation
            int pid = row >> 2;
            if ((my_used[pid >> 6] >> (pid & 63)) & 1ULL) continue;

            unsigned char te = tables[5 * row + 0];
            unsigned char re = tables[5 * row + 1];
            unsigned char be = tables[5 * row + 2];
            unsigned char le = tables[5 * row + 3];

            int score = 0;
            if (tm) { if (te != tn) continue; if (tn != 0) score++; }
            else    { if (te == 0)  continue; }
            if (rm) { if (re != rn) continue; if (rn != 0) score++; }
            else    { if (re == 0)  continue; }
            if (bm) { if (be != bn) continue; if (bn != 0) score++; }
            else    { if (be == 0)  continue; }
            if (lm) { if (le != ln) continue; if (ln != 0) score++; }
            else    { if (le == 0)  continue; }

            // Random tiebreak: hash row*phi ^ beam_seed; higher tag wins.
            unsigned int tag = ((unsigned int)row * 0x9E3779B9u) ^ beam_seed;
            if (score > best_score || (score == best_score && tag > best_tag)) {
                best_score = score;
                best_row   = (unsigned int)row;
                best_tag   = tag;
            }
        }

        if (best_row == 0xFFFFFFFFu) break;  // beam dead — nothing fits here

        my_board[pos] = (int)best_row;
        int pid = (int)(best_row >> 2);
        my_used[pid >> 6] |= 1ULL << (pid & 63);
        placed_count++;
        total_score += (unsigned int)best_score;

        // Advance the seed each step so different depths hash differently.
        beam_seed = beam_seed * 1664525u + 1013904223u;
    }

    beam_placed[tid] = placed_count;
    beam_score[tid]  = total_score;
}
