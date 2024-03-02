#include "board/board.h"

#include <atomic>
#include <functional>
#include <vector>
struct SharedData
{
    Board &max_board;
    int &max_count;
    std::mutex &mutex;
    std::unordered_set<BoardHash> &hashes;
    std::atomic_llong board_count{0};
    std::atomic_llong hash_hit_count{0};
    std::atomic_bool stop{false};
    // callback that is called when you enter the recursive function, takes in a Board
    std::function<void(const Board &)> on_board_update = [](const Board &board) {};
};

SharedData create_shared_data();

std::vector<RotatedPiece> possible_pieces(const Board &board,
                                          const std::vector<PieceWAvailability> &pieces,
                                          Index index);

void solve_board(Board &board, const std::vector<Piece> &pieces, SharedData &shared_data);