// function to create an empty board with the given size and fill it with empty pieces
// a board is a 2D array of pieces

// function to create an empty board with the given size and fill it with empty pieces
// a board is a 2D array of pieces

#ifndef ETERNITY2_BOARD_H
#define ETERNITY2_BOARD_H

#include "../piece_search/piece_search.h"

#include <mutex>
#include <unordered_set>
#include <vector>
int const END_PATH = 2147483647;

struct Board
{
    std::vector<RotatedPiece> board;
    size_t size;
    std::vector<int> next_index_cache;
};

using BoardHash = std::string;
using Index     = std::pair<size_t, size_t>;

struct Neighbor
{
    const RotatedPiece *up;
    const RotatedPiece *right;
    const RotatedPiece *down;
    const RotatedPiece *left;
};

Board create_board(int size);

std::vector<std::string> board_to_string(const Board &board);

void place_piece(Board &board, const RotatedPiece &piece, Index index);

void remove_piece(Board &board, Index index);
auto get_next_scan_row(const Board &board, Index index) -> Index;

Index get_next(const Board &board, Index index);

auto get_next_using_cache(const Board &board, Index index) -> Index;

std::string index_to_string(Index index);

bool is_end(const Board &board, Index index);

const RotatedPiece *get_piece(const Board &board, Index index);

Neighbor get_neighbors(const Board &board, Index index);

void log_board(const Board &board, const std::string &description);

void export_board(const Board &board);

std::string export_board_to_csv_string(const Board &board);
size_t get_1d_board_index(const Board &board, Index index);
Index get_2d_board_index(const Board &board, int index);
#endif //ETERNITY2_BOARD_H