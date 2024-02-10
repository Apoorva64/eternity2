// function to create an empty board with the given size and fill it with empty pieces
// a board is a 2D array of pieces

// function to create an empty board with the given size and fill it with empty pieces
// a board is a 2D array of pieces


#ifndef ETERNITY2_BOARD_H
#define ETERNITY2_BOARD_H

#include <vector>
#include "../piece_search/piece_search.h"

using Board = std::vector<std::vector<RotatedPiece>>;

Board create_board(int size);
std::vector<std::string> board_to_string(const Board &board);
void place_piece(Board board, const RotatedPiece &piece, int x, int y);
void remove_piece(Board board, int x, int y);
std::vector<RotatedPiece> possible_pieces(const Board &board, const std::vector<PIECE> &pieces, int x, int y);
void log_board(const Board &board, const std::string &description);
#endif //ETERNITY2_BOARD_H