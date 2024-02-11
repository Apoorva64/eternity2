//
// Created by appad on 10/02/2024.
//


#include "board.h"
#include "spdlog/spdlog.h"
Board create_board(int size) {
    // function to create an empty board with the given size and fill it with empty pieces
    // a board is a 2D array of pieces
    Board board(size, std::vector<RotatedPiece>(size, RotatedPiece()));
    return board;
}

void place_piece(Board &board, const RotatedPiece &piece, size_t x, size_t y) {
    // function to place a piece on the board at the given position
    // the piece is rotated and placed on the board
    board[y][x] = piece;
}

void remove_piece(Board &board, int x, int y) {
    // function to remove a piece from the board at the given position
    // the piece is replaced with an empty piece
    board[y][x] = RotatedPiece();
}

std::vector<RotatedPiece> possible_pieces(const Board &board, const std::vector<PIECE> &pieces, size_t x, size_t y) {
    // function to get the possible pieces that can be placed at the given position on the board
    // a piece is possible if it does not conflict with the pieces already on the board
    std::vector<RotatedPiece> possible;
    PIECE_PART top;
    PIECE_PART right;
    PIECE_PART bottom;
    PIECE_PART left;
    PIECE mask = EMPTY;

    if (x + 1 < board.size()) {
        right = get_piece_part(apply_rotation(board[y][x + 1]), LEFT_MASK);
        if (right != EMPTY) {
            mask |= RIGHT_MASK;
        }
    } else {
        right = WALL;
        mask |= RIGHT_MASK;
    }
    if (x >= 1) {
        left = get_piece_part(apply_rotation(board[y][x - 1]), RIGHT_MASK);
        if (left != EMPTY) {
            mask |= LEFT_MASK;
        }
    } else {
        left = WALL;
        mask |= LEFT_MASK;
    }

    if (y + 1 < board.size()) {
        bottom = get_piece_part(apply_rotation(board[y + 1][x]), UP_MASK);
        if (bottom != EMPTY) {
            mask |= DOWN_MASK;
        }
    } else {
        bottom = WALL;
        mask |= DOWN_MASK;
    }

    if (y >= 1) {
        top = get_piece_part(apply_rotation(board[y - 1][x]), DOWN_MASK);
        if (top != EMPTY) {
            mask |= UP_MASK;
        }
    } else {
        top = WALL;
        mask |= UP_MASK;
    }

    PIECE piece = make_piece(top, right, bottom, left);
    log_piece(piece, "Query piece");
    const auto query = Query{
        .piece = piece,
        .mask = mask,
        .type = QueryType::POSITIVE
    };
    possible = match_piece_mask({query}, pieces);
    return possible;
}


std::vector<std::string> board_to_string(const Board &board) {
    // function to print the board to the console
    std::vector<std::string> board_lines;
    for (const auto &row: board) {
        std::vector<std::string> row_lines = piece_to_string(rotate_piece_right(row[0].piece, row[0].rotation));
        for (size_t i = 1; i < row.size(); i++) {
            const auto &piece = row[i].piece;
            const auto &piece_lines = piece_to_string(rotate_piece_right(piece, row[i].rotation));
            for (size_t j = 0; j < row_lines.size(); j++) {
                row_lines[j] += piece_lines[j];
            }
        }
        board_lines.insert(board_lines.end(), row_lines.begin(), row_lines.end());
    }
    return board_lines;
}

void log_board(const Board &board, const std::string &description) {
#if SPDLOG_ACTIVE_LEVEL != SPDLOG_LEVEL_OFF
    // function to log the board to the console
    spdlog::info(description);
    for (auto const &line: board_to_string(board)) {
        spdlog::info(line);
    }
#endif
}

bool solve_board_recursive(Board &board, std::vector<PIECE> &pieces, size_t x, size_t y) {
    // function to solve the board recursively
    // the function tries to place a piece at the given position and then calls itself for the next position
    // if the board is solved, the function returns true
    log_board(board, fmt::format("Solving board at x: {}, y: {}", x, y));
    if (y == board.size()) {
        return true;
    }

    if (x == board.size()) {
        return solve_board_recursive(board, pieces, 0, y + 1);
    }

    if (board[y][x].piece != EMPTY) {
        return solve_board_recursive(board, pieces, x + 1, y);
    }

    for (auto possible = possible_pieces(board, pieces, x, y); auto const &rotated_piece: possible) {
        place_piece(board, rotated_piece, x, y);
        // remove placed piece from pieces
        long index = std::distance(pieces.begin(), std::find(pieces.begin(), pieces.end(), rotated_piece.piece));
        pieces.erase(pieces.begin() + index);

        if (solve_board_recursive(board, pieces, x + 1, y)) {
            return true;
        }
        // add placed piece back to pieces
        pieces.push_back(rotated_piece.piece);
        spdlog::info("Backtracking");
        remove_piece(board, x, y);
    }

    return false;

}

void solve_board(Board &board, std::vector<PIECE> &pieces) {
    // function to solve the board
    // the function calls the recursive function to solve the board
    solve_board_recursive(board, pieces, 0, 0);
}
