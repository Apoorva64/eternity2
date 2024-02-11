#include <iostream>
#include <bitset>
#include "fast-cpp-csv-parser/csv.h"


#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_OFF
#include "piece/piece.h"
#include "piece_loader/piece_loader.h"
#include "board/board.h"
#include "spdlog/spdlog.h"


int main() {
    // disable logging
    spdlog::set_level(spdlog::level::off);
    auto start = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < 1000; i++) {
        auto board = Board();
        board = create_board(3);
        std::vector<PIECE> pieces = {
                rotate_piece_right(make_piece(WALL, 2, 1, WALL), 1),
                rotate_piece_right(make_piece(WALL, 5, 1, 2), 2),
                rotate_piece_right(make_piece(WALL, WALL, 1, 5), 3),

                rotate_piece_right(make_piece(1, 2, 3, WALL), 1),
                rotate_piece_right(make_piece(1, 3, 1, 2), 0),
                rotate_piece_right(make_piece(1, WALL, 1, 3), 1),

                rotate_piece_right(make_piece(3, 4, WALL, WALL), 2),
                rotate_piece_right(make_piece(1, 1, WALL, 4), 1),
                rotate_piece_right(make_piece(1, WALL, WALL, 1), 1),


        };

        solve_board(board, pieces);
//    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
//    log_board(board, fmt::format("3x3 board solved in {} seconds", elapsed.count()));
    std::cout << "3x3 board solved in " << elapsed.count() << " seconds" << std::endl;
}
