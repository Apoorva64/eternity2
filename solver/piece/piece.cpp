//
// Created by appad on 10/02/2024.
//

#include "piece.h"
#include "spdlog/spdlog.h"

PIECE make_piece(PIECE_PART top, PIECE_PART right, PIECE_PART down, PIECE_PART left) {
    // concatenate the 4 parts into left single 64-bit integer
    // look at the MASKs to understand the order of the parts
    return (PIECE) left | ((PIECE) down << 16) | ((PIECE) right << 32) | ((PIECE) top << 48);
}

PIECE rotate_piece_right(PIECE piece, size_t n) {
    // rotate the piece n times
    PIECE rotated_piece = piece >> (16 * n);
    // add the bits that were shifted out to the beginning
    rotated_piece |= piece << (64 - 16 * n);
    return rotated_piece;
}

PIECE rotate_piece_left(PIECE piece, size_t n) {
    // rotate the piece n times
    PIECE rotated_piece = piece << (16 * n);
    // add the bits that were shifted out to the end
    rotated_piece |= piece >> (64 - 16 * n);
    return rotated_piece;
}


std::vector<std::string> piece_to_string(PIECE piece) {
    std::bitset<64> bits(piece);

    // print ascii art of the piece to the console with 4 corners
    // --------
    // |            piece[0]              |
    // |  piece[3]              piece[1]  |
    // |            piece[2]              |
    // --------

    const std::string s = bits.to_string();
    const std::string s0 = s.substr(0, 16);
    const std::string s1 = s.substr(16, 16);
    const std::string s2 = s.substr(32, 16);
    const std::string s3 = s.substr(48, 16);
    return {
            fmt::format("-----------------------------------------------------"),
            fmt::format("|                 {}                  |", s0),
            fmt::format("|  {}               {}  |", s3, s1),
            fmt::format("|                 {}                  |", s2),
            fmt::format("-----------------------------------------------------")
    };
}

void log_piece(PIECE piece, const std::string &description) {
#if SPDLOG_ACTIVE_LEVEL != SPDLOG_LEVEL_OFF
    spdlog::info(description);
    for (auto const &line: piece_to_string(piece)) {
        spdlog::info(line);
    }
#endif

}

PIECE_PART get_piece_part(PIECE piece, PIECE mask) {
    // get the part of the piece
    // use the MASKs to extract the part from the piece and shift it to the right position
    return (PIECE_PART) ((piece & mask)
            >> (mask == UP_MASK ? 48 : mask == RIGHT_MASK ? 32 : (mask == DOWN_MASK ? 16 : 0)));
}

