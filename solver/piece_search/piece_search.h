//
// Created by appad on 10/02/2024.
//

#ifndef ETERNITY2_PIECE_SEARCH_H
#define ETERNITY2_PIECE_SEARCH_H

#include <vector>
#include "../piece/piece.h"

struct RotatedPiece {
    PIECE piece;
    int rotation;
    size_t index;
};
enum class QueryType {
    POSITIVE,
    NEGATIVE
};
struct Query {
    PIECE piece;
    PIECE mask;
    QueryType type;
};

struct PieceWAvailability {
    PIECE piece;
    bool available;
};

bool match_piece_mask_internal(PIECE piece_data, PIECE piece_mask, PIECE rotated_piece);

std::vector<RotatedPiece> match_piece_mask(const std::vector<Query> &query, const std::vector<PieceWAvailability> &piecesWAvability);

std::vector<PieceWAvailability> create_pieces_with_availability(const std::vector<PIECE> &pieces);
PIECE apply_rotation(RotatedPiece piece);
std::string csv_piece(RotatedPiece piece);

#endif //ETERNITY2_PIECE_SEARCH_H
