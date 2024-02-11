//
// Created by appad on 10/02/2024.
//

#ifndef ETERNITY2_PIECE_LOADER_H
#define ETERNITY2_PIECE_LOADER_H

#include <vector>
#include "../piece/piece.h"
#include "fast-cpp-csv-parser/csv.h"

std::vector<PIECE> load_from_csv(const std::string& filename);


#endif //ETERNITY2_PIECE_LOADER_H
