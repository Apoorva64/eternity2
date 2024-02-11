cmake_minimum_required(VERSION 3.22)

add_library(solver_lib STATIC
        piece/piece.cpp
        piece/piece.h
        piece_loader/piece_loader.cpp
        piece_loader/piece_loader.h
        piece_search/piece_search.cpp
        piece_search/piece_search.h
        board/board.cpp
        board/board.h)

set_target_properties(solver_lib PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(solver_lib PUBLIC cxx_std_20)

target_include_directories(solver_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(spdlog CONFIG REQUIRED)
target_link_libraries(solver_lib PUBLIC spdlog::spdlog)

find_path(FAST_CPP_CSV_PARSER_INCLUDE_DIRS "fast-cpp-csv-parser/csv.h")
target_include_directories(solver_lib PUBLIC ${FAST_CPP_CSV_PARSER_INCLUDE_DIRS})

add_executable(solver main.cpp)
target_link_libraries(solver PRIVATE solver_lib)

find_package(CLI11 CONFIG REQUIRED)
target_link_libraries(solver PRIVATE CLI11::CLI11)


