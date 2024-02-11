add_executable(tests
        test_piece.cpp
        test_piece_search.cpp
        test_board.cpp
)

find_package(Catch2 CONFIG REQUIRED)

target_link_libraries(tests  Catch2::Catch2 Catch2::Catch2WithMain solver_lib)
include(CTest)
include(Catch)
catch_discover_tests(tests)
