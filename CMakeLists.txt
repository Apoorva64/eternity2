cmake_minimum_required(VERSION 3.22)
project(eternity2)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")

add_subdirectory(tests)
add_subdirectory(solver)
