cmake_minimum_required(VERSION 3.10)
project(islands CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(islands_core islands.cpp)

add_executable(islands main.cpp)
target_link_libraries(islands PRIVATE islands_core)

add_executable(test_islands test_islands.cpp)
target_link_libraries(test_islands PRIVATE islands_core)
