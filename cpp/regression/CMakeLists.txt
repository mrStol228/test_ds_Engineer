cmake_minimum_required(VERSION 3.10)
project(regression CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(regression_core matrix.cpp csv.cpp common.cpp linear_regression.cpp)

add_executable(train train.cpp)
target_link_libraries(train PRIVATE regression_core)

add_executable(predict predict.cpp)
target_link_libraries(predict PRIVATE regression_core)

add_executable(test_regression test_regression.cpp)
target_link_libraries(test_regression PRIVATE regression_core)
