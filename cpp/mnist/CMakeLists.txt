cmake_minimum_required(VERSION 3.10)
project(mnist CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(mnist_core
    tensor3d.cpp
    layers.cpp
    cnn_model.cpp
    decision_tree.cpp
    random_forest_model.cpp
    random_model.cpp
    digit_classifier.cpp
)

add_executable(demo main.cpp)
target_link_libraries(demo PRIVATE mnist_core)

add_executable(test_mnist test_mnist.cpp)
target_link_libraries(test_mnist PRIVATE mnist_core)
