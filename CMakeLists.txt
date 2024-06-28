cmake_minimum_required(VERSION 2.8)
project("Image Edge Detection Using Filters & MultiThread")

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(run main.cpp)
target_link_libraries(run ${OpenCV_LIBS})

