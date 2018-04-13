cmake_minimum_required(VERSION 2.6)
project(rg_ising)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror -Wpedantic")

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)

include_directories(${Boost_INCLUDE_DIR} "/usr/include/eigen3")

add_executable(run main.cpp)
target_link_libraries(run LINK_PUBLIC ${Boost_LIBRARIES})

add_executable(run-tests test.cpp)

add_executable(integral integral.cpp)
