cmake_minimum_required(VERSION 2.6)
project(rg_ising)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost 1.40 COMPONENTS program_options REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

add_executable(run main.cpp)
target_link_libraries(run LINK_PUBLIC ${Boost_LIBRARIES})

add_executable(run-tests test.cpp)

add_executable(integral integral.cpp)
