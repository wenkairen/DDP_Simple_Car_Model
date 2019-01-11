set (CMAKE_CXX_STANDARD 11)
cmake_minimum_required(	VERSION 2.8)

project( main )

add_library( main_shared SHARED hybrid_breadth_first.cpp)
add_executable( main main.cpp)
target_link_libraries( main main_shared)
