cmake_minimum_required(VERSION 3.20)
project(c_rt_itctp_decoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -fopenmp-simd -funroll-loops -fvectorize -fslp-vectorize")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libswscale)

# Shared library for Python ctypes
add_library(c_rt_itctp_decoder SHARED main.cpp)
target_compile_definitions(c_rt_itctp_decoder PRIVATE AS_LIB)
target_link_libraries(c_rt_itctp_decoder PRIVATE PkgConfig::FFMPEG)
