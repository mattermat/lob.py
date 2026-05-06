cmake_minimum_required(VERSION 3.16)
project(get_features VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build Carquet as a static library
set(CARQUET_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(CARQUET_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CARQUET_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CARQUET_BUILD_DEV OFF CACHE BOOL "" FORCE)
add_subdirectory(carquet)

# Our core feature library
add_library(lobcore STATIC
    lobcore.c
)
target_include_directories(lobcore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lobcore PRIVATE m)  # for math functions like exp, log, isnan

# Main executable
add_executable(get_features main.c)
target_include_directories(get_features PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/carquet/include
)
target_link_libraries(get_features PRIVATE
    lobcore
    carquet
    m
)
