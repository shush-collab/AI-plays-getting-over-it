cmake_minimum_required(VERSION 3.22)
project(aiget_game LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
include(FetchContent)
include(CTest)

option(AIGET_SANITIZE "Enable address and undefined-behavior checks" OFF)
option(AIGET_VIEWER "Build the raylib window" ON)

FetchContent_Declare(box2d
    URL https://github.com/erincatto/box2d/archive/refs/tags/v3.1.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(box2d)

add_library(aiget_sim STATIC sim.c)
target_include_directories(aiget_sim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(aiget_sim PUBLIC box2d m)
target_compile_options(aiget_sim PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)

add_executable(aiget_export_level export_level.c)
target_compile_options(aiget_export_level PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(aiget_export_level PRIVATE m)

if(AIGET_SANITIZE)
    target_compile_options(aiget_sim PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(aiget_sim PUBLIC -fsanitize=address,undefined)
endif()

if(AIGET_VIEWER)
    # The upstream Linux binary avoids requiring all GLFW/X11 development headers.
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        message(FATAL_ERROR "The viewer currently supports Linux x86_64; use -DAIGET_VIEWER=OFF for tests elsewhere")
    endif()
    FetchContent_Declare(raylib_dist
        URL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(raylib_dist)
    find_package(X11 REQUIRED)
    add_library(raylib SHARED IMPORTED)
    set_target_properties(raylib PROPERTIES
        IMPORTED_LOCATION "${raylib_dist_SOURCE_DIR}/lib/libraylib.so.5.5.0"
        INTERFACE_INCLUDE_DIRECTORIES "${raylib_dist_SOURCE_DIR}/include")
    add_executable(aiget_game main.c view.c)
    target_link_libraries(aiget_game PRIVATE aiget_sim raylib X11::X11)
    set_target_properties(aiget_game PROPERTIES
        BUILD_RPATH "${raylib_dist_SOURCE_DIR}/lib")
    target_compile_options(aiget_game PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
    add_executable(aiget_sim_test test_sim.c)
    target_link_libraries(aiget_sim_test PRIVATE aiget_sim)
    target_compile_options(aiget_sim_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME physics COMMAND aiget_sim_test)
endif()
