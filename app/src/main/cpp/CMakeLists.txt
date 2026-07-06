cmake_minimum_required(VERSION 3.22)
project(medrag_llama)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# llama.cpp must be cloned as a git submodule into this directory before
# building, i.e. from the android/app/src/main/cpp/ folder run:
#   git submodule add https://github.com/ggerganov/llama.cpp.git llama.cpp
#   git submodule update --init --recursive
# CMake just consumes whatever lives in ./llama.cpp/CMakeLists.txt - we do
# not vendor llama.cpp's source directly in this project.
# ---------------------------------------------------------------------------
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")
if(NOT EXISTS "${LLAMA_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp submodule not found at ${LLAMA_DIR}. Run:\n"
        "  cd android/app/src/main/cpp\n"
        "  git submodule add https://github.com/ggerganov/llama.cpp.git llama.cpp\n"
        "  git submodule update --init --recursive\n"
        "See README.md 'Native build setup' for the full checklist.")
endif()

# Keep llama.cpp's own build lean - we only need the inference engine, not
# its example binaries/tests, and we want a static lib we link into our own
# single .so (simpler for Gradle packaging than shipping two .so files).
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Force-disable OpenMP for both the main engine and the math library
set(LLAMA_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

# ARM-specific perf flags. NEON is mandatory on arm64-v8a; dotprod/i8mm
# requires Armv8.2+ which is common on phones from 2020 onward but NOT
# universal on budget devices - if the resulting .so crashes with an
# illegal-instruction signal on a real low-end test device, drop these two
# flags first.
#if(ANDROID_ABI STREQUAL "arm64-v8a")
#    add_compile_options(-march=armv8.2-a+dotprod+i8mm)
#endif()

add_subdirectory(${LLAMA_DIR})

add_library(medrag_llama SHARED
    llama_jni.cpp
    cpu_affinity.cpp
)

target_include_directories(medrag_llama PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/ggml/include
)

find_library(log-lib log)

target_link_libraries(medrag_llama
    PRIVATE
        llama
        ggml
        ${log-lib}
        android
)
