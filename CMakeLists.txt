cmake_minimum_required(VERSION 3.22)
project(Cumin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

if ( NOT CMAKE_BUILD_TYPE )
    message(STATUS "Build Type not set, defaulting to Release..." )
    set( CMAKE_BUILD_TYPE Release )
else ()
    message(STATUS "Build type is ${CMAKE_BUILD_TYPE}" )
endif()

include_directories(external/parlaylib/include)
include_directories(external/argparse/include)

add_subdirectory(external/kseqpp)
add_subdirectory(external/pybind11)

set(CORE_SOURCES
        src/index.cpp
        src/dyn_index.cpp
        src/tiered_buffer.cpp
)

# The tiered-vector header is configured by compile definitions and leaks
# names, so it is confined to this one translation unit.
set_source_files_properties(src/tiered_buffer.cpp PROPERTIES
        COMPILE_DEFINITIONS "ARRAY;LEVEL;PACK"
        INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/external/tiered-vector/include)

set(PYBIND11_FINDPYTHON ON)
pybind11_add_module(_core
        ${CORE_SOURCES}
        src/pybindings.cpp
)
target_link_libraries(_core PRIVATE kseq++)

add_executable(cumin
        ${CORE_SOURCES}
        src/main.cpp
)
target_link_libraries(cumin PRIVATE kseq++)

add_executable(cumin_smoketest src/smoketest.cpp)

install(TARGETS _core DESTINATION pycumin)
