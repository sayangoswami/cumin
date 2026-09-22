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
)

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
