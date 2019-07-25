project(FastPigDogEgg C CXX)

CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
set(fpde_static "fast_pig_dog_egg_static")
set(fpde_shared "fast_pig_dog_egg_shared")

set(FPDE_ROOT ${PROJECT_SOURCE_DIR})
set(FPDE_SRCS ${FPDE_ROOT}/src)
set(FPDE_TEST ${FPDE_ROOT}/test)
set(FPDE_THIRD_PARTY ${PFDE_ROOT}/third-party)

if (NOT DEFINED FPDE_OUT_ROOT_PATH)
    set(FPDE_OUT_ROOT_PATH ${FPDE_ROOT}/output)
endif()

if (USE_ARM64)
    set(FPDE_OUT_ARM64_PATH ${FPDE_OUT_ROOT_PATH}/arm64)
else()
    set(FPDE_OUT_ARMV7_PATH ${FPDE_OUT_ROOT_PATH}/armv7)
endif()
option(USE_ARM64 "option for aarch64" OFF)
option(USE_CL "option for opencl" OFF)
option(USE_OPENMP "option for openmp" OFF)
option(BUILD_SHARED "build fpde shared" YES)
option(ENABLE_DEBUG "debug mode" OFF)
option(BUILD_WITH_TEST "build FastPigDogEgg with unit test" YES)

include(${FPDE_ROOT}/cmake/compiler_option.cmake)

if (USE_CL)
    set(OPENCL_DIR ${FPDE_THIRD_PARTY}/opencl)
    include_directories(${OPENCL_DIR}/include)
    if (USE_ARM64)
        link_directories(${OPENCL_DIR}/libs/arm64)
    else()
        link_directories(${OPENCL_DIR}/libs/armv7)
    endif()
endif()

include_directories(${FPDE_ROOT}/include)
add_subdirectory(${FPDE_SRCS})

if (BUILD_WITH_TEST)
    add_subdirectory(${FPDE_TEST})
endif()

