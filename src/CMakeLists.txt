set(FPDE_CL_ROOT ${FPDE_SRCS}/cl)
set(FPDE_NEON_ROOT ${FPDE_SRCS}/neon)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
set(MAKE_STATIC_LINKER_FLAGS "${MAKE_STATIC_LINKER_FLAGS} -Wl,--gc-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fno-exceptions")

aux_source_directory(${FPDE_NEON_ROOT} NEON_SRCS)
aux_source_directory(${FPDE_CL_ROOT} CL_SRCS)
if (USE_CL)
    set(SRCS ${NEON_SRCS} ${CL_SRCS})
    set(FPDE_DEP "OpenCL")
else()
    set(SRCS ${NEON_SRCS})
    set(FPDE_DEP "")
endif()

## build shared lib
add_library(${fpde_shared} SHARED ${SRCS})
target_link_libraries(${fpde_shared} ${FPDE_DEP})
if (USE_ARM64)
    set_target_properties(${fpde_shared} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${FPDE_OUT_ARM64_PATH}/lib")
else()
    set_target_properties(${fpde_shared} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${FPDE_OUT_ARMV7_PATH}/lib")
endif()

## build static lib
add_library(${fpde_static} STATIC ${SRCS})
target_link_libraries(${fpde_static} ${FPDE_DEP})
if (USE_ARM64)
    set_target_properties(${fpde_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${FPDE_OUT_ARM64_PATH}/lib")
else()
    set_target_properties(${fpde_static} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${FPDE_OUT_ARMV7_PATH}/lib")
endif()

