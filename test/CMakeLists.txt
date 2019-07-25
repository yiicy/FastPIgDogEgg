set(FPDE_TEST_NEON_PATH ${FPDE_TEST}/neon_test)
set(FPDE_TEST_CL_PATH ${FPDE_TEST}/cl_test)
if (USE_CL)
    aux_source_directory(${FPDE_TEST_CL_PATH} CL_TEST_SRCS)
    foreach (CL_SRC_NAME ${CL_TEST_SRCS})
        #unpack the dir "/"
        string(REPLACE "/" ";" SEXY_LIST ${CL_SRC_NAME})
        list(GET SEXY_LIST -1 TEST_CASE_NAME)
        #get the file name without suffix
        string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
        list(GET SEXY_LIST 0 TEST_CASE_NAME)
        add_executable(${TEST_CASE_NAME}  ${CL_SRC_NAME})
        if(BUILD_SHARED)
            target_link_libraries(${TEST_CASE_NAME} ${fpde_shared} "OpenCL")
        else()
            target_link_libraries(${TEST_CASE_NAME} ${fpde_static} "OpenCL")
        endif()
        if (USE_ARM64)
            set_target_properties(${TEST_CASE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${FPDE_OUT_ARM64_PATH}/cl_test)
        else()
            set_target_properties(${TEST_CASE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${FPDE_OUT_ARMV7_PATH}/cl_test)
        endif()
    endforeach()
else()
    aux_source_directory(${FPDE_TEST_NEON_PATH} NEON_TEST_SRCS)
    message(status "FPDE_TEST_NEON_PATH: ${FPDE_TEST_NEON_PATH}")
    message(status "neon test srcs: ${NEON_TEST_SRCS}")
    foreach (NEON_SRC_NAME ${NEON_TEST_SRCS})
        #unpack the dir "/"
        string(REPLACE "/" ";" SEXY_LIST ${NEON_SRC_NAME})
        list(GET SEXY_LIST -1 TEST_CASE_NAME)
        #get the file name without suffix
        string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
        list(GET SEXY_LIST 0 TEST_CASE_NAME)
        add_executable(${TEST_CASE_NAME}  ${NEON_SRC_NAME})
        if(BUILD_SHARED)
            target_link_libraries(${TEST_CASE_NAME} ${fpde_shared})
        else()
            target_link_libraries(${TEST_CASE_NAME} ${fpde_static})
        endif()
        if (USE_ARM64)
            set_target_properties(${TEST_CASE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${FPDE_OUT_ARM64_PATH}/neon_test)
        else()
            set_target_properties(${TEST_CASE_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${FPDE_OUT_ARMV7_PATH}/neon_test)
        endif()
    endforeach()
endif()


