cmake_minimum_required(VERSION 3.18)
project(cuda-verkle LANGUAGES CUDA CXX)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Default architecture (can be overridden with -DCMAKE_CUDA_ARCHITECTURES=xx)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Enable debug info for Nsight Compute profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

option(ENABLE_PHASE2 "Build MSM and Pedersen commitment support" ON)
option(ENABLE_PHASE3 "Build the simulated tree support" ON)
option(ENABLE_PHASE4 "Enable Phase 4: Benchmarks" OFF)

# The tree calls the MSM implementation.  Keep the options usable when callers
# explicitly enable only the tree target.
if(ENABLE_PHASE3)
    set(ENABLE_PHASE2 ON)
endif()

# Source files (header-only for field/curve, but we need .cu for compilation units)
set(FIELD_SOURCES
    src/field/fp.cu
    src/field/fr.cu
)

set(CURVE_SOURCES
    src/curve/bandersnatch.cu
    src/curve/banderwagon.cu
)

set(LIB_SOURCES ${FIELD_SOURCES} ${CURVE_SOURCES})

if(ENABLE_PHASE2)
    list(APPEND LIB_SOURCES 
        src/msm/msm_kernel.cu 
        src/commitment/pedersen.cu
    )
endif()

if(ENABLE_PHASE3)
    list(APPEND LIB_SOURCES 
        src/tree/verkle_tree.cu
    )
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)

# Library target
add_library(cuda_verkle_lib STATIC ${LIB_SOURCES})
target_include_directories(cuda_verkle_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)

# Test executables
enable_testing()

add_executable(test_field tests/test_field.cu)
target_link_libraries(test_field cuda_verkle_lib)
add_test(NAME TestField COMMAND test_field)

add_executable(test_curve tests/test_curve.cu)
target_link_libraries(test_curve cuda_verkle_lib)
add_test(NAME TestCurve COMMAND test_curve)

if(ENABLE_PHASE2)
    add_executable(test_msm tests/test_msm.cu)
    target_link_libraries(test_msm cuda_verkle_lib)
    add_test(NAME TestMSM COMMAND test_msm)

    add_executable(test_commitment tests/test_commitment.cu)
    target_link_libraries(test_commitment cuda_verkle_lib)
    add_test(NAME TestCommitment COMMAND test_commitment)

    add_executable(test_msm_gpu tests/test_msm_gpu.cu)
    target_link_libraries(test_msm_gpu cuda_verkle_lib)
    add_test(NAME TestMSMGPU COMMAND test_msm_gpu)
    # A CUDA-capable build host may still not have a visible GPU (for example CI compilation jobs). The executable returns 77 in that case.
    set_tests_properties(TestMSMGPU PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
endif()

if(ENABLE_PHASE3)
    add_executable(test_tree tests/test_tree.cu)
    target_link_libraries(test_tree cuda_verkle_lib)
    add_test(NAME TestTree COMMAND test_tree)
endif()

if(ENABLE_PHASE4)
    # Only benchmark sources that exist in this repository.
    add_executable(bench_msm src/benchmark/bench_msm.cu)
    target_link_libraries(bench_msm cuda_verkle_lib)
endif()
