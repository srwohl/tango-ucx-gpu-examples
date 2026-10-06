cmake_minimum_required(VERSION 3.24)
project(tango-ucx-gpu-examples LANGUAGES CXX)
include(CTest)

find_package(tango-ucx CONFIG REQUIRED)
find_package(nlohmann_json 3.12 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
option(EXAMPLES_CUDA "Build the real CUDA publisher path" ON)
option(EXAMPLES_PIPELINE "Build GPU tomography processing device servers" OFF)
set(EXAMPLES_PYTHON_PREFIX "" CACHE PATH "Installed tango-ucx Python prefix for smoke tests")

add_executable(opaque_publisher publisher.cpp)
target_compile_features(opaque_publisher PRIVATE cxx_std_20)
target_link_libraries(opaque_publisher PRIVATE tango-ucx::ucx nlohmann_json::nlohmann_json)
if(EXAMPLES_CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT TARGET tango-ucx::cuda)
        message(FATAL_ERROR "Install tango-ucx with TANGO_UCX_CUDA=ON first")
    endif()
    target_compile_definitions(opaque_publisher PRIVATE EXAMPLES_CUDA)
    # Imported OBJECT libraries need their objects explicitly: linking their interface alone
    # does not include the static initializer that registers the CUDA runtime.
    target_sources(opaque_publisher PRIVATE $<TARGET_OBJECTS:tango-ucx::cuda>)
    target_link_libraries(opaque_publisher PRIVATE tango-ucx::cuda CUDA::cudart)
endif()

if(EXAMPLES_PIPELINE)
    if(NOT EXAMPLES_CUDA)
        message(FATAL_ERROR "The tomography pipeline requires EXAMPLES_CUDA")
    endif()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    add_executable(pipeline_device tomography/pipeline_device.cpp)
    target_compile_features(pipeline_device PRIVATE cxx_std_20)
    target_compile_definitions(pipeline_device PRIVATE
        TOMOGRAPHY_MODULE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tomography")
    target_sources(pipeline_device PRIVATE $<TARGET_OBJECTS:tango-ucx::cuda>)
    target_link_libraries(pipeline_device PRIVATE tango-ucx::ucx tango-ucx::cuda
        CUDA::cudart nlohmann_json::nlohmann_json pybind11::embed)
    if(BUILD_TESTING)
        add_test(NAME tomography_gpu_tcp COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tomography/demo.py
            --device-server $<TARGET_FILE:pipeline_device>
            --pixels 64 --slices 8 --angles 96 --budget 131072 --correction-delay-ms 3)
        set_tests_properties(tomography_gpu_tcp PROPERTIES TIMEOUT 240 LABELS gpu)
        add_test(NAME tomography_stream_gpu_tcp COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tomography/stream_smoke.py
            --device-server $<TARGET_FILE:pipeline_device>)
        set_tests_properties(tomography_stream_gpu_tcp PROPERTIES TIMEOUT 150 LABELS gpu)
        if(EXAMPLES_PYTHON_PREFIX)
            set_tests_properties(tomography_gpu_tcp PROPERTIES
                ENVIRONMENT "TANGO_UCX_PREFIX=${EXAMPLES_PYTHON_PREFIX}")
            set_tests_properties(tomography_stream_gpu_tcp PROPERTIES
                ENVIRONMENT "TANGO_UCX_PREFIX=${EXAMPLES_PYTHON_PREFIX}")
        endif()
    endif()
endif()

if(BUILD_TESTING)
    add_test(NAME host_tcp COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke.py
        --publisher $<TARGET_FILE:opaque_publisher> --host-only)
    set_tests_properties(host_tcp PROPERTIES TIMEOUT 180 LABELS host)
    if(EXAMPLES_CUDA)
        add_test(NAME gpu_tcp COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke.py
            --publisher $<TARGET_FILE:opaque_publisher>)
        set_tests_properties(gpu_tcp PROPERTIES TIMEOUT 300 LABELS gpu)
        add_test(NAME calibration_gpu_tcp COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/smoke.py
            --publisher $<TARGET_FILE:opaque_publisher> --calibration-only)
        set_tests_properties(calibration_gpu_tcp PROPERTIES TIMEOUT 300 LABELS gpu)
    endif()
    if(EXAMPLES_PYTHON_PREFIX)
        set_tests_properties(host_tcp PROPERTIES ENVIRONMENT "TANGO_UCX_PREFIX=${EXAMPLES_PYTHON_PREFIX}")
        if(EXAMPLES_CUDA)
            set_tests_properties(gpu_tcp PROPERTIES ENVIRONMENT "TANGO_UCX_PREFIX=${EXAMPLES_PYTHON_PREFIX}")
            set_tests_properties(calibration_gpu_tcp PROPERTIES ENVIRONMENT "TANGO_UCX_PREFIX=${EXAMPLES_PYTHON_PREFIX}")
        endif()
    endif()
endif()
