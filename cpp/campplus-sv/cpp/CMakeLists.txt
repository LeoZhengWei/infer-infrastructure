cmake_minimum_required(VERSION 3.20)
project(campplus_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_DYLIB "/Users/mac/miniconda3/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.1.32.0.dylib"
    CACHE FILEPATH "ONNX Runtime dylib")
set(ORT_INCLUDE "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ORT headers")

add_executable(campplus_bench main.cpp mlp.cpp)
target_include_directories(campplus_bench PRIVATE "${ORT_INCLUDE}")
target_compile_options(campplus_bench PRIVATE -O3 -DACCELERATE_NEW_LAPACK)
target_link_libraries(campplus_bench PRIVATE "${ORT_DYLIB}" "-framework Accelerate")

# The dylib's install name is @rpath/libonnxruntime.1.dylib. cpp/lib holds that symlink.
set_target_properties(campplus_bench PROPERTIES
    BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib"
    INSTALL_RPATH "${CMAKE_SOURCE_DIR}/lib")
