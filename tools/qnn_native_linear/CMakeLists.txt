cmake_minimum_required(VERSION 3.16)
project(qnn_native_linear LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(qnn_linear_front_probe
  src/qnn_linear_front_probe.cpp
)

target_include_directories(qnn_linear_front_probe PRIVATE
  $ENV{QNN_SDK_ROOT}/include
)

if(ANDROID)
  target_link_libraries(qnn_linear_front_probe PRIVATE dl log)
else()
  target_link_libraries(qnn_linear_front_probe PRIVATE dl pthread)
endif()
