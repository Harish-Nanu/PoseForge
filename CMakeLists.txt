cmake_minimum_required(VERSION 3.22)
project(PoseForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(POSEFORGE_WERROR "Treat warnings as errors" ON)
option(POSEFORGE_BUILD_TESTS "Build unit tests" ON)
set(POSEFORGE_SANITIZER "" CACHE STRING "Sanitizer to enable: address, thread or empty")

find_package(OpenCV 4.7 REQUIRED COMPONENTS core imgproc calib3d objdetect imgcodecs videoio highgui)
find_package(Threads REQUIRED)

add_library(poseforge_warnings INTERFACE)
target_compile_options(poseforge_warnings INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)
if(POSEFORGE_WERROR)
  target_compile_options(poseforge_warnings INTERFACE -Werror)
endif()

if(POSEFORGE_SANITIZER STREQUAL "address")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
elseif(POSEFORGE_SANITIZER STREQUAL "thread")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
  add_link_options(-fsanitize=thread)
endif()

add_library(poseforge
  src/camera.cpp
  src/detector.cpp
  src/synthetic.cpp
  src/pose.cpp
  src/render.cpp
  src/stats.cpp
  src/frame_io.cpp
  src/pipeline.cpp
)
target_include_directories(poseforge PUBLIC include)
# OpenCV headers are treated as system headers so our strict warnings don't fire on them.
target_include_directories(poseforge SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(poseforge PUBLIC ${OpenCV_LIBS} Threads::Threads PRIVATE poseforge_warnings)

if(POSEFORGE_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(poseforge_tests
    tests/main.cpp
    tests/test_camera.cpp
    tests/test_detector.cpp
    tests/test_pose.cpp
    tests/test_render.cpp
    tests/test_stats.cpp
    tests/test_bounded_queue.cpp
    tests/test_pipeline.cpp
  )
  target_link_libraries(poseforge_tests PRIVATE poseforge poseforge_warnings GTest::gtest)
  include(GoogleTest)
  gtest_discover_tests(poseforge_tests)
endif()
