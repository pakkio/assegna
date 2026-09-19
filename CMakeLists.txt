cmake_minimum_required(VERSION 3.14)
project(matcher C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

include(FetchContent)
FetchContent_Declare(
  cjson
  GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
  GIT_TAG v1.7.18
)
FetchContent_MakeAvailable(cjson)

find_package(CURL REQUIRED)

add_executable(match match.c)
target_include_directories(match PRIVATE ${cjson_SOURCE_DIR})
target_link_libraries(match PRIVATE cjson CURL::libcurl m)

add_executable(test_match test_match.c)
target_include_directories(test_match PRIVATE ${cjson_SOURCE_DIR})
target_link_libraries(test_match PRIVATE cjson CURL::libcurl m)
