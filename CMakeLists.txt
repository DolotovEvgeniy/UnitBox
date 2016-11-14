CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(PROJECT_NAME UnitBox)
project(${PROJECT_NAME})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

FILE(GLOB hdrs "*.h*" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h*")
FILE(GLOB srcs "*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

#----------------------------------------------------------------------------------#
option(WITH_CPU_ONLY "Caffe will be used in CPU or GPU mode" ON)
#----------------------------------------------------------------------------------#

add_library(${PROJECT_NAME} STATIC ${srcs} ${hdrs})

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wall")
endif()

set(CMAKE_SCRIPT_DIR ${CMAKE_SOURCE_DIR}/CMakeScripts)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SCRIPT_DIR})

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


find_package(Caffe REQUIRED)
# Seems that after successfull Caffe search CAFFE_FOUND variable is not defined.
if (Caffe_INCLUDE_DIRS)
    include_directories(${Caffe_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES})
endif()
add_definitions(-DHAVE_CAFFE)

if (WITH_CPU_ONLY)
   message(STATUS "Caffe in CPU_ONLY mode.")
   add_definitions(-DCPU_ONLY)
endif()


add_subdirectory(samples)

#----------------------------------------------------------------------------------#
message(STATUS ${PROJECT_NAME})
message(STATUS "-------------------------------------------------------")
message(STATUS "   CPU_ONLY:      ${WITH_CPU_ONLY}")
message(STATUS "-------------------------------------------------------")
