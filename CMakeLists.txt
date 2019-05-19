#
# The interface of pose graph by Ceres And G2O  
# Author: Kang Yifei
# Email: ccyclonel@126.com
#

cmake_minimum_required( VERSION 2.8 )
project(ImageEvaluation)

#set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O0 -Wall -g" )

#ceres
#find_package( Ceres REQUIRED )
#include_directories( ${CERES_INCLUDE_DIRS} )

# Opencv
find_package( OpenCV REQUIRED )
include_directories( ${PROJECT_SOURCE_DIR} 
                     ${OpenCV_INCLUDE_DIRS} )

#G2O
#find_package( G2O REQUIRED )

#Eigen
#include_directories("/usr/include/eigen3")


add_executable(${PROJECT_NAME} image_evaluation.cc
                               create_undistort_rectify_map.cc)
target_link_libraries(${PROJECT_NAME}
                      ${OpenCV_LIBRARIES})

