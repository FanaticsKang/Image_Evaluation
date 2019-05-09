#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "create_undistort_rectify_map.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Please use the following format:\n"
              << "ImageEvaluation path/to/yaml/file" << std::endl;
    return -1;
  }
  std::cout << "File Path: " << argv[1] << std::endl;
  cv::FileStorage fs(argv[1], cv::FileStorage::READ);
  if (!fs.isOpened()) {
    std::cout << "The yaml file is WRONG!" << std::endl;
    return -1;
  }
  int image_cols;
  int image_rows;
  fs["Image.Cols"] >> image_cols;
  fs["Image.Rows"] >> image_rows;

  cv::Mat camera_internal_param;
  cv::Mat camera_distort_param;
  fs["Camera.K"] >> camera_internal_param;
  fs["Camera.Dist"] >> camera_distort_param;
  std::cout << "Image Size: [" << image_cols << "x" << image_rows << "];"
            << std::endl;
  std::cout << "Camera internal parameter:\n"
            << camera_internal_param << std::endl;
  std::cout << "Camera distort paratmeter:\n"
            << camera_distort_param << std::endl;
  cv::Mat rows_map, cols_map;
  CreateUndistortRectifyMap::Init(image_cols, image_rows, camera_internal_param,
                                  camera_distort_param, &rows_map, &cols_map);
}
