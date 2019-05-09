#include <create_undistort_rectify_map.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>

void CreateUndistortRectifyMap::Init(const int image_cols, const int image_rows,
                                     const cv::Mat& camera_matrix,
                                     const cv::Mat& dist_coeffs,
                                     cv::Mat* const cols_map,
                                     cv::Mat* const rows_map) {
  assert(dist_coeffs.size() == cv::Size(1, 4) ||
         dist_coeffs.size() == cv::Size(4, 1) ||
         dist_coeffs.size() == cv::Size(1, 5) ||
         dist_coeffs.size() == cv::Size(5, 1) ||
         dist_coeffs.size() == cv::Size(1, 8) ||
         dist_coeffs.size() == cv::Size(8, 1));

  std::array<double, 8> k{};
  k.fill(0);
  const double* const dist_ptr = dist_coeffs.ptr<double>();
  k[0] = dist_ptr[0];
  k[1] = dist_ptr[1];
  k[2] = dist_ptr[2];
  k[3] = dist_ptr[3];
  k[4] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 5 ? dist_ptr[4] : 0.;
  k[5] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[5] : 0.;
  k[6] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[6] : 0.;
  k[7] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[7] : 0.;

  std::cout << "k: ";
  for (int i = 0; i < 8; ++i) {
    std::cout << k[i] << ", ";
  }
  std::cout << std::endl;
  const double fx = camera_matrix.at<double>(0, 0);
  const double fy = camera_matrix.at<double>(1, 1);
  const double ifx = 1. / fx;
  const double ify = 1. / fy;
  const double cx = camera_matrix.at<double>(0, 2);
  const double cy = camera_matrix.at<double>(1, 2);

  const int max_iter = 10;
  std::cout << "Start..." << std::endl;
  std::fstream file_out("/home/kang/bridge/undistort_data.txt", std::ios::out);
  const long total_size = image_rows * image_cols;
  long index = 0;

  for (int j = 0; j < image_rows; ++j) {
    for (int i = 0; i < image_cols; ++i) {
      double x, y, x0 = 0, y0 = 0, u, v;
      u = x = i;
      v = y = j;
      file_out << std::setprecision(4) << "Compute Point[" << u << ", " << v
               << "]: " << std::endl;
      x = (x - cx) * ifx;
      y = (y - cy) * ify;
      // compensate tilt distortion
      x0 = x;
      y0 = y;
      double error = std::numeric_limits<double>::max();
      // compensate distortion iteratively
      for (int iter = 0; iter < max_iter; ++iter) {
        double r2 = x * x + y * y;
        double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) /
                        (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
        double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
        double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
        x = (x0 - deltaX) * icdist;
        y = (y0 - deltaY) * icdist;
        file_out << std::setprecision(4) <<  "iter(" << iter << "): [" << x << ", " << y << "], "
                 << std::flush;
        file_out << std::setprecision(4) << "icdist: " << icdist << " " << std::flush;
        file_out << std::setprecision(4) << "r2: " << r2 << " " << std::flush;
        //check 
        {
          double r4, r6, a1, a2, a3, cdist, icdist2;
          double xd, yd;
          cv::Vec3d vecTilt;

          r2 = x * x + y * y;
          r4 = r2 * r2;
          r6 = r4 * r2;
          a1 = 2 * x * y;
          a2 = r2 + 2 * x * x;
          a3 = r2 + 2 * y * y;
          cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
          icdist2 = 1. / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
          xd = x * cdist * icdist2 + k[2] * a1 + k[3] * a2;
          yd = y * cdist * icdist2 + k[2] * a3 + k[3] * a1;

          double x_proj = xd * fx + cx;
          double y_proj = yd * fy + cy;

          error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
          file_out << std::setprecision(4) << "error: " << error << " \n" << std::flush;
        }
      }
      ++index;
      std::cout << "\r" << static_cast<int>(index * 100 / total_size) << "% ";
    }
  }
  std::cout << "finished." << std::endl;
}
