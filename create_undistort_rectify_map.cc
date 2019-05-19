#include <create_undistort_rectify_map.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <iostream>

void CreateUndistortRectifyMap::ComputeUndistortXY(
    const double r2, const std::array<double, 8>& k, const double x0,
    const double y0, double* const x_ptr, double* const y_ptr) {
  double& x = *x_ptr;
  double& y = *y_ptr;
  double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) /
                  (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
  double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
  double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
  x = (x0 - deltaX) * icdist;
  y = (y0 - deltaY) * icdist;
}

double CreateUndistortRectifyMap::ComputeError(const double x, const double y,
                                               const std::array<double, 8>& k,
                                               const double fx, const double fy,
                                               const double cx, const double cy,
                                               const double u, const double v,
                                               const double in_r2) {
  double r4, r6, a1, a2, a3, cdist, icdist2;
  double xd, yd;
  cv::Vec3d vecTilt;

  double r2;
  if (in_r2 < 0)
    r2 = x * x + y * y;
  else
    r2 = in_r2;

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

  double error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
  return error;
}

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

  std::cout << "Start..." << std::endl;
  std::fstream file_out("/home/kang/bridge/undistort_data.txt", std::ios::out);

  const int max_iter = 20;
  const long total_size = image_rows * image_cols;
  long index = 0;
  long avalible = 0;
  cv::Mat error_img(image_rows, image_cols, CV_64FC1);
  for (int j = 0; j < image_rows; ++j) {
    for (int i = 0; i < image_cols; ++i) {
      // [x0, y0]是畸变位置(相机坐标系)
      // [x, y]是无畸变位置(相机坐标系), 迭代更新.
      // [u, v]是畸变位置(像素坐标系)
      double x, y, x0 = 0, y0 = 0, u, v;
      // 像素坐标系下
      u = x = i;
      v = y = j;
      file_out << std::setprecision(4) << "Compute Point[" << u << ", " << v
               << "]: " << std::endl;
      //将[x, y]映射到相机坐标系下
      x = (x - cx) * ifx;
      y = (y - cy) * ify;
      // 计算初始位置
      x0 = x;
      y0 = y;
      // 校验误差
      double error = std::numeric_limits<double>::max();

      double pre_r2 = -1;
      double pre_x = 0.0, pre_y = 0.0;
      double pre_error = std::numeric_limits<double>::max();

      // 最小error用来计算最优点
      double min_error = std::numeric_limits<double>::max();
      double min_x;
      double min_y;

      // 数值微分步长
      const double diff_step = 0.01;

      // iter compute
      for (int iter = 0; iter < max_iter; ++iter) {
        double r2 = x * x + y * y;
        ComputeUndistortXY(r2, k, x0, y0, &x, &y);
        // 当前位置的误差
        error = ComputeError(x, y, k, fx, fy, cx, cy, u, v);
        if (pre_r2 > 0) {
          double pre_x_step = x - pre_x;
          double pre_y_step = y - pre_y;
          do {
            // 沿之前梯度方向计算估计值
            double tmp_x = x + diff_step * pre_x_step;
            double tmp_y = y + diff_step * pre_y_step;
            double tmp_error =
                ComputeError(tmp_x, tmp_y, k, fx, fy, cx, cy, u, v);
            // 计算梯度方向
            double direction = (tmp_error - error) / diff_step;
            // 说明误差减少说明方向正确, 可以继续迭代.
            if (direction > 0) {
              x = x - 3.0 / 5 * pre_x_step;
              y = y - 3.0 / 5 * pre_y_step;
              r2 = x * x + y * y;
              error = ComputeError(x, y, k, fx, fy, cx, cy, u, v);
            } else {
              break;
            }
            ++iter;
            // when r2<pre_r2, 迭代点已经早于之前一次, 退出.
          } while (error > pre_error && r2 > pre_r2 && iter < max_iter);
        }
        //记录前一次结果
        pre_r2 = r2;
        pre_x = x;
        pre_y = y;
        pre_error = error;
        file_out << std::setprecision(4) << "iter(" << iter << "): [" << x
                 << ", " << y << "], " << std::flush;
        file_out << std::setprecision(4) << "r2: " << r2 << " " << std::flush;
        file_out << std::setprecision(4) << "error: " << error << " "
                 << std::flush;
        file_out << std::endl;
        if (error < min_error) {
          min_error = error;
          min_x = x;
          min_y = y;
        }
      }
      if (min_error < 1) {
        error_img.at<double>(j, i) = 0;
        ++avalible;
      } else {
        error_img.at<double>(j, i) = 250;
      }
      ++index;
      std::cout << "\r" << static_cast<int>(index * 100 / total_size) << "% ";
    }
  }
  cv::imwrite("/home/kang/bridge/error_image.png", error_img);
  std::cout << "\nAvalible: " << (int)(avalible * 100 / index) << "%"
            << std::endl;
  std::cout << "finished." << std::endl;
}
