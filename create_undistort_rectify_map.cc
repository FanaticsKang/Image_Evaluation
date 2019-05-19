#include <create_undistort_rectify_map.h>
#include <unistd.h>
#include <iomanip>
#include <iostream>
Undistort::Undistort(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                     const std::string out_folder_path)
    : out_folder_path_(out_folder_path) {
  assert(dist_coeffs.size() == cv::Size(1, 4) ||
         dist_coeffs.size() == cv::Size(4, 1) ||
         dist_coeffs.size() == cv::Size(1, 5) ||
         dist_coeffs.size() == cv::Size(5, 1) ||
         dist_coeffs.size() == cv::Size(1, 8) ||
         dist_coeffs.size() == cv::Size(8, 1));

  k_.fill(0);
  const double* const dist_ptr = dist_coeffs.ptr<double>();
  k_[0] = dist_ptr[0];
  k_[1] = dist_ptr[1];
  k_[2] = dist_ptr[2];
  k_[3] = dist_ptr[3];
  k_[4] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 5 ? dist_ptr[4] : 0.;
  k_[5] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[5] : 0.;
  k_[6] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[6] : 0.;
  k_[7] = dist_coeffs.cols + dist_coeffs.rows - 1 >= 8 ? dist_ptr[7] : 0.;

  fx_ = camera_matrix.at<double>(0, 0);
  fy_ = camera_matrix.at<double>(1, 1);
  ifx_ = 1. / fx_;
  ify_ = 1. / fy_;
  cx_ = camera_matrix.at<double>(0, 2);
  cy_ = camera_matrix.at<double>(1, 2);
}

void Undistort::ComputeUndistortXY(const double r2,
                                   const std::array<double, 8>& k,
                                   const double x0, const double y0,
                                   double* const x_ptr, double* const y_ptr) {
  double& x = *x_ptr;
  double& y = *y_ptr;
  double icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) /
                  (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
  double deltaX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
  double deltaY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
  x = (x0 - deltaX) * icdist;
  y = (y0 - deltaY) * icdist;
}

double Undistort::ComputeError(const double x, const double y, const double u,
                               const double v, const double in_r2) {
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
  cdist = 1 + k_[0] * r2 + k_[1] * r4 + k_[4] * r6;
  icdist2 = 1. / (1 + k_[5] * r2 + k_[6] * r4 + k_[7] * r6);
  xd = x * cdist * icdist2 + k_[2] * a1 + k_[3] * a2;
  yd = y * cdist * icdist2 + k_[2] * a3 + k_[3] * a1;

  double x_proj = xd * fx_ + cx_;
  double y_proj = yd * fy_ + cy_;

  double error = sqrt(pow(x_proj - u, 2) + pow(y_proj - v, 2));
  return error;
}

double Undistort::UndistortOnePoint(const int u, const int v,
                                    double* const x_out, double* const y_out,
                                    const size_t max_iter,
                                    std::ofstream* const file_out_ptr) {
  if (file_out_ptr != nullptr) {
    (*file_out_ptr) << std::setprecision(4) << "Compute Point[" << u << ", "
                    << v << "]: " << std::endl;
  }
  double x, y, x0 = 0, y0 = 0;
  // 像素坐标系下
  x = u;
  y = v;
  //将[x, y]映射到相机坐标系下
  x = (x - cx_) * ifx_;
  y = (y - cy_) * ify_;
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
  for (size_t iter = 0; iter < max_iter; ++iter) {
    double r2 = x * x + y * y;
    ComputeUndistortXY(r2, k_, x0, y0, &x, &y);
    // 当前位置的误差
    error = ComputeError(x, y, u, v);
    if (pre_r2 > 0) {
      double pre_x_step = x - pre_x;
      double pre_y_step = y - pre_y;
      do {
        // 沿之前梯度方向计算估计值
        double tmp_x = x + diff_step * pre_x_step;
        double tmp_y = y + diff_step * pre_y_step;
        double tmp_error = ComputeError(tmp_x, tmp_y, u, v);
        // 计算梯度方向
        double direction = (tmp_error - error) / diff_step;
        // 说明误差减少说明方向正确, 可以继续迭代.
        if (direction > 0) {
          x = x - 3.0 / 5 * pre_x_step;
          y = y - 3.0 / 5 * pre_y_step;
          r2 = x * x + y * y;
          error = ComputeError(x, y, u, v);
          if (file_out_ptr != nullptr) {
            (*file_out_ptr) << std::setprecision(4) << "iter(" << iter << "): ["
                            << x << "," << y << "], ";
            (*file_out_ptr) << std::setprecision(4) << "r2: " << r2 << " ";
            (*file_out_ptr) << std::setprecision(4) << "error: " << error
                            << " ";
            (*file_out_ptr) << std::endl;
          }
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
    if (file_out_ptr != nullptr) {
      (*file_out_ptr) << std::setprecision(4) << "iter(" << iter << "): [" << x
                      << "," << y << "], ";
      (*file_out_ptr) << std::setprecision(4) << "r2: " << r2 << " ";
      (*file_out_ptr) << std::setprecision(4) << "error: " << error << " ";
      (*file_out_ptr) << std::endl;
    }
    if (error < min_error) {
      min_error = error;
      min_x = x;
      min_y = y;
    }
  }
  *x_out = min_x;
  *y_out = min_y;
  return min_error;
}

void Undistort::InitRectifyMap(const int image_cols, const int image_rows,
                               cv::Mat* const cols_map, cv::Mat* const rows_map,
                               const int max_iter) {
  std::ofstream file_out(out_folder_path_ + log_file_name_, std::ios::out);

  const long total_size = image_rows * image_cols;
  long index = 0;
  long avalible = 0;
  cv::Mat error_img(image_rows, image_cols, CV_64FC1);
  int complete_ratio = -1;
  for (int j = 0; j < image_rows; ++j) {
    for (int i = 0; i < image_cols; ++i) {
      double undist_x, undist_y;
      double min_error =
          UndistortOnePoint(i, j, &undist_x, &undist_y, max_iter, &file_out);
      if (min_error < 1) {
        error_img.at<double>(j, i) = 0;
        ++avalible;
      } else {
        error_img.at<double>(j, i) = 250;
      }
      ++index;
      int current_ratio = static_cast<int>(index * 100 / total_size);
      if (current_ratio != complete_ratio) {
        complete_ratio = current_ratio;
        std::cout << "\r" << static_cast<int>(index * 100 / total_size) << "% "
                  << std::flush;
      }
    }
  }
  cv::imwrite("/home/kang/bridge/error_image.png", error_img);
  std::cout << "\nAvalible: " << (int)(avalible * 100 / index) << "%"
            << std::endl;
  std::cout << "finished." << std::endl;
}
