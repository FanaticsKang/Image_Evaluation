#include <array>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

class Undistort {
 public:
  Undistort(const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
            const std::string out_folder_path);

  void InitRectifyMap(const int image_cols, const int image_rows,
                      cv::Mat* const cols_map, cv::Mat* const rows_map,
                      const int max_iter = 10);

  // [u,v]是在像素坐标系下, [min_x, min_y]在相机坐标系下.
  double UndistortOnePoint(const int u, const int v, double* const x_out,
                           double* const y_out, const size_t max_iter = 10,
                           std::ofstream* const file_out_ptr = nullptr);

  double ComputeError(const double x, const double y, const double u,
                      const double v, const double in_r2 = -1);

  // 迭代计算XY, 从原始点[x0, y0开始], 根据半径r2计算
  // 圆形畸变(radial distortion)主要和x0, y0,r2有关. [k1, k2, k3,...] in opencv
  // 正切畸变(tangential distortion)和当前迭代位置[x,y]有关.
  void ComputeUndistortXY(const double r2, const std::array<double, 8>& k,
                          const double x0, const double y0, double* const x_ptr,
                          double* const y_ptr);

 private:
  std::array<double, 8> k_{};
  double fx_;
  double fy_;
  double ifx_;
  double ify_;
  double cx_;
  double cy_;
  const std::string out_folder_path_;
  const std::string log_file_name_ = "undistort_data_log.txt";
};
