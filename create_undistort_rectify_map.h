#include <array>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

class CreateUndistortRectifyMap {
 public:
  static void Init(const int image_cols, const int image_rows,
                   const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                   cv::Mat* const cols_map, cv::Mat* const rows_map);
  static double ComputeError(const double x, const double y,
                             const std::array<double, 8>& k, const double fx,
                             const double fy, const double cx, const double cy,
                             const double u, const double v,
                             const double in_r2 = -1);

  // 迭代计算XY, 从原始点[x0, y0开始], 根据半径r2计算
  // 圆形畸变(radial distortion)主要和x0, y0,r2有关. [k1, k2, k3,...] in opencv
  // 正切畸变(tangential distortion)和当前迭代位置[x,y]有关.
  static void ComputeUndistortXY(const double r2,
                                 const std::array<double, 8>& k,
                                 const double x0, const double y0,
                                 double* const x_ptr, double* const y_ptr);
};
