#include <array>
#include <opencv2/core/core.hpp>

class CreateUndistortRectifyMap {
 public:
  static void Init(const int image_cols, const int image_rows,
                   const cv::Mat& camera_matrix, const cv::Mat& dist_coeffs,
                   cv::Mat* const cols_map, cv::Mat* const rows_map);
};
