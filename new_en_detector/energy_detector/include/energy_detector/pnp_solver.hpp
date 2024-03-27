#ifndef PNP_SLOVER_HPP_
#define PNP_SLOVER_HPP_

#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include "Leaf.hpp"
namespace rm_auto_aim
{
  class PnPSolver
  {
  public:
    PnPSolver(
        const std::array<double, 9> &camera_matrix, const std::vector<double> &dist_coeffs);
        
    // Calculate the distance between attack center and image center
    float calculateDistanceToCenter(const cv::Point2f &image_point);
    bool solvePnP_(const Leaf &leaf, cv::Mat &rvec, cv::Mat &tvec);

  private:
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    std::vector<cv::Point3f> energy_leaf_points_;
    // Unit: mm
    static constexpr float ENERGY_LEAF_WIDTH = 332.94;
    static constexpr float ENERGY_LEAF_HEIGHT = 125.27;
  };
}
#endif