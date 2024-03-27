#ifndef ENERGY__KALMAN_FILTER_HPP_
#define ENERGY__KALMAN_FILTER_HPP_

#include <Eigen/Dense>
#include <functional>
#include <rclcpp/logger.hpp>
#include "rclcpp/rclcpp.hpp"
namespace rm_auto_aim
{

class ExtendedKalmanFilter
{
public:
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;
  ExtendedKalmanFilter() = default;
  explicit ExtendedKalmanFilter(const VoidMatFunc & u_f);

  
  // Set the initial state
  void setState(const Eigen::VectorXd & x0);
  // Compute a predicted state
  Eigen::MatrixXd predict();
  float t;  //  predict time
  // Update the estimated state based on measurement
  Eigen::MatrixXd update(const Eigen::VectorXd & z);
  
private:
  
  // Posteriori state
  VoidMatFunc u_f;
  Eigen::VectorXd x_post; // State vector
  // Priori state
  Eigen::VectorXd x_pri;
  Eigen::MatrixXd F; // State transition matrix
  Eigen::MatrixXd P; // Estimate error covariance
  Eigen::MatrixXd Q; // Process noise covariance
  Eigen::MatrixXd H; // Measurement matrix
  Eigen::MatrixXd R; // Measurement noise covariance
};

}  // namespace rm_auto_aim

#endif  // ENERGY__KALMAN_FILTER_HPP_
