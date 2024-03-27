#include "energy_tracker/energy_extended_kalman_filter.hpp"
namespace rm_auto_aim
{
ExtendedKalmanFilter::ExtendedKalmanFilter(const VoidMatFunc & u_f):u_f(u_f)
{
  int n = 3; // State vector size: [angle, Angular_velocity,Angular_a]
  int m = 1; // Measurement vector size: [angle]

  x_post = Eigen::VectorXd(n);
  x_pri = Eigen::VectorXd(n);
  F = Eigen::MatrixXd(n, n);
  P = Eigen::MatrixXd(n, n);
  Q = Eigen::MatrixXd(n, n);
  H = Eigen::MatrixXd(m, n);
  R = Eigen::MatrixXd(m, m);

  // Initial state
  x_post << 0, 0, 0;

  x_pri << 0, 0, 0;


  // State transition matrix
  // F << 1*t, 1*t, 0.5*t*t,
  //      0,   1,    1*t,
  //      0,   0,    1;

  // Process covariance matrix
  Q << 0.1, 0, 0,
       0, 0.1, 0,
       0, 0, 0.1;

  // Measurement matrix
  H << 1, 0, 0;

  // Measurement covariance matrix
  R << 0.1;

  // Initial estimate error covariance
  P << 0.05, 0, 0,
       0, 0.05, 0,
       0, 0, 0.05;
      
  
}

void ExtendedKalmanFilter::setState(const Eigen::VectorXd & x0) {x_post = x0; }

Eigen::MatrixXd ExtendedKalmanFilter::predict()
{
  F=u_f();
  x_pri = F * x_post;
  P = F * P * F.transpose() + Q;
  x_post = x_pri;
  return x_pri;
}

Eigen::MatrixXd ExtendedKalmanFilter::update(const Eigen::VectorXd & z)
{
  RCLCPP_DEBUG(rclcpp::get_logger("energy_tracker"), "EKF update");
  Eigen::VectorXd y = z - H * x_post;
  Eigen::MatrixXd S = H * P * H.transpose() + R;
  Eigen::MatrixXd K = P * H.transpose() * S.inverse();

  x_post = x_post + K * y;
  P = (Eigen::MatrixXd::Identity(x_post.size(), x_post.size()) - K * H) * P;

  return x_post;
}

}
