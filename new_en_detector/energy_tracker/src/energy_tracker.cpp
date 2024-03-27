#include "energy_tracker/energy_tracker.hpp"
namespace rm_auto_aim
{
    EnTracker::EnTracker()
    {
        tracker_state = LOST;
    }

    float EnTracker::angleSolver(auto_aim_interfaces::msg::Leaf leaf)
    {
        float y=leaf.leaf_center.z-leaf.r_center.z;
        float x=leaf.leaf_center.y-leaf.r_center.y;
        float angle=cv::fastAtan2(y,x)/180.0*M_PI;
        return angle;
    }

    void EnTracker::init(const Leaf &l)
    {
        initEKF(l);
        RCLCPP_DEBUG(rclcpp::get_logger("energy_tracker"), "Init EKF!");
    }

    void EnTracker::update(const Leaf &l)
    {
        RCLCPP_INFO(rclcpp::get_logger("energy_tracker"), "current angle:%f,e_x:%f,e_y:%f", angleSolver(l)*180/M_PI,l.leaf_center.y-l.r_center.y,l.leaf_center.z-l.r_center.z);
        Eigen::VectorXd ekf_prediction = ekf.predict();
        target_state = ekf_prediction;
        measurement = Eigen::VectorXd(1);
        measurement << angleSolver(l);
        target_state = ekf.update(measurement);
    }

    void EnTracker::initEKF(const Leaf &l)
    {
        target_state = Eigen::VectorXd::Zero(3);
        float angle = angleSolver(l);
        target_state << angle, 0, 0;
        ekf.setState(target_state);
    }
}
