// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

// C++ system
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/en_target.hpp"
#include "auto_aim_interfaces/srv/tracking_mode.hpp"
namespace rm_serial_driver
{
class RMSerialDriver : public rclcpp::Node
{
public:
  explicit RMSerialDriver(const rclcpp::NodeOptions & options);

  ~RMSerialDriver() override;

private:
  void initParams();

  void getParams();

  void receiveData();

  void sendData(auto_aim_interfaces::msg::Target::SharedPtr msg);

  void reopenPort();

  void setParam(const rclcpp::Parameter & param);

  void resetTracker(uint8_t mode);

  void sendEnData(auto_aim_interfaces::msg::EnTarget::SharedPtr msg);

  void resetEnMode(uint8_t mode);

  void setCamParam(uint8_t mode);
  //params
  std::vector<rclcpp::Parameter> Armor_parameters;
  std::vector<rclcpp::Parameter> Energy_parameters;
  // Serial port
  std::unique_ptr<IoContext> owned_ctx_;
  std::string device_name_;
  std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
  std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

  // Param client to set detect_colr
  using ResultFuturePtr = std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;
  bool initial_set_param_ = false;
  uint8_t previous_receive_color_ = 0;
  uint8_t previous_receive_tracker_mode_=0;
  rclcpp::AsyncParametersClient::SharedPtr detector_param_client_;
  rclcpp::AsyncParametersClient::SharedPtr Endetector_param_client_;
  rclcpp::AsyncParametersClient::SharedPtr cam_parameter_client;
  ResultFuturePtr set_param_future_;

  // Service client to reset tracker
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_tracker_client_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_EnTracker_client_;
  rclcpp::Client<auto_aim_interfaces::srv::TrackingMode>::SharedPtr EnTracker_mode_client;
  
  // Aimimg point receiving from serial port for visualization
  visualization_msgs::msg::Marker aiming_point_;

  // Broadcast tf from odom to gimbal_link
  double timestamp_offset_ = 0;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;
  rclcpp::Subscription<auto_aim_interfaces::msg::EnTarget>::SharedPtr en_target_sub_;

  // For debug usage
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  std::thread receive_thread_;
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
