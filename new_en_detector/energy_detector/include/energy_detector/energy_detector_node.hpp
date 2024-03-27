#ifndef ENERGY_DETECTOR_NODE_HPP_
#define ENERGY_DETECTOR_NODE_HPP_

#include <cv_bridge/cv_bridge.h>

// ROS
#include <rmw/qos_profiles.h>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

// STD
#include <memory>
#include <string>
#include <vector>

#include "energy_detector/En_detector.hpp"
#include "energy_detector/pnp_solver.hpp"
#include "auto_aim_interfaces/msg/leafs.hpp"
#include "auto_aim_interfaces/msg/debug_leafs.hpp"
namespace rm_auto_aim{
class EnergyDetector : public rclcpp::Node {
public:
    EnergyDetector(const rclcpp::NodeOptions& options);
    cv::Mat VideoTest(cv::Mat &img,cv::Mat &bin);
    long long int count=0;
    long double sum_latency=0;
private:
    void ImageCallBack(const sensor_msgs::msg::Image::SharedPtr _ros_image);
    void createDebugPublishers();
    void destroyDebugPublishers();
    std::unique_ptr<En_Detector> initDetector();
    std::vector<Leaf> detectLeafs(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg);
    void publishMarkers();
    // Energy_leaf Detector
    std::unique_ptr<En_Detector>detector_;

    // Detected leafs && R publisher
    auto_aim_interfaces::msg::Leafs leafs_msg_;
    rclcpp::Publisher<auto_aim_interfaces::msg::Leafs>::SharedPtr leafs_pub_;

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    cv::Point2f cam_center_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    std::unique_ptr<PnPSolver> pnp_solver_;

    //  Visualization marker publisher
    visualization_msgs::msg::Marker leaf_marker_;
    visualization_msgs::msg::Marker text_marker_;
    visualization_msgs::msg::MarkerArray marker_array_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    //  Debug information
    bool debug_;
    std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
    std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
    rclcpp::Publisher<auto_aim_interfaces::msg::DebugLeafs>::SharedPtr leafs_data_pub;
    image_transport::Publisher result_img_pub_;
    image_transport::Publisher binary_img_pub_;

    // Image subscrpition
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

};
}
#endif