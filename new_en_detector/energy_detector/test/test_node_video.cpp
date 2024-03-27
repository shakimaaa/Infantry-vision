#include <string>
#include "energy_detector/En_detector.hpp"
#include "gtest/gtest.h"
#include <rclcpp/executors.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/utilities.hpp>
#include "energy_detector/energy_detector_node.hpp"
#include "auto_aim_interfaces/msg/tracker2_d.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
using namespace rm_auto_aim;

class Test_node : public rclcpp::Node
{
public:
    cv::Point2f pre_Point;
    Test_node(const rclcpp::NodeOptions &options) : Node("TestNode", options)
    {
        timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.0);
        Test_Sub = this->create_subscription<auto_aim_interfaces::msg::Tracker2D>("tracker/LeafTarget2D", rclcpp::SensorDataQoS(), std::bind(&Test_node::Test_SubCallback, this, std::placeholders::_1));
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        receive_thread_ = std::thread(&Test_node::receiveData, this);
    }

private:
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    void receiveData()
    {
        while (rclcpp::ok)
        {
            geometry_msgs::msg::TransformStamped t;
            timestamp_offset_ = get_parameter("timestamp_offset").as_double();
            t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
            t.header.frame_id = "odom";
            t.child_frame_id = "gimbal_link";
            tf2::Quaternion q;
            q.setRPY(0, 0, 0);
            q.setX(0);
            q.setY(0);
            q.setZ(0);
            t.transform.rotation = tf2::toMsg(q);
            tf_broadcaster_->sendTransform(t);
        }
    }
    double timestamp_offset_ = 0.05;
    rclcpp::Subscription<auto_aim_interfaces::msg::Tracker2D>::SharedPtr Test_Sub;
    void Test_SubCallback(const auto_aim_interfaces::msg::Tracker2D::SharedPtr Point)
    {
        RCLCPP_INFO(this->get_logger(), "x:%.2f y:%.2f", Point->x, Point->y);
        pre_Point.x = Point->x;
        pre_Point.y = Point->y;
    }
    std::thread receive_thread_;
};

TEST(energy_detector, test_node_video)
{
#ifndef VIDEO
#define VIDEO
#endif
#ifndef TEST_DIR
#define TEST_DIR
#endif
    rclcpp::NodeOptions options;

    auto node = std::make_shared<rm_auto_aim::EnergyDetector>(options);
    auto test_node = std::make_shared<Test_node>(options);
    std::string video_name[] = {"buff_blue.mp4", "2xfile.mp4", "file.mp4", "比赛打符高环数合集.mp4", "环数检测视频.mp4", "快速激活大能量机关视频.mp4", "en1.mp4"};
    std::string video_path = std::string(TEST_DIR) + "/video/" + video_name[1];
    cv::VideoCapture cap(video_path);
    cv::Mat bin;
    while (true)
    {
        rclcpp::spin_some(node);
        rclcpp::spin_some(test_node);
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        cv::imshow("src", frame);
        cv::waitKey(10);
        cv::Mat color_frame = frame.clone();
        color_frame = node->VideoTest(color_frame, bin);
        cv::circle(color_frame, test_node->pre_Point, 5, cv::Scalar(0, 0, 255), -1);
        cv::imshow("color_frame", color_frame);
        cv::waitKey(10);
        cv::imshow("bin", bin);
        cv::waitKey(10);
    }
    std::cout << "avg latency:" << node->sum_latency / node->count << std::endl;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    auto result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
