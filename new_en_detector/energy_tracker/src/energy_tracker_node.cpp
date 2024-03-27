#include "energy_tracker/energy_tracker_node.hpp"

// STD
#include <memory>
#include <vector>

namespace rm_auto_aim
{

   EnergyTrackerNode::EnergyTrackerNode(const rclcpp::NodeOptions &options)
       : Node("energy_tracker", options)
   {
      RCLCPP_INFO(this->get_logger(), "Starting EnergyTarckerNode!");

      tracker_ = std::make_unique<EnTracker>();
      auto f = [this]()
      {
         Eigen::MatrixXd F(3, 3);
         // clang-format off
         F << 1*dt_, 1*dt_, 0.5*dt_*dt_,
              0,    1,      1*dt_,
              0,    0,      1;
         // clang-format on
         return F;
      };
      tracker_->ekf = ExtendedKalmanFilter(f);
      // Subscriber with tf2 message_filter
      // tf2 relevant
      tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
      // Create the timer interface before call to waitForTransform,
      // to avoid a tf2_ros::CreateTimerInterfaceException exception
      auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
          this->get_node_base_interface(), this->get_node_timers_interface());
      tf2_buffer_->setCreateTimerInterface(timer_interface);
      tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
      // subscriber and filter
      leafs_sub_.subscribe(this, "/detector/leafs", rmw_qos_profile_sensor_data);
      target_frame_ = this->declare_parameter("target_frame", "odom");
      tf2_filter_ = std::make_shared<tf2_filter>(
          leafs_sub_, *tf2_buffer_, target_frame_, 10, this->get_node_logging_interface(),
          this->get_node_clock_interface(), std::chrono::duration<int>(1));
      // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
      tf2_filter_->registerCallback(&EnergyTrackerNode::LeafsCallback, this);
      // Publisher
      target_pub_ = this->create_publisher<auto_aim_interfaces::msg::EnTarget>(
          "/tracker/EnTarget", rclcpp::SensorDataQoS());
      target_2d_pub_ = this->create_publisher<auto_aim_interfaces::msg::Tracker2D>("tracker/LeafTarget2D", rclcpp::SensorDataQoS());
   }
   float EnergyTrackerNode::Gravity_compensation(double bottom_len, double angle_0, double z)
   {
      const double g = 9.788; // m/s
      double t = bottom_len / (v * cos(angle_0));
      double h = g * t * t / 2.0;
      return atan2(z + h, bottom_len) * 180 / M_PI;
   }
   void EnergyTrackerNode::LeafsCallback(const auto_aim_interfaces::msg::Leafs::SharedPtr leafs_msg)
   {
      if (leafs_msg->leafs.empty())
         return;

      // find the best match leaf
      auto leaf_ = leafs_msg->leafs[0];
      for (auto &leaf : leafs_msg->leafs)
      {
         if (leaf_.prob > leaf.prob)
         {
            leaf_ = leaf;
         }
      }
      geometry_msgs::msg::PoseStamped ps;
      ps.header = leafs_msg->header;
      ps.pose = leaf_.pose;
      if (!leaf_.type)
         return;
      try
      {
         leaf_.pose = tf2_buffer_->transform(ps, target_frame_).pose;
      }
      catch (const tf2::ExtrapolationException &ex)
      {
         RCLCPP_ERROR(get_logger(), "Error while transforming %s", ex.what());
         return;
      }
      // Init message
      // auto_aim_interfaces::msg::TrackerInfo info_msg;
      auto_aim_interfaces::msg::EnTarget target_msg;
      auto_aim_interfaces::msg::Tracker2D target_msg_2d;
      rclcpp::Time time = leafs_msg->header.stamp;
      target_msg.header.stamp = time;
      target_msg.header.frame_id = target_frame_;
      // Update tracker
      if (tracker_->tracker_state == EnTracker::LOST)
      {
         tracker_->init(leaf_);
         angle0 = tracker_->angleSolver(leaf_);
         RCLCPP_INFO(this->get_logger(), "angle0:%f", angle0);
         last_time_ = time;
         tracker_->tracker_state = EnTracker::TRACKING;
         return;
      }
      angle1 = tracker_->angleSolver(leaf_);
      if (rad_destation == 0)
      {
         rad_destation = angle1 - angle0 < 0 ? 1 : -1;
         RCLCPP_INFO(this->get_logger(), "angle1:%f", angle1);
      }
      dt_ = (time.seconds() - last_time_.seconds())/9;
      tracker_->ekf.t = dt_;
      tracker_->update(leaf_);
      const auto &state = tracker_->target_state;
      double angle_ = state(0), angle_v = state(1);
      RCLCPP_INFO(rclcpp::get_logger("energy_tracker"), "predict angle:%f,angle_v:%f,dt_:%f", angle_ * 180 / M_PI, angle_v * tracker_->ekf.t, dt_);
      Eigen::Vector2d p1(leaf_.leaf_center.z, leaf_.leaf_center.y);
      Eigen::Vector2d p2(leaf_.r_center.z, leaf_.r_center.y);
      double r_distance = (p1 - p2).norm();
      target_msg.position.x = leaf_.pose.position.x;
      target_msg.position.y = leaf_.r_center.y + r_distance * cos(angle_) * rad_destation;
      target_msg.position.z = leaf_.r_center.z + r_distance * sin(angle_) * rad_destation;
      RCLCPP_INFO(rclcpp::get_logger("energy_tracker"), "predict x:%f,predict y:%f", target_msg.position.z, target_msg.position.y);
      target_msg.angle = angle_;
      target_msg_2d.x = target_msg.position.y;
      target_msg_2d.y = target_msg.position.z;
      target_msg.yaw = atan2(target_msg.position.x, target_msg.position.y) * 180 / M_PI;
      double bottom_len = sqrt(pow(target_msg.position.x, 2.0) + pow(target_msg.position.y, 2.0));
      target_msg.pitch = atan2(target_msg.position.z, bottom_len); //   pitch angle
      target_msg.pitch = Gravity_compensation(bottom_len, target_msg.pitch, target_msg.position.x);
      last_time_ = time;
      target_pub_->publish(target_msg);
      target_2d_pub_->publish(target_msg_2d);
   }

} // namespace rm_auto_aim
#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::EnergyTrackerNode)