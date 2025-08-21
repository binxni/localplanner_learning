#include <memory>
#include <optional>
#include <filesystem>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "global_to_polar_cpp/msg/polar_grid.hpp"
#include "f1tenth_planning_custom_msgs/msg/path_with_velocity.hpp"
#include "f1tenth_planning_custom_msgs/msg/waypoint.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "ament_index_cpp/get_package_share_directory.hpp"

#include <torch/torch.h>
#include <torch/script.h>

class LocalLearningPlanner : public rclcpp::Node
{
public:
  using Scan = sensor_msgs::msg::LaserScan;
  using Grid = global_to_polar_cpp::msg::PolarGrid;
  using PathWithVelocity = f1tenth_planning_custom_msgs::msg::PathWithVelocity;
  using Waypoint = f1tenth_planning_custom_msgs::msg::Waypoint;

  LocalLearningPlanner()
  : Node("local_learning"),
    scan_sub_(this, "/scan"),
    grid_sub_(this, "/polar_grid"),
    sync_(SyncPolicy(10), scan_sub_, grid_sub_)
  {
    auto package_dir = ament_index_cpp::get_package_share_directory("local_learning");
    std::string default_model_path = package_dir + "/model/mobilenet_trained_updated.pt";
    this->declare_parameter("model_path", default_model_path);
    std::string model_path = this->get_parameter("model_path").as_string();
    if (!std::filesystem::path(model_path).is_absolute()) {
      model_path = package_dir + std::string("/") + model_path;
    }

    if (std::filesystem::exists(model_path)) {
      try {
        model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path));
        model_->eval();
      } catch (const c10::Error &e) {
        RCLCPP_WARN(this->get_logger(), "Failed to load model: %s", e.what());
        model_.reset();
      }
    } else {
      RCLCPP_WARN(this->get_logger(),
                  "Model file not found at %s, planner will not run.",
                  model_path.c_str());
    }

    sync_.registerCallback(std::bind(&LocalLearningPlanner::synced_callback,
                                     this,
                                     std::placeholders::_1,
                                     std::placeholders::_2));

    publisher_ = this->create_publisher<PathWithVelocity>(
        "/planned_path_with_velocity", 10);
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Scan, Grid>;

  std::optional<torch::Tensor> preprocess(const Scan &scan, const Grid &grid)
  {
    if (scan.ranges.size() != 1080 || grid.ranges.size() != 1080) {
      RCLCPP_WARN(this->get_logger(),
                  "Expected 1080 points in both scan and grid.");
      return std::nullopt;
    }
    auto scan_tensor = torch::from_blob(
        const_cast<float *>(scan.ranges.data()), {1080}, torch::kFloat32).clone();
    auto grid_tensor = torch::from_blob(
        const_cast<float *>(grid.ranges.data()), {1080}, torch::kFloat32).clone();

    scan_tensor = torch::nan_to_num(scan_tensor, 0.0, 0.0, 0.0);
    grid_tensor = torch::nan_to_num(grid_tensor, 0.0, 0.0, 0.0);

    auto stacked = torch::stack({scan_tensor, grid_tensor}).unsqueeze(0);
    return stacked;
  }

  void synced_callback(const Scan::ConstSharedPtr scan,
                       const Grid::ConstSharedPtr grid)
  {
    if (!model_) {
      return;
    }
    auto input_opt = preprocess(*scan, *grid);
    if (!input_opt) {
      return;
    }

    torch::NoGradGuard no_grad;
    auto output = model_->forward({*input_opt}).toTensor();
    output = output.squeeze(0).to(torch::kCPU);

    PathWithVelocity msg;
    msg.header.stamp = this->now();
    msg.header.frame_id = "map";

    auto accessor = output.accessor<float, 2>();
    for (int i = 0; i < output.size(0); ++i) {
      Waypoint wp;
      wp.x = accessor[i][0];
      wp.y = accessor[i][1];
      wp.yaw = accessor[i][2];
      wp.velocity = accessor[i][3];
      msg.points.push_back(wp);
    }

    while (msg.points.size() < 10) {
      msg.points.emplace_back();
    }

    publisher_->publish(msg);
  }

  message_filters::Subscriber<Scan> scan_sub_;
  message_filters::Subscriber<Grid> grid_sub_;
  message_filters::Synchronizer<SyncPolicy> sync_;
  rclcpp::Publisher<PathWithVelocity>::SharedPtr publisher_;
  std::shared_ptr<torch::jit::script::Module> model_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalLearningPlanner>());
  rclcpp::shutdown();
  return 0;
}

