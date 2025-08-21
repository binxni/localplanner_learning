#!/usr/bin/env python3
import os
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from global_to_polar_cpp.msg import PolarGrid
from f1tenth_planning_custom_msgs.msg import PathWithVelocity, Waypoint
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory
import torch


class LocalLearningPlanner(Node):
    def __init__(self) -> None:
        super().__init__('local_learning')

        package_dir = get_package_share_directory('local_learning')
        default_model_path = os.path.join(
            package_dir, 'model', 'mobilenet_trained_updated.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.isabs(model_path):
            model_path = os.path.join(package_dir, model_path)

        self.model: Optional[torch.jit.ScriptModule] = None
        if os.path.exists(model_path):
            try:
                self.model = torch.jit.load(model_path)
                self.model.eval()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f'Failed to load model: {e}')
                self.model = None
        else:
            self.get_logger().warn(
                f'Model file not found at {model_path}, planner will not run.'
            )

        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.grid_sub = Subscriber(self, PolarGrid, '/polar_grid')
        self.sync = ApproximateTimeSynchronizer(
            [self.scan_sub, self.grid_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.synced_callback)

        self.publisher = self.create_publisher(
            PathWithVelocity, '/planned_path_with_velocity', 10
        )

    def preprocess(self, scan: LaserScan, grid: PolarGrid) -> Optional[torch.Tensor]:
        if len(scan.ranges) != 1080 or len(grid.ranges) != 1080:
            self.get_logger().warn(
                'Expected 1080 points in both scan and grid.'
            )
            return None

        scan_tensor = torch.tensor(scan.ranges, dtype=torch.float32)
        grid_tensor = torch.tensor(grid.ranges, dtype=torch.float32)
        scan_tensor = torch.nan_to_num(scan_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        stacked = torch.stack((scan_tensor, grid_tensor)).unsqueeze(0)
        return stacked

    def synced_callback(self, scan: LaserScan, grid: PolarGrid) -> None:
        if self.model is None:
            return

        input_tensor = self.preprocess(scan, grid)
        if input_tensor is None:
            return

        with torch.no_grad():
            output = self.model(input_tensor).squeeze(0).cpu()

        msg = PathWithVelocity()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for row in output:
            wp = Waypoint()
            wp.x = float(row[0])
            wp.y = float(row[1])
            wp.yaw = float(row[2])
            wp.velocity = float(row[3])
            msg.points.append(wp)

        while len(msg.points) < 10:
            msg.points.append(Waypoint())

        self.publisher.publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = LocalLearningPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
