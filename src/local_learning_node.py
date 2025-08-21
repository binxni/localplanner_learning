#!/usr/bin/env python3
"""Local planner node that uses a Torch model to predict waypoints.

The node subscribes to ``/scan`` and ``/polar_grid`` and publishes a
``PathWithVelocity`` message on ``/planned_path_with_velocity``. Incoming scan
and grid messages are cached and processed at a fixed rate when both are
available and close in time.
"""

from __future__ import annotations

import os
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from global_to_polar_cpp.msg import PolarGrid
from f1tenth_planning_custom_msgs.msg import PathWithVelocity, Waypoint
from ament_index_python.packages import get_package_share_directory

import torch


class LocalLearningPlanner(Node):
    """ROS2 node wrapping a Torch model for local planning."""

    def __init__(self) -> None:
        super().__init__("local_learning")

        package_dir = get_package_share_directory("local_learning")
        default_model = os.path.join(
            package_dir, "model", "mobilenet_trained_updated.pt"
        )
        self.declare_parameter("model_path", default_model)
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not os.path.isabs(model_path):
            model_path = os.path.join(package_dir, model_path)

        self.model: Optional[torch.jit.ScriptModule] = None
        if os.path.exists(model_path):
            try:
                self.model = torch.jit.load(model_path)
                self.model.eval()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"Could not load model: {exc}")
        else:
            self.get_logger().error(f"Model file not found: {model_path}")

        qos = QoSProfile(depth=10)
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_cb, qos)
        self.grid_sub = self.create_subscription(
            PolarGrid, "/polar_grid", self._grid_cb, qos
        )
        self.publisher = self.create_publisher(
            PathWithVelocity, "/planned_path_with_velocity", qos
        )

        self._latest_scan: Optional[LaserScan] = None
        self._latest_grid: Optional[PolarGrid] = None

        # Process input at 20 Hz
        self.timer = self.create_timer(0.05, self._on_timer)

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg

    def _grid_cb(self, msg: PolarGrid) -> None:
        self._latest_grid = msg

    def _on_timer(self) -> None:
        if self.model is None or self._latest_scan is None or self._latest_grid is None:
            return

        # Ensure the two inputs are approximately synchronized
        dt = self._time_diff(self._latest_scan, self._latest_grid)
        if dt > 0.1:
            return

        input_tensor = self._preprocess(self._latest_scan, self._latest_grid)
        if input_tensor is None:
            return

        with torch.no_grad():
            output = self.model(input_tensor).squeeze(0).cpu()

        msg = PathWithVelocity()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        for row in output:
            wp = Waypoint()
            wp.x = float(row[0])
            wp.y = float(row[1])
            wp.yaw = float(row[2])
            wp.velocity = float(row[3])
            msg.points.append(wp)

        self.publisher.publish(msg)

    @staticmethod
    def _time_diff(scan: LaserScan, grid: PolarGrid) -> float:
        scan_time = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        grid_time = grid.header.stamp.sec + grid.header.stamp.nanosec * 1e-9
        return abs(scan_time - grid_time)

    def _preprocess(
        self, scan: LaserScan, grid: PolarGrid
    ) -> Optional[torch.Tensor]:
        if len(scan.ranges) != len(grid.ranges):
            self.get_logger().warn("Scan and grid lengths differ")
            return None

        scan_tensor = torch.tensor(scan.ranges, dtype=torch.float32)
        grid_tensor = torch.tensor(grid.ranges, dtype=torch.float32)
        scan_tensor = torch.nan_to_num(scan_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        grid_tensor = torch.nan_to_num(grid_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.stack((scan_tensor, grid_tensor)).unsqueeze(0)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = LocalLearningPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()

