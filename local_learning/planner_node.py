import math
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import LaserScan
from global_to_polar_cpp.msg import PolarGrid
from f1tenth_planning_custom_msgs.msg import PathWithVelocity, Waypoint

import torch


class LocalLearningPlanner(Node):
    """ROS2 node using a pre-trained PyTorch model for local planning."""

    def __init__(self) -> None:
        super().__init__('local_learning')
        package_dir = Path(__file__).resolve().parent.parent
        default_model_path = str(package_dir / 'model/mobilenet_trained_updated.pt')
        self.declare_parameter('model_path', default_model_path)
        model_path_param = (
            self.get_parameter('model_path').get_parameter_value().string_value
        )
        model_path = Path(model_path_param)
        if not model_path.is_absolute():
            model_path = package_dir / model_path
        if model_path.exists():
            self.model = torch.jit.load(str(model_path))
            self.model.eval()
        else:
            self.get_logger().warn(
                f'Model file not found at {model_path}, planner will not run.'
            )
            self.model = None

        # Synchronize LaserScan and PolarGrid messages
        self.scan_sub = Subscriber(self, LaserScan, '/scan')
        self.grid_sub = Subscriber(self, PolarGrid, '/polar_grid')
        self.sync = ApproximateTimeSynchronizer(
            [self.scan_sub, self.grid_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self.synced_callback)

        self.publisher = self.create_publisher(
            PathWithVelocity, '/planned_path_with_velocity', 10
        )

    def preprocess(self, scan: LaserScan, grid: PolarGrid) -> torch.Tensor:
        scan_ranges = np.array(scan.ranges, dtype=np.float32)
        grid_ranges = np.array(grid.ranges, dtype=np.float32)

        if scan_ranges.shape[0] != 1080 or grid_ranges.shape[0] != 1080:
            self.get_logger().warn('Expected 1080 points in both scan and grid.')
            return None

        # Replace NaN/Inf and clip to valid range
        scan_ranges = np.nan_to_num(scan_ranges, nan=0.0, posinf=0.0, neginf=0.0)
        grid_ranges = np.nan_to_num(grid_ranges, nan=0.0, posinf=0.0, neginf=0.0)

        tensor = torch.from_numpy(np.stack([scan_ranges, grid_ranges])).unsqueeze(0)
        return tensor

    def synced_callback(self, scan: LaserScan, grid: PolarGrid) -> None:
        if self.model is None:
            return
        inputs = self.preprocess(scan, grid)
        if inputs is None:
            return

        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = outputs.squeeze(0).cpu().numpy()  # (10, 4)

        msg = PathWithVelocity()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        for row in outputs:
            waypoint = Waypoint()
            waypoint.x = float(row[0])
            waypoint.y = float(row[1])
            waypoint.yaw = float(row[2])
            waypoint.velocity = float(row[3])
            msg.points.append(waypoint)

        # Ensure the message always has 10 waypoints
        if len(msg.points) < 10:
            for _ in range(10 - len(msg.points)):
                msg.points.append(Waypoint())

        self.publisher.publish(msg)


def main(args: List[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LocalLearningPlanner()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
