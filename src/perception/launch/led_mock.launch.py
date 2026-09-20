from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Odometry relayed into TF as map -> base_link. Ground truth by default; the
        # navigation-error injector's output when a level is set (#69).
        DeclareLaunchArgument('robot_odom_topic', default_value='/model/bluerov2_heavy/odometry'),
        Node(
            package='perception',
            executable='led_mock_publisher',
            name='led_mock_publisher',
            parameters=[{
                'noise_stddev_m': 0.01,
                'detection_distance_m': 10.0,
                'robot_odom_topic': LaunchConfiguration('robot_odom_topic'),
            }],
            output='screen',
        ),
    ])
