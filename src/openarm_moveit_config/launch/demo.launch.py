import os

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def _generate_robot_description(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    can_interface,
    arm_prefix,
):
    description_package_str = context.perform_substitution(description_package)
    description_file_str = context.perform_substitution(description_file)
    arm_type_str = context.perform_substitution(arm_type)
    use_fake_hardware_str = context.perform_substitution(use_fake_hardware)
    can_interface_str = context.perform_substitution(can_interface)
    arm_prefix_str = context.perform_substitution(arm_prefix)

    xacro_path = os.path.join(
        get_package_share_directory(description_package_str),
        "urdf",
        "robot",
        description_file_str,
    )

    robot_description = xacro.process_file(
        xacro_path,
        mappings={
            "arm_type": arm_type_str,
            "bimanual": "false",
            "use_fake_hardware": use_fake_hardware_str,
            "ros2_control": "true",
            "can_interface": can_interface_str,
            "arm_prefix": arm_prefix_str,
        },
    ).toprettyxml(indent="  ")

    return robot_description


def _robot_nodes_spawner(
    context: LaunchContext,
    description_package,
    description_file,
    arm_type,
    use_fake_hardware,
    controllers_file,
    can_interface,
    arm_prefix,
):
    robot_description = _generate_robot_description(
        context,
        description_package,
        description_file,
        arm_type,
        use_fake_hardware,
        can_interface,
        arm_prefix,
    )

    controllers_file_str = context.perform_substitution(controllers_file)
    robot_description_param = {"robot_description": robot_description}

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description_param],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="both",
        parameters=[robot_description_param, controllers_file_str],
    )

    static_world_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_openarm_base",
        arguments=["0", "0", "0", "0", "0", "0", "world", "openarm_link0"],
        output="screen",
    )

    return [robot_state_pub_node, control_node, static_world_tf]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument(
            "description_package",
            default_value="openarm_description",
        ),
        DeclareLaunchArgument(
            "description_file",
            default_value="v10.urdf.xacro",
        ),
        DeclareLaunchArgument(
            "arm_type",
            default_value="v10",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true",
            description="true=mock_components, false=real OpenArm hardware plugin",
        ),
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="openarm_bringup",
        ),
        DeclareLaunchArgument(
            "can_interface",
            default_value="can0",
        ),
        DeclareLaunchArgument(
            "arm_prefix",
            default_value="",
        ),
        DeclareLaunchArgument(
            "controllers_file",
            default_value="openarm_v10_controllers.yaml",
        ),
    ]

    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    arm_type = LaunchConfiguration("arm_type")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    can_interface = LaunchConfiguration("can_interface")
    arm_prefix = LaunchConfiguration("arm_prefix")

    controllers_file = PathJoinSubstitution(
        [
            FindPackageShare(runtime_config_package),
            "config",
            "v10_controllers",
            controllers_file,
        ]
    )

    robot_nodes_spawner_func = OpaqueFunction(
        function=_robot_nodes_spawner,
        args=[
            description_package,
            description_file,
            arm_type,
            use_fake_hardware,
            controllers_file,
            can_interface,
            arm_prefix,
        ],
    )

    jsb_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    arm_ctrl_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    gripper_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "-c", "/controller_manager"],
    )

    delayed_jsb = TimerAction(period=2.0, actions=[jsb_spawner])
    delayed_arm_ctrl = TimerAction(period=2.0, actions=[arm_ctrl_spawner])
    delayed_gripper = TimerAction(period=2.0, actions=[gripper_spawner])

    moveit_config = MoveItConfigsBuilder(
        "openarm", package_name="openarm_moveit_config"
    ).to_moveit_configs()
    moveit_params = moveit_config.to_dict()

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_params],
    )

    rviz_cfg = os.path.join(
        get_package_share_directory("openarm_moveit_config"), "config", "moveit.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_cfg],
        parameters=[moveit_params],
    )

    return LaunchDescription(
        declared_arguments
        + [
            robot_nodes_spawner_func,
            delayed_jsb,
            delayed_arm_ctrl,
            delayed_gripper,
            move_group_node,
            rviz_node,
        ]
    )

