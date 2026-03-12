# OpenArm YOLO Pose Teleoperation

OpenArm mirror control using YOLOv8 pose

#실행
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000 dbitrate 5000000 fd on sample-point 0.750

sudo ip link set can1 down
sudo ip link set can1 up type can bitrate 1000000 dbitrate 5000000 fd on sample-point 0.750

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/src/openarm_portfolio_tools/scripts/setup_ros2_only.sh

ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=v10 use_fake_hardware:=false


#yolo 실행
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/src/openarm_portfolio_tools/scripts/setup_ros2_only.sh
python3 ~/ros2_ws/pose_control/human_to_openarm_bimanual.py
