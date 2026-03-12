#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint

import os
import urllib.request

import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
import mediapipe as mp


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_norm(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return n if n > eps else eps


class SimpleArmControllerRealSense(Node):
    def __init__(self):
        super().__init__('simple_yolo_arm_realsense')

        # =========================
        # 1) YOLO 설정
        # =========================
        self.model = YOLO("yolov8m-pose.pt")

        # =========================
        # 2) RealSense (D455) 설정
        # =========================
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.width = 1280
        self.height = 720
        self.fps = 30

        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        # depth를 color에 정렬 (color 픽셀 좌표로 depth 조회)
        self.align = rs.align(rs.stream.color)

        # 워밍업
        for _ in range(10):
            self.pipeline.wait_for_frames()

        self.get_logger().info("✅ RealSense 스트림 시작됨 (Color+Depth)")

        # 컬러 intrinsics (deproject용)
        color_stream = self.profile.get_stream(rs.stream.color)
        color_vsp = color_stream.as_video_stream_profile()
        self.color_intr = color_vsp.get_intrinsics()

        # =========================
        # 3) 로봇 팔 액션 클라이언트
        # =========================
        self.left_client = ActionClient(
            self, FollowJointTrajectory,
            '/left_joint_trajectory_controller/follow_joint_trajectory'
        )
        self.right_client = ActionClient(
            self, FollowJointTrajectory,
            '/right_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.get_logger().info("⏳ Trajectory action server 대기 중...")
        self.left_client.wait_for_server()
        self.right_client.wait_for_server()
        self.get_logger().info("✅ Action server 연결 완료")

        # =========================
        # 4) 스무딩 버퍼 (크게 해서 흔들림 감소)
        # =========================
        self.left_buffer = deque(maxlen=12)
        self.right_buffer = deque(maxlen=12)

        # =========================
        # 5) 캘리브레이션
        # =========================
        self.calibration_frames = []
        self.is_calibrated = False
        self.reference_pose_3d = None
        self.calib_needed = 30

        # =========================
        # 6) 관절 방향 보정 (왼팔만 반전 가정)
        # =========================
        self.ARM_JOINT_SIGN = {
            'left':  np.array([-1, -1, +1, -1, +1, +1, +1], dtype=float),
            'right': np.array([+1, +1, +1, +1, +1, +1, +1], dtype=float),
        }

        # =========================
        # 7) 관절 제한 (뒤로 굽힘/뒤로 젖힘 허용 범위 확장)
        #    * 실제 로봇 한계에 맞게 꼭 조정하세요.
        # =========================
        self.J_LIMITS = {
            'j1': (-1.6, 1.6),   # shoulder yaw
            'j2': (-1.2, 1.8),   # shoulder pitch
            'j3': (-1.2, 1.2),   # shoulder roll (depth 포함)
            'j4': (-1.2, 2.2),   # elbow flex: 음수=반대방향(“뒤로”) 굽힘 허용
            'j5': (-2.5, 2.5),
            'j6': (-1.8, 1.8),
            'j7': (-2.8, 2.8),
        }
        # OPENARM 중앙 기둥 관통 방지: 왼팔 j1≤0, 오른팔 j1≥0
        self.J1_LEFT_MAX = 0.0
        self.J1_RIGHT_MIN = 0.0

        # =========================
        # 8) MediaPipe 0.10 Tasks API — HandLandmarker (손 모양: 펴기/주먹 → 그리퍼)
        # =========================
        _model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        _cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(_cache_dir, exist_ok=True)
        _model_path = os.path.join(_cache_dir, "hand_landmarker.task")
        if not os.path.isfile(_model_path):
            self.get_logger().info("Downloading hand_landmarker.task...")
            urllib.request.urlretrieve(_model_url, _model_path)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=VisionRunningMode.IMAGE,
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)

        # 그리퍼 액션 클라이언트 (control_msgs/GripperCommand)
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')
        self.gripper_open_m = 0.04   # 열린 위치(m), 로봇에 맞게 조정
        self.gripper_max_effort = 50.0

        self.gripper_left_buffer = deque(maxlen=5)
        self.gripper_right_buffer = deque(maxlen=5)

        # =========================
        # 9) 타이머
        # =========================
        self.timer = self.create_timer(0.1, self.run)  # 10Hz

        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 RealSense + YOLO Pose + MediaPipe Hands (그리퍼 확장)")
        self.get_logger().info("📹 카메라 앞에서 차렷 자세로 2~3초간 서주세요")
        self.get_logger().info("💡 손 펴기=그리퍼 열림, 주먹=그리퍼 쥠")
        self.get_logger().info("=" * 60)

    # -------------------------
    # Robot control helpers
    # -------------------------
    def send_command(self, client, positions):
        if not client.server_is_ready():
            return

        goal = FollowJointTrajectory.Goal()
        arm_side = 'left' if client == self.left_client else 'right'

        goal.trajectory.joint_names = [
            f"openarm_{arm_side}_joint{i}" for i in range(1, 8)
        ]

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 350_000_000  # 0.35s (조금 길게 해서 흔들림 감소)

        goal.trajectory.points = [point]
        client.send_goal_async(goal)

    def smooth_positions(self, buffer, new_pos):
        buffer.append(new_pos)
        n = len(buffer)
        if n == 0:
            return new_pos
        # 단순 이동 평균으로 흔들림 감소 (가중치 균일)
        arr = np.stack(buffer, axis=0)
        return np.mean(arr, axis=0)

    # -------------------------
    # Depth + 3D helpers
    # -------------------------
    def get_depth_m(self, depth_frame: rs.depth_frame, x, y):
        """(x,y)에서 depth(m) 읽기. 0이면 주변 평균으로 보정."""
        x = int(clamp(x, 0, self.width - 1))
        y = int(clamp(y, 0, self.height - 1))

        d = float(depth_frame.get_distance(x, y))
        if d > 0.01:
            return d

        vals = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xx = int(clamp(x + dx, 0, self.width - 1))
                yy = int(clamp(y + dy, 0, self.height - 1))
                dd = float(depth_frame.get_distance(xx, yy))
                if dd > 0.01:
                    vals.append(dd)

        if vals:
            return float(np.mean(vals))
        return 0.0

    def deproject_color_pixel_to_3d(self, depth_frame, x_color, y_color):
        """
        aligned된 depth_frame + color intrinsics 기준으로,
        color 픽셀(x,y)의 depth를 읽고 3D point (meters) 반환.
        """
        d = self.get_depth_m(depth_frame, x_color, y_color)
        if d <= 0.01:
            return None, 0.0

        # rs2_deproject_pixel_to_point: [X,Y,Z] in meters (camera coord)
        pt = rs.rs2_deproject_pixel_to_point(self.color_intr, [float(x_color), float(y_color)], float(d))
        return np.array(pt, dtype=float), float(d)

    # -------------------------
    # 3D Angle estimation helpers
    # 카메라 좌표:
    #   +X: 오른쪽, +Y: 아래, +Z: 카메라에서 멀어짐
    # -------------------------
    def shoulder_yaw_pitch_roll_from_upper(self, upper3):
        """
        upper3: elbow - shoulder (3D)
        대략적인 3DOF 어깨각(ya/pitch/roll)을 추정.
        로봇 관절 정의와 1:1로 완벽히 매칭되진 않으니 스케일은 조정 대상입니다.
        """
        x, y, z = float(upper3[0]), float(upper3[1]), float(upper3[2])

        # yaw: 좌/우(수평)로 벌어짐 -> atan2(x, z)
        yaw = math.atan2(x, max(z, 1e-6))

        # pitch: 위/아래 -> -y 기준 (카메라 Y는 아래로 증가)
        horiz = math.sqrt(x * x + z * z)
        pitch = math.atan2(-y, max(horiz, 1e-6))

        # roll(깊이 포함): 팔이 몸 앞/뒤로 들어오는 느낌을 upper의 z 성분으로 반영
        # upper가 카메라 쪽(+z 작아짐)으로 오면 roll이 변하도록
        # 여기서는 간단히 atan2(z, |x|+|y|)로 만들어 "앞/뒤" 변화를 잡습니다.
        roll = math.atan2(z, abs(x) + abs(y) + 1e-6) - (math.pi / 4.0)
        # (pi/4 오프셋은 “정면 차렷”에서 roll이 0 근처로 오게 하려는 경험적 값)

        return yaw, pitch, roll

    def signed_elbow_flex(self, upper3, fore3):
        """
        팔꿈치 굽힘을 부호 포함해서 계산.
        - 크기: 두 벡터 각도 기반 flex
        - 부호: 팔의 평면 법선과 카메라 전방축(+Z) 관계로 결정(간단한 구분)
        """
        u = upper3 / safe_norm(upper3)
        f = fore3 / safe_norm(fore3)

        dot = float(np.clip(np.dot(u, f), -1.0, 1.0))
        angle = math.acos(dot)          # 0..pi
        flex = math.pi - angle          # 0(펴짐) .. pi(완전굽힘)

        # 부호 결정: cross(u, f)의 z성분으로 방향 구분
        # (완벽하진 않지만, “앞/뒤 굽힘”을 살리기엔 꽤 효과적입니다.)
        n = np.cross(u, f)              # 법선
        sign = 1.0 if float(n[2]) >= 0.0 else -1.0

        return float(sign * flex)

    def calculate_angles_3d(self, shoulder3, elbow3, wrist3, ref=None):
        """
        3D(미터) 기준 각도 계산.
        ref(캘리브 기준)가 있으면 delta로 출력.
        """
        shoulder3 = np.asarray(shoulder3, dtype=float)
        elbow3 = np.asarray(elbow3, dtype=float)
        wrist3 = np.asarray(wrist3, dtype=float)

        upper = elbow3 - shoulder3
        fore = wrist3 - elbow3

        if np.linalg.norm(upper) < 0.05 or np.linalg.norm(fore) < 0.05:
            return np.zeros(7, dtype=float)

        yaw, pitch, roll = self.shoulder_yaw_pitch_roll_from_upper(upper)
        eflex = self.signed_elbow_flex(upper, fore)

        # 스케일(로봇 관절 범위에 맞춰 경험적으로 조정)
        j1 = yaw * 1.4
        j2 = pitch * 1.6
        j3 = roll * 1.6

        # elbow: pi에 대해 2.2rad 스케일 -> 부호 포함
        j4 = (eflex / math.pi) * 2.2

        # 손목(추정): fore 벡터의 방향으로 wrist pitch/yaw를 대충 맞춤 (불안정하면 0으로 두세요)
        fx, fy, fz = float(fore[0]), float(fore[1]), float(fore[2])
        j7 = math.atan2(fx, max(fz, 1e-6)) * 0.8      # wrist yaw(대충)
        j6 = math.atan2(-fy, math.sqrt(fx*fx + fz*fz) + 1e-6) * 0.8  # wrist pitch(대충)
        j5 = 0.0  # pronation은 3점만으로 거의 불가 → 0 고정

        if ref is not None:
            ref_upper = ref['elbow'] - ref['shoulder']
            ref_fore = ref['wrist'] - ref['elbow']

            ryaw, rpitch, rroll = self.shoulder_yaw_pitch_roll_from_upper(ref_upper)
            ref_eflex = self.signed_elbow_flex(ref_upper, ref_fore)

            rj1 = ryaw * 1.4
            rj2 = rpitch * 1.6
            rj3 = rroll * 1.6
            rj4 = (ref_eflex / math.pi) * 2.2

            # 기준 자세 대비 delta
            j1 -= rj1
            j2 -= rj2
            j3 -= rj3
            j4 -= rj4

            # 손목은 기준을 빼면 더 흔들릴 수 있어 기본은 그대로 둡니다.
            # 필요하면 아래 2줄도 빼는 방식으로 바꾸세요.
            # j6 -= (math.atan2(-float(ref_fore[1]), math.sqrt(float(ref_fore[0])**2 + float(ref_fore[2])**2) + 1e-6) * 0.8)
            # j7 -= (math.atan2(float(ref_fore[0]), max(float(ref_fore[2]), 1e-6)) * 0.8)

        # clamp
        j1 = clamp(j1, *self.J_LIMITS['j1'])
        j2 = clamp(j2, *self.J_LIMITS['j2'])
        j3 = clamp(j3, *self.J_LIMITS['j3'])
        j4 = clamp(j4, *self.J_LIMITS['j4'])
        j5 = clamp(j5, *self.J_LIMITS['j5'])
        j6 = clamp(j6, *self.J_LIMITS['j6'])
        j7 = clamp(j7, *self.J_LIMITS['j7'])

        return np.array([j1, j2, j3, j4, j5, j6, j7], dtype=float)

    def apply_arm_sign(self, arm_name, joints7):
        return joints7 * self.ARM_JOINT_SIGN[arm_name]

    # -------------------------
    # MediaPipe Hands: 손 펴기(0.0) vs 주먹(1.0) → 그리퍼 명령
    # -------------------------
    def _hand_openness_from_landmarks(self, landmarks, image_shape):
        """
        MediaPipe hand landmarks (list of .x,.y,.z) → 0.0(펴기) ~ 1.0(주먹).
        손가락 끝(4,8,12,16,20)이 손목(0)에서 멀면 펴기, 가까우면 주먹.
        """
        h, w = image_shape[:2]
        def px(lm):
            return (lm.x * w, lm.y * h)
        if len(landmarks) < 21:
            return 0.0
        wrist = np.array(px(landmarks[0]))
        tips = [np.array(px(landmarks[i])) for i in [4, 8, 12, 16, 20]]
        palm_ref = np.array(px(landmarks[5]))
        hand_size = float(np.linalg.norm(wrist - palm_ref)) + 1e-6
        spread = float(np.mean([np.linalg.norm(wrist - t) for t in tips]))
        ratio = spread / hand_size
        open_val = clamp((ratio - 0.8) / (2.2 - 0.8), 0.0, 1.0)
        return 1.0 - open_val  # 그리퍼: 0=열림, 1=쥠

    def _run_hands_and_gripper(self, frame_rgb, display_frame):
        """
        MediaPipe 0.10 HandLandmarker: frame_rgb(RGB) → 손 2개 감지 → 그리퍼 0~1 스무딩·퍼블리시.
        """
        robot_left_gripper = 0.0
        robot_right_gripper = 0.0
        hand_poses = []

        try:
            frame_rgb = np.ascontiguousarray(frame_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.hand_landmarker.detect(mp_image)
        except Exception:
            result = None

        if result and result.hand_landmarks and result.handedness:
            for idx in range(len(result.hand_landmarks)):
                landmarks = result.hand_landmarks[idx]
                label = result.handedness[idx][0].category_name if result.handedness[idx] else "Right"
                openness = self._hand_openness_from_landmarks(landmarks, display_frame.shape)
                cx = np.mean([lm.x for lm in landmarks]) * display_frame.shape[1]
                hand_poses.append((cx, openness, landmarks, label))
                # 왼손↔오른손 반대 매핑: 화면 Left → 로봇 오른쪽 그리퍼, Right → 로봇 왼쪽 그리퍼
                if label == "Left":
                    robot_right_gripper = openness
                else:
                    robot_left_gripper = openness

        self.gripper_left_buffer.append(robot_left_gripper)
        self.gripper_right_buffer.append(robot_right_gripper)
        smooth_left = float(np.mean(self.gripper_left_buffer))
        smooth_right = float(np.mean(self.gripper_right_buffer))

        # GripperCommand 액션: position(m) = (1 - openness) * open_m → 0=쥠, open_m=열림
        pos_left = (1.0 - smooth_left) * self.gripper_open_m
        pos_right = (1.0 - smooth_right) * self.gripper_open_m
        for client, pos in [(self.left_gripper_client, pos_left), (self.right_gripper_client, pos_right)]:
            if client.server_is_ready():
                goal = GripperCommand.Goal()
                goal.command.position = float(pos)
                goal.command.max_effort = self.gripper_max_effort
                client.send_goal_async(goal)

        for cx, openness, landmarks, label in hand_poses:
            for lm in landmarks:
                x, y = int(lm.x * display_frame.shape[1]), int(lm.y * display_frame.shape[0])
                cv2.circle(display_frame, (x, y), 2, (0, 255, 0), -1)
            txt = "OPEN" if openness < 0.5 else "FIST"
            color = (0, 255, 0) if openness < 0.5 else (0, 0, 255)
            cv2.putText(display_frame, f"{label} {txt}", (int(cx) - 40, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return smooth_left, smooth_right

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        # === RealSense 프레임 수신 ===
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        except Exception as e:
            self.get_logger().warn(f"⚠️ RealSense 프레임 수신 실패: {e}")
            return

        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("⚠️ color/depth 프레임이 비어 있습니다.")
            return

        frame_unflipped = np.asanyarray(color_frame.get_data())  # 원본(color)
        frame = cv2.flip(frame_unflipped, 1)                     # 표시/YOLO는 거울 모드

        results = self.model.predict(
            frame,
            conf=0.25,
            iou=0.45,
            verbose=False,
            imgsz=640
        )

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        smooth_left_gripper, smooth_right_gripper = self._run_hands_and_gripper(frame_rgb, display_frame)

        if (not results) or (len(results[0].keypoints.xy) == 0):
            cv2.putText(display_frame, "No person detected",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
            cv2.waitKey(1)
            return

        kps = results[0].keypoints.xy[0].cpu().numpy()
        conf = results[0].keypoints.conf[0].cpu().numpy()

        if len(kps) < 11:
            cv2.putText(display_frame, "Incomplete keypoints",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
            cv2.waitKey(1)
            return

        # COCO keypoints (거울 모드 프레임 기준)
        screen_left_shoulder = kps[5]
        screen_left_elbow = kps[7]
        screen_left_wrist = kps[9]

        screen_right_shoulder = kps[6]
        screen_right_elbow = kps[8]
        screen_right_wrist = kps[10]

        key_indices = [5, 6, 7, 8, 9, 10]
        avg_conf = float(np.mean([conf[i] for i in key_indices]))

        if avg_conf < 0.35:
            cv2.putText(display_frame, f"Low confidence: {avg_conf:.2f}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
            cv2.waitKey(1)
            return

        # === 거울모드 좌표 -> 원본(color/depth 정렬 좌표)로 환산 ===
        # depth_frame은 align(color 원본)에 맞춰져 있으므로 "원본 픽셀"로 depth를 읽어야 합니다.
        def to_unflipped_pixel(pt_flipped):
            x = float(pt_flipped[0])
            y = float(pt_flipped[1])
            x_un = (self.width - 1) - x
            return x_un, y

        # 3D 점 만들기
        def get_3d_from_flipped(pt_flipped):
            x_un, y_un = to_unflipped_pixel(pt_flipped)
            p3, d = self.deproject_color_pixel_to_3d(depth_frame, x_un, y_un)
            return p3, d

        Ls3, dLs = get_3d_from_flipped(screen_left_shoulder)
        Le3, dLe = get_3d_from_flipped(screen_left_elbow)
        Lw3, dLw = get_3d_from_flipped(screen_left_wrist)

        Rs3, dRs = get_3d_from_flipped(screen_right_shoulder)
        Re3, dRe = get_3d_from_flipped(screen_right_elbow)
        Rw3, dRw = get_3d_from_flipped(screen_right_wrist)

        # depth가 너무 많이 비면 스킵
        if any(p is None for p in [Ls3, Le3, Lw3, Rs3, Re3, Rw3]):
            cv2.putText(display_frame, "Depth missing on keypoints",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
            cv2.waitKey(1)
            return

        # === 캘리브레이션(3D 기준) ===
        if not self.is_calibrated:
            self.calibration_frames.append({
                'L': {'shoulder': Ls3.copy(), 'elbow': Le3.copy(), 'wrist': Lw3.copy()},
                'R': {'shoulder': Rs3.copy(), 'elbow': Re3.copy(), 'wrist': Rw3.copy()},
            })

            progress = len(self.calibration_frames)
            cv2.putText(display_frame, f"Calibrating... {progress}/{self.calib_needed}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(display_frame, "Stand still in attention pose!",
                        (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            if progress >= self.calib_needed:
                self.reference_pose_3d = {
                    'L': {
                        'shoulder': np.mean([f['L']['shoulder'] for f in self.calibration_frames], axis=0),
                        'elbow':    np.mean([f['L']['elbow'] for f in self.calibration_frames], axis=0),
                        'wrist':    np.mean([f['L']['wrist'] for f in self.calibration_frames], axis=0),
                    },
                    'R': {
                        'shoulder': np.mean([f['R']['shoulder'] for f in self.calibration_frames], axis=0),
                        'elbow':    np.mean([f['R']['elbow'] for f in self.calibration_frames], axis=0),
                        'wrist':    np.mean([f['R']['wrist'] for f in self.calibration_frames], axis=0),
                    }
                }
                self.is_calibrated = True
                self.get_logger().info("✅ 캘리브레이션 완료!(3D)")

            cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
            cv2.waitKey(1)
            return

        # === 각도 계산(3D) ===
        # 화면 왼쪽 팔 -> 로봇 오른팔
        right_raw = self.calculate_angles_3d(
            Ls3, Le3, Lw3,
            ref=self.reference_pose_3d['L']
        )
        right_cmd = self.apply_arm_sign('right', right_raw)
        right_cmd[0] = clamp(right_cmd[0], self.J1_RIGHT_MIN, self.J_LIMITS['j1'][1])

        # 화면 오른쪽 팔 -> 로봇 왼팔
        left_raw = self.calculate_angles_3d(
            Rs3, Re3, Rw3,
            ref=self.reference_pose_3d['R']
        )
        left_cmd = self.apply_arm_sign('left', left_raw)
        left_cmd[0] = clamp(left_cmd[0], self.J_LIMITS['j1'][0], self.J1_LEFT_MAX)

        robot_left_smooth = self.smooth_positions(self.left_buffer, left_cmd)
        robot_right_smooth = self.smooth_positions(self.right_buffer, right_cmd)

        self.send_command(self.left_client, robot_left_smooth)
        self.send_command(self.right_client, robot_right_smooth)

        # === 시각화(2D) ===
        skeleton = [
            (5, 7, (255, 0, 0)),
            (7, 9, (255, 100, 0)),
            (6, 8, (0, 0, 255)),
            (8, 10, (0, 100, 255))
        ]
        for s, e, color in skeleton:
            if conf[s] > 0.3 and conf[e] > 0.3:
                start = tuple(kps[s].astype(int))
                end = tuple(kps[e].astype(int))
                cv2.line(display_frame, start, end, color, 5)

        joints = [
            (screen_left_shoulder, "->Robot RIGHT", (255, 0, 0)),
            (screen_left_elbow, "", (255, 100, 0)),
            (screen_left_wrist, "", (255, 200, 0)),
            (screen_right_shoulder, "->Robot LEFT", (0, 0, 255)),
            (screen_right_elbow, "", (0, 100, 255)),
            (screen_right_wrist, "", (0, 200, 255))
        ]
        for p, label, color in joints:
            x, y = int(p[0]), int(p[1])
            cv2.circle(display_frame, (x, y), 10, color, -1)
            cv2.circle(display_frame, (x, y), 12, (255, 255, 255), 2)
            if label:
                cv2.putText(display_frame, label, (x + 15, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.rectangle(display_frame, (10, 10), (1180, 265), (0, 0, 0), -1)

        cv2.putText(display_frame, f"MIRROR MODE (RealSense 3D) | Conf: {avg_conf:.2f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(display_frame,
                    "LEFT  J1/J2/J3/J4/J5/J6/J7 = "
                    f"{robot_left_smooth[0]:+.2f} {robot_left_smooth[1]:+.2f} {robot_left_smooth[2]:+.2f} "
                    f"{robot_left_smooth[3]:+.2f} {robot_left_smooth[4]:+.2f} {robot_left_smooth[5]:+.2f} {robot_left_smooth[6]:+.2f}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)

        cv2.putText(display_frame,
                    "RIGHT J1/J2/J3/J4/J5/J6/J7 = "
                    f"{robot_right_smooth[0]:+.2f} {robot_right_smooth[1]:+.2f} {robot_right_smooth[2]:+.2f} "
                    f"{robot_right_smooth[3]:+.2f} {robot_right_smooth[4]:+.2f} {robot_right_smooth[5]:+.2f} {robot_right_smooth[6]:+.2f}",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 0), 2)

        cv2.putText(display_frame,
                    f"Depth(m)  LW={dLw:.2f}  RW={dRw:.2f}  (shoulder/elbow/wrist 3D used)",
                    (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display_frame,
                    f"Gripper  LEFT={smooth_left_gripper:.2f}  RIGHT={smooth_right_gripper:.2f}  (0=open 1=grip)",
                    (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

        cv2.putText(display_frame, "Press 'r' to recalibrate | 'q' to quit",
                    (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Mirror Mode Control (RealSense)", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info("종료 요청됨")
            raise KeyboardInterrupt
        elif key == ord('r'):
            self.calibration_frames = []
            self.is_calibrated = False
            self.reference_pose_3d = None
            self.left_buffer.clear()
            self.right_buffer.clear()
            self.get_logger().info("🔄 재캘리브레이션 시작!")


def main():
    rclpy.init()
    node = None
    try:
        node = SimpleArmControllerRealSense()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("종료 중...")
    except Exception as e:
        if node:
            node.get_logger().error(f"오류 발생: {e}")
        print(f"오류: {e}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if node:
            try:
                node.pipeline.stop()
            except Exception:
                pass
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()

