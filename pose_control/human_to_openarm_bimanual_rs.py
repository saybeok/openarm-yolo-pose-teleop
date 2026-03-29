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


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


class Kalman1D:
    """
    단순 1차원 칼만 필터
    """
    def __init__(self, q=1e-3, r=1e-2, x0=0.0, p0=1.0):
        self.Q = float(q)
        self.R = float(r)
        self.x = float(x0)
        self.P = float(p0)
        self.initialized = False

    def reset(self, x0=0.0, p0=1.0):
        self.x = float(x0)
        self.P = float(p0)
        self.initialized = False

    def update(self, z):
        z = float(z)

        if not self.initialized:
            self.x = z
            self.P = 1.0
            self.initialized = True
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (z - x_pred)
        self.P = (1.0 - K) * P_pred
        return self.x


class SimpleArmControllerRealSense(Node):
    def __init__(self):
        super().__init__('simple_yolo_arm_realsense_palm_wrist')

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
        self.align = rs.align(rs.stream.color)

        for _ in range(10):
            self.pipeline.wait_for_frames()

        self.get_logger().info("✅ RealSense 스트림 시작됨 (Color+Depth)")

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
        # 4) 보조 버퍼
        # =========================
        self.left_buffer = deque(maxlen=12)
        self.right_buffer = deque(maxlen=12)

        # =========================
        # 4-1) 칼만 필터
        # =========================
        self.kf_3d = {
            'L_shoulder': [Kalman1D(q=1e-4, r=2e-3) for _ in range(3)],
            'L_elbow':    [Kalman1D(q=1e-4, r=3e-3) for _ in range(3)],
            'L_wrist':    [Kalman1D(q=2e-4, r=5e-3) for _ in range(3)],
            'R_shoulder': [Kalman1D(q=1e-4, r=2e-3) for _ in range(3)],
            'R_elbow':    [Kalman1D(q=1e-4, r=3e-3) for _ in range(3)],
            'R_wrist':    [Kalman1D(q=2e-4, r=5e-3) for _ in range(3)],
            'L_hip':      [Kalman1D(q=1e-4, r=2e-3) for _ in range(3)],
            'R_hip':      [Kalman1D(q=1e-4, r=2e-3) for _ in range(3)],
        }

        self.kf_joint_left = [Kalman1D(q=2e-3, r=1.5e-2) for _ in range(7)]
        self.kf_joint_right = [Kalman1D(q=2e-3, r=1.5e-2) for _ in range(7)]

        # 손바닥 기반 손목각 안정화용
        self.kf_palm_left = [Kalman1D(q=2e-3, r=2e-2) for _ in range(3)]   # j5, j6, j7
        self.kf_palm_right = [Kalman1D(q=2e-3, r=2e-2) for _ in range(3)]  # j5, j6, j7

        # =========================
        # 5) 캘리브레이션
        # =========================
        self.calibration_frames = []
        self.is_calibrated = False
        self.reference_pose_3d = None
        self.ref_torso_R = None
        self.ref_torso_origin = None
        self.calib_needed = 30

        # 손바닥 기준 자세 저장
        self.reference_palm = {
            'L': None,
            'R': None,
        }

        # =========================
        # 6) 관절 방향 보정
        # =========================
        self.ARM_JOINT_SIGN = {
            'left':  np.array([-1, +1, -1, -1, +1, +1, +1], dtype=float),
            'right': np.array([+1, +1, +1, +1, +1, +1, +1], dtype=float),
        }

        # =========================
        # 7) 관절 제한
        # =========================
        self.J_LIMITS = {
            'j1': (-1.6, 1.6),
            'j2': (-1.2, 1.8),
            'j3': (-1.2, 1.2),
            'j4': (-1.2, 2.2),
            'j5': (-2.5, 2.5),
            'j6': (-1.8, 1.8),
            'j7': (-2.8, 2.8),
        }
        self.J1_LEFT_MAX = 1.6
        self.J1_RIGHT_MIN = -1.6

        # =========================
        # 8) MediaPipe Hands
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

        # 그리퍼 액션 클라이언트
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')
        self.gripper_open_m = 0.04
        self.gripper_max_effort = 50.0

        self.gripper_left_buffer = deque(maxlen=5)
        self.gripper_right_buffer = deque(maxlen=5)

        self.window_name = "Mirror Mode Control"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

        # =========================
        # 9) 타이머
        # =========================
        self.timer = self.create_timer(0.1, self.run)  # 10Hz

        self.get_logger().info("=" * 60)
        self.get_logger().info("🤖 RealSense + YOLO Pose + MediaPipe Hands + Palm Wrist Mapping")
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
        point.time_from_start.nanosec = 350_000_000

        goal.trajectory.points = [point]
        client.send_goal_async(goal)

    def kalman_filter_vec3(self, key, vec3):
        vec3 = np.asarray(vec3, dtype=float)
        return np.array([
            self.kf_3d[key][0].update(vec3[0]),
            self.kf_3d[key][1].update(vec3[1]),
            self.kf_3d[key][2].update(vec3[2]),
        ], dtype=float)

    def kalman_filter_joints(self, filters, joints7):
        joints7 = np.asarray(joints7, dtype=float)
        return np.array([filters[i].update(joints7[i]) for i in range(7)], dtype=float)

    def kalman_filter_palm_wrist(self, arm_name, j5, j6, j7):
        arr = np.array([j5, j6, j7], dtype=float)
        filters = self.kf_palm_left if arm_name == 'left' else self.kf_palm_right
        out = np.array([filters[i].update(arr[i]) for i in range(3)], dtype=float)
        return out[0], out[1], out[2]

    def reset_kalman_filters(self):
        for key in self.kf_3d:
            for f in self.kf_3d[key]:
                f.reset()
        for f in self.kf_joint_left:
            f.reset()
        for f in self.kf_joint_right:
            f.reset()
        for f in self.kf_palm_left:
            f.reset()
        for f in self.kf_palm_right:
            f.reset()

    def reset_runtime_state(self):
        self.calibration_frames = []
        self.is_calibrated = False
        self.reference_pose_3d = None
        self.ref_torso_R = None
        self.ref_torso_origin = None
        self.reference_palm = {'L': None, 'R': None}

        self.left_buffer.clear()
        self.right_buffer.clear()
        self.gripper_left_buffer.clear()
        self.gripper_right_buffer.clear()
        self.reset_kalman_filters()

    def handle_hotkeys(self, key):
        if key == ord('q'):
            self.get_logger().info("종료 요청됨")
            raise KeyboardInterrupt
        if key == ord('r'):
            self.reset_runtime_state()
            self.get_logger().info("🔄 재캘리브레이션 시작!")

    def show_and_handle(self, display_frame):
        cv2.imshow(self.window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        self.handle_hotkeys(key)

    # -------------------------
    # Depth + 3D helpers
    # -------------------------
    def get_depth_m(self, depth_frame: rs.depth_frame, x, y):
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
        d = self.get_depth_m(depth_frame, x_color, y_color)
        if d <= 0.01:
            return None, 0.0

        pt = rs.rs2_deproject_pixel_to_point(self.color_intr, [float(x_color), float(y_color)], float(d))
        return np.array(pt, dtype=float), float(d)

    # -------------------------
    # 몸통 로컬 좌표계 (★ 추가)
    # -------------------------
    def build_torso_frame(self, l_shoulder, r_shoulder, l_hip, r_hip):
        """
        몸통 로컬 좌표계:
          origin = 양쪽 어깨 중점
          X = 오른어깨 → 왼어깨
          Y = 위쪽 (엉덩이→어깨)
          Z = 전방 (X × Y)
        """
        origin = (l_shoulder + r_shoulder) / 2.0
        hip_center = (l_hip + r_hip) / 2.0

        x_axis = normalize(l_shoulder - r_shoulder)
        spine = normalize(origin - hip_center)
        y_axis = normalize(spine - np.dot(spine, x_axis) * x_axis)

        if np.linalg.norm(x_axis) < 1e-6 or np.linalg.norm(y_axis) < 1e-6:
            return None, None

        z_axis = normalize(np.cross(x_axis, y_axis))
        R = np.column_stack((x_axis, y_axis, z_axis))
        return R, origin

    def to_torso_local(self, point_3d, torso_R, torso_origin):
        return torso_R.T @ (point_3d - torso_origin)

    # -------------------------
    # 3D angle helpers (★ 몸통 로컬 기반으로 수정)
    # -------------------------
    def shoulder_yaw_pitch_roll_from_upper(self, upper_local):
        """
        몸통 로컬 좌표(X=좌, Y=위, Z=전방)에서의 상완 벡터 → yaw, pitch, roll

        차렷 자세: 상완 = -Y 방향(아래)
        j1(yaw):   전후 움직임 → atan2(-z, -y)
        j2(pitch):  좌우 벌림  → atan2(x, -y)
        """
        u = normalize(upper_local)
        if np.linalg.norm(u) < 1e-6:
            return 0.0, 0.0, 0.0

        yaw = math.atan2(-u[2], -u[1])
        pitch = math.atan2(u[0], -u[1])
        roll = 0.0

        return yaw, pitch, roll

    def signed_elbow_flex(self, upper3, fore3):
        u = upper3 / safe_norm(upper3)
        f = fore3 / safe_norm(fore3)

        dot = float(np.clip(np.dot(u, f), -1.0, 1.0))
        angle = math.acos(dot)
        flex = math.pi - angle

        n = np.cross(u, f)
        sign = 1.0 if float(n[2]) >= 0.0 else -1.0

        return float(sign * flex)

    def compute_shoulder_roll(self, upper, fore):
        u_hat = normalize(upper)
        if np.linalg.norm(u_hat) < 1e-6:
            return 0.0

        n = np.cross(upper, fore)
        n_norm = float(np.linalg.norm(n))
        if n_norm < 1e-6:
            return 0.0
        n = n / n_norm

        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        y_perp = y_axis - np.dot(y_axis, u_hat) * u_hat

        if np.linalg.norm(y_perp) < 1e-6:
            y_perp = np.array([0.0, 0.0, 1.0], dtype=float)
            y_perp = y_perp - np.dot(y_perp, u_hat) * u_hat

        y_perp = normalize(y_perp)
        x_perp = np.cross(u_hat, y_perp)

        roll = math.atan2(float(np.dot(n, x_perp)), float(np.dot(n, y_perp)))

        bend = math.acos(clamp(float(np.dot(u_hat, normalize(fore))), -1.0, 1.0))
        alpha = clamp((bend - 0.2) / 0.3, 0.0, 1.0)

        return roll * alpha

    def orthonormalize_frame(self, x_axis, y_hint):
        x_axis = normalize(x_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            return None

        z_axis = np.cross(x_axis, y_hint)
        z_axis = normalize(z_axis)
        if np.linalg.norm(z_axis) < 1e-6:
            alt = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(x_axis, alt)) > 0.9:
                alt = np.array([1.0, 0.0, 0.0], dtype=float)
            z_axis = normalize(np.cross(x_axis, alt))
            if np.linalg.norm(z_axis) < 1e-6:
                return None

        y_axis = normalize(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            return None

        return np.column_stack((x_axis, y_axis, z_axis))

    def rotation_matrix_to_euler_xyz(self, R):
        sy = clamp(R[0, 2], -1.0, 1.0)
        b = math.asin(sy)

        cb = math.cos(b)
        if abs(cb) > 1e-6:
            a = math.atan2(-R[1, 2], R[2, 2])
            c = math.atan2(-R[0, 1], R[0, 0])
        else:
            a = math.atan2(R[2, 1], R[1, 1])
            c = 0.0

        return a, b, c

    def calculate_angles_3d(self, shoulder3, elbow3, wrist3,
                            torso_R, torso_origin,
                            ref=None, ref_torso_R=None, ref_torso_origin=None):
        """
        ★ 변경: 카메라 절대좌표 → 몸통 로컬좌표 변환 후 IK

        나머지 로직(yaw/pitch/roll, elbow flex, fallback wrist)은 동일하되
        입력 벡터만 로컬 좌표계 기준으로 바뀜
        """
        # 현재 프레임 → 몸통 로컬
        s_loc = self.to_torso_local(shoulder3, torso_R, torso_origin)
        e_loc = self.to_torso_local(elbow3, torso_R, torso_origin)
        w_loc = self.to_torso_local(wrist3, torso_R, torso_origin)

        upper = e_loc - s_loc
        fore = w_loc - e_loc

        if np.linalg.norm(upper) < 0.05 or np.linalg.norm(fore) < 0.05:
            return np.zeros(7, dtype=float)

        yaw, pitch, _ = self.shoulder_yaw_pitch_roll_from_upper(upper)
        roll = self.compute_shoulder_roll(upper, fore)
        eflex = self.signed_elbow_flex(upper, fore)

        j1 = yaw * 1.4
        j2 = pitch * 1.6
        j3 = roll * 1.2
        j4 = (eflex / math.pi) * 2.2

        # 손목 fallback (로컬 좌표 기반)
        fx, fy, fz = float(fore[0]), float(fore[1]), float(fore[2])
        j7 = math.atan2(fx, max(-fy, 1e-6)) * 0.8
        j6 = math.atan2(-fz, math.sqrt(fx * fx + fy * fy) + 1e-6) * 0.8
        j5 = 0.0

        # 기준 자세 빼기 (로컬 좌표 기반)
        if ref is not None and ref_torso_R is not None and ref_torso_origin is not None:
            rs_loc = self.to_torso_local(ref['shoulder'], ref_torso_R, ref_torso_origin)
            re_loc = self.to_torso_local(ref['elbow'], ref_torso_R, ref_torso_origin)
            rw_loc = self.to_torso_local(ref['wrist'], ref_torso_R, ref_torso_origin)

            ref_upper = re_loc - rs_loc
            ref_fore = rw_loc - re_loc

            ryaw, rpitch, _ = self.shoulder_yaw_pitch_roll_from_upper(ref_upper)
            rroll = self.compute_shoulder_roll(ref_upper, ref_fore)
            ref_eflex = self.signed_elbow_flex(ref_upper, ref_fore)

            rj1 = ryaw * 1.4
            rj2 = rpitch * 1.6
            rj3 = rroll * 1.2
            rj4 = (ref_eflex / math.pi) * 2.2

            j1 -= rj1
            j2 -= rj2
            j3 -= rj3
            j4 -= rj4
            
            # fallback wrist도 기준 빼기
            rfx, rfy, rfz = float(ref_fore[0]), float(ref_fore[1]), float(ref_fore[2])
            j7 -= math.atan2(rfx, max(-rfy, 1e-6)) * 0.8
            j6 -= math.atan2(-rfz, math.sqrt(rfx * rfx + rfy * rfy) + 1e-6) * 0.8

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
    # MediaPipe Hands helpers (변경 없음)
    # -------------------------
    def _hand_openness_from_landmarks(self, landmarks, image_shape):
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
        return 1.0 - open_val

    def _to_unflipped_pixel(self, pt_flipped):
        x = float(pt_flipped[0])
        y = float(pt_flipped[1])
        x_un = (self.width - 1) - x
        return x_un, y

    def _deproject_flipped_point(self, depth_frame, x_flip, y_flip):
        x_un, y_un = self._to_unflipped_pixel((x_flip, y_flip))
        p3, d = self.deproject_color_pixel_to_3d(depth_frame, x_un, y_un)
        return p3, d

    def _extract_palm_pose_from_landmarks(self, landmarks, depth_frame, display_shape):
        h, w = display_shape[:2]

        def pt2(i):
            return np.array([
                float(landmarks[i].x * w),
                float(landmarks[i].y * h)
            ], dtype=float)

        ids = {
            'wrist': 0,
            'index_mcp': 5,
            'middle_mcp': 9,
            'ring_mcp': 13,
            'pinky_mcp': 17
        }

        pts2 = {k: pt2(v) for k, v in ids.items()}
        pts3 = {}

        for k, p2 in pts2.items():
            p3, _ = self._deproject_flipped_point(depth_frame, p2[0], p2[1])
            if p3 is None:
                return None
            pts3[k] = p3

        wrist3 = pts3['wrist']
        index3 = pts3['index_mcp']
        middle3 = pts3['middle_mcp']
        pinky3 = pts3['pinky_mcp']
        ring3 = pts3['ring_mcp']

        palm_center3 = (index3 + middle3 + ring3 + pinky3) / 4.0

        palm_y = normalize(palm_center3 - wrist3)
        if np.linalg.norm(palm_y) < 1e-6:
            return None

        palm_x = normalize(index3 - pinky3)
        if np.linalg.norm(palm_x) < 1e-6:
            return None

        palm_z = normalize(np.cross(palm_x, palm_y))
        if np.linalg.norm(palm_z) < 1e-6:
            return None

        palm_x = normalize(np.cross(palm_y, palm_z))
        palm_R = np.column_stack((palm_x, palm_y, palm_z))

        return {
            'wrist3': wrist3,
            'center3': palm_center3,
            'palm_x': palm_x,
            'palm_y': palm_y,
            'palm_z': palm_z,
            'R': palm_R,
        }

    def _run_hands_and_gripper(self, frame_rgb, depth_frame, display_frame):
        robot_left_gripper = 0.0
        robot_right_gripper = 0.0
        hand_poses = []
        palm_info = {'Left': None, 'Right': None}

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
                ui_label = "Left" if label == "Right" else "Right"

                openness = self._hand_openness_from_landmarks(landmarks, display_frame.shape)
                cx = np.mean([lm.x for lm in landmarks]) * display_frame.shape[1]
                hand_poses.append((cx, openness, landmarks, ui_label))

                palm_pose = self._extract_palm_pose_from_landmarks(
                    landmarks, depth_frame, display_frame.shape
                )
                palm_info[label] = palm_pose

                if label == "Left":
                    robot_right_gripper = openness
                else:
                    robot_left_gripper = openness

        self.gripper_left_buffer.append(robot_left_gripper)
        self.gripper_right_buffer.append(robot_right_gripper)
        smooth_left = float(np.mean(self.gripper_left_buffer))
        smooth_right = float(np.mean(self.gripper_right_buffer))

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

        return smooth_left, smooth_right, palm_info

    # -------------------------
    # Palm wrist estimation (변경 없음)
    # -------------------------
    def estimate_wrist_from_palm(self, arm_name, elbow3, wrist3, palm_pose, ref_palm_pose=None):
        if palm_pose is None:
            return None

        fore = wrist3 - elbow3
        fore_y = normalize(fore)
        if np.linalg.norm(fore_y) < 1e-6:
            return None

        fore_R = self.orthonormalize_frame(fore_y, palm_pose['palm_x'])
        if fore_R is None:
            return None

        palm_R = palm_pose['R']
        R_rel = fore_R.T @ palm_R

        if ref_palm_pose is not None:
            ref_fore = normalize(ref_palm_pose['wrist3'] - ref_palm_pose['elbow3'])
            ref_fore_R = self.orthonormalize_frame(ref_fore, ref_palm_pose['palm_x'])
            if ref_fore_R is not None:
                R_ref_rel = ref_fore_R.T @ ref_palm_pose['R']
                R_rel = R_ref_rel.T @ R_rel

        a, b, c = self.rotation_matrix_to_euler_xyz(R_rel)

        j5 = clamp(c * 1.2, *self.J_LIMITS['j5'])
        j6 = clamp(a * 1.0, *self.J_LIMITS['j6'])
        j7 = clamp(b * 1.0, *self.J_LIMITS['j7'])

        if arm_name == 'left':
            j5 = -j5
            j7 = -j7

        j5, j6, j7 = self.kalman_filter_palm_wrist(arm_name, j5, j6, j7)
        return np.array([j5, j6, j7], dtype=float)

    # -------------------------
    # Main loop (★ 힙 키포인트 + 몸통 좌표계 추가)
    # -------------------------
    def run(self):
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

        frame_unflipped = np.asanyarray(color_frame.get_data())
        frame = cv2.flip(frame_unflipped, 1)

        results = self.model.predict(
            frame,
            conf=0.25,
            iou=0.45,
            verbose=False,
            imgsz=640
        )

        display_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        smooth_left_gripper, smooth_right_gripper, palm_info = self._run_hands_and_gripper(
            frame_rgb, depth_frame, display_frame
        )

        if (not results) or (len(results[0].keypoints.xy) == 0):
            cv2.putText(display_frame, "No person detected",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.show_and_handle(display_frame)
            return

        kps = results[0].keypoints.xy[0].cpu().numpy()
        conf = results[0].keypoints.conf[0].cpu().numpy()

        if len(kps) < 13:  # ★ 13으로 변경 (힙까지 필요)
            cv2.putText(display_frame, "Incomplete keypoints",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            self.show_and_handle(display_frame)
            return

        # COCO keypoints (거울 모드 기준)
        screen_left_shoulder = kps[5]
        screen_left_elbow = kps[7]
        screen_left_wrist = kps[9]

        screen_right_shoulder = kps[6]
        screen_right_elbow = kps[8]
        screen_right_wrist = kps[10]

        screen_left_hip = kps[11]   # ★ 추가
        screen_right_hip = kps[12]  # ★ 추가

        key_indices = [5, 6, 7, 8, 9, 10, 11, 12]  # ★ 힙 추가
        avg_conf = float(np.mean([conf[i] for i in key_indices]))

        if avg_conf < 0.35:
            cv2.putText(display_frame, f"Low confidence: {avg_conf:.2f}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            self.show_and_handle(display_frame)
            return

        def to_unflipped_pixel(pt_flipped):
            x = float(pt_flipped[0])
            y = float(pt_flipped[1])
            x_un = (self.width - 1) - x
            return x_un, y

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

        Lh3, _ = get_3d_from_flipped(screen_left_hip)   # ★ 추가
        Rh3, _ = get_3d_from_flipped(screen_right_hip)  # ★ 추가

        if any(p is None for p in [Ls3, Le3, Lw3, Rs3, Re3, Rw3, Lh3, Rh3]):
            cv2.putText(display_frame, "Depth missing on keypoints",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            self.show_and_handle(display_frame)
            return

        # 3D point 칼만
        Ls3 = self.kalman_filter_vec3('L_shoulder', Ls3)
        Le3 = self.kalman_filter_vec3('L_elbow', Le3)
        Lw3 = self.kalman_filter_vec3('L_wrist', Lw3)

        Rs3 = self.kalman_filter_vec3('R_shoulder', Rs3)
        Re3 = self.kalman_filter_vec3('R_elbow', Re3)
        Rw3 = self.kalman_filter_vec3('R_wrist', Rw3)

        Lh3 = self.kalman_filter_vec3('L_hip', Lh3)  # ★ 추가
        Rh3 = self.kalman_filter_vec3('R_hip', Rh3)  # ★ 추가

        # ★ 몸통 좌표계 구성
        torso_R, torso_origin = self.build_torso_frame(Ls3, Rs3, Lh3, Rh3)
        if torso_R is None:
            cv2.putText(display_frame, "Torso frame failed",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            self.show_and_handle(display_frame)
            return

        # 캘리브레이션
        if not self.is_calibrated:
            self.calibration_frames.append({
                'L': {'shoulder': Ls3.copy(), 'elbow': Le3.copy(), 'wrist': Lw3.copy()},
                'R': {'shoulder': Rs3.copy(), 'elbow': Re3.copy(), 'wrist': Rw3.copy()},
                'torso_R': torso_R.copy(),          # ★ 추가
                'torso_origin': torso_origin.copy(), # ★ 추가
            })

            if palm_info['Left'] is not None:
                palm_info['Left']['elbow3'] = Le3.copy()
            if palm_info['Right'] is not None:
                palm_info['Right']['elbow3'] = Re3.copy()

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

                # ★ 기준 몸통 좌표계 (평균 후 SVD 재직교화)
                avg_torso_R = np.mean([f['torso_R'] for f in self.calibration_frames], axis=0)
                U, _, Vt = np.linalg.svd(avg_torso_R)
                self.ref_torso_R = U @ Vt
                self.ref_torso_origin = np.mean(
                    [f['torso_origin'] for f in self.calibration_frames], axis=0
                )

                if palm_info['Right'] is not None:
                    self.reference_palm['L'] = {
                        **palm_info['Right'],
                        'elbow3': Re3.copy()
                    }
                if palm_info['Left'] is not None:
                    self.reference_palm['R'] = {
                        **palm_info['Left'],
                        'elbow3': Le3.copy()
                    }

                self.is_calibrated = True
                self.get_logger().info("✅ 캘리브레이션 완료!(3D + Torso-Local + Palm Wrist)")

            self.show_and_handle(display_frame)
            return

        # ★ 화면 왼쪽 팔 -> 로봇 오른팔 (몸통 좌표계 전달)
        right_raw = self.calculate_angles_3d(
            Ls3, Le3, Lw3,
            torso_R, torso_origin,
            ref=self.reference_pose_3d['L'],
            ref_torso_R=self.ref_torso_R,
            ref_torso_origin=self.ref_torso_origin,
        )

        # ★ 화면 오른쪽 팔 -> 로봇 왼팔 (몸통 좌표계 전달)
        left_raw = self.calculate_angles_3d(
            Rs3, Re3, Rw3,
            torso_R, torso_origin,
            ref=self.reference_pose_3d['R'],
            ref_torso_R=self.ref_torso_R,
            ref_torso_origin=self.ref_torso_origin,
        )

        # 손바닥 기반 손목 개선
        if palm_info['Left'] is not None:
            right_wrist_from_palm = self.estimate_wrist_from_palm(
                arm_name='right',
                elbow3=Le3,
                wrist3=Lw3,
                palm_pose={**palm_info['Left'], 'elbow3': Le3.copy()},
                ref_palm_pose=self.reference_palm['R']
            )
            if right_wrist_from_palm is not None:
                right_raw[4] = right_wrist_from_palm[0]
                right_raw[5] = right_wrist_from_palm[1]
                right_raw[6] = right_wrist_from_palm[2]

        if palm_info['Right'] is not None:
            left_wrist_from_palm = self.estimate_wrist_from_palm(
                arm_name='left',
                elbow3=Re3,
                wrist3=Rw3,
                palm_pose={**palm_info['Right'], 'elbow3': Re3.copy()},
                ref_palm_pose=self.reference_palm['L']
            )
            if left_wrist_from_palm is not None:
                left_raw[4] = left_wrist_from_palm[0]
                left_raw[5] = left_wrist_from_palm[1]
                left_raw[6] = left_wrist_from_palm[2]

        right_cmd = self.apply_arm_sign('right', right_raw)
        right_cmd[0] = clamp(right_cmd[0], self.J1_RIGHT_MIN, self.J_LIMITS['j1'][1])

        left_cmd = self.apply_arm_sign('left', left_raw)
        left_cmd[0] = clamp(left_cmd[0], self.J_LIMITS['j1'][0], self.J1_LEFT_MAX)

        robot_left_smooth = self.kalman_filter_joints(self.kf_joint_left, left_cmd)
        robot_right_smooth = self.kalman_filter_joints(self.kf_joint_right, right_cmd)

        self.send_command(self.left_client, robot_left_smooth)
        self.send_command(self.right_client, robot_right_smooth)

        # 시각화
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

        cv2.putText(display_frame, f"Conf:{avg_conf:.2f}  r:recalibrate  q:quit",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        self.show_and_handle(display_frame)

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
