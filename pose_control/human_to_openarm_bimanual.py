#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory

from trajectory_msgs.msg import JointTrajectoryPoint



from ultralytics import YOLO

import cv2

import numpy as np

from collections import deque

import math





def clamp(x, lo, hi):

    return max(lo, min(hi, x))





class SimpleArmController(Node):

    def __init__(self):

        super().__init__('simple_yolo_arm')



        # 1) YOLO 및 카메라 설정

        self.model = YOLO("yolov8m-pose.pt")

        self.cap = cv2.VideoCapture(0)



        if not self.cap.isOpened():

            self.get_logger().error("❌ 카메라를 열 수 없습니다!")

            raise RuntimeError("Camera failed to open")



        # 카메라 해상도 설정

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cap.set(cv2.CAP_PROP_FPS, 30)



        # 2) 로봇 팔 액션 클라이언트

        self.left_client = ActionClient(

            self, FollowJointTrajectory,

            '/left_joint_trajectory_controller/follow_joint_trajectory'

        )

        self.right_client = ActionClient(

            self, FollowJointTrajectory,

            '/right_joint_trajectory_controller/follow_joint_trajectory'

        )



        # 서버 연결 대기(초기 한 번)

        self.get_logger().info("⏳ Trajectory action server 대기 중...")

        self.left_client.wait_for_server()

        self.right_client.wait_for_server()

        self.get_logger().info("✅ Action server 연결 완료")



        # 3) 스무딩 버퍼

        self.left_buffer = deque(maxlen=5)

        self.right_buffer = deque(maxlen=5)



        # 4) 캘리브레이션

        self.calibration_frames = []

        self.is_calibrated = False

        self.reference_pose = None



        # === 로봇 관절 방향 보정 ===

        # ✅ 요청대로 "왼팔만" 방향 보정 적용

        # 왼팔에서 J1/J2/J4가 반대라고 가정한 가장 흔한 조합

        self.ARM_JOINT_SIGN = {

            'left':  np.array([-1, -1, +1, -1, +1, +1, +1], dtype=float),  # ✅ LEFT만 변경

            'right': np.array([+1, +1, +1, +1, +1, +1, +1], dtype=float),  # RIGHT는 그대로

        }



        # 관절 각도 안전 범위(로봇에 맞게 조정 가능)

        self.J_LIMITS = {

            'j1': (-1.5, 1.5),

            'j2': (-0.8, 2.0),

            'j4': (0.0, 2.2),

        }



        # 5) 타이머

        self.timer = self.create_timer(0.1, self.run)



        self.get_logger().info("=" * 60)

        self.get_logger().info("🤖 거울 모드 로봇 컨트롤러 시작!")

        self.get_logger().info("📹 카메라 앞에서 차렷 자세로 3초간 서주세요")

        self.get_logger().info("💡 당신의 왼쪽 팔 → 로봇의 오른쪽 팔 (거울처럼)")

        self.get_logger().info("=" * 60)



    def send_command(self, client, positions):

        """로봇 팔에 명령 전송"""

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

        point.time_from_start.nanosec = 200_000_000  # 0.2s



        goal.trajectory.points = [point]

        client.send_goal_async(goal)



    def smooth_positions(self, buffer, new_pos):

        """움직임 스무딩(가중 이동 평균)"""

        buffer.append(new_pos)

        n = len(buffer)

        if n == 0:

            return new_pos

        weights = np.linspace(1, n, n, dtype=float)

        weights /= weights.sum()

        arr = np.stack(buffer, axis=0)

        return np.sum(arr * weights[:, None], axis=0)



    def elbow_flex_from_vectors(self, a, b):

        """

        팔꿈치 굽힘(라디안)을 내적 기반으로 계산.

        a: shoulder->elbow

        b: elbow->wrist

        반환: 0(펴짐) ~ pi(완전 굽힘) 형태로 나오게 구성

        """

        na = np.linalg.norm(a)

        nb = np.linalg.norm(b)

        if na < 1e-6 or nb < 1e-6:

            return 0.0

        ua = a / na

        ub = b / nb

        dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))

        angle = math.acos(dot)  # 0..pi

        flex = math.pi - angle

        return clamp(flex, 0.0, math.pi)



    def shoulder_pitch_from_upperarm(self, upper_vec):

        """

        어깨 pitch를 화면 좌표에서 각도로 계산.

        이미지 좌표: y가 아래로 증가.

        팔을 올리면 dy가 음수 -> pitch 증가하도록 -dy 사용.

        """

        dx = float(upper_vec[0])

        dy = float(upper_vec[1])

        pitch = math.atan2(-dy, abs(dx) + 60.0)

        return pitch



    def shoulder_yaw_from_upperarm(self, upper_vec):

        """어깨 yaw(J1)을 화면 좌표에서 계산"""

        dx = float(upper_vec[0])

        dy = float(upper_vec[1])

        yaw = math.atan2(dx, abs(dy) + 60.0)

        return yaw



    def calculate_angles(self, shoulder, elbow, wrist, reference=None):

        """2D 키포인트로부터 [j1..j7] 생성"""

        shoulder = np.asarray(shoulder, dtype=float)

        elbow = np.asarray(elbow, dtype=float)

        wrist = np.asarray(wrist, dtype=float)



        upper = elbow - shoulder

        fore = wrist - elbow



        upper_len = np.linalg.norm(upper)

        fore_len = np.linalg.norm(fore)

        if upper_len < 10 or fore_len < 10:

            return np.zeros(7, dtype=float)



        j1 = self.shoulder_yaw_from_upperarm(upper) * 1.6

        j2 = self.shoulder_pitch_from_upperarm(upper) * 2.2



        flex = self.elbow_flex_from_vectors(upper, fore)

        j4 = (flex / math.pi) * 2.2



        if reference is not None:

            ref_upper = reference['elbow'] - reference['shoulder']

            ref_fore = reference['wrist'] - reference['elbow']



            ref_j1 = self.shoulder_yaw_from_upperarm(ref_upper) * 1.6

            ref_j2 = self.shoulder_pitch_from_upperarm(ref_upper) * 2.2

            ref_flex = self.elbow_flex_from_vectors(ref_upper, ref_fore)

            ref_j4 = (ref_flex / math.pi) * 2.2



            j1 -= ref_j1

            j2 -= ref_j2

            j4 -= ref_j4



        j1 = clamp(j1, *self.J_LIMITS['j1'])

        j2 = clamp(j2, *self.J_LIMITS['j2'])

        j4 = clamp(j4, *self.J_LIMITS['j4'])



        return np.array([j1, j2, 0.0, j4, 0.0, 0.0, 0.0], dtype=float)



    def apply_arm_sign(self, arm_name, joints7):

        """로봇 좌/우팔 관절 방향(부호) 적용"""

        return joints7 * self.ARM_JOINT_SIGN[arm_name]



    def run(self):

        """메인 루프"""

        ret, frame = self.cap.read()

        if not ret:

            self.get_logger().warn("⚠️ 프레임 읽기 실패")

            blank = np.zeros((480, 640, 3), dtype=np.uint8)

            cv2.putText(blank, "Camera Error - Retrying...", (50, 240),

                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Mirror Mode Control", blank)

            cv2.waitKey(1)

            return



        frame = cv2.flip(frame, 1)



        results = self.model.predict(

            frame,

            conf=0.25,

            iou=0.45,

            verbose=False,

            imgsz=640

        )



        display_frame = frame.copy()



        if not results or len(results[0].keypoints.xy) == 0:

            cv2.putText(display_frame, "No person detected",

                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.putText(display_frame, "Stand in front of camera",

                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.imshow("Mirror Mode Control", display_frame)

            cv2.waitKey(1)

            return



        kps = results[0].keypoints.xy[0].cpu().numpy()

        conf = results[0].keypoints.conf[0].cpu().numpy()



        if len(kps) < 11:

            cv2.putText(display_frame, "Incomplete keypoints",

                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            cv2.imshow("Mirror Mode Control", display_frame)

            cv2.waitKey(1)

            return



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

            cv2.imshow("Mirror Mode Control", display_frame)

            cv2.waitKey(1)

            return



        # === 캘리브레이션 ===

        if not self.is_calibrated:

            self.calibration_frames.append({

                'screen_left': {

                    'shoulder': screen_left_shoulder.copy(),

                    'elbow': screen_left_elbow.copy(),

                    'wrist': screen_left_wrist.copy()

                },

                'screen_right': {

                    'shoulder': screen_right_shoulder.copy(),

                    'elbow': screen_right_elbow.copy(),

                    'wrist': screen_right_wrist.copy()

                }

            })



            progress = len(self.calibration_frames)

            cv2.putText(display_frame, f"Calibrating... {progress}/30",

                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

            cv2.putText(display_frame, "Stand still in attention pose!",

                        (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)



            if len(self.calibration_frames) >= 30:

                self.reference_pose = {

                    'screen_left': {

                        'shoulder': np.mean([f['screen_left']['shoulder'] for f in self.calibration_frames], axis=0),

                        'elbow': np.mean([f['screen_left']['elbow'] for f in self.calibration_frames], axis=0),

                        'wrist': np.mean([f['screen_left']['wrist'] for f in self.calibration_frames], axis=0)

                    },

                    'screen_right': {

                        'shoulder': np.mean([f['screen_right']['shoulder'] for f in self.calibration_frames], axis=0),

                        'elbow': np.mean([f['screen_right']['elbow'] for f in self.calibration_frames], axis=0),

                        'wrist': np.mean([f['screen_right']['wrist'] for f in self.calibration_frames], axis=0)

                    }

                }

                self.is_calibrated = True

                self.get_logger().info("✅ 캘리브레이션 완료!")



            cv2.imshow("Mirror Mode Control", display_frame)

            cv2.waitKey(1)

            return



        # === 각도 계산 ===

        # 화면 왼쪽 -> 로봇 오른팔

        right_raw = self.calculate_angles(

            screen_left_shoulder, screen_left_elbow, screen_left_wrist,

            reference=self.reference_pose['screen_left']

        )

        right_cmd = self.apply_arm_sign('right', right_raw)



        # 화면 오른쪽 -> 로봇 왼팔

        left_raw = self.calculate_angles(

            screen_right_shoulder, screen_right_elbow, screen_right_wrist,

            reference=self.reference_pose['screen_right']

        )

        left_cmd = self.apply_arm_sign('left', left_raw)



        robot_left_smooth = self.smooth_positions(self.left_buffer, left_cmd)

        robot_right_smooth = self.smooth_positions(self.right_buffer, right_cmd)



        self.send_command(self.left_client, robot_left_smooth)

        self.send_command(self.right_client, robot_right_smooth)



        # === 시각화 ===

        skeleton = [

            (5, 7, (255, 0, 0)),

            (7, 9, (255, 100, 0)),

            (6, 8, (0, 0, 255)),

            (8, 10, (0, 100, 255))

        ]

        for start_idx, end_idx, color in skeleton:

            if conf[start_idx] > 0.3 and conf[end_idx] > 0.3:

                start = tuple(kps[start_idx].astype(int))

                end = tuple(kps[end_idx].astype(int))

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



        info_y = 40

        cv2.rectangle(display_frame, (10, 10), (820, 160), (0, 0, 0), -1)



        cv2.putText(display_frame, f"MIRROR MODE | Confidence: {avg_conf:.2f}",

                    (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)



        cv2.putText(

            display_frame,

            f"Robot LEFT:  J1={robot_left_smooth[0]:+.2f} J2={robot_left_smooth[1]:+.2f} J4={robot_left_smooth[3]:+.2f}",

            (20, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2

        )

        cv2.putText(

            display_frame,

            f"Robot RIGHT: J1={robot_right_smooth[0]:+.2f} J2={robot_right_smooth[1]:+.2f} J4={robot_right_smooth[3]:+.2f}",

            (20, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2

        )



        cv2.putText(display_frame, "Press 'r' to recalibrate | 'q' to quit",

                    (20, info_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



        cv2.imshow("Mirror Mode Control", display_frame)

        key = cv2.waitKey(1) & 0xFF



        if key == ord('q'):

            self.get_logger().info("종료 요청됨")

            raise KeyboardInterrupt

        elif key == ord('r'):

            self.calibration_frames = []

            self.is_calibrated = False

            self.left_buffer.clear()

            self.right_buffer.clear()

            self.get_logger().info("🔄 재캘리브레이션 시작!")





def main():

    rclpy.init()

    node = None

    try:

        node = SimpleArmController()

        rclpy.spin(node)

    except KeyboardInterrupt:

        if node:

            node.get_logger().info("종료 중...")

    except Exception as e:

        if node:

            node.get_logger().error(f"오류 발생: {e}")

        print(f"오류: {e}")

    finally:

        if node:

            if node.cap.isOpened():

                node.cap.release()

            node.destroy_node()

        cv2.destroyAllWindows()

        rclpy.shutdown()





if __name__ == "__main__":

    main()
