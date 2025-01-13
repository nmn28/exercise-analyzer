import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List
import argparse
from tqdm import tqdm
from collections import deque
import os

class RepCounter:
    def __init__(self):
        self.is_in_rep = False
        self.rep_count = 0
        self.rep_scores = []
        self.current_rep_data = {
            'lowest_knee_angle': float('inf'),
            'hip_angles': [],
            'knee_angles': [],
            'back_angles': [],
            'lowest_back_angle': None  # Store the back angle at the bottom of the rep
        }
        self.threshold_angle = 160
        
    def process_angles(self, knee_angle: float, hip_angle: float, back_angle: float) -> bool:
        if not self.is_in_rep and knee_angle < self.threshold_angle:
            self.is_in_rep = True
            self.current_rep_data = {
                'lowest_knee_angle': float('inf'),
                'hip_angles': [],
                'knee_angles': [],
                'back_angles': [],
                'lowest_back_angle': None
            }
        
        if self.is_in_rep:
            self.current_rep_data['knee_angles'].append(knee_angle)
            self.current_rep_data['hip_angles'].append(hip_angle)
            self.current_rep_data['back_angles'].append(back_angle)

            if knee_angle < self.current_rep_data['lowest_knee_angle']:
                self.current_rep_data['lowest_knee_angle'] = knee_angle
                self.current_rep_data['lowest_back_angle'] = back_angle  # Store back angle at lowest point

        if self.is_in_rep and knee_angle > self.threshold_angle:
            score, deductions = self._calculate_rep_score()
            self.rep_scores.append((score, deductions))
            self.rep_count += 1
            self.is_in_rep = False

        return self.is_in_rep
    
    def _calculate_rep_score(self) -> Tuple[float, List[str]]:
        score = 100
        deductions = []

        # Check squat depth
        lowest_knee = self.current_rep_data['lowest_knee_angle']
        if lowest_knee > 90:
            deduction = min((lowest_knee - 90) * 2, 40)
            score -= deduction
            deductions.append(f"Depth: -{deduction:.1f} (didn't reach parallel)")

        # Check hip hinge
        hip_angles = self.current_rep_data['hip_angles']
        avg_hip_angle = sum(hip_angles) / len(hip_angles)
        if avg_hip_angle < 45:
            deduction = min((45 - avg_hip_angle) * 1.5, 30)
            score -= deduction
            deductions.append(f"Hip hinge: -{deduction:.1f} (excessive forward lean)")

        # Check back angle at bottom of squat (should be between 20-50 degrees from vertical)
        if self.current_rep_data['lowest_back_angle'] is not None:
            back_angle = self.current_rep_data['lowest_back_angle']
            if back_angle < 20 or back_angle > 50:
                # Calculate deduction based on how far outside the acceptable range
                if back_angle < 20:
                    angle_diff = 20 - back_angle
                    message = f"torso too vertical (angle: {back_angle:.1f}°)"
                else:
                    angle_diff = back_angle - 50
                    message = f"torso too horizontal (angle: {back_angle:.1f}°)"
                deduction = min(angle_diff * 2, 30)  # 2 points per degree outside range, max 30
                score -= deduction
                deductions.append(f"Back angle: -{deduction:.1f} ({message})")
        
        # Check back angle stability with reduced impact
        back_angles = self.current_rep_data['back_angles']
        back_angle_variance = np.var(back_angles)
        if back_angle_variance > 100:
            # Reduced impact - max deduction of 10 points instead of original 20
            deduction = min(back_angle_variance / 20, 10)
            score -= deduction
            deductions.append(f"Back stability: -{deduction:.1f} (inconsistent back angle)")

        score = max(0, score)
        return score, deductions

class FormAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.rep_counter = RepCounter()
        self.landmark_buffer = deque(maxlen=6)

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points."""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _calculate_angle_from_vertical(self, top_point: np.ndarray, bottom_point: np.ndarray) -> float:
        """Calculate angle from vertical axis.
        Returns angle in degrees where:
        0 = vertical (top point directly above bottom point)
        90 = horizontal (top point level with bottom point)
        The angle measures forward lean, so a greater angle means more forward lean.
        """
        dx = top_point[0] - bottom_point[0]
        dy = top_point[1] - bottom_point[1]
        raw_angle = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)
        
        # Convert to forward lean angle
        if raw_angle > 90:
            # If raw_angle > 90, the torso is leaning forward
            forward_lean_angle = 180 - raw_angle
        else:
            # If raw_angle <= 90, the torso is leaning backward (rare in squats)
            forward_lean_angle = raw_angle
            
        return forward_lean_angle

    def analyze_squat_form(self, avg_landmarks) -> Tuple[float, List[str]]:
        def get_landmark(idx):
            return avg_landmarks[idx]

        left_shoulder = get_landmark(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_landmark(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_landmark(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_landmark(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_knee = get_landmark(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        right_knee = get_landmark(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        left_ankle = get_landmark(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_landmark(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)

        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)

        shoulder_midpoint = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_midpoint = np.mean([left_hip, right_hip], axis=0)
        knee_midpoint = np.mean([left_knee, right_knee], axis=0)
        
        # Calculate torso angle from vertical
        back_angle = self._calculate_angle_from_vertical(shoulder_midpoint, hip_midpoint)

        knee_angle = (left_knee_angle + right_knee_angle) / 2
        hip_angle = (left_hip_angle + right_hip_angle) / 2

        is_in_rep = self.rep_counter.process_angles(knee_angle, hip_angle, back_angle)

        if self.rep_counter.rep_scores:
            return self.rep_counter.rep_scores[-1]
        return 0, []

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        input_file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{input_file_name}_mediapipe_analyzed.mp4"

        out = self._create_video_writer(output_path, fps, width, height)

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    self.landmark_buffer.append(landmarks)

                    # Average landmarks over the buffer
                    avg_landmarks = [
                        np.mean([[lm.x, lm.y] for lm in frame_landmarks], axis=0)
                        for frame_landmarks in zip(*self.landmark_buffer)
                    ]

                    current_score, feedback = self.analyze_squat_form(avg_landmarks)

                    # Draw skeleton
                    for connection in self.mp_pose.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        start_point = (int(avg_landmarks[start_idx][0] * width), int(avg_landmarks[start_idx][1] * height))
                        end_point = (int(avg_landmarks[end_idx][0] * width), int(avg_landmarks[end_idx][1] * height))
                        cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

                    # Draw joints as dots
                    for landmark in avg_landmarks:
                        point = (int(landmark[0] * width), int(landmark[1] * height))
                        cv2.circle(frame, point, 6, (255, 0, 0), -1)  # Blue dots for joints

                    # Draw padded bounding box
                    x_coords = [lm[0] for lm in avg_landmarks]
                    y_coords = [lm[1] for lm in avg_landmarks]
                    x_min = int(min(x_coords) * width) - 50
                    x_max = int(max(x_coords) * width) + 50
                    y_min = int(min(y_coords) * height) - 50
                    y_max = int(max(y_coords) * height) + 50
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)

                    # Display rep count at top left with larger font
                    font_scale = 2.0
                    thickness = 3
                    padding = 20
                    
                    rep_text = f"Reps: {self.rep_counter.rep_count}"
                    cv2.putText(frame, rep_text, (padding, padding + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                    # Display feedback in red below rep count
                    for i, fb in enumerate(feedback):
                        cv2.putText(frame, fb, (padding, padding + 100 + i * 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                    # Display "IN REP" or "REST" at top right
                    status_text = "IN REP" if self.rep_counter.is_in_rep else "REST"
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    status_x = width - status_size[0] - padding
                    cv2.putText(frame, status_text, (status_x, padding + 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              (0, 255, 0) if self.rep_counter.is_in_rep else (0, 0, 255), 
                              thickness)

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()

        # Output details to console
        print(f"Analysis complete! Output saved to: {output_path}")
        print(f"Total Reps: {self.rep_counter.rep_count}")
        if self.rep_counter.rep_count > 0:
            total_score = sum(score for score, _ in self.rep_counter.rep_scores)
            avg_score = total_score / self.rep_counter.rep_count
            print(f"Average Score: {avg_score:.1f}%")
            for i, (score, deductions) in enumerate(self.rep_counter.rep_scores, 1):
                print(f"Rep {i}: Score: {score:.1f}%")
                if deductions:
                    for d in deductions:
                        print(f"  - {d}")

    def _create_video_writer(self, output_path: str, fps: int, width: int, height: int):
        """Create a video writer with the best available codec"""
        codecs = ['avc1', 'h264', 'XVID', 'mp4v']

        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Using {codec} codec")
                    return out
            except Exception as e:
                print(f"Failed to use {codec} codec: {str(e)}")
                continue

        raise RuntimeError("No working codec found")

def main():
    parser = argparse.ArgumentParser(description='Analyze exercise form from video using MediaPipe Pose')
    parser.add_argument('video_path', type=str, help='Path to input video file')

    args = parser.parse_args()

    analyzer = FormAnalyzer()
    analyzer.process_video(args.video_path)

if __name__ == "__main__":
    main()