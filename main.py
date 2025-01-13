import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import argparse
from tqdm import tqdm
from collections import deque
import os

@dataclass
class PersonDetection:
    landmarks: List[np.ndarray]
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max

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
            'lowest_back_angle': None
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
                self.current_rep_data['lowest_back_angle'] = back_angle

        if self.is_in_rep and knee_angle > self.threshold_angle:
            score, deductions = self._calculate_rep_score()
            self.rep_scores.append((score, deductions))
            self.rep_count += 1
            self.is_in_rep = False

        return self.is_in_rep
    
    def _calculate_rep_score(self) -> Tuple[float, List[str]]:
        score = 100
        deductions = []

        # squat depth
        lowest_knee = self.current_rep_data['lowest_knee_angle']
        if lowest_knee > 90:
            deduction = min((lowest_knee - 90) * 2, 40)
            score -= deduction
            deductions.append(f"Depth: -{deduction:.1f} (didn't reach parallel)")

        # hip hinge
        hip_angles = self.current_rep_data['hip_angles']
        avg_hip_angle = sum(hip_angles) / len(hip_angles)
        if avg_hip_angle < 45:
            deduction = min((45 - avg_hip_angle) * 1.5, 30)
            score -= deduction
            deductions.append(f"Hip hinge: -{deduction:.1f} (excessive forward lean)")

        # back angle
        if self.current_rep_data['lowest_back_angle'] is not None:
            back_angle = self.current_rep_data['lowest_back_angle']
            if back_angle < 15 or back_angle > 50:
                if back_angle < 15:
                    angle_diff = 15 - back_angle
                    message = f"torso too vertical (angle: {back_angle:.1f}°)"
                else:
                    angle_diff = back_angle - 50
                    message = f"torso too horizontal (angle: {back_angle:.1f}°)"
                deduction = min(angle_diff * 2, 30)
                score -= deduction
                deductions.append(f"Back angle: -{deduction:.1f} ({message})")
        
        # back stability
        back_angles = self.current_rep_data['back_angles']
        back_angle_variance = np.var(back_angles)
        if back_angle_variance > 100:
            deduction = min(back_angle_variance / 20, 10)
            score -= deduction
            deductions.append(f"Back stability: -{deduction:.1f} (inconsistent back angle)")

        score = max(0, score)
        return score, deductions

class RobustFormAnalyzer:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=2
        )
        self.rep_counter = RepCounter()
        self.landmark_buffer = deque(maxlen=6)
        self.previous_person = None
        self.person_lost_frames = 0
        self.MAX_LOST_FRAMES = 10

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def _calculate_angle_from_vertical(self, top_point: np.ndarray, bottom_point: np.ndarray) -> float:
        dx = top_point[0] - bottom_point[0]
        dy = top_point[1] - bottom_point[1]
        raw_angle = np.abs(np.arctan2(dx, dy) * 180.0 / np.pi)
        
        return raw_angle if raw_angle <= 90 else 180 - raw_angle

    def _get_person_detections(self, results, frame_width, frame_height) -> List[PersonDetection]:
        if not results.pose_landmarks:
            return []

        persons = []
        landmarks = results.pose_landmarks.landmark
        
        landmark_points = []
        total_confidence = 0
        visible_landmarks = 0
        
        for landmark in landmarks:
            landmark_points.append(np.array([landmark.x, landmark.y]))
            if landmark.visibility > 0.5:
                total_confidence += landmark.visibility
                visible_landmarks += 1
        
        avg_confidence = total_confidence / max(visible_landmarks, 1)
        
        x_coords = [lm[0] for lm in landmark_points]
        y_coords = [lm[1] for lm in landmark_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        persons.append(PersonDetection(
            landmarks=landmark_points,
            confidence=avg_confidence,
            bounding_box=(x_min, y_min, x_max, y_max)
        ))
        
        return persons

    def _calculate_person_similarity(self, person1: PersonDetection, person2: PersonDetection) -> float:
        box1 = person1.bounding_box
        box2 = person2.bounding_box
        
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        landmark_distance = np.mean([np.linalg.norm(np.array(l1) - np.array(l2)) 
                                   for l1, l2 in zip(person1.landmarks, person2.landmarks)])
        landmark_similarity = 1 / (1 + landmark_distance)
        
        return 0.6 * iou + 0.4 * landmark_similarity

    def _select_target_person(self, persons: List[PersonDetection]) -> Optional[PersonDetection]:
        if not persons:
            self.person_lost_frames += 1
            return None
            
        if self.previous_person is None:
            target_person = max(persons, key=lambda p: p.confidence)
        else:
            similarity_scores = [
                (person, self._calculate_person_similarity(person, self.previous_person))
                for person in persons
            ]
            target_person = max(similarity_scores, key=lambda x: x[1] * x[0].confidence)[0]
        
        self.previous_person = target_person
        self.person_lost_frames = 0
        return target_person

    def analyze_squat_form(self, landmarks) -> Tuple[float, List[str]]:
        def get_point(idx):
            return landmarks[idx]

        # Get key points
        left_shoulder = get_point(self.mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        right_shoulder = get_point(self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        left_hip = get_point(self.mp_pose.PoseLandmark.LEFT_HIP.value)
        right_hip = get_point(self.mp_pose.PoseLandmark.RIGHT_HIP.value)
        left_knee = get_point(self.mp_pose.PoseLandmark.LEFT_KNEE.value)
        right_knee = get_point(self.mp_pose.PoseLandmark.RIGHT_KNEE.value)
        left_ankle = get_point(self.mp_pose.PoseLandmark.LEFT_ANKLE.value)
        right_ankle = get_point(self.mp_pose.PoseLandmark.RIGHT_ANKLE.value)

        # Calculate angles
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)

        shoulder_midpoint = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_midpoint = np.mean([left_hip, right_hip], axis=0)
        
        back_angle = self._calculate_angle_from_vertical(shoulder_midpoint, hip_midpoint)
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        hip_angle = (left_hip_angle + right_hip_angle) / 2

        is_in_rep = self.rep_counter.process_angles(knee_angle, hip_angle, back_angle)

        if self.rep_counter.rep_scores:
            return self.rep_counter.rep_scores[-1]
        return 0, []

    def _create_video_writer(self, output_path: str, fps: int, width: int, height: int):
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

    def _calculate_text_scale(self, width: int, height: int) -> Tuple[float, int, int]:
        """Calculate dynamic text scale and thickness based on video resolution."""
        # Base this on the video's diagonal size
        diagonal = np.sqrt(width * width + height * height)
        base_diagonal = 1920  # Base resolution diagonal (1920x1080)
        
        # Scale factor based on resolution
        scale_factor = diagonal / base_diagonal
        
        # Calculate font scale, thickness, and padding
        font_scale = max(0.6, min(2.5, scale_factor))
        thickness = max(1, min(4, int(scale_factor * 2)))
        padding = max(10, min(40, int(scale_factor * 20)))
        
        return font_scale, thickness, padding

    def _draw_padded_bbox(self, frame, landmarks: List[np.ndarray], width: int, height: int):
        """Draw a padded bounding box around the detected person."""
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = [(int(lm[0] * width), int(lm[1] * height)) for lm in landmarks]
        
        # Get bounding box coordinates
        x_coords = [x for x, _ in pixel_coords]
        y_coords = [y for _, y in pixel_coords]
        
        # Calculate box with padding (proportional to frame size)
        padding_x = int(width * 0.05)  # 5% of frame width
        padding_y = int(height * 0.05)  # 5% of frame height
        
        x_min = max(0, min(x_coords) - padding_x)
        x_max = min(width, max(x_coords) + padding_x)
        y_min = max(0, min(y_coords) - padding_y)
        y_max = min(height, max(y_coords) + padding_y)
        
        # Draw the bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)  # Red color
        
        return (x_min, y_min, x_max, y_max)

    def process_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Calculate dynamic text parameters based on resolution
        font_scale, text_thickness, padding = self._calculate_text_scale(width, height)

        input_file_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{input_file_name}_analyzed.mp4"
        out = self._create_video_writer(output_path, fps, width, height)

        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    persons = self._get_person_detections(results, width, height)
                    target_person = self._select_target_person(persons)
                    
                    if target_person and target_person.confidence > 0.6:
                        # Store landmarks in buffer for smoothing
                        self.landmark_buffer.append(target_person.landmarks)
                        
                        if len(self.landmark_buffer) == self.landmark_buffer.maxlen:
                            # Average landmarks over the buffer
                            avg_landmarks = [
                                np.mean([frame[i] for frame in self.landmark_buffer], axis=0)
                                for i in range(len(target_person.landmarks))
                            ]
                            
                            # Convert to pixel coordinates for drawing
                            pixel_landmarks = [
                                (int(lm[0] * width), int(lm[1] * height))
                                for lm in avg_landmarks
                            ]
                            
                            # Draw skeleton
                            for connection in self.mp_pose.POSE_CONNECTIONS:
                                start_idx, end_idx = connection
                                cv2.line(frame, pixel_landmarks[start_idx], 
                                    pixel_landmarks[end_idx], (0, 255, 0), 3)
                            
                            for point in pixel_landmarks:
                                cv2.circle(frame, point, 6, (255, 0, 0), -1)
                            
                            current_score, feedback = self.analyze_squat_form(avg_landmarks)
                            
                            bbox = self._draw_padded_bbox(frame, avg_landmarks, width, height)
                            
                            feedback_scale = font_scale * 0.6
                            
                            # rep counter
                            rep_text = f"Reps: {self.rep_counter.rep_count}"
                            cv2.putText(frame, rep_text, (padding, padding + int(40 * font_scale)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thickness)
                            
                            # feedback
                            line_spacing = int(40 * feedback_scale)
                            for i, fb in enumerate(feedback):
                                cv2.putText(frame, fb, 
                                        (padding, padding + int(40 * font_scale) + (i + 1) * line_spacing), 
                                        cv2.FONT_HERSHEY_SIMPLEX, feedback_scale, (0, 0, 255), 
                                        max(1, text_thickness - 1))
                            
                            # rep status
                            status_text = "IN REP" if self.rep_counter.is_in_rep else "REST"
                            status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                        font_scale, text_thickness)[0]
                            status_x = width - status_size[0] - padding
                            cv2.putText(frame, status_text, (status_x, padding + int(40 * font_scale)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                    (0, 255, 0) if self.rep_counter.is_in_rep else (0, 0, 255),
                                    text_thickness)
                            
                            angles_scale = font_scale 
                            line_height = int(40 * angles_scale)
                            
                            # left knee and right knee angles
                            left_knee = self._calculate_angle(avg_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value])
                            right_knee = self._calculate_angle(avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                            
                            # hip angles
                            left_hip = self._calculate_angle(avg_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                        avg_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                        avg_landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value])
                            right_hip = self._calculate_angle(avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                            avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value])
                            
                            # back angle
                            shoulder_midpoint = np.mean([avg_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                    avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]], axis=0)
                            hip_midpoint = np.mean([avg_landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                                                avg_landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]], axis=0)
                            back_angle = self._calculate_angle_from_vertical(shoulder_midpoint, hip_midpoint)
                            
                            # angles in bottom right
                            angles_text = [
                                f"L Knee: {left_knee:.1f} deg",
                                f"R Knee: {right_knee:.1f} deg",
                                f"L Hip: {left_hip:.1f} deg",
                                f"R Hip: {right_hip:.1f} deg",
                                f"Back: {back_angle:.1f} deg"
                            ]
                            
                            max_width = 0
                            for text in angles_text:
                                (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                angles_scale, max(1, text_thickness - 1))
                                max_width = max(max_width, text_width)
                            
                            for i, text in enumerate(angles_text):
                                (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                                angles_scale, max(1, text_thickness - 1))
                                x_pos = width - padding - max_width
                                y_pos = height - padding - (len(angles_text) - 1 - i) * line_height
                                cv2.putText(frame, text, (x_pos, y_pos), 
                                        cv2.FONT_HERSHEY_SIMPLEX, angles_scale, (0, 0, 255), 
                                        text_thickness)
                            
                            # conf score
                            conf_text = f"Tracking Confidence: {target_person.confidence:.2f}"
                            cv2.putText(frame, conf_text, (padding, height - padding), 
                                    cv2.FONT_HERSHEY_SIMPLEX, feedback_scale, (255, 255, 255), 
                                    max(1, text_thickness - 1))
                    
                    elif self.person_lost_frames > self.MAX_LOST_FRAMES:
                        cv2.putText(frame, "Target Lost - Please Reset", 
                                (int(width/4), int(height/2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                out.write(frame)
                pbar.update(1)

        cap.release()
        out.release()
        
        print(f"\nAnalysis complete! Output saved to: {output_path}")
        print(f"Total Reps: {self.rep_counter.rep_count}")
        if self.rep_counter.rep_count > 0:
            total_score = sum(score for score, _ in self.rep_counter.rep_scores)
            avg_score = total_score / self.rep_counter.rep_count
            print(f"Average Score: {avg_score:.1f}%")
            for i, (score, deductions) in enumerate(self.rep_counter.rep_scores, 1):
                print(f"\nRep {i}: Score: {score:.1f}%")
                if deductions:
                    for d in deductions:
                        print(f"  - {d}")

def main():
    parser = argparse.ArgumentParser(description='Analyze squat form from video using MediaPipe Pose with robust person tracking')
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--min-detection-confidence', type=float, default=0.7,
                      help='Minimum confidence value for the detection to be considered successful')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.7,
                      help='Minimum confidence value to consider tracked landmarks valid')
    
    args = parser.parse_args()

    analyzer = RobustFormAnalyzer(
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    
    analyzer.process_video(args.video_path)

if __name__ == "__main__":
    main()