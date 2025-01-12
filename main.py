from collections import deque
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List, Dict, Deque
import argparse
from enum import Enum
from tqdm import tqdm

class Exercise(Enum):
    SQUATS = "squats"
    SHOULDER_PRESS = "dumbbell_shoulder_press"

class RepCounter:
    def __init__(self):
        self.is_in_rep = False
        self.rep_count = 0
        self.rep_scores = []
        self.current_rep_data = {
            'lowest_knee_angle': float('inf'),
            'hip_angles': [],
            'knee_angles': [],
            'back_angles': []  # Angle between shoulders, hips, and knees
        }
        self.threshold_angle = 160  # More lenient threshold to detect start of rep
        
    def process_angles(self, knee_angle: float, hip_angle: float, back_angle: float) -> bool:
        """
        Track rep progress and store metrics throughout the movement
        Returns True if currently in a rep
        """
        if not self.is_in_rep and knee_angle < self.threshold_angle:
            # Starting a new rep
            self.is_in_rep = True
            self.current_rep_data = {
                'lowest_knee_angle': float('inf'),
                'hip_angles': [],
                'knee_angles': [],
                'back_angles': []
            }
            
        if self.is_in_rep:
            # Store all angles during the rep
            self.current_rep_data['knee_angles'].append(knee_angle)
            self.current_rep_data['hip_angles'].append(hip_angle)
            self.current_rep_data['back_angles'].append(back_angle)
            
            # Track lowest point of squat
            if knee_angle < self.current_rep_data['lowest_knee_angle']:
                self.current_rep_data['lowest_knee_angle'] = knee_angle
            
        if self.is_in_rep and knee_angle > self.threshold_angle:
            # Finishing a rep - calculate overall score
            score, deductions = self._calculate_rep_score()
            self.rep_scores.append((score, deductions))
            self.rep_count += 1
            self.is_in_rep = False
            
        return self.is_in_rep
    
    def _calculate_rep_score(self) -> Tuple[float, List[str]]:
        """
        Calculate overall rep score based on:
        1. Depth (knee angle at bottom)
        2. Hip hinge form throughout movement
        3. Back angle consistency
        4. Symmetry of descent and ascent
        """
        score = 100
        deductions = []
        
        # Check depth
        lowest_knee = self.current_rep_data['lowest_knee_angle']
        if lowest_knee > 90:  # Didn't reach parallel
            deduction = min((lowest_knee - 90) * 2, 40)  # Up to 40 point deduction
            score -= deduction
            deductions.append(f"Depth: -{deduction:.1f} (didn't reach parallel)")
            
        # Check hip hinge
        hip_angles = self.current_rep_data['hip_angles']
        avg_hip_angle = sum(hip_angles) / len(hip_angles)
        if avg_hip_angle < 45:  # Too much forward lean
            deduction = min((45 - avg_hip_angle) * 1.5, 30)
            score -= deduction
            deductions.append(f"Hip hinge: -{deduction:.1f} (excessive forward lean)")
            
        # Check back angle consistency
        back_angles = self.current_rep_data['back_angles']
        back_angle_variance = np.var(back_angles)
        if back_angle_variance > 100:  # High variance indicates unstable back angle
            deduction = min(back_angle_variance / 10, 20)
            score -= deduction
            deductions.append(f"Back stability: -{deduction:.1f} (inconsistent back angle)")
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        return score, deductions

class FormAnalyzer:
    def __init__(self, model_size='x'):
        # Load YOLOv8 pose estimation model based on size
        model_name = f'yolov8{model_size}-pose.pt'
        self.model = YOLO(model_name)
        # Buffer for landmark smoothing (store last 7 frames)
        self.landmark_buffer: Dict[int, Deque] = {}
        self.buffer_size = 7
        self.rep_counter = RepCounter()
        
        # YOLO keypoint indices
        self.KEYPOINTS = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

    def _init_landmark_buffer(self, num_keypoints: int):
        """Initialize circular buffers for each keypoint"""
        self.landmark_buffer = {
            i: deque(maxlen=self.buffer_size) for i in range(num_keypoints)
        }

    def _smooth_landmarks(self, current_landmarks: np.ndarray) -> np.ndarray:
        """Average landmarks over last 7 frames"""
        num_keypoints = len(current_landmarks)
        
        # Initialize buffer if not done
        if not self.landmark_buffer:
            self._init_landmark_buffer(num_keypoints)
        
        # Add current landmarks to buffer
        for i, landmark in enumerate(current_landmarks):
            self.landmark_buffer[i].append(landmark)
        
        # Calculate smoothed landmarks
        smoothed = np.zeros_like(current_landmarks)
        for i in range(num_keypoints):
            if len(self.landmark_buffer[i]) > 0:
                smoothed[i] = np.mean(self.landmark_buffer[i], axis=0)
            else:
                smoothed[i] = current_landmarks[i]
                
        return smoothed

    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def analyze_squat_form(self, keypoints: np.ndarray) -> Tuple[float, List[str]]:
        """Analyze squat form using YOLO keypoints"""
        # Get relevant keypoints
        left_shoulder = keypoints[self.KEYPOINTS['left_shoulder']][:2]
        right_shoulder = keypoints[self.KEYPOINTS['right_shoulder']][:2]
        left_hip = keypoints[self.KEYPOINTS['left_hip']][:2]
        right_hip = keypoints[self.KEYPOINTS['right_hip']][:2]
        left_knee = keypoints[self.KEYPOINTS['left_knee']][:2]
        right_knee = keypoints[self.KEYPOINTS['right_knee']][:2]
        left_ankle = keypoints[self.KEYPOINTS['left_ankle']][:2]
        right_ankle = keypoints[self.KEYPOINTS['right_ankle']][:2]
        
        # Calculate angles for both sides
        left_knee_angle = self._calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self._calculate_angle(right_hip, right_knee, right_ankle)
        
        left_hip_angle = self._calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = self._calculate_angle(right_shoulder, right_hip, right_knee)
        
        # Calculate back angle (angle between shoulders, hips, and knees)
        shoulder_midpoint = np.mean([left_shoulder, right_shoulder], axis=0)
        hip_midpoint = np.mean([left_hip, right_hip], axis=0)
        knee_midpoint = np.mean([left_knee, right_knee], axis=0)
        back_angle = self._calculate_angle(shoulder_midpoint, hip_midpoint, knee_midpoint)
        
        # Average the angles from both sides
        knee_angle = (left_knee_angle + right_knee_angle) / 2
        hip_angle = (left_hip_angle + right_hip_angle) / 2
        
        # Process the rep and get scoring information
        is_in_rep = self.rep_counter.process_angles(knee_angle, hip_angle, back_angle)
        
        # Only return the last rep score if we have one
        if self.rep_counter.rep_scores:
            return self.rep_counter.rep_scores[-1]
        return 0, []  # Return 0 and empty feedback if no reps completed yet

    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, confidence: float):
        """Draw skeleton with larger points and connections"""
        # Define YOLO pose pairs for drawing
        skeleton = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                   [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        
        # Draw points
        for x, y, conf in keypoints:
            if conf > confidence:
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)  # Larger circles
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 0), 2)   # Add outline
        
        # Draw connections
        for pair in skeleton:
            p1, p2 = pair
            if (keypoints[p1-1][2] > confidence and 
                keypoints[p2-1][2] > confidence):
                cv2.line(frame, 
                        (int(keypoints[p1-1][0]), int(keypoints[p1-1][1])),
                        (int(keypoints[p2-1][0]), int(keypoints[p2-1][1])),
                        (0, 255, 0), 3)  # Thicker lines

    def process_video(self, exercise: Exercise, video_path: str, model_size: str) -> Tuple[str, List[Tuple[float, List[str]]], int]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Include model size in output filename
        output_path = f"analyzed_{exercise.value}_yolov8{model_size}.mp4"
        out = create_video_writer(output_path, fps, width, height)
        # out = cv2.VideoWriter(output_path, 
        #                     cv2.VideoWriter_fourcc(*'mp4v'),
        #                     fps, (width, height))
        
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run YOLOv8 pose estimation
                results = self.model(frame, conf=0.5)[0]
                if len(results.keypoints) > 0:
                    keypoints = results.keypoints[0].data[0].cpu().numpy()
                    smoothed_keypoints = self._smooth_landmarks(keypoints)
                    self.draw_skeleton(frame, smoothed_keypoints, confidence=0.5)
                    
                    # Analyze form
                    if exercise == Exercise.SQUATS:
                        current_score, feedback = self.analyze_squat_form(smoothed_keypoints)
                    
                    # Add visual feedback
                    cv2.putText(frame, f"Current Score: {current_score:.1f}%", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"Reps: {self.rep_counter.rep_count}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Display feedback
                    y_pos = 180
                    for fb in feedback:
                        cv2.putText(frame, fb, (10, y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        y_pos += 40
                    
                    # Add rep tracking visualization
                    cv2.putText(frame, "IN REP" if self.rep_counter.is_in_rep else "REST", 
                            (width - 200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (0, 255, 0) if self.rep_counter.is_in_rep else (0, 0, 255), 2)
                
                out.write(frame)
                pbar.update(1)
                
        cap.release()
        out.release()
        
        return output_path, self.rep_counter.rep_scores, self.rep_counter.rep_count

def create_video_writer(output_path, fps, width, height):
    """Create a video writer with the best available codec"""
    # Try codecs in order of preference
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
   parser = argparse.ArgumentParser(description='Analyze exercise form from video')
   parser.add_argument('exercise', type=str, choices=['squats', 'dumbbell_shoulder_press'],
                      help='Type of exercise to analyze')
   parser.add_argument('video_path', type=str, help='Path to input video file')
   parser.add_argument('--model-size', type=str, choices=['n', 's', 'm', 'l', 'x'],
                      default='x', help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=extra large)')
   
   args = parser.parse_args()
   
   analyzer = FormAnalyzer(model_size=args.model_size)
   exercise = Exercise(args.exercise)
   
   output_path, rep_results, rep_count = analyzer.process_video(exercise, args.video_path, args.model_size)
   print(f"\nAnalysis complete! Output saved to: {output_path}")
   print(f"Number of reps detected: {rep_count}")
   
   if rep_count > 0:
       print("\nDetailed rep analysis:")
       total_score = 0
       for i, (score, deductions) in enumerate(rep_results, 1):
           print(f"\nRep {i}:")
           print(f"Score: {score:.1f}%")
           if deductions:
               print("Deductions:")
               for d in deductions:
                   print(f"  - {d}")
           total_score += score
           
       avg_score = total_score / rep_count
       print(f"\nOverall average score: {avg_score:.1f}%")
   else:
       print("\nNo reps detected in the video.")

if __name__ == "__main__":
    main()

# alter to detect which person is which and track the person doing the exercise