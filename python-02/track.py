"""
MediaPipe Video Processing Module

Processes video frames with MediaPipe Pose detection and draws skeleton overlays.
"""

import cv2
import json
import os
import subprocess
import numpy as np
import mediapipe as mp
from typing import List, Dict, Any, Tuple
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class VideoAnnotator:
    """Processes videos with MediaPipe pose detection and skeleton drawing."""
    
    # MediaPipe pose connections for skeleton drawing
    # Landmark names for mapping indices to strings
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]
    
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """Initialize MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def process_video(
        self,
        input_path: str,
        output_path: str
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process video with MediaPipe pose detection.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save annotated video
            
        Returns:
            Tuple of (landmarks_list, metadata)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        landmarks_list = []
        frame_index = 0
        
        print(f"Processing {total_frames} frames at {fps} FPS...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            # Extract landmarks with names
            frame_landmarks_list = None
            if results.pose_landmarks:
                frame_landmarks_list = self._extract_landmarks(results.pose_landmarks)
                # Draw skeleton on frame (using MP utils)
                self._draw_skeleton(frame, results.pose_landmarks)
            
            # Store landmarks data matching original format
            # Original: frames object with keys like frameIdx, timestampSec, landmarks
            landmarks_list.append({
                "frameIdx": frame_index,
                "timestampSec": round(frame_index / fps, 3),
                "landmarks": frame_landmarks_list
            })
            
            # Write annotated frame
            out.write(frame)
            frame_index += 1
            
            # Progress logging
            if frame_index % 100 == 0:
                print(f"  Processed {frame_index}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        # Convert to H.264 for browser compatibility
        print("Converting to H.264...")
        temp_raw = output_path + ".raw.mp4"
        try:
            if os.path.exists(output_path):
                os.rename(output_path, temp_raw)
                self._convert_to_h264(temp_raw, output_path)
                # Clean up raw file
                if os.path.exists(temp_raw):
                    os.remove(temp_raw)
        except Exception as e:
            print(f"Warning: H.264 conversion failed, falling back to raw mp4v: {e}")
            if os.path.exists(temp_raw) and not os.path.exists(output_path):
                os.rename(temp_raw, output_path)
        
        # Original script returns full structure, but here we return components
        # to be assembled by handler.py
        
        metadata = {
            "fps": int(fps),
            "total_frames": total_frames,
            "duration_sec": round(total_frames / fps, 2),
            "width": width,
            "height": height
        }
        
        return landmarks_list, metadata
    
    def _extract_landmarks(self, pose_landmarks) -> List[Dict[str, Any]]:
        """Extract landmarks with names to list of dicts."""
        landmarks = []
        for i, lm in enumerate(pose_landmarks.landmark):
            name = self.LANDMARK_NAMES[i] if i < len(self.LANDMARK_NAMES) else ""
            landmarks.append({
                "name": name,
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility)
            })
        return landmarks
    
    def _draw_skeleton(
        self,
        frame: np.ndarray,
        pose_landmarks
    ) -> None:
        """Draw skeleton overlay on frame using MediaPipe drawing utils."""
        
        # Visualization specs from original script
        landmark_spec = mp_drawing.DrawingSpec(
            color=(0, 0, 255),     # RED joints
            thickness=4,
            circle_radius=4
        )
        connection_spec = mp_drawing.DrawingSpec(
            color=(0, 255, 255),   # YELLOW connections
            thickness=3,
            circle_radius=2
        )
        
        # Filter connections to exclude head (indices 0-10)
        filtered_connections = [
            c for c in mp_pose.POSE_CONNECTIONS 
            if c[0] > 10 and c[1] > 10
        ]
        
        # Hide head landmarks (visibility=0) strictly for drawing (copy not needed if we don't return these modified objects)
        # However, MP modifies in-place usually. 
        # The extract_landmarks was called BEFORE this, so we are safe to modify visibility now if we want.
        # But to be safe and avoid side effects on any future usage, we can just let MP draw and it might draw head.
        # Original script explicitly set visibility to 0.0 for idx < 11.
        
        # Create a copy or just accept we modify the object for drawing
        # Since we extracted data already, modifying visibility here is fine.
        for idx in range(11): 
            if idx < len(pose_landmarks.landmark):
                pose_landmarks.landmark[idx].visibility = 0.0
                
        try:
            mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                filtered_connections,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )
        except Exception as e:
            print(f"Drawing error: {e}")
    
    def _convert_to_h264(self, input_path: str, output_path: str) -> None:
        """Convert video to H.264 using FFmpeg for browser compatibility."""
        try:
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'fast',
                '-y',
                '-loglevel', 'error',
                output_path
            ]
            subprocess.run(cmd, check=True)
            print("✓ Converted to H.264")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed with code {e.returncode}")
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
