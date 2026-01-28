import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

print("\n--- Testing Imports ---")

# 1. MediaPipe
try:
    import mediapipe as mp
    print(f"✓ MediaPipe loaded: {mp.__version__}")
    if hasattr(mp, 'solutions'):
        print("  mp.solutions available")
        import mediapipe.python.solutions.pose as mp_pose
        print("  mp.solutions.pose available")
    else:
        print("  WARN: mp.solutions NOT available (using Tasks API)")
except Exception as e:
    print(f"✗ MediaPipe import error: {e}")

# 2. OpenCV
try:
    import cv2
    print(f"✓ OpenCV loaded: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV import error: {e}")

# 3. NumPy
try:
    import numpy as np
    print(f"✓ NumPy loaded: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy import error: {e}")

# 4. Track module
try:
    from track import VideoAnnotator, NumpyEncoder
    print("✓ VideoAnnotator loaded")
except ImportError as e:
    print(f"✗ VideoAnnotator import error: {e}")
except Exception as e:
    print(f"✗ VideoAnnotator other error: {e}")

# 5. Supabase client
try:
    from supabase_client import get_uploader
    uploader = get_uploader()
    print(f"✓ SupabaseUploader loaded (enabled: {uploader.enabled})")
except ImportError as e:
    print(f"✗ SupabaseUploader import error: {e}")
except Exception as e:
    print(f"✗ SupabaseUploader other error: {e}")

print("\n--- Environment ---")
print(f"SUPABASE_URL: {'SET' if os.environ.get('SUPABASE_URL') else 'NOT SET'}")
print(f"SUPABASE_SERVICE_KEY: {'SET' if os.environ.get('SUPABASE_SERVICE_KEY') else 'NOT SET'}")

print("\n--- Done ---")
