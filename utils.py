import os
import sys
import subprocess

def install_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        import cv2, face_recognition, numpy, pandas, playsound
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
