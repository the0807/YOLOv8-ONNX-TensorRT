import argparse
from utils.detect import detect_camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object detection on camera stream.')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .pt or .engine')
    args = parser.parse_args()
    
    detect_camera(args.model)
