import argparse
from utils.detect import detect_camera


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object detection on camera stream.')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .pt or .engine')
    parser.add_argument('--q', type=str, default='', required=False, help='[fp16, int8]')
    args = parser.parse_args()
    
    detect_camera(args.model, args.q)
