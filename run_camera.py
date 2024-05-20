import argparse
from utils.detect import detect_camera
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run object detection on camera stream.')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .onnx or .engine')
    parser.add_argument('--q', type=str, default='', required=False, help='[fp16, int8]')
    args = parser.parse_args()
    
    file_extension = pathlib.Path(args.model).suffix
    
    if file_extension == '.engine':
        detect_camera(args.model, args.q)
    elif file_extension == '.onnx':
        if args.q == 'fp16':
            detect_camera(args.model, args.q)
        else:
            print('\n[ERROR] ONNX supports only FP16 quantization!')
    elif file_extension == '.pt':
        args.q = ''
        detect_camera(args.model, args.q)
    else:
        print('\n[ERROR] File extensions only support [.engine], [.onnx], [.pt]')

