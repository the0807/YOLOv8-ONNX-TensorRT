from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation model')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=True, help='[fp16, int8]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    args = parser.parse_args()
    
    model = YOLO(args.model, task='detect')
    if args.q == 'fp16':
        results = model.val(
            data=args.data,
            batch=1,
            imgsz=640,
            verbose=False,
            half = True
        )
    if args.q == 'int8':
        results = model.val(
            data=args.data,
            batch=1,
            imgsz=640,
            verbose=False,
            int8 = True
        )
