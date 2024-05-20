from ultralytics import YOLO
import argparse
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='validation model')
    parser.add_argument('--model', type=str, default='model/yolov8n.engine', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=False, help='[fp16, int8]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    args = parser.parse_args()
    
    model = YOLO(args.model, task='detect')
    
    file_extension = pathlib.Path(args.model).suffix

    if file_extension == '.engine':
        if args.q == 'fp16':
            results = model.val(
                data=args.data,
                batch=1,
                imgsz=640,
                verbose=False,
                half = True
            )
        elif args.q == 'int8':
            results = model.val(
                data=args.data,
                batch=1,
                imgsz=640,
                verbose=False,
                int8 = True
            )
        else:
            print('\n[ERROR] Make sure "--q" parameter is entered correctly!')
    elif file_extension == '.onnx':
        if args.q == 'fp16':
            results = model.val(
                data=args.data,
                batch=1,
                imgsz=640,
                verbose=False,
            )
        else:
            print('\n[ERROR] ONNX supports only FP16 quantization!')
    elif file_extension == '.pt':
        results = model.val(
            data=args.data,
            batch=1,
            imgsz=640,
            verbose=False,
        )
    else:
        print('\n[ERROR] File extensions only support [.engine], [.onnx], [.pt]')
