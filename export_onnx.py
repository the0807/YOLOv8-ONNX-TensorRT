from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model', type=str, default='model/yolov8n.pt', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=True, help='[fp16]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size')
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    # Export the model
    if args.q == 'fp16':
        model.export(format = 'onnx', data = args.data, batch = args.batch, half = True)
    else:
        print('\n[ERROR] Make sure "--q" parameter is entered correctly!')