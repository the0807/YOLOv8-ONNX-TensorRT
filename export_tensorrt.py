from ultralytics import YOLO
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to TensorRT')
    parser.add_argument('--model', type=str, default='model/yolov8n.pt', required=True, help='Path to the .pt')
    parser.add_argument('--q', type=str, default='fp16', required=True, help='[fp16, int8]')
    parser.add_argument('--data', type=str, default='coco.yaml', required=True, help='Dataset')
    parser.add_argument('--batch', type=int, default=1, required=False, help='batch size')
    parser.add_argument('--workspace', type=int, default=4, required=False, help='workspace')
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)

    # Export the model
    if args.q == 'fp16':
        model.export(format = 'engine', data = args.data, batch = args.batch, workspace = args.workspace, half = True)
    elif args.q == 'int8':
        model.export(format = 'engine', data = args.data, batch = args.batch, workspace = args.workspace, int8 = True)
    else:
        print('\n[ERROR] Make sure "--q" parameter is entered correctly!')