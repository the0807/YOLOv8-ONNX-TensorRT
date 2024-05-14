from ultralytics import YOLO

# Load a model
model = YOLO('model/yolov8n.pt')

# Export the model
model.export(format='TensorRT', half = True)