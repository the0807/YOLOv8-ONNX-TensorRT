# YOLOv8-TensorRT
👀 YOLOv8 using TensorRT in real-time camera


# Prepare
1. Install `CUDA`

    🚀 [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit)

2. Install `PyTorch`

    🚀 [`PyTorch official website`](https://pytorch.org/get-started/locally/)

3. Install `TensorRT`

    🚀 [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

4. Install python requirements
 ``` shell
 pip install -r requirements.txt
 ```

5. Install [`ultralytics`](https://github.com/ultralytics/ultralytics) package
 ``` shell
 pip install ultralytics
 ```

6. Prepare your own PyTorch weight such as `yolov8n.pt`


***NOTICE:***

Install compatible `PyTorch` in the `CUDA` version follow [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)

Tested on Nvidia Jetson Orin Nano


# Usage
### 1. Turn the PyTorch model into TensorRT engine

Make sure the model path is correct before running

 ``` shell
 python3 export_tensorrt.py
 ```
Please see more information in [`ultralytics_export`](https://docs.ultralytics.com/modes/export/)

### 2. Real-time camera inference

``` shell
python3 run_camera.py --model 'model/yolov8n.engine'
```
Please see more information in [`ultralytics_predict`](https://docs.ultralytics.com/modes/predict/)

#### Description of all arguments

- `--model` : The PyTorch model you trained such as `yolov8n.pt` or `yolov8n.engine`
