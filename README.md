# YOLOv8-TensorRT
👀 YOLOv8 using TensorRT in real-time camera

# FPS
#### Tested on `Nvidia Jetson Orin Nano`
|Model|Quantization method|FPS|
|:---|:---:|:---:|
|yolov8n.pt||35
|yolov8n.engine|FP16|60
|yolov8n.engine|Int8|80

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

5. Install or upgrade [`ultralytics`](https://github.com/ultralytics/ultralytics) package
 
     ``` shell
     # Install
     pip install ultralytics

     # Upgrade
     pip install -U ultralytics
     ```

6. Prepare your own PyTorch weight such as `yolov8n.pt`

***NOTICE:***

Install compatible `PyTorch` in the `CUDA` version

🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)



# Jetson (optional but recommend for high speed)
- Enable MAX Power Mode and Jetson Clocks

     ``` shell
     # MAX Power Mode
     sudo nvpmodel -m 0
    
     # Enable Clocks
     sudo jetson_clocks
     ```

- Install Jetson Stats Application

     ``` shell
     sudo apt update
     sudo pip install jetson-stats
     sudo reboot
     jtop
     ```


# Usage
### 1. Turn the PyTorch model into TensorRT engine

#### Make sure the model path is correct before running

 ``` shell
 python3 export_tensorrt.py —model 'model/yolov8n.pt' —q int8
 ```
Please see more information in [`ultralytics_export`](https://docs.ultralytics.com/modes/export/)

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.pt`
- `--q` : Quantization method `[fp16, int8]`
- `--batch` : Specifies export model batch inference size or the max number of images the exported model will process concurrently in predict mode.
- `--workspace` : Sets the maximum workspace size in GiB for TensorRT optimizations, balancing memory usage and performance.


### 2. Real-time camera inference

``` shell
python3 run_camera.py --model 'model/yolov8n.engine' --q int8
```
Please see more information in [`ultralytics_predict`](https://docs.ultralytics.com/modes/predict/)

#### Description of all arguments:
- `--model` : The PyTorch model you trained such as `yolov8n.pt` or `yolov8n.engine`
- `--q` : Quantization method `[fp16, int8]`
