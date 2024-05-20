<div align="center">

# YOLOv8-ONNX-TensorRT
### 👀 YOLOv8 optimized with ONNX or TensorRT and applied to Real-time camera

</div>

# 🏆 Performance

### ⭐ TensorRT

#### Tested on `Nvidia Jetson Orin Nano`

<details>
<summary>yolov8n</summary>
<!-- - #### yolov8n -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)||32|21.9|37.1
|yolov8n.engine|FP16|65|7.3|37
|yolov8n.engine|INT8|70|4.3|26.2

</details>

<details>
<summary>yolov8s</summary>
<!-- - #### yolov8s -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)||27|33.1|44.7
|yolov8s.engine|FP16|50|11.4|44.7
|yolov8s.engine|INT8|60|5.6|37.8

</details>

<details open>
<summary>yolov8m</summary>
<!-- - #### yolov8m -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)||15|66.5|50.1
|yolov8m.engine|FP16|30|23.6|50
|yolov8m.engine|INT8|40|13.9|44.1

</details>

<details>
<summary>yolov8l</summary>
<!-- - #### yolov8l -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8l.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)||10||
|yolov8l.engine|FP16|25||
|yolov8l.engine|INT8|32|18.3|46

</details>

<details>
<summary>yolov8x</summary>
<!-- - #### yolov8x -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)||||
|yolov8x.engine|FP16|||
|yolov8x.engine|INT8||33.4|38.4

- YOLOv8x has limitations in optimizing

</details>


***NOTICE:***

- Use optimal parameters for each model
- **FPS** is based on when an object is detected. It increases by about 5 frames when an object is not detected.
- **Speed** average and **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org) dataset.


# ✏️ Prepare
1. Install `CUDA`

    🚀 [`CUDA official website`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#download-the-nvidia-cuda-toolkit)

2. Install `PyTorch`

    🚀 [`PyTorch official website`](https://pytorch.org/get-started/locally/)

3. Install if using `TensorRT`

    🚀 [`TensorRT official website`](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

4. Git clone and Install python requirements
     ``` shell
     git clone https://github.com/the0807/YOLOv8-ONNX-TensorRT
     cd YOLOv8-ONNX-TensorRT
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

- Install compatible `PyTorch` in the `CUDA` version

    🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)


# ⚡️ Optional (recommend for high speed)

### ⭐ Jetson
<details>
<summary>details</summary>

- Enable MAX Power Mode and Jetson Clocks

     ``` shell
     # MAX Power Mode
     sudo nvpmodel -m 0
    
     # Enable Clocks (Do it again when you reboot)
     sudo jetson_clocks
     ```

- Install Jetson Stats Application

     ``` shell
     sudo apt update
     sudo pip install jetson-stats
     sudo reboot
     jtop
     ```

</details>


# 📚 Usage

### ⭐ ONNX
<details>
<summary>details</summary>

### 1. Turn the PyTorch model into ONNX

#### Make sure the model path is correct before running

 ``` shell
 python3 export_onnx.py --model 'model/yolov8n.pt' --q fp16 --data='coco8.yaml'
 ```

Please see more information in [`ultralytics_export`](https://docs.ultralytics.com/modes/export/)

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.pt`
- `--q` : Quantization method `[fp16]`
- `--data` : Path to your data.yaml
- `--batch` : Specifies export model batch inference size or the max number of images the exported model will process concurrently in predict mode.


### 2. Real-time camera inference

``` shell
python3 run_camera.py --model 'model/yolov8n.onnx' --q fp16
```
Please see more information in [`ultralytics_predict`](https://docs.ultralytics.com/modes/predict/)

#### Description of all arguments:
- `--model` : The PyTorch model you trained such as `yolov8n.onnx`
- `--q` : Quantization method `[fp16]`

</details>


### ⭐ TensorRT
<details open>
<summary>details</summary>

### 1. Turn the PyTorch model into TensorRT engine

#### Make sure the model path is correct before running

 ``` shell
 python3 export_tensorrt.py --model 'model/yolov8n.pt' --q int8 --data='coco8.yaml' --workspace 2 --batch 3
 ```
If aborted or killed appears, reduce the `--batch` and `--workspace`

Please see more information in [`ultralytics_export`](https://docs.ultralytics.com/modes/export/)

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.pt`
- `--q` : Quantization method `[fp16, int8]`
- `--data` : Path to your data.yaml
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

</details>


# 🧐 Validation

### ⭐ ONNX
<details>
<summary>details</summary>

 ``` shell
 python3 validation.py --model 'model/yolov8n.onnx' --q fp16 --data 'coco.yaml'
 ```

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.onnx`
- `--q` : Quantization method `[fp16]`
- `--data` : Path to your validata.yaml

</details>

### ⭐ TensorRT
<details open>
<summary>details</summary>

 ``` shell
 python3 validation.py --model 'model/yolov8n.engine' --q int8 --data 'coco.yaml'
 ```

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.engine`
- `--q` : Quantization method `[fp16, int8]`
- `--data` : Path to your validata.yaml

</details>
