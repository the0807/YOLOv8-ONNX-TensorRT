<div align="center">

# YOLOv8-ONNX-TensorRT

[![Python](https://img.shields.io/badge/Python-3.8.10-yellow)](https://www.python.org/downloads/release/python-3810/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-orange)](https://pytorch.org/)
[![JetPack](https://img.shields.io/badge/JetPack-5.1.2-green)](https://developer.nvidia.com/embedded/jetpack-sdk-512)
[![CUDA](https://img.shields.io/badge/CUDA-11.4-green)](https://developer.nvidia.com/cuda-downloads)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.5.2-green)](https://developer.nvidia.com/tensorrt)
[![GitHub all releases](https://img.shields.io/github/downloads/the0807/YOLOv8-ONNX-TensorRT/total)](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases)
<!-- [![GitHub Repo stars](https://img.shields.io/github/stars/the0807/YOLOv8-ONNX-TensorRT)](https://github.com/the0807/YOLOv8-ONNX-TensorRT/stargazers) -->

### 👀 Apply YOLOv8 exported with ONNX or TensorRT(FP16, INT8) to the Real-time camera

</div>

# 🏆 Performance

> [!Note]
> -  Tested on `Nvidia Jetson Orin Nano`

### ⭐ ONNX (CPU)
<details>
<summary>details</summary>

<details open>
<summary>YOLOv8n</summary>
<!-- - #### yolov8n -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8n.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8n.zip)||2|535.8|37.1
|yolov8n.onnx|FP16|7|146|37

</details>

<details>
<summary>YOLOv8s</summary>
<!-- - #### yolov8s -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8s.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8s.zip)||1|943.9|44.7
|yolov8s.onnx|FP16|3|347.6|44.7

</details>

<details>
<summary>YOLOv8m</summary>
<!-- - #### yolov8m -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8m.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8m.zip)||0.5|1745.2|50.1
|yolov8m.onnx|FP16|1.2|1126.3|50.1

</details>

`YOLOv8l` and `YOLOv8x` were too slow to measure


</details>


### ⭐ TensorRT (GPU)
<details open>
<summary>details</summary>

<details>
<summary>YOLOv8n</summary>
<!-- - #### yolov8n -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8n.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8n.zip)||36|21.9|37.1
|yolov8n.engine|FP16|60|7.3|37
|yolov8n.engine|INT8|68|4.3|26.2

</details>

<details open>
<summary>YOLOv8s</summary>
<!-- - #### yolov8s -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8s.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8s.zip)||27|33.1|44.7
|yolov8s.engine|FP16|48|11.4|44.7
|yolov8s.engine|INT8|57|5.6|37.8

</details>

<details>
<summary>YOLOv8m</summary>
<!-- - #### yolov8m -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8m.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8m.zip)||14|66.5|50.1
|yolov8m.engine|FP16|30|23.6|50
|yolov8m.engine|INT8|37|13.9|44.1

</details>

<details>
<summary>YOLOv8l</summary>
<!-- - #### yolov8l -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8l.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8l.zip)||9|103.2|52.9
|yolov8l.engine|FP16|22|35.5|52.6
|yolov8l.engine|INT8|31|18.3|46

</details>

<details>
<summary>YOLOv8x</summary>
<!-- - #### yolov8x -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8x.pt](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases/download/v1.0/YOLOv8x.zip)||6|160.2|54.0
|yolov8x.engine|FP16|15|56.6|53.7
|yolov8x.engine|INT8|24|34.2|38.4

</details>

</details>

> [!Note]
> - Use optimal parameters for each model
> - **FPS** is based on when an object is detected
> - **Speed** average and **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](https://cocodataset.org) dataset

> [!Tip]
> - You can download the ONNX and TensorRT files from the [release](https://github.com/the0807/YOLOv8-ONNX-TensorRT/releases)

> [!Caution]
> - Optimizing and exporting models on your own devices will give you the best results

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

6. Prepare your own datasets with PyTorch weights such as 'yolov8n.pt '

7. (Optional) If you want to test with YOLOv8 base model rather than custom model, please run the code and prepare the 'coco' dataset

     ``` shell
     cd datasets

     # It will take time to download
     python3 coco_download.py
     ```

> [!Important]
> - Install compatible `PyTorch` in the `CUDA` version
> 
>     🚀 [`PyTorch Version Check`](https://pytorch.org/get-started/previous-versions/)


# ⚡️ Optional (recommend for high speed)

### ⭐ Jetson

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


# 📚 Usage

### ⭐ ONNX
<details>
<summary>details</summary>

### 1. Turn the PyTorch model into ONNX

 ``` shell
 python3 export_onnx.py --model 'model/yolov8n.pt' --q fp16 --data='coco.yaml'
 ```

#### Description of all arguments:
- `--model` : required The PyTorch model you trained such as `yolov8n.pt`
- `--q` : Quantization method `[fp16]`
- `--data` : Path to your data.yaml
- `--batch` : Specifies export model batch inference size or the max number of images the exported model will process concurrently in predict mode.


### 2. Real-time camera inference

``` shell
python3 run_camera.py --model 'model/yolov8n.onnx' --q fp16
```

#### Description of all arguments:
- `--model` : The PyTorch model you trained such as `yolov8n.onnx`
- `--q` : Quantization method `[fp16]`

</details>


### ⭐ TensorRT
<details open>
<summary>details</summary>

### 1. Turn the PyTorch model into TensorRT engine

 ``` shell
 python3 export_tensorrt.py --model 'model/yolov8n.pt' --q int8 --data='coco.yaml' --workspace 2 --batch 3
 ```

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

#### Description of all arguments:
- `--model` : The PyTorch model you trained such as `yolov8n.pt` or `yolov8n.engine`
- `--q` : Quantization method `[fp16, int8]`

</details>

> [!Tip]
> - You can get more information
> 
>     🚀 [`ultralytics_export`](https://docs.ultralytics.com/modes/export/)
> 
>     🚀 [`ultralytics_predict`](https://docs.ultralytics.com/modes/predict/)

> [!Warning]
> - If aborted or killed appears, reduce the `--batch` and `--workspace`

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


