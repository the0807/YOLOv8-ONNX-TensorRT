<div align="center">

# YOLOv8-ONNX-TensorRT
### 👀 YOLOv8 optimized with ONNX or TensorRT and applied to Real-time camera

</div>

# 🏆 Performance

### ⭐ ONNX 
<details>
<summary>details</summary>

#### Tested on `Raspberry Pi 4B`

<details>
<summary>yolov8n</summary>
<!-- - #### yolov8n -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)||||
|yolov8n.onnx|FP16|||

</details>

<details>
<summary>yolov8s</summary>
<!-- - #### yolov8s -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)||||
|yolov8s.onnx|FP16|||

</details>

<details>
<summary>yolov8m</summary>
<!-- - #### yolov8m -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)||||
|yolov8m.onnx|FP16|||

</details>

<details>
<summary>yolov8l</summary>
<!-- - #### yolov8l -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8l.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)||||
|yolov8l.onnx|FP16|||

</details>

<details>
<summary>yolov8x</summary>
<!-- - #### yolov8x -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)||||
|yolov8x.onnx|FP16|||

</details>

</details>

### ⭐ TensorRT
<details open>
<summary>details</summary>

#### Tested on `Nvidia Jetson Orin Nano`

<details>
<summary>yolov8n</summary>
<!-- - #### yolov8n -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)||40|20|37.1
|yolov8n.engine|FP16|70|7|37.1
|yolov8n.engine|INT8|80|4.3|26.2

</details>

<details>
<summary>yolov8s</summary>
<!-- - #### yolov8s -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)||30|30|44.7
|yolov8s.engine|FP16|55|11|44.7
|yolov8s.engine|INT8|70|5.6|37.8

</details>

<details open>
<summary>yolov8m</summary>
<!-- - #### yolov8m -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)||15|53|50
|yolov8m.engine|FP16|30|23|50
|yolov8m.engine|INT8|45|13.9|44.1

</details>

<details>
<summary>yolov8l</summary>
<!-- - #### yolov8l -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8l.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)||10|90|52.7
|yolov8l.engine|FP16|25|35|52.7
|yolov8l.engine|INT8|35|18.3|46

</details>

<details>
<summary>yolov8x</summary>
<!-- - #### yolov8x -->

|Model|Quantization|FPS|Speed<sup><br>(ms)|mAP<sup>val<br>50-95|
|:---:|:---:|:---:|:---:|:---:|
|[yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)||6|131|53.7
|yolov8x.engine|FP16|16|56|53.7
|yolov8x.engine|INT8|25|33|

</details>

</details>


***NOTICE:***

- Use optimal parameters for each model
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
