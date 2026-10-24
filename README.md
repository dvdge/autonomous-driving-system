# 基于深度学习的无人驾驶仿真项目

使用 Python、TensorFlow / Keras 和卷积神经网络，根据模拟器的摄像头图像预测转向角，并通过 Socket.IO 将转向与油门指令发送给 Udacity 驾驶模拟器。

项目包含湖泊、山地赛道的训练脚本、图像增强与预处理、三种 CNN 结构，以及基础控制、PI 控制和自适应 PID 控制的驾驶脚本。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `build_model.py` | 定义 `build_model1`、`build_model2`、`build_model3` 三种网络 |
| `preprocessing_2.py` | 摄像头图像选择、平移、翻转、裁剪、缩放与颜色空间转换 |
| `train_model_3.py` | 山地赛道训练，读取 `data_mountain/` |
| `train_model_4.py` | 湖泊赛道训练，读取 `data_lake/` |
| `drving_1.py` | 基础通信与固定转向控制演示（保留原文件名） |
| `driving_2.py` | 使用山地模型预测转向，配合 PI 速度控制 |
| `driving_3.py` | 使用湖泊模型，包含自适应速度、转向平滑及 PID 控制 |
| `test_file.py` | 依赖导入检查脚本 |
| `test/` | 三个摄像头的预处理示例图片 |
| `*.h5` | 湖泊、山地模型及训练检查点 |

## 环境

本地原环境记录为 Python 3.8.10，TensorFlow 2.3.0。`requirements.txt` 记录了现有环境中与项目直接相关的包版本，便于迁移时参考；尚未验证这些版本在全新环境中的完整安装与运行兼容性。

在兼容的 Python 环境中安装依赖：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 仿真驾驶

在项目根目录运行一个驾驶脚本，例如：

```powershell
python driving_2.py
```

再启动本地 Udacity 模拟器，选择与模型相应的赛道和自动驾驶模式。服务监听 TCP 端口 `4567`。驾驶脚本会打开 OpenCV 图像窗口，因此需要桌面图形环境。

`driving_2.py` 默认读取 `mountain_model2.h5`，`driving_3.py` 默认读取 `lake_model2.h5`。如需更换检查点，请修改脚本中的 `load_model(...)` 路径。

## 训练

完整训练数据和模拟器程序保留在本地，不包含在此仓库。训练数据需要按以下结构放置：

```text
data_lake/
  driving_log.csv
  IMG/
data_mountain/
  driving_log.csv
  IMG/
```

`driving_log.csv` 不含表头，列顺序为中心摄像头、左摄像头、右摄像头、转向角、油门、刹车、速度。图像路径应相对于对应的数据目录；如果本地数据存在额外嵌套层，请调整目录结构或脚本中的 `data_path`。

```powershell
# 山地赛道
python train_model_3.py

# 湖泊赛道
python train_model_4.py
```

两个训练脚本均使用 `build_model2()`，训练过程中保存验证损失改善时的检查点，并写入 `logs/`。

## 上传范围与验证状态

仓库包含原有 Python 源码、15 个 HDF5 模型文件和 3 张示例图片。`.gitignore` 排除了虚拟环境、IDE 设置、Python 缓存、TensorBoard 日志、完整数据集和模拟器二进制文件。

本次整理保留原有算法实现；尚未重新训练模型，也未验证模拟器闭环驾驶效果。`driving_3.py` 使用独立预处理流程，与训练脚本的预处理存在差异，使用时需结合实际模型验证。
