# YOLOv12 訓練模組

本目錄的文件以 [opt_train12n.py](opt_train12n.py) 為準，描述「LVIS Custom」資料集的防過擬合（freeze + 保守優化器）訓練配置與使用方式。

## 🎯 目標與策略

- 目標：最大化泛化能力、最小化 overfitting 風險
- 核心：凍結大部分 backbone（`freeze=10`）、搭配 AdamW + 較強正則化、關閉 mosaic/mixup

## 📋 訓練配置（對齊腳本）

### 模型與資料

- **模型權重**：`yolo12l.pt`（腳本目前載入 Large 權重）
- **資料集**：`datasets/lvis_custom_yolo/data.yaml`
- **輸出位置**：`FindForYou/runs/train/<run_name>/`
- **run name**：`lvis_yolov12n_freeze20_anti_overfit`

### 偵測類別（13 類，來自 data.yaml）

1. cellular phone
2. remote control
3. backpack
4. handbag
5. book
6. bottle
7. cup
8. key
9. watch
10. earphone
11. glasses
12. notebook
13. mask

## 🚀 快速開始

### 1) 環境準備

```bash
# (可選) 進入你的環境
conda activate d2_final

# 安裝/更新依賴
pip install -U ultralytics torch
```

### 2) 確認資料集路徑

腳本會讀取：`../../datasets/lvis_custom_yolo/data.yaml`（以 repo root 為基準）。

資料集結構應如下：

```
datasets/lvis_custom_yolo/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

快速檢查：

```bash
ls ../../datasets/lvis_custom_yolo/data.yaml
```

### 3) 執行訓練

```bash
cd /path/to/FP/FindForYou/train
python opt_train12n.py
```

## 📊 主要訓練參數（對齊腳本）

| 類別 | 參數 | 值 |
|---|---:|---:|
| 核心 | imgsz | 640 |
| 核心 | batch | 32 |
| 核心 | freeze | 10 |
| 迭代 | epochs | 150 |
| 迭代 | patience | 50 |
| 最佳化 | optimizer | AdamW |
| 最佳化 | lr0 | 0.005 |
| 最佳化 | lrf | 0.001 |
| 最佳化 | warmup_epochs | 5.0 |
| 正則化 | weight_decay | 0.001 |
| 正則化 | dropout | 0.1 |
| 增強 | mosaic / mixup / copy_paste | 0.0 / 0.0 / 0.0 |
| 增強 | degrees / translate / scale / shear | 10 / 0.1 / 0.3 / 2.0 |
| 增強 | perspective / fliplr / flipud | 0.0001 / 0.5 / 0.0 |
| 增強 | hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| Loss | box / cls / dfl | 7.5 / 0.5 / 1.5 |
| 系統 | device | 0 |
| 系統 | workers | 8 |
| 系統 | cache | True |
| 系統 | amp | True |
| 輸出 | save_period | 10 |

## 💾 訓練輸出

訓練完成後，輸出會在：`FindForYou/runs/train/lvis_yolov12n_freeze20_anti_overfit/`

常見檔案：

```
FindForYou/runs/train/lvis_yolov12n_freeze20_anti_overfit/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── args.yaml
```

## 📝 使用訓練好的模型

```python
from ultralytics import YOLO

model = YOLO("FindForYou/runs/train/lvis_yolov12n_freeze20_anti_overfit/weights/best.pt")
results = model.predict("path/to/image.jpg", conf=0.5)
```

## ⚠️ 注意事項（與腳本一致）

- 腳本載入的是 `yolo12l.pt`，但 run name/列印文字仍寫「v12n」；若你是要訓練 nano，請同步調整權重檔名與 run name。
- run name 內含 `freeze20`，但實際參數是 `freeze=10`；建議將 name 改成和實際 freeze 一致，方便管理實驗。
- `cache=True` 會加速資料載入但可能增加記憶體壓力；若遇到 RAM/VRAM 壓力，可嘗試改為 `cache=False`。

## 🐛 常見問題

### CUDA Out of Memory

優先調整：降低 `batch` 或 `imgsz`。

### 找不到資料集

```bash
ls ../../datasets/lvis_custom_yolo/data.yaml
```

### ModuleNotFoundError: ultralytics

```bash
pip install -U ultralytics
```
