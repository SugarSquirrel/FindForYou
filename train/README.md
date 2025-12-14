# YOLOv12 訓練模組

本目錄包含 FindForYou 專案的 YOLOv12 物品偵測模型訓練腳本。

## 📋 訓練配置

### 模型規格
- **架構**: YOLOv12 Medium (yolo12m)
- **輸入解析度**: 1024x1024 pixels
- **參數量**: ~25M parameters
- **Batch Size**: 4
- **訓練 Epochs**: 100
- **Early Stopping**: 50 epochs patience

### 偵測類別 (8 類)
1. 📱 cell_phone (手機)
2. 👛 wallet (錢包)
3. 🔑 key (鑰匙)
4. 📺 remote_control (遙控器)
5. ⌚ watch (手錶)
6. 🎧 earphone (耳機)
7. ☕ cup (杯子)
8. 🍶 bottle (瓶子)

## 🚀 快速開始

### 1. 環境準備

```bash
# 激活 conda 環境
conda activate d2_final

# 安裝依賴 (如果尚未安裝)
pip install ultralytics
```

### 2. 資料集準備

資料集應放置於：`../../datasets/findyou_yolo_clean/`

資料集結構：
```
datasets/findyou_yolo_clean/
├── data.yaml          # 資料集配置檔
├── images/
│   ├── train/        # 訓練影像 (7,782 張)
│   └── val/          # 驗證影像 (1,100 張)
└── labels/
    ├── train/        # 訓練標註 (YOLO 格式)
    └── val/          # 驗證標註
```

**注意**: 資料集檔案因體積過大不包含在 Git 倉庫中。

### 3. 執行訓練

```bash
cd /path/to/FindForYou/train
python train_yolov12m.py
```

## 📊 訓練參數

| 參數 | 值 | 說明 |
|------|-----|------|
| imgsz | 1024 | 訓練影像尺寸 |
| batch | 4 | Batch size |
| epochs | 100 | 訓練輪數 |
| patience | 50 | Early stopping |
| lr0 | 0.01 | 初始學習率 |
| lrf | 0.01 | 最終學習率 |
| weight_decay | 0.0005 | 權重衰減 |
| optimizer | auto | 自動選擇優化器 (SGD) |
| amp | True | 混合精度訓練 |

## 💾 訓練輸出

訓練完成後，輸出檔案位於：`../../runs/train/findyou_yolov12m_1024/`

```
runs/train/findyou_yolov12m_1024/
├── weights/
│   ├── best.pt      # 最佳模型 (mAP 最高)
│   └── last.pt      # 最後一個 epoch 的模型
├── results.png      # 訓練曲線圖
├── confusion_matrix.png  # 混淆矩陣
├── labels.jpg       # 標籤統計圖
└── args.yaml        # 訓練參數記錄
```

## 🎯 效能指標

訓練完成後會顯示以下指標：
- **mAP50**: IoU=0.5 的平均精度
- **mAP50-95**: IoU=0.5-0.95 的平均精度
- **Precision**: 精確率
- **Recall**: 召回率

## 🔧 調整訓練參數

若遇到 GPU 記憶體不足，可調整以下參數：

```python
# 在 train_yolov12m.py 中修改
training_args = {
    "imgsz": 640,      # 降低解析度
    "batch": 2,        # 降低 batch size
    # ...
}
```

## 📝 使用訓練好的模型

訓練完成後，可將模型整合回 backend：

```python
from ultralytics import YOLO

# 載入自訓練模型
model = YOLO('../../runs/train/findyou_yolov12m_1024/weights/best.pt')

# 進行推論
results = model.predict(image_path, conf=0.5)
```

## ⚠️ 注意事項

1. **GPU 需求**: 建議使用至少 16GB VRAM 的 GPU
2. **訓練時間**: RTX 4090 約需 2-3 小時完成 100 epochs
3. **記憶體管理**: 訓練時會自動使用 AMP (混合精度) 以節省記憶體
4. **資料快取**: 首次執行會創建標籤快取檔案，加快後續訓練

## 🐛 常見問題

### CUDA Out of Memory
```bash
# 解決方案：降低 batch size 或解析度
batch: 2
imgsz: 640
```

### 找不到資料集
```bash
# 確認資料集路徑正確
ls ../../datasets/findyou_yolo_clean/data.yaml
```

### ModuleNotFoundError: ultralytics
```bash
# 重新安裝 ultralytics
pip install ultralytics --upgrade
```

## 📚 相關資源

- [Ultralytics YOLOv12 文檔](https://docs.ultralytics.com/)
- [YOLO 格式標註說明](https://docs.ultralytics.com/datasets/detect/)
- [模型訓練最佳實踐](https://docs.ultralytics.com/modes/train/)

## 📧 支援

如有問題，請提交 Issue 或聯繫專案維護者。
