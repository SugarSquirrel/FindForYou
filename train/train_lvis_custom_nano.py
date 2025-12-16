"""
YOLOv12n Training Script - Configurable Image Size
支援 640x640 或 1024x1024 訓練模式

Usage:
    python train_lvis_custom_nano.py --img 640
    python train_lvis_custom_nano.py --img 1024
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import torch
import sys

def check_gpu():
    if not torch.cuda.is_available():
        print("❌ 錯誤: 未檢測到 GPU！")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ 檢測到 GPU: {gpu_name}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv12n Training Script')
    parser.add_argument('--img', type=int, default=640, choices=[640, 1024], help='Image size (640 or 1024)')
    parser.add_argument('--batch', type=int, default=-1, help='Batch size (-1 for auto/default)')
    args = parser.parse_args()

    # ===== 1. 設定參數 =====
    img_size = args.img
    
    # 根據解析度設定預設 Batch Size (針對 RTX 4090)
    if args.batch == -1:
        if img_size == 640:
            batch_size = 128  # Nano 640px 可以開很大
        else:
            batch_size = 64   # Nano 1024px
    else:
        batch_size = args.batch

    print("\n" + "="*60)
    print(f"🚀 啟動 YOLOv12n 訓練")
    print(f"   - Image Size: {img_size}x{img_size}")
    print(f"   - Batch Size: {batch_size}")
    print("="*60)

    # ===== 2. 路徑設定 =====
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    
    data_yaml = repo_root / "datasets" / "lvis_custom_yolo" / "data.yaml"
    runs_root = repo_root / "FindForYou" / "runs" / "train"
    
    if not data_yaml.exists():
        print(f"❌ 找不到資料集: {data_yaml}")
        sys.exit(1)
        
    check_gpu()

    # ===== 3. 載入模型 =====
    model_name = "yolov12n.pt"
    try:
        model = YOLO(model_name)
    except:
        print(f"⚠️ 載入 {model_name} 失敗，嘗試 yolo12n.pt...")
        model = YOLO("yolo12n.pt")

    # ===== 4. 訓練參數 =====
    project_name = f"lvis_custom_yolov12n_{img_size}"
    
    training_args = {
        "data": str(data_yaml),
        "project": str(runs_root),
        "name": project_name,
        
        # 核心參數
        "epochs": 150,
        "imgsz": img_size,
        "batch": batch_size,
        "patience": 40,
        
        # 優化參數 (RTX 4090)
        "device": 0,
        "workers": 16,
        "cache": True,
        "amp": True,
        
        # Transfer Learning
        "pretrained": True,
        "freeze": 10,           # 凍結 Backbone
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "warmup_epochs": 5.0,
        
        # Augmentation
        # 1024px 時稍微降低 Mosaic 以保留小物件細節
        "mosaic": 1.0 if img_size == 640 else 0.8,
        "close_mosaic": 10,
        
        "exist_ok": True,
        "save": True,
        "val": True,
    }

    # ===== 5. 開始訓練 =====
    model.train(**training_args)
    
    print("\n✅ 訓練完成！")
    print(f"最佳權重位置: {runs_root}/{project_name}/weights/best.pt")

if __name__ == "__main__":
    main()
