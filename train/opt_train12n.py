# optimize_training_v1.py
"""
優化版訓練腳本 - 主要修正 Freeze 策略
"""
from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    data_yaml = repo_root / "datasets" / "lvis_custom_yolo" / "data.yaml"
    runs_root = repo_root / "FindForYou" / "runs" / "train"
    
    model = YOLO("yolo12l.pt")
    
    training_args = {
        "data": str(data_yaml),
        "project": str(runs_root),
        "name": "lvis_yolov12n_freeze20_anti_overfit",
        
        # === 核心配置 - 極致防止 overfitting ===
        "freeze": 10,            # 🔥 只訓練最後 2 層 (最小化可訓練參數)
        "batch": 32,             # 🔧 降低 batch (更多梯度更新，更好泛化)
        "imgsz": 640,            # 🔧 降低解析度 (減少過擬合，加快訓練)
        
        # === 訓練策略 - 充分訓練但防止過擬合 ===
        "epochs": 150,           # 增加 epochs (少量參數需要更多時間)
        "patience": 50,          # 增加 patience (給模型更多機會找到最佳點)
        
        # === 優化器 - 保守策略 ===
        "lr0": 0.005,            # 🔧 降低學習率 (更穩定，防止震盪)
        "lrf": 0.001,            # 🔧 更低的最終 lr (細緻調整)
        "warmup_epochs": 5.0,    # 🔧 延長 warmup (更穩定的開始)
        "optimizer": "AdamW",    # 🔧 AdamW 有更好的正則化
        "weight_decay": 0.001,   # 🔧 增加 L2 正則化 (防止權重過大)
        "momentum": 0.937,
        
        # === 資料增強 - 適度增強提升泛化 ===
        "mosaic": 0.0,           # 關閉 (freeze 多層時效果不佳)
        "mixup": 0.0,            # 關閉 (同上)
        "copy_paste": 0.0,       # 關閉 (同上)
        
        # 🔧 幾何增強 - 適度增加
        "degrees": 10.0,         # 旋轉 ±10°
        "translate": 0.1,        # 平移 10%
        "scale": 0.3,            # 縮放 ±30%
        "shear": 2.0,            # 剪切 ±2°
        "perspective": 0.0001,   # 輕微透視
        "fliplr": 0.5,           # 水平翻轉
        "flipud": 0.0,           # 不垂直翻轉
        
        # 🔧 顏色增強 - 適度增加
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        
        # === Loss - 平衡配置 ===
        "box": 7.5,              # 🔧 提高 box loss (更重視定位)
        "cls": 0.5,              # 🔧 降低 cls loss (避免過度自信)
        "dfl": 1.5,
        
        # === Dropout (額外正則化) ===
        "dropout": 0.1,          # 🔥 啟用 dropout (如果模型支援)
        
        # === 硬體 ===
        "device": 0,
        "workers": 8,            # 🔧 降低 workers (更穩定)
        "cache": True,          # 🔧 關閉快取 (避免 OOM)
        "amp": True,
        
        # === 其他 ===
        "exist_ok": True,
        "save": True,
        "save_period": 10,       # 🔧 每 10 epochs 保存一次
        "val": True,
        "plots": True,
        "close_mosaic": 0,
    }
    
    print("\n" + "="*80)
    print("🛡️  極致防 Overfitting 訓練策略 (YOLOv12n)")
    print("="*80)
    print(f"📊 數據集: LVIS Custom (~8K train, ~2K val)")
    print(f"🎯 目標: 最大化泛化能力，最小化過擬合風險")
    print("")
    print(f"🔒 模型凍結策略:")
    print(f"  ├─ Freeze Layers: {training_args['freeze']}/22 (只訓練最後 2 層)")
    print(f"  ├─ 可訓練參數: ~5-10% (極少量參數)")
    print(f"  └─ 效果: 強制模型使用預訓練特徵，降低過擬合風險")
    print("")
    print(f"⚙️  訓練配置:")
    print(f"  ├─ Image Size: {training_args['imgsz']} (降低複雜度)")
    print(f"  ├─ Batch Size: {training_args['batch']} (更多梯度更新)")
    print(f"  ├─ Epochs: {training_args['epochs']} (充分訓練)")
    print(f"  ├─ Patience: {training_args['patience']} (避免過早停止)")
    print(f"  └─ Optimizer: {training_args['optimizer']} (內建正則化)")
    print("")
    print(f"📈 正則化技術:")
    print(f"  ├─ Weight Decay: {training_args['weight_decay']} (L2 正則化)")
    print(f"  ├─ Learning Rate: {training_args['lr0']} → {training_args['lrf']} (保守策略)")
    print(f"  ├─ Data Augmentation: 適度幾何+顏色增強")
    print(f"  └─ Mosaic/Mixup: 關閉 (freeze 多層時不適用)")
    print("="*80 + "\n")
    
    model.train(**training_args)
    
    print("\n✅ 訓練完成！")

if __name__ == "__main__":
    main()