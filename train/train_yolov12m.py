"""
YOLOv12m Training Script - Optimized for Transfer Learning
針對 findyou_yolo_clean 資料集進行 fine-tuning 優化

主要改進：
1. 確保正確載入 YOLOv12 預訓練權重
2. 調整學習率策略以適應 transfer learning
3. 優化資料增強以改善小物件檢測
4. 調整 loss 權重以平衡定位與分類
"""
from ultralytics import YOLO
from pathlib import Path
import torch


def check_pretrained_model():
    """檢查並下載正確的預訓練模型"""
    # YOLOv12 官方模型名稱列表（按優先順序）
    model_candidates = [
        "yolo12m.pt",       # 可能的替代命名
        "yolov12m.pt",      # YOLOv12 medium (官方命名)
        "yolov11m.pt",      # Fallback to YOLOv11
    ]
    
    for model_name in model_candidates:
        try:
            print(f"嘗試載入模型: {model_name}")
            model = YOLO(model_name)
            print(f"✓ 成功載入: {model_name}")
            
            # 驗證模型資訊
            print(f"  - 模型類型: {type(model.model).__name__}")
            print(f"  - 任務類型: {model.task}")
            
            return model, model_name
        except Exception as e:
            print(f"✗ 載入 {model_name} 失敗: {e}")
            continue
    
    raise RuntimeError("無法載入任何預訓練模型！請檢查 ultralytics 版本。")


def main():
    # ===== 路徑設定 =====
    script_dir = Path(__file__).resolve().parent
    findforyou_root = script_dir.parent
    repo_root = findforyou_root.parent
    runs_root = findforyou_root / "runs" / "train"
    runs_root.mkdir(parents=True, exist_ok=True)

    # 資料集路徑
    data_yaml = repo_root / "datasets" / "findyou_yolo_clean_cleaned_no_key" / "data.yaml"
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到資料集設定檔: {data_yaml}")
    
    print(f"資料集路徑: {data_yaml}")
    
    # ===== 檢查 GPU =====
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("警告: 未檢測到 GPU，將使用 CPU 訓練（速度會很慢）")
    
    # ===== 載入預訓練模型 =====
    print("\n" + "="*60)
    print("載入預訓練模型...")
    print("="*60)
    
    model, model_name = check_pretrained_model()
    
    # ===== 訓練參數 - Transfer Learning 優化配置 =====
    training_args = {
        "data": str(data_yaml),
        
        # ===== 基本訓練設定 =====
        "epochs": 150,               # 適中的訓練輪數
        "imgsz": 640,                # 維持 640（小物件建議用更高解析度，但需平衡記憶體）
        "batch": 16,                 # batch size
        "patience": 50,              # early stopping patience（給模型足夠時間學習）
        
        # ===== 裝置與輸出設定 =====
        "device": 0,
        "workers": 8,
        "project": str(runs_root),
        "name": "findyou_yolov12m_transfer_v2",
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "save_period": 10,           # 每 10 個 epoch 儲存一次 checkpoint
        
        # ===== 確保使用預訓練權重 =====
        "pretrained": True,          # 使用預訓練權重
        
        # ===== 學習率設定 - Transfer Learning 關鍵 =====
        # Fine-tuning 時應使用較低的學習率，避免破壞預訓練特徵
        "lr0": 0.001,                # 初始學習率 (降低 10 倍，從 0.01 -> 0.001)
        "lrf": 0.01,                 # 最終學習率為 lr0 * lrf = 0.00001
        "optimizer": "AdamW",        # AdamW 對 fine-tuning 效果較好
        "momentum": 0.937,           # Adam beta1
        "weight_decay": 0.0005,      # L2 正則化
        
        # ===== Warmup 設定 =====
        "warmup_epochs": 5.0,        # 增加 warmup 輪數（讓模型慢慢適應新資料）
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.01,      # 降低 warmup bias lr
        
        # ===== 學習率排程 =====
        "cos_lr": True,              # 使用 cosine annealing（更平滑的學習率衰減）
        
        # ===== Loss 權重調整 =====
        # 對於小物件檢測，需要適當調整各項 loss 的權重
        "box": 7.5,                  # box loss gain
        "cls": 1.0,                  # 分類 loss（提高以改善多類別區分）
        "dfl": 1.5,                  # distribution focal loss
        
        # ===== 資料增強 - 針對小物件優化 =====
        # Mosaic 對小物件可能有害，因為會縮小物件
        "mosaic": 0.8,               # 降低 mosaic 機率（預設 1.0）
        "mixup": 0.0,                # 關閉 mixup（對小物件效果不好）
        "close_mosaic": 20,          # 最後 20 個 epoch 關閉 mosaic
        "degrees": 5.0,              # 旋轉角度（降低以保持物件形狀）
        "translate": 0.1,            # 平移
        "scale": 0.3,                # 縮放範圍（降低以避免物件過小）
        "shear": 2.0,                # 剪切
        "perspective": 0.0,          # 透視變換（關閉）
        "flipud": 0.0,               # 上下翻轉（關閉，物件有方向性）
        "fliplr": 0.5,               # 左右翻轉
        "hsv_h": 0.015,              # HSV 色調增強
        "hsv_s": 0.5,                # HSV 飽和度增強
        "hsv_v": 0.3,                # HSV 亮度增強
        "copy_paste": 0.0,           # Copy-paste 增強（對小物件可能有害）
        
        # ===== 其他訓練設定 =====
        "rect": False,               # 矩形訓練
        "label_smoothing": 0.1,      # 標籤平滑（輕微正則化，改善泛化）
        "nbs": 64,                   # nominal batch size
        "amp": True,                 # 自動混合精度
        "cache": False,              # 不快取（節省記憶體）
        "val": True,                 # 訓練時進行驗證
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "dropout": 0.0,
        "freeze": 10,                # 凍結 Backbone (前 10 層) 以防止 Overfitting
    }
    
    # ===== 打印訓練配置 =====
    print("\n" + "="*60)
    print("訓練配置")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"資料集: {data_yaml}")
    print(f"Epochs: {training_args['epochs']}")
    print(f"Image Size: {training_args['imgsz']}x{training_args['imgsz']}")
    print(f"Batch Size: {training_args['batch']}")
    print(f"初始學習率: {training_args['lr0']}")
    print(f"最終學習率: {training_args['lr0'] * training_args['lrf']}")
    print(f"優化器: {training_args['optimizer']}")
    print(f"Cosine LR: {training_args['cos_lr']}")
    print(f"Mosaic: {training_args['mosaic']} (最後 {training_args['close_mosaic']} epochs 關閉)")
    print(f"Label Smoothing: {training_args['label_smoothing']}")
    print("="*60 + "\n")
    
    # ===== 開始訓練 =====
    try:
        results = model.train(**training_args)
        
        print("\n" + "="*60)
        print("✓ 訓練完成！")
        print("="*60)
        
        # ===== 驗證模型 =====
        print("\n驗證訓練後的模型...")
        metrics = model.val()
        
        print("\n" + "="*60)
        print("驗證結果")
        print("="*60)
        print(f"mAP50:    {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # 嘗試打印各類別的 AP
        if hasattr(metrics.box, 'ap_class_index') and hasattr(metrics.box, 'ap50'):
            print("\n各類別 AP50:")
            class_names = model.names
            for i, ap in enumerate(metrics.box.ap50):
                class_name = class_names.get(i, f"class_{i}")
                print(f"  {class_name}: {ap:.4f}")
        
        # 儲存路徑
        save_dir = runs_root / training_args['name']
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        
        print(f"\n模型儲存位置:")
        print(f"  Best: {best_path}")
        print(f"  Last: {last_path}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ 訓練過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise


def train_with_frozen_backbone():
    """
    替代方案：凍結 backbone 的兩階段訓練
    
    這種方法對於資料量較少的情況特別有效：
    1. 第一階段：凍結 backbone，只訓練 head
    2. 第二階段：解凍全部，用更低的學習率微調
    """
    script_dir = Path(__file__).resolve().parent
    findforyou_root = script_dir.parent
    repo_root = findforyou_root.parent
    runs_root = findforyou_root / "runs" / "train"
    data_yaml = repo_root / "datasets" / "findyou_yolo_clean" / "data.yaml"
    
    print("="*60)
    print("兩階段訓練模式")
    print("="*60)
    
    # 載入模型
    model, model_name = check_pretrained_model()
    
    # ===== 第一階段：凍結 backbone =====
    print("\n[階段 1/2] 凍結 Backbone，訓練 Detection Head")
    
    stage1_args = {
        "data": str(data_yaml),
        "epochs": 50,
        "imgsz": 640,
        "batch": 16,
        "patience": 20,
        "device": 0,
        "workers": 8,
        "project": str(runs_root),
        "name": "findyou_stage1_frozen",
        "exist_ok": True,
        "pretrained": True,
        "lr0": 0.01,                 # 較高學習率（因為只訓練 head）
        "lrf": 0.1,
        "optimizer": "AdamW",
        "freeze": 10,                # 凍結前 10 層（backbone）
        "mosaic": 1.0,
        "close_mosaic": 10,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "amp": True,
        "val": True,
    }
    
    results_stage1 = model.train(**stage1_args)
    
    # ===== 第二階段：解凍全部，微調 =====
    print("\n[階段 2/2] 解凍全部層，Fine-tuning")
    
    # 載入第一階段的最佳權重
    stage1_best = runs_root / "findyou_stage1_frozen" / "weights" / "best.pt"
    model = YOLO(str(stage1_best))
    
    stage2_args = {
        "data": str(data_yaml),
        "epochs": 100,
        "imgsz": 640,
        "batch": 16,
        "patience": 30,
        "device": 0,
        "workers": 8,
        "project": str(runs_root),
        "name": "findyou_stage2_finetune",
        "exist_ok": True,
        "lr0": 0.0001,               # 非常低的學習率
        "lrf": 0.01,
        "optimizer": "AdamW",
        "freeze": None,              # 不凍結
        "mosaic": 0.5,               # 降低 mosaic
        "close_mosaic": 20,
        "cos_lr": True,
        "warmup_epochs": 2.0,
        "amp": True,
        "val": True,
    }
    
    results_stage2 = model.train(**stage2_args)
    
    print("\n" + "="*60)
    print("✓ 兩階段訓練完成！")
    print("="*60)
    
    return results_stage2


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--two-stage":
        # 使用兩階段訓練
        train_with_frozen_backbone()
    else:
        # 使用標準訓練
        main()