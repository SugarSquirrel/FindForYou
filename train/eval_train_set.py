"""
評估訓練集以檢查 Overfitting
對比訓練集與驗證集的表現差距
"""
from ultralytics import YOLO
from pathlib import Path
import yaml

def main():
    # 讀取訓練配置取得資料集路徑
    run_dir = Path("../runs/train/lvis_yolov12l_freeze10")
    args_yaml = run_dir / "args.yaml"
    
    with open(args_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    data_yaml_path = config['data']
    best_weights = run_dir / "weights" / "best.pt"
    
    print("="*70)
    print("🔍 Overfitting 檢測：評估訓練集表現")
    print("="*70)
    print(f"模型權重: {best_weights}")
    print(f"資料集配置: {data_yaml_path}")
    print(f"解析度: {config['imgsz']}")
    print(f"Freeze layers: {config.get('freeze', 'None')}")
    print("="*70 + "\n")
    
    # 載入訓練好的模型
    model = YOLO(str(best_weights))
    
    # 1. 先對驗證集評估（參考基準）
    print("\n📊 驗證集評估（參考基準）...")
    val_metrics = model.val(
        data=data_yaml_path,
        split='val',
        imgsz=config['imgsz'],
        batch=config.get('batch', 16),
        device=config.get('device', 0),
        plots=False,
        save_json=False,
        verbose=False
    )
    
    # 2. 對訓練集評估（檢查 overfitting）
    print("\n📊 訓練集評估（檢查過擬合）...")
    train_metrics = model.val(
        data=data_yaml_path,
        split='train',  # 關鍵：改成訓練集
        imgsz=config['imgsz'],
        batch=config.get('batch', 16),
        device=config.get('device', 0),
        plots=False,
        save_json=False,
        verbose=False
    )
    
    # 3. 對比結果
    print("\n" + "="*70)
    print("📈 Overfitting 分析結果")
    print("="*70)
    
    metrics_names = [
        ('Precision', 'metrics/precision(B)'),
        ('Recall', 'metrics/recall(B)'),
        ('mAP50', 'metrics/mAP50(B)'),
        ('mAP50-95', 'metrics/mAP50-95(B)')
    ]
    
    print(f"{'指標':<15} {'訓練集':>12} {'驗證集':>12} {'差距':>12} {'狀態'}")
    print("-" * 70)
    
    for name, key in metrics_names:
        # 從 metrics 物件取值（Ultralytics 回傳的格式）
        train_val = getattr(train_metrics.box, key.split('/')[-1].replace('(B)', ''), 0)
        val_val = getattr(val_metrics.box, key.split('/')[-1].replace('(B)', ''), 0)
        
        diff = train_val - val_val
        diff_pct = (diff / val_val * 100) if val_val > 0 else 0
        
        # 判斷 overfitting 程度
        if diff_pct > 20:
            status = "⚠️  嚴重過擬合"
        elif diff_pct > 10:
            status = "⚠️  中度過擬合"
        elif diff_pct > 5:
            status = "⚠️  輕微過擬合"
        else:
            status = "✅ 正常"
        
        print(f"{name:<15} {train_val:>12.4f} {val_val:>12.4f} {diff:>+12.4f} ({diff_pct:+.1f}%)  {status}")
    
    print("="*70)
    print("\n💡 判讀建議:")
    print("  - 差距 < 5%  : 正常，泛化良好")
    print("  - 差距 5-10% : 輕微過擬合，可接受")
    print("  - 差距 10-20%: 中度過擬合，建議增加正則化或資料增強")
    print("  - 差距 > 20% : 嚴重過擬合，需檢討訓練策略")
    print("\n建議對策（若過擬合）：")
    print("  1. 增加 dropout / weight_decay")
    print("  2. 增強資料增強（mosaic, mixup, augment）")
    print("  3. Early stopping (降低 epochs 或 patience)")
    print("  4. 增加訓練資料量")
    print("="*70)

if __name__ == "__main__":
    main()
