#!/bin/bash
# 評估訓練集與驗證集以檢查 overfitting

RUN_DIR="/home/ryueee17/114-1/DLCV/FP/FindForYou/runs/train/lvis_yolov12n_freeze3"
WEIGHTS="${RUN_DIR}/weights/best.pt"
DATA="/home/ryueee17/114-1/DLCV/FP/datasets/lvis_custom_yolo/data.yaml"
IMGSZ=1024
BATCH=24

echo "======================================================================"
echo "🔍 Overfitting 檢測：比對訓練集 vs 驗證集表現"
echo "======================================================================"
echo "模型: ${WEIGHTS}"
echo "資料: ${DATA}"
echo "======================================================================"

# 1. 評估驗證集（參考基準）
echo ""
echo "📊 步驟 1/2: 評估驗證集..."
yolo val model="${WEIGHTS}" data="${DATA}" split=val imgsz=${IMGSZ} batch=${BATCH} device=0 \
    plots=False save_json=False > /tmp/val_result.txt 2>&1

# 提取驗證集指標
VAL_P=$(grep -oP 'all\s+\d+\s+\d+\s+\K[0-9.]+' /tmp/val_result.txt | head -1)
VAL_R=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+\K[0-9.]+' /tmp/val_result.txt | head -1)
VAL_MAP50=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+[0-9.]+\s+\K[0-9.]+' /tmp/val_result.txt | head -1)
VAL_MAP=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+\K[0-9.]+' /tmp/val_result.txt | head -1)

echo "✓ 驗證集完成"

# 2. 評估訓練集
echo ""
echo "📊 步驟 2/2: 評估訓練集..."
yolo val model="${WEIGHTS}" data="${DATA}" split=train imgsz=${IMGSZ} batch=${BATCH} device=0 \
    plots=False save_json=False > /tmp/train_result.txt 2>&1

# 提取訓練集指標
TRAIN_P=$(grep -oP 'all\s+\d+\s+\d+\s+\K[0-9.]+' /tmp/train_result.txt | head -1)
TRAIN_R=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+\K[0-9.]+' /tmp/train_result.txt | head -1)
TRAIN_MAP50=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+[0-9.]+\s+\K[0-9.]+' /tmp/train_result.txt | head -1)
TRAIN_MAP=$(grep -oP 'all\s+\d+\s+\d+\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+\K[0-9.]+' /tmp/train_result.txt | head -1)

echo "✓ 訓練集完成"

# 3. 計算差異並判斷
echo ""
echo "======================================================================"
echo "📈 Overfitting 分析結果"
echo "======================================================================"
printf "%-15s %12s %12s %12s\n" "指標" "訓練集" "驗證集" "狀態"
echo "----------------------------------------------------------------------"

# 使用 awk 計算百分比差異
check_overfitting() {
    local name=$1
    local train=$2
    local val=$3
    
    if [ -z "$train" ] || [ -z "$val" ]; then
        printf "%-15s %12s %12s %12s\n" "$name" "N/A" "N/A" "⚠️  資料不足"
        return
    fi
    
    local diff=$(awk "BEGIN {printf \"%.4f\", $train - $val}")
    local pct=$(awk "BEGIN {if($val>0) printf \"%.1f\", ($train-$val)/$val*100; else print \"N/A\"}")
    
    local status
    if awk "BEGIN {exit !($pct > 20)}"; then
        status="⚠️  嚴重過擬合"
    elif awk "BEGIN {exit !($pct > 10)}"; then
        status="⚠️  中度過擬合"
    elif awk "BEGIN {exit !($pct > 5)}"; then
        status="⚠️  輕微過擬合"
    else
        status="✅ 正常"
    fi
    
    printf "%-15s %12.4f %12.4f %12s\n" "$name" "$train" "$val" "$status (+${pct}%)"
}

check_overfitting "Precision" "$TRAIN_P" "$VAL_P"
check_overfitting "Recall" "$TRAIN_R" "$VAL_R"
check_overfitting "mAP50" "$TRAIN_MAP50" "$VAL_MAP50"
check_overfitting "mAP50-95" "$TRAIN_MAP" "$VAL_MAP"

echo "======================================================================"
echo ""
echo "💡 判讀建議:"
echo "  - 差距 < 5%  : 正常，泛化良好"
echo "  - 差距 5-10% : 輕微過擬合，可接受"
echo "  - 差距 10-20%: 中度過擬合，建議增加正則化"
echo "  - 差距 > 20% : 嚴重過擬合，需檢討訓練策略"
echo ""
echo "建議對策（若過擬合）："
echo "  1. 增加 dropout (目前=0.0) 或 weight_decay"
echo "  2. 增強資料增強 (mosaic, mixup, cutout)"
echo "  3. Early stopping (降低 patience)"
echo "  4. 收集更多訓練資料"
echo "======================================================================"
