"""
找東西助手 - 後端 API 服務
FastAPI 提供偵測服務和 API 端點
使用 YOLO12 + DINOv2 個人化物件偵測
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np

from detector import ObjectDetector
from scheduler import DetectionScheduler


# ========================================
# 資料模型
# ========================================

class Detection(BaseModel):
    """單一偵測結果"""
    object_class: str
    object_class_zh: Optional[str] = None
    confidence: float
    bbox: List[float]
    matched_object_id: Optional[str] = None
    matched_object_name: Optional[str] = None
    matched_object_name_zh: Optional[str] = None
    similarity: Optional[float] = None
    surface: Optional[str] = None
    region: Optional[str] = None
    timestamp: Optional[int] = None


class DetectionResponse(BaseModel):
    """偵測回應"""
    success: bool
    detections: List[Detection]
    timestamp: int
    message: Optional[str] = None
    image_path: Optional[str] = None


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    version: str
    detector_ready: bool
    scheduler_running: bool
    registered_objects: int


class RegisterObjectRequest(BaseModel):
    """註冊物品請求"""
    name: str
    name_zh: str


# ========================================
# 全域變數
# ========================================

detector: Optional[ObjectDetector] = None
scheduler: Optional[DetectionScheduler] = None
connected_websockets: List[WebSocket] = []
latest_detections: List[Detection] = []


# ========================================
# 生命週期管理
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global detector, scheduler
    
    print("🚀 啟動找東西助手後端服務 (YOLO12 + DINOv2)...")
    
    # 初始化偵測器
    try:
        detector = ObjectDetector()
        print("✅ 物件偵測器已載入")
    except Exception as e:
        print(f"⚠️ 偵測器載入失敗: {e}")
        import traceback
        traceback.print_exc()
        detector = None
    
    # 初始化排程器
    scheduler = DetectionScheduler(
        detector=detector,
        on_detection=broadcast_detection,
        interval_seconds=30
    )
    
    yield
    
    # 清理資源
    print("🛑 關閉服務...")
    if scheduler:
        scheduler.stop()


# ========================================
# FastAPI 應用程式
# ========================================

app = FastAPI(
    title="FindForYou API",
    description="物品定位服務後端 API (YOLO12 + DINOv2)",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# API 端點
# ========================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    registered_count = 0
    if detector and detector.object_registry:
        registered_count = len(detector.object_registry.objects)
    
    return HealthResponse(
        status="ok",
        version="2.0.0",
        detector_ready=detector is not None and detector.is_ready,
        scheduler_running=scheduler is not None and scheduler.is_running,
        registered_objects=registered_count
    )


# ========================================
# 物品註冊 API (新增)
# ========================================

@app.get("/api/objects")
async def list_objects():
    """列出已註冊物品"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    return {
        "success": True,
        "objects": detector.get_registered_objects()
    }


@app.get("/api/objects/{obj_id}")
async def get_object(obj_id: str):
    """取得單一物品詳情"""
    if detector is None or detector.object_registry is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    obj = detector.object_registry.get(obj_id)
    if not obj:
        raise HTTPException(status_code=404, detail=f"物品 {obj_id} 不存在")
    
    # 取得圖片列表 (相對路徑)
    images = []
    for img_path in obj.images:
        if os.path.exists(img_path):
            images.append(f"/object_images/{os.path.basename(img_path)}")
    
    return {
        "success": True,
        "object": {
            "id": obj.id,
            "name": obj.name,
            "name_zh": obj.name_zh,
            "images": images,
            "embedding_count": len(obj.embeddings),
            "created_at": obj.created_at,
            "updated_at": obj.updated_at
        }
    }


@app.post("/api/objects/register")
async def register_object(
    name: str = Form(...),
    name_zh: str = Form(...),
    image: UploadFile = File(...)
):
    """註冊新物品"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    # 檢查檔案類型
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        # 讀取圖片
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        # 註冊物品
        result = detector.register_object(
            name=name,
            name_zh=name_zh,
            image=img
        )
        
        if result:
            return {
                "success": True,
                "message": f"已註冊物品: {name_zh}",
                "object": result
            }
        else:
            raise HTTPException(status_code=500, detail="註冊失敗")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/objects/{obj_id}/images")
async def add_object_image(
    obj_id: str,
    image: UploadFile = File(...)
):
    """為物品新增照片"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        result = detector.add_object_image(obj_id=obj_id, image=img)
        
        if result:
            return {
                "success": True,
                "message": f"已為物品新增照片",
                "object": result
            }
        else:
            raise HTTPException(status_code=404, detail=f"物品 {obj_id} 不存在")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/objects/{obj_id}")
async def update_object(
    obj_id: str,
    name: Optional[str] = Form(None),
    name_zh: Optional[str] = Form(None)
):
    """更新物品資訊"""
    if detector is None or detector.object_registry is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    result = detector.object_registry.update(
        obj_id=obj_id,
        name=name,
        name_zh=name_zh
    )
    
    if result:
        return {
            "success": True,
            "message": f"已更新物品: {result.name_zh}",
            "object": {
                "id": result.id,
                "name": result.name,
                "name_zh": result.name_zh
            }
        }
    else:
        raise HTTPException(status_code=404, detail=f"物品 {obj_id} 不存在")


@app.delete("/api/objects/{obj_id}")
async def delete_object(obj_id: str):
    """刪除物品"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    success = detector.delete_object(obj_id)
    
    if success:
        return {
            "success": True,
            "message": f"已刪除物品: {obj_id}"
        }
    else:
        raise HTTPException(status_code=404, detail=f"物品 {obj_id} 不存在")


# ========================================
# 攝影機管理 API
# ========================================

CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")


def load_camera_config():
    """載入攝影機配置"""
    if os.path.exists(CAMERA_CONFIG_PATH):
        with open(CAMERA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"cameras": {}, "default_camera": 0}


def save_camera_config(config):
    """儲存攝影機配置"""
    with open(CAMERA_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class CameraConfigRequest(BaseModel):
    """攝影機配置請求"""
    camera_id: str
    name: str
    location: str
    enabled: bool = True


@app.get("/api/cameras")
async def list_cameras():
    """列出可用的攝影機"""
    cameras = []
    config = load_camera_config()
    
    # 測試攝影機 0-5
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cam_config = config.get("cameras", {}).get(str(i), {})
                name = cam_config.get("name", f"攝影機 {i}")
                location = cam_config.get("location", "")
                
                cameras.append({
                    "id": i,
                    "name": name,
                    "location": location,
                    "display": f"{name} ({location})" if location else name
                })
            cap.release()
    
    return {
        "cameras": cameras,
        "current": detector.camera_source if detector else 0
    }


@app.get("/api/cameras/{camera_id}/preview")
async def camera_preview(camera_id: int):
    """取得攝影機預覽圖片"""
    import base64
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"攝影機 {camera_id} 無法開啟")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="無法擷取畫面")
    
    # 縮小圖片
    height, width = frame.shape[:2]
    scale = 640 / width
    new_size = (640, int(height * scale))
    frame = cv2.resize(frame, new_size)
    
    # 轉換為 base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "success": True,
        "camera_id": camera_id,
        "image": f"data:image/jpeg;base64,{img_base64}"
    }


@app.post("/api/cameras/{camera_id}")
async def set_camera(camera_id: int):
    """設定使用的攝影機"""
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail=f"攝影機 {camera_id} 無法開啟")
    cap.release()
    
    detector.camera_source = camera_id
    return {
        "success": True,
        "message": f"已切換到攝影機 {camera_id}",
        "current": camera_id
    }


@app.get("/api/cameras/config")
async def get_camera_config():
    """取得攝影機配置"""
    return load_camera_config()


@app.post("/api/cameras/config")
async def set_camera_config(request: CameraConfigRequest):
    """設定單一攝影機配置"""
    config = load_camera_config()
    
    config["cameras"][request.camera_id] = {
        "name": request.name,
        "location": request.location,
        "enabled": request.enabled
    }
    
    save_camera_config(config)
    
    return {
        "success": True,
        "message": f"攝影機 {request.camera_id} 配置已儲存",
        "config": config
    }


@app.delete("/api/cameras/config/{camera_id}")
async def delete_camera_config(camera_id: str):
    """刪除攝影機配置"""
    config = load_camera_config()
    
    if camera_id in config["cameras"]:
        del config["cameras"][camera_id]
        save_camera_config(config)
        return {"success": True, "message": f"攝影機 {camera_id} 配置已刪除"}
    
    return {"success": False, "message": f"找不到攝影機 {camera_id}"}


# ========================================
# 偵測 API
# ========================================

@app.post("/api/snapshot", response_model=DetectionResponse)
async def trigger_snapshot():
    """手動觸發快照偵測"""
    global latest_detections
    
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    try:
        raw_detections, image_path = await detector.detect_snapshot()
        
        # 取得當前攝影機的位置配置
        camera_config = load_camera_config()
        current_camera = str(detector.camera_source)
        camera_location = "unknown"
        
        if current_camera in camera_config.get("cameras", {}):
            camera_location = camera_config["cameras"][current_camera].get("location", "unknown")
        
        # 轉換為 Pydantic 模型
        detections = []
        for d in raw_detections:
            det = Detection(
                object_class=d.object_class,
                object_class_zh=d.object_class_zh,
                confidence=d.confidence,
                bbox=d.bbox,
                matched_object_id=d.matched_object_id,
                matched_object_name=d.matched_object_name,
                matched_object_name_zh=d.matched_object_name_zh,
                similarity=d.similarity,
                surface=camera_location,
                region=d.region,
                timestamp=d.timestamp
            )
            detections.append(det)
        
        latest_detections = detections
        
        # 廣播給所有連線的 WebSocket
        await broadcast_detection(detections, image_path)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=int(datetime.now().timestamp() * 1000),
            message=f"快照偵測完成，找到 {len(detections)} 個物品",
            image_path=image_path
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """上傳圖片進行偵測"""
    global latest_detections
    
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        raw_detections = detector._detect_frame(frame)
        
        detections = [
            Detection(
                object_class=d.object_class,
                object_class_zh=d.object_class_zh,
                confidence=d.confidence,
                bbox=d.bbox,
                matched_object_id=d.matched_object_id,
                matched_object_name=d.matched_object_name,
                matched_object_name_zh=d.matched_object_name_zh,
                similarity=d.similarity,
                surface=d.surface,
                region=d.region,
                timestamp=d.timestamp
            ) for d in raw_detections
        ]
        
        latest_detections = detections
        
        await broadcast_detection(detections)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=int(datetime.now().timestamp() * 1000),
            message=f"偵測完成，找到 {len(detections)} 個物品"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detections/latest", response_model=DetectionResponse)
async def get_latest_detections():
    """取得最新偵測結果"""
    return DetectionResponse(
        success=True,
        detections=latest_detections,
        timestamp=int(datetime.now().timestamp() * 1000)
    )


# ========================================
# 排程器 API
# ========================================

@app.post("/api/scheduler/start")
async def start_scheduler():
    """啟動定時偵測"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    scheduler.start()
    return {"success": True, "message": "定時偵測已啟動"}


@app.post("/api/scheduler/stop")
async def stop_scheduler():
    """停止定時偵測"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    scheduler.stop()
    return {"success": True, "message": "定時偵測已停止"}


@app.get("/api/scheduler/status")
async def scheduler_status():
    """取得排程器狀態"""
    if scheduler is None:
        return {"is_running": False, "interval_seconds": 0}
    
    return {
        "is_running": scheduler.is_running,
        "interval_seconds": scheduler.interval_seconds
    }


class IntervalRequest(BaseModel):
    """間隔設定請求"""
    interval: int


@app.post("/api/scheduler/interval")
async def set_scheduler_interval(request: IntervalRequest):
    """設定偵測間隔"""
    if scheduler is None:
        raise HTTPException(status_code=503, detail="排程器未初始化")
    
    if request.interval < 5 or request.interval > 300:
        raise HTTPException(status_code=400, detail="間隔必須在 5-300 秒之間")
    
    scheduler.set_interval(request.interval)
    return {
        "success": True, 
        "message": f"偵測間隔已設為 {request.interval} 秒",
        "interval": request.interval
    }


# ========================================
# WebSocket
# ========================================

@app.websocket("/ws/detections")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端點，用於即時推送偵測結果"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)


async def broadcast_detection(detections_input, image_path=None):
    """廣播偵測結果給所有連線的 WebSocket"""
    global latest_detections
    
    # 處理不同輸入格式
    if isinstance(detections_input, tuple):
        detections = detections_input[0] if detections_input[0] else []
        if len(detections_input) > 1 and detections_input[1]:
            image_path = detections_input[1]
    else:
        detections = detections_input if detections_input else []
    
    # 取得當前攝影機的位置配置
    camera_location = "unknown"
    if detector:
        camera_config = load_camera_config()
        current_camera = str(detector.camera_source)
        if current_camera in camera_config.get("cameras", {}):
            camera_location = camera_config["cameras"][current_camera].get("location", "unknown")
    
    # 轉換為可序列化格式
    def to_serializable(d):
        if hasattr(d, 'dict'):
            data = d.dict()
        elif hasattr(d, 'to_dict'):
            data = d.to_dict()
        elif hasattr(d, '__dataclass_fields__'):
            from dataclasses import asdict
            data = asdict(d)
        else:
            data = d if isinstance(d, dict) else {}
        
        # 設定位置
        if not data.get('surface') or data.get('surface') == 'unknown':
            data['surface'] = camera_location if camera_location != 'unknown' else '未知位置'
        
        if image_path:
            data['image_path'] = image_path
            
        return data
    
    message = json.dumps({
        "type": "detection",
        "data": [to_serializable(d) for d in detections],
        "timestamp": int(datetime.now().timestamp() * 1000)
    })
    
    for ws in connected_websockets.copy():
        try:
            await ws.send_text(message)
        except Exception:
            connected_websockets.remove(ws)


# ========================================
# 靜態檔案服務
# ========================================

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
css_path = os.path.join(frontend_path, "css")
js_path = os.path.join(frontend_path, "js")

# 分別掛載目錄
if os.path.exists(css_path):
    app.mount("/css", StaticFiles(directory=css_path), name="css")
if os.path.exists(js_path):
    app.mount("/js", StaticFiles(directory=js_path), name="js")

# 掛載截圖資料夾
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# 掛載物品圖片資料夾
object_images_path = os.path.join(os.path.dirname(__file__), "object_images")
os.makedirs(object_images_path, exist_ok=True)
app.mount("/object_images", StaticFiles(directory=object_images_path), name="object_images")


@app.get("/")
async def serve_frontend():
    """服務前端首頁"""
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/settings")
async def serve_settings():
    """服務設定頁面"""
    return FileResponse(os.path.join(frontend_path, "settings.html"))


# ========================================
# 主程式入口
# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
