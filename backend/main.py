"""
找東西助手 - 後端 API 服務
FastAPI 提供偵測服務和 API 端點
使用 YOLO12 + DINOv2 個人化物件偵測
架構：前端擷取圖片 → 後端推論
"""

import os
import json
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
    image_base64: Optional[str] = None  # 返回帶標註的圖片


class HealthResponse(BaseModel):
    """健康檢查回應"""
    status: str
    version: str
    detector_ready: bool
    registered_objects: int


class RegisterObjectRequest(BaseModel):
    """註冊物品請求"""
    name: str
    name_zh: str


# ========================================
# 全域變數
# ========================================

detector: Optional[ObjectDetector] = None
connected_websockets: List[WebSocket] = []


# ========================================
# 生命週期管理
# ========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用程式生命週期管理"""
    global detector
    
    print("🚀 啟動找東西助手後端服務 (YOLO12 + DINOv2)...")
    print("📡 架構：前端攝影機 → API 推論")
    
    # 初始化偵測器
    try:
        detector = ObjectDetector()
        print("✅ 物件偵測器已載入")
    except Exception as e:
        print(f"⚠️ 偵測器載入失敗: {e}")
        import traceback
        traceback.print_exc()
        detector = None
    
    yield
    
    # 清理資源
    print("🛑 關閉服務...")


# ========================================
# FastAPI 應用程式
# ========================================

app = FastAPI(
    title="FindForYou API",
    description="物品定位服務後端 API (YOLO12 + DINOv2) - 前端攝影機架構",
    version="2.1.0",
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
        version="2.1.0",
        detector_ready=detector is not None and detector.is_ready,
        registered_objects=registered_count
    )


# ========================================
# 物品註冊 API
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
    
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        result = detector.register_object(name=name, name_zh=name_zh, image=img)
        
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


class RegisterCroppedRequest(BaseModel):
    """從偵測結果註冊物品的請求"""
    image_base64: str  # 完整圖片的 base64
    bbox: List[float]  # [x1, y1, x2, y2]
    name: str
    name_zh: str


@app.post("/api/objects/register-cropped")
async def register_object_cropped(request: RegisterCroppedRequest):
    """
    從偵測結果中註冊物品
    接收完整圖片的 base64 和 bbox，裁切後進行註冊
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    try:
        import base64
        
        # 解析 base64 圖片
        image_data = request.image_base64
        if image_data.startswith("data:"):
            # 移除 data:image/jpeg;base64, 前綴
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        # 使用 bbox 裁切圖片
        x1, y1, x2, y2 = [int(v) for v in request.bbox]
        h, w = img.shape[:2]
        
        # 邊界檢查
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        cropped = img[y1:y2, x1:x2]
        
        if cropped.size == 0:
            raise HTTPException(status_code=400, detail="裁切區域無效")
        
        # 註冊物品（不再使用 YOLO 裁切，直接使用已裁切的圖片）
        result = detector.register_object_direct(
            name=request.name,
            name_zh=request.name_zh,
            image=cropped
        )
        
        if result:
            return {
                "success": True,
                "message": f"已註冊物品: {request.name_zh}",
                "object": result
            }
        else:
            raise HTTPException(status_code=500, detail="註冊失敗")
            
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/objects/{obj_id}/images")
async def add_object_image(obj_id: str, image: UploadFile = File(...)):
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
            return {"success": True, "message": f"已為物品新增照片", "object": result}
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
    
    result = detector.object_registry.update(obj_id=obj_id, name=name, name_zh=name_zh)
    
    if result:
        return {
            "success": True,
            "message": f"已更新物品: {result.name_zh}",
            "object": {"id": result.id, "name": result.name, "name_zh": result.name_zh}
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
        return {"success": True, "message": f"已刪除物品: {obj_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"物品 {obj_id} 不存在")


# ========================================
# 偵測 API (主要推論入口)
# ========================================

@app.post("/api/detect/image", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...)):
    """
    偵測圖片中的物品 (主要 API)
    前端擷取攝影機畫面後傳送至此 API 進行推論
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="偵測器未就緒")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="請上傳圖片檔案")
    
    try:
        import base64
        
        # 讀取圖片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="無法解析圖片")
        
        # 執行偵測
        raw_detections = detector.detect_frame(frame)
        
        # 在圖片上畫框
        annotated_frame = detector.annotate_frame(frame, raw_detections)
        
        # 轉換為 base64 返回
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 轉換為 Pydantic 模型
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
        
        # 廣播給 WebSocket 連線
        await broadcast_detection(detections)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            timestamp=int(datetime.now().timestamp() * 1000),
            message=f"偵測完成，找到 {len(detections)} 個物品",
            image_base64=f"data:image/jpeg;base64,{img_base64}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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


async def broadcast_detection(detections):
    """廣播偵測結果給所有連線的 WebSocket"""
    if not detections:
        return
    
    def to_serializable(d):
        if hasattr(d, 'dict'):
            return d.dict()
        elif hasattr(d, 'to_dict'):
            return d.to_dict()
        elif hasattr(d, '__dataclass_fields__'):
            from dataclasses import asdict
            return asdict(d)
        return d if isinstance(d, dict) else {}
    
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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
