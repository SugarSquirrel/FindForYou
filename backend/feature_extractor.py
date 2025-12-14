"""
DINOv2 特徵提取器模組
使用 Meta 的 DINOv2 模型提取視覺特徵
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List
import os


class FeatureExtractor:
    """DINOv2 特徵提取器"""
    
    def __init__(self, model_name: str = "dinov2_vits14", device: str = None):
        """
        初始化 DINOv2 模型
        
        Args:
            model_name: 模型名稱，選項：
                - dinov2_vits14 (384 維, 最輕量)
                - dinov2_vitb14 (768 維)
                - dinov2_vitl14 (1024 維)
                - dinov2_vitg14 (1536 維, 最大)
            device: 運算裝置 ('cuda' 或 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.is_ready = False
        
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            print(f"🔄 載入 DINOv2 模型: {self.model_name}...")
            
            # 從 torch hub 載入 DINOv2
            self.model = torch.hub.load(
                'facebookresearch/dinov2', 
                self.model_name,
                pretrained=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 設定圖片轉換
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self.is_ready = True
            print(f"✅ DINOv2 模型已載入: {self.model_name} (裝置: {self.device})")
            
        except Exception as e:
            print(f"❌ DINOv2 模型載入失敗: {e}")
            self.is_ready = False
    
    def extract_features(
        self, 
        image: Union[np.ndarray, Image.Image, str]
    ) -> np.ndarray:
        """
        提取圖片的特徵向量
        
        Args:
            image: 輸入圖片 (numpy array, PIL Image, 或檔案路徑)
            
        Returns:
            特徵向量 (numpy array)
        """
        if not self.is_ready:
            raise RuntimeError("DINOv2 模型未就緒")
        
        # 轉換為 PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]
            pil_image = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"不支援的圖片類型: {type(image)}")
        
        # 轉換並提取特徵
        with torch.no_grad():
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            features = self.model(img_tensor)
            
            # 正規化特徵
            features = F.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy().flatten()
    
    def extract_features_batch(
        self, 
        images: List[Union[np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """
        批次提取多張圖片的特徵
        
        Args:
            images: 圖片列表
            
        Returns:
            特徵矩陣 (N x feature_dim)
        """
        if not self.is_ready:
            raise RuntimeError("DINOv2 模型未就緒")
        
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]
                pil_img = Image.fromarray(img).convert('RGB')
            else:
                pil_img = img.convert('RGB')
            
            tensors.append(self.transform(pil_img))
        
        with torch.no_grad():
            batch = torch.stack(tensors).to(self.device)
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy()
    
    @staticmethod
    def cosine_similarity(
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        計算兩個特徵向量的餘弦相似度
        
        Args:
            embedding1: 第一個特徵向量
            embedding2: 第二個特徵向量
            
        Returns:
            相似度 (0~1)
        """
        # 確保是 1D 向量
        e1 = embedding1.flatten()
        e2 = embedding2.flatten()
        
        # 計算餘弦相似度
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    @staticmethod
    def find_best_match(
        query_embedding: np.ndarray,
        embeddings_list: List[np.ndarray],
        threshold: float = 0.7
    ) -> tuple:
        """
        在嵌入列表中找到最佳匹配
        
        Args:
            query_embedding: 查詢特徵向量
            embeddings_list: 候選特徵向量列表
            threshold: 相似度閾值
            
        Returns:
            (最佳匹配索引, 相似度) 或 (-1, 0.0) 若無匹配
        """
        if not embeddings_list:
            return (-1, 0.0)
        
        best_idx = -1
        best_sim = 0.0
        
        for idx, emb in enumerate(embeddings_list):
            sim = FeatureExtractor.cosine_similarity(query_embedding, emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        
        if best_sim >= threshold:
            return (best_idx, best_sim)
        else:
            return (-1, best_sim)
    
    def get_feature_dim(self) -> int:
        """取得特徵維度"""
        dims = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        return dims.get(self.model_name, 384)
