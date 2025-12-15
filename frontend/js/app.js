/**
 * 找東西助手 - 主程式
 * 前端攝影機架構：使用 getUserMedia 擷取畫面，傳送至後端推論
 */

class ObjectFinderApp {
    constructor() {
        this.db = window.objectFinderDB;
        this.api = window.objectFinderAPI;
        this.ui = window.objectFinderUI;
        this.isInitialized = false;
        this.isDetecting = false;
        
        // 攝影機相關
        this.videoStream = null;
        this.videoElement = null;
        this.canvasElement = null;
        this.canvasContext = null;
        
        // 自動偵測
        this.autoDetectInterval = null;
        this.autoDetectSeconds = 5;
    }

    async init() {
        try {
            // 初始化 UI
            this.ui.init();
            
            // 初始化 IndexedDB
            await this.db.init();
            
            // 綁定事件
            this.bindEvents();
            
            // 檢查後端連線
            await this.checkConnection();
            
            // 連接 WebSocket
            this.connectWebSocket();
            
            // 列舉可用攝影機
            await this.enumerateCameras();
            
            // 載入最近記錄
            await this.loadRecentDetections();
            
            // 載入自訂常用物品
            this.loadQuickItems();
            
            // 初始化 canvas
            this.videoElement = document.getElementById('cameraVideo');
            this.canvasElement = document.getElementById('previewCanvas');
            this.canvasContext = this.canvasElement.getContext('2d');
            
            this.isInitialized = true;
            console.log('✅ App 初始化完成');
            
        } catch (error) {
            console.error('初始化失敗:', error);
            this.ui.showToast('初始化失敗，請重新整理頁面', 'error');
        }
    }

    bindEvents() {
        // 搜尋按鈕
        this.ui.elements.searchBtn.addEventListener('click', () => this.handleSearch());
        
        // Enter 鍵搜尋
        this.ui.elements.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.handleSearch();
        });
        
        // 語音輸入
        this.ui.elements.voiceBtn.addEventListener('click', () => this.handleVoiceInput());
        
        // 快捷按鈕
        this.ui.elements.quickItemsGrid.querySelectorAll('.quick-item-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const item = btn.dataset.item;
                this.ui.setSearchValue(item);
                this.handleSearch();
            });
        });
        
        // 攝影機控制
        const startCameraBtn = document.getElementById('startCameraBtn');
        const detectBtn = document.getElementById('detectBtn');
        const autoDetectToggle = document.getElementById('autoDetectToggle');
        const intervalInput = document.getElementById('intervalInput');
        
        if (startCameraBtn) {
            startCameraBtn.addEventListener('click', () => this.toggleCamera());
        }
        
        if (detectBtn) {
            detectBtn.addEventListener('click', () => this.detectCurrentFrame());
        }
        
        if (autoDetectToggle) {
            autoDetectToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.startAutoDetection();
                } else {
                    this.stopAutoDetection();
                }
            });
        }
        
        if (intervalInput) {
            intervalInput.addEventListener('change', (e) => {
                this.autoDetectSeconds = Math.max(1, Math.min(60, parseInt(e.target.value) || 5));
                e.target.value = this.autoDetectSeconds;
                
                // 如果自動偵測中，重新啟動
                if (this.autoDetectInterval) {
                    this.stopAutoDetection();
                    this.startAutoDetection();
                }
            });
        }
        
        // 歷史記錄
        this.ui.elements.historyBtn.addEventListener('click', () => this.showHistory());
        
        // 清空資料
        const clearDataBtn = document.getElementById('clearDataBtn');
        if (clearDataBtn) {
            clearDataBtn.addEventListener('click', () => this.clearAllData());
        }
        
        // 設定
        this.ui.elements.settingsBtn.addEventListener('click', () => this.showSettings());
        
        // 最近偵測項目點擊
        this.ui.elements.recentList.addEventListener('click', (e) => {
            const item = e.target.closest('.recent-item');
            if (item) {
                this.showDetectionDetail(item);
            }
        });
    }

    // ========================================
    // 攝影機控制
    // ========================================

    async enumerateCameras() {
        try {
            // 先請求權限
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(track => track.stop());
            
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(d => d.kind === 'videoinput');
            
            const select = document.getElementById('cameraSelect');
            if (select) {
                select.innerHTML = cameras.map((cam, idx) => 
                    `<option value="${cam.deviceId}">${cam.label || `攝影機 ${idx}`}</option>`
                ).join('');
            }
            
            console.log(`📹 發現 ${cameras.length} 個攝影機`);
            
        } catch (error) {
            console.warn('無法列舉攝影機:', error);
        }
    }

    async toggleCamera() {
        const btn = document.getElementById('startCameraBtn');
        const offMessage = document.getElementById('cameraOffMessage');
        
        if (this.videoStream) {
            // 關閉攝影機
            this.stopCamera();
            btn.innerHTML = '<span class="btn-icon">📹</span><span class="btn-text">開啟攝影機</span>';
            offMessage.style.display = 'flex';
            this.videoElement.style.display = 'none';
            this.canvasElement.style.display = 'none';
        } else {
            // 開啟攝影機
            await this.startCamera();
            btn.innerHTML = '<span class="btn-icon">⏹️</span><span class="btn-text">關閉攝影機</span>';
            offMessage.style.display = 'none';
            this.videoElement.style.display = 'block';
        }
    }

    async startCamera() {
        try {
            const select = document.getElementById('cameraSelect');
            const deviceId = select?.value;
            
            const constraints = {
                video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'environment' },
                audio: false
            };
            
            this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.videoStream;
            
            // 等待 video 載入
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = resolve;
            });
            
            // 設定 canvas 尺寸
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;
            
            this.ui.showToast('攝影機已開啟', 'success');
            console.log('✅ 攝影機已開啟');
            
        } catch (error) {
            console.error('開啟攝影機失敗:', error);
            this.ui.showToast('無法開啟攝影機，請確認權限設定', 'error');
        }
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
            this.videoElement.srcObject = null;
        }
        
        // 停止自動偵測
        this.stopAutoDetection();
        document.getElementById('autoDetectToggle').checked = false;
        
        this.ui.showToast('攝影機已關閉', 'info');
    }

    captureFrame() {
        if (!this.videoStream || !this.videoElement.videoWidth) {
            return null;
        }
        
        // 將 video 畫到 canvas
        this.canvasContext.drawImage(
            this.videoElement,
            0, 0,
            this.canvasElement.width,
            this.canvasElement.height
        );
        
        // 轉成 Blob
        return new Promise((resolve) => {
            this.canvasElement.toBlob(resolve, 'image/jpeg', 0.9);
        });
    }

    // ========================================
    // 偵測功能
    // ========================================

    async detectCurrentFrame() {
        if (this.isDetecting) return;
        
        if (!this.videoStream) {
            this.ui.showToast('請先開啟攝影機', 'warning');
            return;
        }
        
        this.isDetecting = true;
        this.ui.showLoading('偵測中...');
        
        try {
            // 擷取畫面
            const blob = await this.captureFrame();
            if (!blob) {
                this.ui.showToast('無法擷取畫面', 'error');
                return;
            }
            
            // 傳送到後端偵測
            const result = await this.api.detectImage(blob);
            
            if (result && result.success) {
                // 顯示標註後的圖片
                if (result.image_base64) {
                    this.showAnnotatedImage(result.image_base64);
                }
                
                // 儲存偵測結果
                if (result.detections && result.detections.length > 0) {
                    // 去重
                    const deduped = this.deduplicateDetections(result.detections);
                    
                    // 確認有圖片資料
                    console.log('📷 image_base64 長度:', result.image_base64?.length || 0);
                    
                    for (const det of deduped) {
                        await this.db.saveDetection({
                            objectClass: det.object_class,
                            objectClassZh: det.object_class_zh || det.matched_object_name_zh,
                            confidence: det.similarity || det.confidence,
                            bbox: det.bbox,
                            surface: det.surface || '攝影機',
                            region: det.region || '',
                            timestamp: det.timestamp || Date.now(),
                            matchedObjectId: det.matched_object_id,
                            matchedObjectName: det.matched_object_name_zh,
                            imagePath: result.image_base64  // 儲存完整 base64 圖片
                        });
                    }
                    
                    this.ui.showToast(`偵測到 ${deduped.length} 個物品`, 'success');
                } else {
                    this.ui.showToast('未偵測到物品', 'info');
                }
                
                await this.loadRecentDetections();
            } else {
                this.ui.showToast('偵測失敗', 'error');
            }
            
        } catch (error) {
            console.error('偵測失敗:', error);
            this.ui.showToast('偵測失敗', 'error');
        } finally {
            this.ui.hideLoading();
            this.isDetecting = false;
        }
    }

    showAnnotatedImage(base64) {
        // 暫時顯示標註後的圖片在 canvas 上
        const img = new Image();
        img.onload = () => {
            this.canvasElement.style.display = 'block';
            this.canvasContext.drawImage(img, 0, 0, this.canvasElement.width, this.canvasElement.height);
            
            // 3 秒後恢復顯示 video
            setTimeout(() => {
                this.canvasElement.style.display = 'none';
            }, 3000);
        };
        img.src = base64;
    }

    deduplicateDetections(detections) {
        const deduped = {};
        for (const det of detections) {
            const key = det.matched_object_id || det.object_class;
            if (!deduped[key] || (det.similarity || det.confidence) > (deduped[key].similarity || deduped[key].confidence)) {
                deduped[key] = det;
            }
        }
        return Object.values(deduped);
    }

    // ========================================
    // 自動偵測
    // ========================================

    startAutoDetection() {
        if (!this.videoStream) {
            this.ui.showToast('請先開啟攝影機', 'warning');
            document.getElementById('autoDetectToggle').checked = false;
            return;
        }
        
        this.stopAutoDetection();
        
        this.autoDetectInterval = setInterval(() => {
            if (!this.isDetecting && this.videoStream) {
                this.detectCurrentFrame();
            }
        }, this.autoDetectSeconds * 1000);
        
        this.ui.showToast(`自動偵測已啟動 (${this.autoDetectSeconds}秒)`, 'success');
        console.log(`⏱️ 自動偵測已啟動，間隔 ${this.autoDetectSeconds} 秒`);
    }

    stopAutoDetection() {
        if (this.autoDetectInterval) {
            clearInterval(this.autoDetectInterval);
            this.autoDetectInterval = null;
            console.log('⏹️ 自動偵測已停止');
        }
    }

    // ========================================
    // 搜尋功能
    // ========================================

    async handleSearch() {
        const query = this.ui.getSearchValue();
        if (!query) {
            this.ui.showToast('請輸入要搜尋的物品', 'warning');
            return;
        }
        
        this.ui.showLoading('正在搜尋...');
        
        try {
            const result = await this.db.getLastLocation(query);
            
            if (result) {
                this.ui.showResult(result);
                this.ui.showToast(`找到 ${result.objectClassZh}！`, 'success');
            } else {
                this.ui.showNotFound(query);
                this.ui.showToast('找不到該物品', 'warning');
            }
        } catch (error) {
            console.error('搜尋失敗:', error);
            this.ui.showToast('搜尋時發生錯誤', 'error');
        } finally {
            this.ui.hideLoading();
        }
    }

    handleVoiceInput() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            this.ui.showToast('您的瀏覽器不支援語音輸入', 'error');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'zh-TW';
        recognition.continuous = false;
        
        recognition.onstart = () => {
            this.ui.showToast('請說出物品名稱...', 'info');
            this.ui.elements.voiceBtn.style.background = 'var(--success-gradient)';
        };
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            this.ui.setSearchValue(transcript);
            this.handleSearch();
        };
        
        recognition.onerror = () => {
            this.ui.showToast('語音辨識失敗', 'error');
        };
        
        recognition.onend = () => {
            this.ui.elements.voiceBtn.style.background = 'var(--secondary-gradient)';
        };
        
        recognition.start();
    }

    // ========================================
    // 其他功能
    // ========================================

    async checkConnection() {
        const health = await this.api.checkHealth();
        this.ui.updateStatus(!!health, health ? '已連線至偵測服務' : '離線模式');
    }

    async loadRecentDetections() {
        try {
            const detections = await this.db.getRecentDetections(5);
            this.ui.updateRecentList(detections);
        } catch (error) {
            console.error('載入記錄失敗:', error);
        }
    }

    loadQuickItems() {
        const DEFAULT_QUICK_ITEMS = [
            { name: '手機', icon: '📱', order: 1 },
            { name: '鑰匙', icon: '🔑', order: 2 },
            { name: '眼鏡', icon: '👓', order: 3 },
            { name: '錢包', icon: '👛', order: 4 }
        ];
        
        const saved = localStorage.getItem('quickItems');
        const items = saved ? JSON.parse(saved) : DEFAULT_QUICK_ITEMS;
        
        const container = this.ui.elements.quickItemsGrid;
        if (!container) return;
        
        container.innerHTML = items.map(item => `
            <button class="quick-item-btn" data-item="${item.name}">
                <span class="item-icon">${item.icon}</span>
                <span class="item-name">${item.name}</span>
            </button>
        `).join('');
        
        // 重新綁定事件
        container.querySelectorAll('.quick-item-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const itemName = btn.dataset.item;
                this.ui.setSearchValue(itemName);
                this.handleSearch();
            });
        });
    }

    async clearAllData() {
        if (!confirm('確定要清空所有偵測記錄嗎？此操作無法復原！')) {
            return;
        }
        
        try {
            await this.db.clearAll();
            await this.loadRecentDetections();
            this.ui.hideResult();
            this.ui.showToast('已清空所有偵測記錄', 'success');
        } catch (error) {
            console.error('清空資料失敗:', error);
            this.ui.showToast('清空失敗', 'error');
        }
    }

    async showHistory() {
        try {
            const allDetections = await this.db.getAllDetections(200);
            
            if (allDetections.length === 0) {
                this.ui.showToast('尚無歷史記錄', 'info');
                return;
            }
            
            // 按物品分類
            const grouped = {};
            for (const det of allDetections) {
                const key = det.matchedObjectId || det.objectClass;
                if (!grouped[key]) {
                    grouped[key] = {
                        objectClass: det.objectClass,
                        objectClassZh: det.objectClassZh || det.matchedObjectName,
                        records: []
                    };
                }
                grouped[key].records.push(det);
            }
            
            this.showHistoryModal(Object.values(grouped));
            
        } catch (error) {
            console.error('載入歷史記錄失敗:', error);
            this.ui.showToast('載入歷史記錄失敗', 'error');
        }
    }

    showHistoryModal(groupedData) {
        const existing = document.getElementById('historyModal');
        if (existing) existing.remove();
        
        const modal = document.createElement('div');
        modal.id = 'historyModal';
        modal.style.cssText = `
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9); z-index: 9999;
            display: flex; flex-direction: column;
            padding: 20px; overflow: hidden;
        `;
        
        const formatTime = (timestamp) => {
            const date = new Date(timestamp);
            return date.toLocaleString('zh-TW', { 
                month: 'short', day: 'numeric', 
                hour: '2-digit', minute: '2-digit' 
            });
        };
        
        modal.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <h2 style="color:#fff; margin:0;">📋 歷史記錄</h2>
                <button id="closeHistoryBtn" style="
                    background: rgba(255,255,255,0.1); border: none; color: #fff;
                    width: 40px; height: 40px; border-radius: 50%; font-size: 20px; cursor: pointer;
                ">✕</button>
            </div>
            <div style="flex:1; overflow-y:auto; padding-right:10px;">
                ${groupedData.map(group => `
                    <div style="margin-bottom:20px;">
                        <h3 style="color:#ffd700; margin-bottom:10px; font-size:16px;">
                            ${this.ui.getObjectIcon(group.objectClass)} ${group.objectClassZh}
                            <span style="color:#888; font-size:12px; margin-left:8px;">(${group.records.length} 筆)</span>
                        </h3>
                        <div style="display:flex; flex-direction:column; gap:8px;">
                            ${group.records.slice(0, 10).map(record => `
                                <div style="
                                    background: rgba(255,255,255,0.05); 
                                    padding: 12px 16px; border-radius: 8px;
                                    display: flex; justify-content: space-between; align-items: center;
                                ">
                                    <div>
                                        <div style="color:#fff;">${record.surfaceZh || record.surface || '攝影機'} ${record.regionZh || record.region || ''}</div>
                                        <div style="color:#888; font-size:12px;">${formatTime(record.timestamp)}</div>
                                    </div>
                                    <div style="color:#38ef7d; font-size:14px;">${Math.round(record.confidence * 100)}%</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        document.body.appendChild(modal);
        
        document.getElementById('closeHistoryBtn').addEventListener('click', () => modal.remove());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
    }

    showSettings() {
        window.location.href = '/settings';
    }

    showDetectionDetail(item) {
        // 從 recentDetections 陣列取得完整資料（包含 imagePath）
        const index = parseInt(item.dataset.index);
        const detection = this.ui.recentDetections?.[index];
        
        if (detection) {
            // 使用完整的偵測資料
            const result = {
                objectClassZh: detection.objectClassZh,
                objectClass: detection.objectClass,
                surfaceZh: detection.surfaceZh,
                regionZh: detection.regionZh,
                lastSeen: detection.timestamp,
                confidence: detection.confidence,
                description: `${detection.objectClassZh}在${detection.surfaceZh || ''}${detection.regionZh || ''}`,
                imagePath: detection.imagePath  // 包含完整 base64 圖片
            };
            this.ui.showResult(result);
        } else {
            // 後備：使用 data 屬性
            const result = {
                objectClassZh: item.dataset.classZh,
                objectClass: item.dataset.class,
                surfaceZh: item.dataset.surface,
                regionZh: item.dataset.region,
                lastSeen: parseInt(item.dataset.time),
                confidence: parseFloat(item.dataset.confidence),
                description: `${item.dataset.classZh}在${item.dataset.surface}${item.dataset.region || ''}`
            };
            this.ui.showResult(result);
        }
    }

    connectWebSocket() {
        // 根據頁面協議自動選擇 ws:// 或 wss://
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/detections`;
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket 連線成功');
            };
            
            this.ws.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'detection' && message.data && message.data.length > 0) {
                        console.log(`📡 收到偵測結果: ${message.data.length} 個物品`);
                        await this.loadRecentDetections();
                    }
                } catch (e) {
                    console.error('WebSocket 訊息處理錯誤:', e);
                }
            };
            
            this.ws.onclose = () => {
                console.log('⚠️ WebSocket 連線關閉，5秒後重試...');
                setTimeout(() => this.connectWebSocket(), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket 錯誤:', error);
            };
            
        } catch (error) {
            console.error('WebSocket 連線失敗:', error);
        }
    }
}

// 頁面載入後初始化
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ObjectFinderApp();
    window.app.init();
});
