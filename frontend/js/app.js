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
        
        // 多攝影機支援
        this.multiCameraStreams = {};  // { deviceId: { stream, video, canvas } }
        this.availableCameras = [];
        this.currentCameraDeviceId = null;
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
                const multiCameraSelect = document.getElementById('multiCameraSelect');
                if (e.target.checked) {
                    if (multiCameraSelect) multiCameraSelect.style.display = 'block';
                    this.startAutoDetection();
                } else {
                    if (multiCameraSelect) multiCameraSelect.style.display = 'none';
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
            
            // 儲存攝影機列表
            this.availableCameras = cameras;
            
            // 取得攝影機設定
            const cameraSettings = this.getCameraSettings();
            
            const select = document.getElementById('cameraSelect');
            if (select) {
                select.innerHTML = cameras.map((cam, idx) => {
                    const selected = cameraSettings.defaultCamera === cam.deviceId ? 'selected' : '';
                    const location = cameraSettings.locations[cam.deviceId];
                    const label = location ? `${cam.label || `攝影機 ${idx}`} (${location})` : (cam.label || `攝影機 ${idx}`);
                    return `<option value="${cam.deviceId}" ${selected}>${label}</option>`;
                }).join('');
                
                // 儲存當前選中的攝影機
                this.currentCameraDeviceId = select.value;
                
                // 監聽切換
                select.addEventListener('change', () => {
                    this.currentCameraDeviceId = select.value;
                });
            }
            
            // 填充多攝影機選擇 checkbox 列表
            const checkboxList = document.getElementById('cameraCheckboxList');
            if (checkboxList) {
                checkboxList.innerHTML = cameras.map((cam, idx) => {
                    const location = cameraSettings.locations[cam.deviceId];
                    const label = location || cam.label || `攝影機 ${idx + 1}`;
                    return `
                        <label style="display:flex; align-items:center; gap:8px; background:rgba(255,255,255,0.1); padding:8px 12px; border-radius:8px; cursor:pointer;">
                            <input type="checkbox" class="camera-checkbox" value="${cam.deviceId}" checked 
                                style="width:18px; height:18px; accent-color:#667eea;">
                            <span style="color:#fff; font-size:13px;">${label}</span>
                        </label>
                    `;
                }).join('');
            }
            
            console.log(`📹 發現 ${cameras.length} 個攝影機`);
            
        } catch (error) {
            console.warn('無法列舉攝影機:', error);
        }
    }
    
    getCameraSettings() {
        try {
            const saved = localStorage.getItem('cameraSettings');
            return saved ? JSON.parse(saved) : { defaultCamera: '', locations: {} };
        } catch {
            return { defaultCamera: '', locations: {} };
        }
    }
    
    getCameraLocation() {
        const settings = this.getCameraSettings();
        const deviceId = this.currentCameraDeviceId || '';
        return settings.locations[deviceId] || '攝影機';
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
                        const cameraLocation = this.getCameraLocation();
                        console.log(`📍 位置: ${cameraLocation}, 攝影機ID: ${this.currentCameraDeviceId}`);
                        
                        await this.db.saveDetection({
                            objectClass: det.object_class,
                            objectClassZh: det.object_class_zh || det.matched_object_name_zh,
                            confidence: det.similarity || det.confidence,
                            bbox: det.bbox,
                            surface: cameraLocation,
                            region: det.region || '',
                            timestamp: det.timestamp || Date.now(),
                            matchedObjectId: det.matched_object_id,
                            matchedObjectName: det.matched_object_name_zh,
                            imagePath: result.image_base64,  // 帶標註的圖片 (顯示用)
                            imageOriginal: result.image_original_base64  // 原始圖片 (註冊用)
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
    
    // 多攝影機輪流偵測索引
    currentCameraIndex = 0;
    
    getSelectedCameras() {
        const checkboxes = document.querySelectorAll('.camera-checkbox:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }

    async startAutoDetection() {
        const selectedCameras = this.getSelectedCameras();
        
        if (selectedCameras.length === 0) {
            this.ui.showToast('請至少選擇一個攝影機', 'warning');
            document.getElementById('autoDetectToggle').checked = false;
            return;
        }
        
        this.stopAutoDetection();
        
        // 同時開啟所有選中的攝影機
        await this.openMultipleCameras(selectedCameras);
        
        // 設定定時器：每次同時擷取所有攝影機
        const runSimultaneousDetection = async () => {
            if (!this.autoDetectInterval) return;
            if (this.isDetecting) return;
            
            this.isDetecting = true;
            
            try {
                await this.detectAllCameras();
            } catch (error) {
                console.error('多攝影機偵測失敗:', error);
            } finally {
                this.isDetecting = false;
            }
        };
        
        // 立即執行一次
        await runSimultaneousDetection();
        
        // 設定定時器
        this.autoDetectInterval = setInterval(runSimultaneousDetection, this.autoDetectSeconds * 1000);
        
        this.ui.showToast(`同時偵測 ${selectedCameras.length} 個攝影機 (${this.autoDetectSeconds}秒)`, 'success');
        console.log(`⏱️ 同時多攝影機偵測已啟動，${selectedCameras.length} 個攝影機，間隔 ${this.autoDetectSeconds} 秒`);
    }
    
    async openMultipleCameras(deviceIds) {
        // 關閉現有的多攝影機串流
        this.closeMultipleCameras();
        
        // 隱藏多攝影機選擇區的 checkbox 部分，改顯示預覽格
        const checkboxList = document.getElementById('cameraCheckboxList');
        const container = document.getElementById('multiCameraSelect');
        
        // 建立預覽格容器
        let previewGrid = document.getElementById('multiCameraPreviewGrid');
        if (!previewGrid) {
            previewGrid = document.createElement('div');
            previewGrid.id = 'multiCameraPreviewGrid';
            previewGrid.style.cssText = 'display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:10px; margin-top:15px;';
            container.appendChild(previewGrid);
        }
        
        previewGrid.innerHTML = '';
        
        const cameraSettings = this.getCameraSettings();
        
        for (const deviceId of deviceIds) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: deviceId } },
                    audio: false
                });
                
                // 建立 video 元素
                const video = document.createElement('video');
                video.autoplay = true;
                video.playsInline = true;
                video.muted = true;
                video.srcObject = stream;
                video.style.cssText = 'width:100%; border-radius:8px; background:#000;';
                
                // 建立 canvas 用於擷取
                const canvas = document.createElement('canvas');
                
                // 取得位置名稱
                const location = cameraSettings.locations[deviceId] || '攝影機';
                
                // 建立預覽卡片
                const previewCard = document.createElement('div');
                previewCard.style.cssText = 'background:rgba(0,0,0,0.3); border-radius:12px; padding:8px; text-align:center;';
                previewCard.innerHTML = `<div style="margin-bottom:5px; color:#aaa; font-size:12px;">📹 ${location}</div>`;
                previewCard.appendChild(video);
                previewGrid.appendChild(previewCard);
                
                // 儲存串流資訊
                this.multiCameraStreams[deviceId] = { stream, video, canvas, location };
                
                console.log(`📹 已開啟攝影機: ${location}`);
                
            } catch (error) {
                console.error(`無法開啟攝影機 ${deviceId}:`, error);
            }
        }
        
        console.log(`📹 已開啟 ${Object.keys(this.multiCameraStreams).length} 個攝影機`);
    }
    
    closeMultipleCameras() {
        for (const [deviceId, cam] of Object.entries(this.multiCameraStreams)) {
            if (cam.stream) {
                cam.stream.getTracks().forEach(track => track.stop());
            }
        }
        this.multiCameraStreams = {};
        
        // 清除預覽格
        const previewGrid = document.getElementById('multiCameraPreviewGrid');
        if (previewGrid) previewGrid.innerHTML = '';
    }
    
    async detectAllCameras() {
        const cameras = Object.entries(this.multiCameraStreams);
        if (cameras.length === 0) return;
        
        // 同時擷取所有攝影機畫面並發送偵測
        const detectionPromises = cameras.map(async ([deviceId, cam]) => {
            try {
                // 等待 video 準備好
                if (cam.video.readyState < 2) {
                    await new Promise(resolve => {
                        cam.video.onloadeddata = resolve;
                    });
                }
                
                // 設定 canvas 尺寸
                cam.canvas.width = cam.video.videoWidth;
                cam.canvas.height = cam.video.videoHeight;
                
                // 擷取畫面
                const ctx = cam.canvas.getContext('2d');
                ctx.drawImage(cam.video, 0, 0);
                
                // 轉換為 blob
                const blob = await new Promise(resolve => {
                    cam.canvas.toBlob(resolve, 'image/jpeg', 0.9);
                });
                
                if (!blob) return;
                
                // 發送到後端偵測
                const result = await this.api.detectImage(blob);
                
                if (result && result.success && result.detections && result.detections.length > 0) {
                    const deduped = this.deduplicateDetections(result.detections);
                    
                    console.log(`📍 ${cam.location}: 偵測到 ${deduped.length} 個物品`);
                    
                    for (const det of deduped) {
                        await this.db.saveDetection({
                            objectClass: det.object_class,
                            objectClassZh: det.object_class_zh || det.matched_object_name_zh,
                            confidence: det.similarity || det.confidence,
                            bbox: det.bbox,
                            surface: cam.location,  // 使用此攝影機的位置
                            region: det.region || '',
                            timestamp: det.timestamp || Date.now(),
                            matchedObjectId: det.matched_object_id,
                            matchedObjectName: det.matched_object_name_zh,
                            imagePath: result.image_base64,
                            imageOriginal: result.image_original_base64
                        });
                    }
                }
                
            } catch (error) {
                console.error(`攝影機 ${cam.location} 偵測失敗:`, error);
            }
        });
        
        // 等待所有偵測完成
        await Promise.all(detectionPromises);
        
        // 更新最近偵測列表
        await this.loadRecentDetections();
    }
    
    async switchToCamera(deviceId) {
        // 儲存當前攝影機 ID
        this.currentCameraDeviceId = deviceId;
        
        // 更新下拉選單顯示
        const select = document.getElementById('cameraSelect');
        if (select) select.value = deviceId;
        
        // 如果攝影機已開啟，切換到新攝影機
        if (this.videoStream) {
            // 停止舊的串流
            this.videoStream.getTracks().forEach(track => track.stop());
            
            // 開啟新攝影機
            const constraints = {
                video: { deviceId: { exact: deviceId } },
                audio: false
            };
            
            this.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
            this.videoElement.srcObject = this.videoStream;
            
            // 等待 video 載入
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = resolve;
            });
            
            // 更新 canvas 尺寸
            this.canvasElement.width = this.videoElement.videoWidth;
            this.canvasElement.height = this.videoElement.videoHeight;
        } else {
            // 如果攝影機未開啟，開啟它
            await this.toggleCamera();
        }
    }

    stopAutoDetection() {
        if (this.autoDetectInterval) {
            clearInterval(this.autoDetectInterval);
            this.autoDetectInterval = null;
            console.log('⏹️ 自動偵測已停止');
        }
        
        // 關閉多攝影機
        this.closeMultipleCameras();
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

    // ========================================
    // 從偵測結果註冊物品
    // ========================================

    showRegisterModal(detectionIndex) {
        const detection = this.ui.recentDetections?.[detectionIndex];
        if (!detection) {
            this.ui.showToast('找不到偵測資料', 'error');
            return;
        }

        // 儲存當前要註冊的偵測
        this.pendingRegistration = {
            detection: detection,
            imageBase64: detection.imageOriginal || detection.imagePath,  // 優先使用原始圖片
            bbox: detection.bbox
        };

        // 顯示 modal
        const modal = document.getElementById('registerModal');
        const cropImage = document.getElementById('registerCropImage');
        const nameEn = document.getElementById('registerNameEn');
        const nameZh = document.getElementById('registerNameZh');

        // 顯示帶標註的圖片 (預覽用)
        if (detection.imagePath) {
            cropImage.src = detection.imagePath;
        } else {
            cropImage.src = '';
        }

        // 預填名稱建議
        nameEn.value = detection.objectClass?.replace(/\s+/g, '_') || '';
        nameZh.value = detection.objectClassZh || '';

        modal.style.display = 'flex';

        // 綁定事件
        const closeBtn = document.getElementById('closeRegisterModal');
        const confirmBtn = document.getElementById('confirmRegisterBtn');

        closeBtn.onclick = () => {
            modal.style.display = 'none';
            this.pendingRegistration = null;
        };

        confirmBtn.onclick = () => this.registerFromDetection();

        // 點擊背景關閉
        modal.onclick = (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
                this.pendingRegistration = null;
            }
        };
    }

    async registerFromDetection() {
        if (!this.pendingRegistration) {
            this.ui.showToast('沒有待註冊的物品', 'error');
            return;
        }

        const nameEn = document.getElementById('registerNameEn').value.trim();
        const nameZh = document.getElementById('registerNameZh').value.trim();

        if (!nameEn || !nameZh) {
            this.ui.showToast('請填寫英文和中文名稱', 'warning');
            return;
        }

        this.ui.showLoading('正在註冊...');

        try {
            const response = await fetch('/api/objects/register-cropped', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: this.pendingRegistration.imageBase64,
                    bbox: this.pendingRegistration.bbox || [0, 0, 100, 100],
                    name: nameEn,
                    name_zh: nameZh
                })
            });

            const result = await response.json();

            if (result.success) {
                this.ui.showToast(`已註冊: ${nameZh}`, 'success');
                document.getElementById('registerModal').style.display = 'none';
                this.pendingRegistration = null;
                
                // 重新載入最近偵測列表
                await this.loadRecentDetections();
            } else {
                throw new Error(result.detail || '註冊失敗');
            }

        } catch (error) {
            console.error('註冊失敗:', error);
            this.ui.showToast('註冊失敗: ' + error.message, 'error');
        } finally {
            this.ui.hideLoading();
        }
    }
}

// 頁面載入後初始化
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ObjectFinderApp();
    window.app.init();
});
