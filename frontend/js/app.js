/**
 * 找東西助手 - 主程式
 */

class ObjectFinderApp {
    constructor() {
        this.db = window.objectFinderDB;
        this.api = window.objectFinderAPI;
        this.ui = window.objectFinderUI;
        this.isInitialized = false;
        this.isScanning = false;
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
            
            // 連接 WebSocket 接收即時偵測結果
            this.connectWebSocket();
            
            // 載入攝影機清單
            await this.loadCameras();
            
            // 載入最近記錄
            await this.loadRecentDetections();
            
            // 載入自訂常用物品
            this.loadQuickItems();
            
            // 添加測試資料（開發用）
            // await this.addDemoData();
            
            this.isInitialized = true;
            console.log('App 初始化完成');
            
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
        
        // 手動掃描
        this.ui.elements.manualScanBtn.addEventListener('click', () => this.handleManualScan());
        
        // 歷史記錄
        this.ui.elements.historyBtn.addEventListener('click', () => this.showHistory());
        
        // 清空資料
        const clearDataBtn = document.getElementById('clearDataBtn');
        if (clearDataBtn) {
            clearDataBtn.addEventListener('click', () => this.clearAllData());
        }
        
        // 設定
        this.ui.elements.settingsBtn.addEventListener('click', () => this.showSettings());
        
        // 攝影機選擇
        const cameraSelect = document.getElementById('cameraSelect');
        if (cameraSelect) {
            cameraSelect.addEventListener('change', (e) => this.handleCameraChange(e.target.value));
        }
        
        // 最近偵測項目點擊
        this.ui.elements.recentList.addEventListener('click', (e) => {
            const item = e.target.closest('.recent-item');
            if (item) {
                this.showDetectionDetail(item);
            }
        });
    }

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

    async handleManualScan() {
        if (this.isScanning) return;
        this.isScanning = true;
        this.ui.showLoading('正在掃描...');
        
        try {
            if (this.api.isConnected) {
                const result = await this.api.triggerSnapshot();
                
                // 顯示截圖
                if (result && result.image_path) {
                    this.ui.showSnapshot(result.image_path);
                }
                
                // 儲存偵測結果到本地
                if (result && result.detections && result.detections.length > 0) {
                    // 去重：同一物品類別只保留信心度最高的
                    const deduped = {};
                    for (const det of result.detections) {
                        const key = det.object_class;
                        if (!deduped[key] || det.confidence > deduped[key].confidence) {
                            deduped[key] = det;
                        }
                    }
                    
                    const dedupedList = Object.values(deduped);
                    for (const det of dedupedList) {
                        await this.db.saveDetection({
                            objectClass: det.object_class,
                            confidence: det.confidence,
                            bbox: det.bbox,
                            surface: det.surface || '未知',
                            region: det.region || '',
                            timestamp: det.timestamp || Date.now(),
                            imagePath: result.image_path  // 儲存截圖路徑
                        });
                    }
                    this.ui.showToast(`掃描完成！找到 ${dedupedList.length} 個物品`, 'success');
                } else {
                    this.ui.showToast('掃描完成，但未偵測到物品', 'info');
                }
                
                await this.loadRecentDetections();
            } else {
                this.ui.showToast('後端服務未連線，無法掃描', 'warning');
            }
        } catch (error) {
            console.error('掃描失敗:', error);
            this.ui.showToast('掃描失敗', 'error');
        } finally {
            this.ui.hideLoading();
            this.isScanning = false;
        }
    }

    async checkConnection() {
        const health = await this.api.checkHealth();
        this.ui.updateStatus(!!health, health ? '已連線至偵測服務' : '離線模式（使用本地資料）');
    }

    async loadCameras() {
        try {
            const result = await this.api.getCameras();
            if (result && result.cameras) {
                const select = document.getElementById('cameraSelect');
                if (select) {
                    select.innerHTML = result.cameras.map(cam => 
                        `<option value="${cam.id}" ${cam.id === result.current ? 'selected' : ''}>${cam.display || cam.name}</option>`
                    ).join('');
                    
                    if (result.cameras.length > 1) {
                        this.ui.showToast(`發現 ${result.cameras.length} 個攝影機`, 'info');
                    }
                }
            }
        } catch (error) {
            console.error('載入攝影機失敗:', error);
        }
    }

    async handleCameraChange(cameraId) {
        try {
            this.ui.showLoading('切換攝影機...');
            const result = await this.api.setCamera(parseInt(cameraId));
            if (result && result.success) {
                this.ui.showToast(`已切換到攝影機 ${cameraId}`, 'success');
            }
        } catch (error) {
            console.error('切換攝影機失敗:', error);
            this.ui.showToast('切換攝影機失敗', 'error');
        } finally {
            this.ui.hideLoading();
        }
    }

    async loadRecentDetections() {
        try {
            const detections = await this.db.getRecentDetections(5);
            this.ui.updateRecentList(detections);
        } catch (error) {
            console.error('載入記錄失敗:', error);
        }
    }

    async addDemoData() {
        const objects = await this.db.getAllObjects();
        if (objects.length === 0) {
            const demoData = [
                { objectClass: 'cell phone', confidence: 0.95, surface: 'sofa', region: 'left', timestamp: Date.now() - 300000 },
                { objectClass: 'remote', confidence: 0.88, surface: 'table', region: 'center', timestamp: Date.now() - 600000 },
                { objectClass: 'bottle', confidence: 0.92, surface: 'desk', region: 'right', timestamp: Date.now() - 900000 }
            ];
            
            // 優先使用 API 寫入
            if (this.api.isConnected) {
                try {
                    // 使用批次 API 寫入
                    await this.api.saveDetectionsBatch(demoData);
                    console.log('Demo 資料已透過 API 寫入');
                    
                    // 同時儲存到本地 IndexedDB 作為快取
                    for (const data of demoData) {
                        await this.db.saveDetection(data);
                    }
                } catch (error) {
                    console.warn('API 寫入失敗，改用本地儲存:', error);
                    // Fallback: 直接寫入本地 IndexedDB
                    for (const data of demoData) {
                        await this.db.saveDetection(data);
                    }
                }
            } else {
                // 後端未連線，直接寫入本地 IndexedDB
                for (const data of demoData) {
                    await this.db.saveDetection(data);
                }
                console.log('Demo 資料已寫入本地 IndexedDB（離線模式）');
            }
            
            await this.loadRecentDetections();
        }
    }

    loadQuickItems() {
        const DEFAULT_QUICK_ITEMS = [
            { name: '手機', icon: '📱', order: 1 },
            { name: '鑰匙', icon: '🔑', order: 2 },
            { name: '眼鏡', icon: '👓', order: 3 },
            { name: '錢包', icon: '👛', order: 4 },
            { name: '耳機', icon: '🎧', order: 5 },
            { name: '遙控器', icon: '📺', order: 6 }
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
                const key = det.objectClass;
                if (!grouped[key]) {
                    grouped[key] = {
                        objectClass: det.objectClass,
                        objectClassZh: det.objectClassZh,
                        records: []
                    };
                }
                grouped[key].records.push(det);
            }
            
            // 建立 Modal
            this.showHistoryModal(Object.values(grouped));
            
        } catch (error) {
            console.error('載入歷史記錄失敗:', error);
            this.ui.showToast('載入歷史記錄失敗', 'error');
        }
    }

    showHistoryModal(groupedData) {
        // 移除舊的 Modal
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
        
        // 格式化時間
        const formatTime = (timestamp) => {
            const date = new Date(timestamp);
            return date.toLocaleString('zh-TW', { 
                month: 'short', day: 'numeric', 
                hour: '2-digit', minute: '2-digit' 
            });
        };
        
        // 處理區域顯示
        const getRegionDisplay = (regionZh) => {
            if (!regionZh || regionZh === 'unknown' || regionZh === 'undefined') return '';
            return ' ' + regionZh;
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
                                " data-image="${record.imagePath || ''}" class="history-item">
                                    <div>
                                        <div style="color:#fff;">${record.surfaceZh || '未知位置'}${getRegionDisplay(record.regionZh)}</div>
                                        <div style="color:#888; font-size:12px;">${formatTime(record.timestamp)}</div>
                                    </div>
                                    <div style="color:#38ef7d; font-size:14px;">${Math.round(record.confidence * 100)}%</div>
                                </div>
                            `).join('')}
                            ${group.records.length > 10 ? `
                                <div style="color:#888; font-size:12px; text-align:center;">
                                    還有 ${group.records.length - 10} 筆記錄...
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // 關閉按鈕
        document.getElementById('closeHistoryBtn').addEventListener('click', () => modal.remove());
        
        // 點擊背景關閉
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.remove();
        });
        
        // 點擊歷史項目顯示截圖
        modal.querySelectorAll('.history-item').forEach(item => {
            item.style.cursor = 'pointer';
            item.addEventListener('click', () => {
                const imagePath = item.dataset.image;
                if (imagePath) {
                    this.ui.showSnapshot(imagePath);
                }
            });
        });
    }

    showSettings() {
        window.location.href = '/settings';
    }

    showDetectionDetail(item) {
        // 從 data 屬性取得資料
        const result = {
            objectClassZh: item.dataset.classZh,
            objectClass: item.dataset.class,
            surfaceZh: item.dataset.surface,
            regionZh: item.dataset.region,
            lastSeen: parseInt(item.dataset.time),
            confidence: parseFloat(item.dataset.confidence),
            imagePath: item.dataset.image || null,
            description: `${item.dataset.classZh}在${item.dataset.surface}${item.dataset.region}`
        };
        
        // 使用和搜尋結果一樣的顯示方式
        this.ui.showResult(result);
    }

    connectWebSocket() {
        const wsUrl = 'ws://localhost:8000/ws/detections';
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('✅ WebSocket 連線成功');
            };
            
            this.ws.onmessage = async (event) => {
                try {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'detection' && message.data && message.data.length > 0) {
                        console.log(`📡 收到定時偵測: ${message.data.length} 個物品`);
                        
                        // 去重：同一物品類別只保留信心度最高的
                        const deduped = {};
                        for (const det of message.data) {
                            const key = det.object_class;
                            if (!deduped[key] || det.confidence > deduped[key].confidence) {
                                deduped[key] = det;
                            }
                        }
                        
                        // 儲存到 IndexedDB
                        for (const det of Object.values(deduped)) {
                            await this.db.saveDetection({
                                objectClass: det.object_class,
                                confidence: det.confidence,
                                bbox: det.bbox,
                                surface: det.surface || '未知',
                                region: det.region || '',
                                timestamp: det.timestamp || Date.now(),
                                imagePath: det.image_path || null
                            });
                        }
                        
                        // 更新最近偵測列表
                        await this.loadRecentDetections();
                        
                        // 顯示通知
                        this.ui.showToast(`自動偵測到 ${message.data.length} 個物品`, 'info');
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
