/**
 * SiteGuard AI - Modern Dashboard Frontend Logic
 * Handles real-time polling, webcam stream toggling, video processing,
 * image detection analysis, and violation history management.
 */

document.addEventListener('DOMContentLoaded', () => {
    initDashboardPolling();
    initWebcamControls();
    initVideoUpload();
    initImageUpload();
    initHistoryControls();
});

// 1. DASHBOARD POLLING
function initDashboardPolling() {
    const workersEl = document.getElementById('metric-workers');
    const compliantEl = document.getElementById('metric-compliant');
    const violationsEl = document.getElementById('metric-violations');
    const fpsEl = document.getElementById('metric-fps');

    if (!workersEl) return;

    async function fetchStats() {
        try {
            const res = await fetch('/api/stats');
            if (res.ok) {
                const data = await res.json();
                if (workersEl) workersEl.textContent = data.workers_detected || 0;
                if (compliantEl) compliantEl.textContent = data.compliant_workers || 0;
                if (violationsEl) violationsEl.textContent = data.ppe_violations || 0;
                if (fpsEl) fpsEl.textContent = (data.current_fps || 0).toFixed(1);
            }
        } catch (err) {
            console.warn('Dashboard stats fetch error:', err);
        }
    }

    fetchStats();
    setInterval(fetchStats, 2500);
}

// 2. WEBCAM STREAM CONTROLS & SAFETY ALARM SYSTEM
class SafetyAlarmSystem {
    constructor() {
        this.audioCtx = null;
        this.isPlaying = false;
        this.isMuted = false;
        this.alarmInterval = null;
        this.stopTimeout = null;
    }

    initAudio() {
        if (!this.audioCtx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioCtx();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    }

    playSirenBurst() {
        if (!this.audioCtx || this.isMuted) return;
        try {
            const now = this.audioCtx.currentTime;
            
            // Primary urgent siren oscillator (sawtooth for industrial sound)
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = 'sawtooth';

            // Two-tone emergency pitch modulation: 760 Hz to 990 Hz
            osc.frequency.setValueAtTime(760, now);
            osc.frequency.linearRampToValueAtTime(1020, now + 0.16);
            osc.frequency.linearRampToValueAtTime(760, now + 0.32);
            osc.frequency.linearRampToValueAtTime(990, now + 0.48);

            gain.gain.setValueAtTime(0.24, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.62);

            osc.connect(gain);
            gain.connect(this.audioCtx.destination);

            osc.start(now);
            osc.stop(now + 0.64);
        } catch (err) {
            console.warn('Audio siren synthesis error:', err);
        }
    }

    startAlarm() {
        if (this.stopTimeout) {
            clearTimeout(this.stopTimeout);
            this.stopTimeout = null;
        }
        if (this.isPlaying) return;
        this.initAudio();
        this.isPlaying = true;
        this.playSirenBurst();
        this.alarmInterval = setInterval(() => {
            if (this.isPlaying && !this.isMuted) {
                this.playSirenBurst();
            }
        }, 680);
    }

    stopAlarm(immediate = false) {
        if (!this.isPlaying) return;
        if (immediate) {
            if (this.stopTimeout) clearTimeout(this.stopTimeout);
            this.stopTimeout = null;
            this.isPlaying = false;
            if (this.alarmInterval) {
                clearInterval(this.alarmInterval);
                this.alarmInterval = null;
            }
            return;
        }

        if (this.stopTimeout) return;
        // 1.0s debounce to avoid flutter when worker glances or changes pose
        this.stopTimeout = setTimeout(() => {
            this.isPlaying = false;
            if (this.alarmInterval) {
                clearInterval(this.alarmInterval);
                this.alarmInterval = null;
            }
            this.stopTimeout = null;
        }, 1000);
    }

    testAlarm() {
        this.initAudio();
        const prevMuted = this.isMuted;
        this.isMuted = false;
        this.playSirenBurst();
        setTimeout(() => {
            this.playSirenBurst();
            setTimeout(() => {
                this.isMuted = prevMuted;
            }, 720);
        }, 700);
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        return this.isMuted;
    }
}

function initWebcamControls() {
    const startBtn = document.getElementById('btn-start-camera');
    const stopBtn = document.getElementById('btn-stop-camera');
    const toggleAlarmBtn = document.getElementById('btn-toggle-alarm');
    const testAlarmBtn = document.getElementById('btn-test-alarm');
    const videoStream = document.getElementById('webcam-stream');
    const placeholder = document.getElementById('webcam-placeholder');

    const alarmBanner = document.getElementById('live-alarm-banner');
    const bannerTitle = document.getElementById('alarm-banner-title');
    const bannerDesc = document.getElementById('alarm-banner-desc');
    const bannerBadge = document.getElementById('alarm-banner-badge');
    const alarmPill = document.getElementById('alarm-status-pill');
    const alarmPillText = document.getElementById('alarm-status-text');

    const violatorsContainer = document.getElementById('violators-container');
    const violatorsBadge = document.getElementById('violators-count-badge');
    const personnelContainer = document.getElementById('personnel-container');
    const workersBadge = document.getElementById('workers-count-badge');
    const incidentLog = document.getElementById('incident-log-container');
    const clearLogBtn = document.getElementById('btn-clear-alarm-log');

    if (!startBtn || !stopBtn) return;

    const alarmSystem = new SafetyAlarmSystem();
    let pollInterval = null;
    let hadViolationLastFrame = false;
    let lastLoggedViolationKey = '';
    let lastLoggedTime = 0;

    function addIncidentLog(message, isViolation = false) {
        if (!incidentLog) return;
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const entry = document.createElement('div');
        entry.className = `incident-log-entry ${isViolation ? 'log-violation' : 'log-resolved'}`;
        entry.innerHTML = `
            <span>${message}</span>
            <span class="incident-time">${timeStr}</span>
        `;
        incidentLog.insertBefore(entry, incidentLog.firstChild);
        // Keep maximum 25 log items
        while (incidentLog.children.length > 25) {
            incidentLog.removeChild(incidentLog.lastChild);
        }
    }

    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', () => {
            if (incidentLog) {
                incidentLog.innerHTML = `
                    <div class="incident-log-entry">
                        <span>Incident log cleared.</span>
                        <span class="incident-time">${new Date().toLocaleTimeString()}</span>
                    </div>
                `;
            }
        });
    }

    // Toggle Alarm Mute / Unmute
    if (toggleAlarmBtn) {
        toggleAlarmBtn.addEventListener('click', () => {
            alarmSystem.initAudio();
            const muted = alarmSystem.toggleMute();
            const toggleText = document.getElementById('alarm-toggle-text');
            if (toggleText) toggleText.textContent = muted ? 'Alarm Sound: MUTED' : 'Alarm Sound: ON';

            if (muted) {
                toggleAlarmBtn.classList.remove('btn-secondary');
                toggleAlarmBtn.classList.add('btn-secondary');
                toggleAlarmBtn.style.opacity = '0.75';
                if (alarmPill) {
                    alarmPill.className = 'alarm-status-pill muted';
                    if (alarmPillText) alarmPillText.textContent = 'Alarm Siren Muted 🔇';
                }
            } else {
                toggleAlarmBtn.style.opacity = '1';
                if (alarmPill) {
                    alarmPill.className = alarmSystem.isPlaying ? 'alarm-status-pill sounding' : 'alarm-status-pill armed';
                    if (alarmPillText) alarmPillText.textContent = alarmSystem.isPlaying ? 'Alarm Sounding 🚨' : 'Alarm Armed & Ready';
                }
            }
        });
    }

    // Test Alarm Sound Preview
    if (testAlarmBtn) {
        testAlarmBtn.addEventListener('click', () => {
            alarmSystem.testAlarm();
            addIncidentLog('🔊 Audio alarm siren test sounded.', false);
        });
    }

    // Camera Status Polling
    async function checkCameraStatus() {
        try {
            const res = await fetch('/api/camera/status');
            if (!res.ok) return;
            const data = await res.json();

            if (!data.is_running) return;

            const violations = data.violations || [];
            const hasViolation = data.has_violation && violations.length > 0;
            const violatorIds = data.violator_ids || [];
            const violatorInfo = data.violator_info || (violatorIds.length ? '#' + violatorIds.join(', #') : '');

            // 1. Audio Siren Alarm Trigger
            if (hasViolation) {
                alarmSystem.startAlarm();

                // Update Alarm Status Pill
                if (alarmPill) {
                    alarmPill.className = alarmSystem.isMuted ? 'alarm-status-pill muted' : 'alarm-status-pill sounding';
                    if (alarmPillText) {
                        alarmPillText.textContent = alarmSystem.isMuted 
                            ? `Siren Muted 🔇 (Worker ${violatorInfo} Violation)` 
                            : `🚨 ALARM SOUNDING – Worker ${violatorInfo}!`;
                    }
                }

                // Update Emergency Alert Banner
                if (alarmBanner) {
                    alarmBanner.classList.add('alarm-active');
                    if (bannerTitle) {
                        bannerTitle.innerHTML = `🚨 CRITICAL SAFETY ALARM – WORKER ${violatorInfo} VIOLATION!`;
                    }
                    if (bannerDesc) {
                        const items = [...new Set(violations.map(v => v.missing_ppe))].join(', ');
                        bannerDesc.innerHTML = `Immediate safety breach detected: <strong>Worker ${violatorInfo}</strong> is missing mandatory PPE (<strong>${items}</strong>).`;
                    }
                    if (bannerBadge) {
                        bannerBadge.innerHTML = `<span style="color: var(--danger)">ALARM SOUNDING</span>`;
                    }
                }

                // Update Active Violators Spotlight Card
                if (violatorsBadge) {
                    violatorsBadge.className = 'badge badge-non-compliant';
                    violatorsBadge.textContent = `${violations.length} Violation${violations.length > 1 ? 's' : ''}`;
                }

                if (violatorsContainer) {
                    violatorsContainer.innerHTML = violations.map(v => `
                        <div class="violator-card">
                            <div class="violator-header">
                                <span class="worker-id-badge">🚨 WORKER #${v.worker_id}</span>
                                <span class="violation-tag">NON-COMPLIANT</span>
                            </div>
                            <div class="violation-details">
                                <div>Infraction: <strong style="color: var(--danger)">${v.missing_ppe}</strong></div>
                                <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 3px;">
                                    Confidence: ${(v.confidence * 100).toFixed(0)}% • Alarm Siren Active
                                </div>
                            </div>
                        </div>
                    `).join('');
                }

                // Log incident with anti-spam throttle (3.0s)
                const currentViolationKey = `${violatorInfo}_${violations.map(v => v.missing_ppe).join(',')}`;
                const nowSec = Date.now() / 1000;
                if (!hadViolationLastFrame || (currentViolationKey !== lastLoggedViolationKey && nowSec - lastLoggedTime > 3.0)) {
                    lastLoggedViolationKey = currentViolationKey;
                    lastLoggedTime = nowSec;
                    const items = [...new Set(violations.map(v => v.missing_ppe))].join(', ');
                    addIncidentLog(`🚨 ALARM TRIGGERED: Worker ${violatorInfo} broke safety rule (${items})`, true);
                }

                hadViolationLastFrame = true;

            } else {
                // No Violations
                alarmSystem.stopAlarm();

                if (hadViolationLastFrame) {
                    addIncidentLog(`🟢 Alarm Cleared: All personnel compliance restored.`, false);
                    hadViolationLastFrame = false;
                }

                // Reset Status Pill
                if (alarmPill) {
                    alarmPill.className = alarmSystem.isMuted ? 'alarm-status-pill muted' : 'alarm-status-pill armed';
                    if (alarmPillText) {
                        alarmPillText.textContent = alarmSystem.isMuted ? 'Alarm Siren Muted 🔇' : 'Alarm Armed & Ready';
                    }
                }

                // Reset Banner
                if (alarmBanner) {
                    alarmBanner.classList.remove('alarm-active');
                    if (bannerTitle) {
                        bannerTitle.innerHTML = `🟢 SAFETY SYSTEM ARMED`;
                    }
                    if (bannerDesc) {
                        bannerDesc.innerHTML = `Active camera monitoring: ${data.total_workers || 0} worker(s) in frame, all compliant with site safety standards.`;
                    }
                    if (bannerBadge) {
                        bannerBadge.innerHTML = `<span style="color: var(--success)">MONITORING</span>`;
                    }
                }

                // Reset Violators Spotlight
                if (violatorsBadge) {
                    violatorsBadge.className = 'badge badge-compliant';
                    violatorsBadge.textContent = `0 Violations`;
                }

                if (violatorsContainer) {
                    violatorsContainer.innerHTML = `
                        <div style="padding: 18px 12px; text-align: center; color: var(--text-muted); font-size: 0.88rem;">
                            <div style="font-size: 1.6rem; margin-bottom: 6px;">🛡️</div>
                            <strong style="color: var(--success);">All Personnel Safe</strong>
                            <div style="font-size: 0.78rem; margin-top: 4px;">No safety compliance violations detected in active stream.</div>
                        </div>
                    `;
                }
            }

            // Update Tracked Personnel in Frame List
            if (workersBadge) {
                workersBadge.textContent = `${data.total_workers || 0} Worker${(data.total_workers || 0) === 1 ? '' : 's'}`;
            }

            if (personnelContainer) {
                const workers = data.workers || [];
                if (workers.length > 0) {
                    personnelContainer.innerHTML = workers.map(w => {
                        const isViolator = w.overall_status === 'NON-COMPLIANT';
                        const isCompliant = w.overall_status === 'COMPLIANT';
                        const badgeClass = isViolator ? 'badge-non-compliant' : (isCompliant ? 'badge-compliant' : 'badge-unverified');
                        const statusLabel = isViolator ? 'VIOLATION' : (isCompliant ? 'COMPLIANT' : 'UNVERIFIED');
                        const details = isViolator ? (w.missing_items || []).join(', ') : 'Safety gear verified';
                        
                        return `
                            <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-size: 0.82rem;">
                                <span><strong>Worker #${w.worker_id}</strong> (${details})</span>
                                <span class="badge ${badgeClass}">${statusLabel}</span>
                            </div>
                        `;
                    }).join('');
                } else {
                    personnelContainer.innerHTML = `
                        <div style="font-size: 0.82rem; color: var(--text-muted); text-align: center; padding: 12px 0;">
                            No workers currently in camera view
                        </div>
                    `;
                }
            }

        } catch (err) {
            console.warn('Status poll error:', err);
        }
    }

    // Start Camera Handler
    startBtn.addEventListener('click', async () => {
        try {
            alarmSystem.initAudio(); // Initialize Web Audio Context on user click!
            startBtn.disabled = true;
            startBtn.textContent = 'Connecting...';
            
            const res = await fetch('/api/camera/start', { method: 'POST' });
            const data = await res.json();

            if (res.ok && data.status === 'started') {
                if (videoStream) {
                    videoStream.src = '/video_feed';
                    videoStream.style.display = 'block';
                }
                if (placeholder) placeholder.style.display = 'none';
                startBtn.style.display = 'none';
                stopBtn.style.display = 'inline-flex';

                addIncidentLog('📹 Live camera connected. AI safety compliance monitoring active.', false);

                // Start status polling
                if (pollInterval) clearInterval(pollInterval);
                pollInterval = setInterval(checkCameraStatus, 350);
            } else {
                alert(data.message || 'Unable to access webcam. Please verify camera permissions.');
            }
        } catch (err) {
            alert('Error connecting to camera server: ' + err.message);
        } finally {
            startBtn.disabled = false;
            startBtn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg> Start Camera`;
        }
    });

    // Stop Camera Handler
    stopBtn.addEventListener('click', async () => {
        try {
            stopBtn.disabled = true;
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
            alarmSystem.stopAlarm(true);

            const res = await fetch('/api/camera/stop', { method: 'POST' });
            if (res.ok) {
                if (videoStream) {
                    videoStream.src = '';
                    videoStream.style.display = 'none';
                }
                if (placeholder) placeholder.style.display = 'flex';
                stopBtn.style.display = 'none';
                startBtn.style.display = 'inline-flex';

                addIncidentLog('⏹️ Live camera stopped. Monitoring paused.', false);

                // Reset UI to standby
                if (alarmBanner) {
                    alarmBanner.classList.remove('alarm-active');
                    if (bannerTitle) bannerTitle.innerHTML = `🟢 SAFETY SYSTEM ARMED`;
                    if (bannerDesc) bannerDesc.innerHTML = `AI camera engine is ready. Connect camera to monitor safety compliance and sound sudden alarms.`;
                    if (bannerBadge) bannerBadge.innerHTML = `STANDBY`;
                }
                if (alarmPill) {
                    alarmPill.className = 'alarm-status-pill armed';
                    if (alarmPillText) alarmPillText.textContent = 'Alarm Armed & Ready';
                }
                if (personnelContainer) {
                    personnelContainer.innerHTML = `
                        <div style="font-size: 0.82rem; color: var(--text-muted); text-align: center; padding: 12px 0;">
                            Camera not streaming
                        </div>
                    `;
                }
                if (workersBadge) workersBadge.textContent = '0 Workers';
            }
        } catch (err) {
            console.error('Stop camera error:', err);
        } finally {
            stopBtn.disabled = false;
        }
    });
}

// 3. VIDEO UPLOAD & PROCESSING
function initVideoUpload() {
    const uploadForm = document.getElementById('upload-video-form');
    const fileInput = document.getElementById('video-file-input');
    const dropzone = document.getElementById('upload-dropzone');
    const fileNameDisplay = document.getElementById('selected-filename');
    const processBtn = document.getElementById('btn-process-video');
    const progressContainer = document.getElementById('upload-progress-container');
    const progressBar = document.getElementById('upload-progress-bar');
    const statusText = document.getElementById('processing-status-text');
    const resultPanel = document.getElementById('upload-results-panel');
    const resultVideo = document.getElementById('processed-video-player');
    const downloadLink = document.getElementById('btn-download-video');

    if (!uploadForm || !fileInput) return;

    if (dropzone) {
        dropzone.addEventListener('click', () => fileInput.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--primary)';
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = 'rgba(14, 165, 233, 0.35)';
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect(fileInput.files[0]);
            }
        });
    }

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (fileNameDisplay) {
            fileNameDisplay.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        }
        if (processBtn) processBtn.disabled = false;
    }

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!fileInput.files.length) return;

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('video', file);

        if (processBtn) processBtn.disabled = true;
        if (progressContainer) progressContainer.style.display = 'block';
        if (statusText) statusText.textContent = 'Uploading video file...';

        try {
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/api/video/process', true);

            xhr.upload.onprogress = (event) => {
                if (event.lengthComputable && progressBar) {
                    const percent = Math.round((event.loaded / event.total) * 40);
                    progressBar.style.width = percent + '%';
                }
            };

            xhr.onload = function () {
                if (xhr.status === 200) {
                    const data = JSON.parse(xhr.responseText);
                    if (progressBar) progressBar.style.width = '100%';
                    if (statusText) statusText.textContent = 'Processing completed successfully!';
                    
                    if (resultPanel) resultPanel.style.display = 'block';
                    if (resultVideo && data.download_url) {
                        resultVideo.src = data.download_url;
                        resultVideo.load();
                    }
                    if (downloadLink && data.download_url) {
                        downloadLink.href = data.download_url;
                    }

                    const statsDiv = document.getElementById('processed-stats');
                    if (statsDiv && data.stats) {
                        statsDiv.innerHTML = `
                            <div style="display: flex; gap: 20px; margin-top: 15px; font-size: 0.9rem;">
                                <span>Total Frames: <strong>${data.stats.total_frames}</strong></span>
                                <span>Inference FPS: <strong>${data.stats.fps}</strong></span>
                                <span>Workers Detected: <strong>${data.stats.workers_detected}</strong></span>
                                <span style="color: var(--danger)">Violations Logged: <strong>${data.stats.violations_detected}</strong></span>
                            </div>
                        `;
                    }
                } else {
                    const err = JSON.parse(xhr.responseText);
                    alert(err.message || 'Error processing video.');
                    if (statusText) statusText.textContent = 'Processing failed.';
                }
                if (processBtn) processBtn.disabled = false;
            };

            xhr.onerror = function () {
                alert('Network error while processing video.');
                if (processBtn) processBtn.disabled = false;
            };

            if (statusText) statusText.textContent = 'AI is analyzing video frames with YOLOv8... (Please wait)';
            if (progressBar) progressBar.style.width = '60%';
            xhr.send(formData);

        } catch (err) {
            alert('Failed to initiate video processing: ' + err.message);
            if (processBtn) processBtn.disabled = false;
        }
    });
}

// 4. IMAGE UPLOAD & DETECTION (NEW)
function initImageUpload() {
    const form = document.getElementById('upload-image-form');
    const input = document.getElementById('image-file-input');
    const dropzone = document.getElementById('upload-image-dropzone');
    const nameLabel = document.getElementById('selected-image-filename');
    const btn = document.getElementById('btn-process-image');
    const previewContainer = document.getElementById('image-client-preview-container');
    const previewImg = document.getElementById('image-client-preview');
    const statusText = document.getElementById('image-status-text');

    const resultPanel = document.getElementById('image-results-panel');
    const resultImg = document.getElementById('processed-image-result');
    const downloadBtn = document.getElementById('btn-download-image');

    const statWorkers = document.getElementById('img-stat-workers');
    const statCompliant = document.getElementById('img-stat-compliant');
    const statViolations = document.getElementById('img-stat-violations');
    const statStatus = document.getElementById('img-stat-status');
    const breakdownDiv = document.getElementById('image-worker-breakdown');

    if (!form || !input) return;

    if (dropzone) {
        dropzone.addEventListener('click', () => input.click());
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.style.borderColor = 'var(--primary)';
        });
        dropzone.addEventListener('dragleave', () => {
            dropzone.style.borderColor = 'rgba(14, 165, 233, 0.35)';
        });
        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                input.files = e.dataTransfer.files;
                handleSelect(input.files[0]);
            }
        });
    }

    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            handleSelect(input.files[0]);
        }
    });

    function handleSelect(file) {
        if (nameLabel) {
            nameLabel.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        }
        if (btn) btn.disabled = false;

        // Show client preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg && previewContainer) {
                previewImg.src = e.target.result;
                previewContainer.style.display = 'block';
            }
        };
        reader.readAsDataURL(file);
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!input.files.length) return;

        const file = input.files[0];
        const formData = new FormData();
        formData.append('image', file);

        if (btn) btn.disabled = true;
        if (statusText) statusText.textContent = 'Analyzing image with YOLOv8 & Compliance Engine...';

        try {
            const res = await fetch('/api/image/process', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok && data.status === 'success') {
                if (statusText) statusText.textContent = 'Analysis complete!';
                if (resultPanel) resultPanel.style.display = 'block';
                if (resultImg) resultImg.src = data.download_url + '?t=' + new Date().getTime();
                if (downloadBtn) downloadBtn.href = data.download_url;

                // Update KPI Cards
                if (statWorkers) statWorkers.textContent = data.summary.total_workers;
                if (statCompliant) statCompliant.textContent = data.summary.compliant;
                if (statViolations) statViolations.textContent = data.summary.non_compliant;
                if (statStatus) {
                    statStatus.textContent = data.overall_status;
                    if (data.overall_status === 'COMPLIANT') {
                        statStatus.style.color = 'var(--success)';
                    } else if (data.overall_status === 'NON-COMPLIANT') {
                        statStatus.style.color = 'var(--danger)';
                    } else {
                        statStatus.style.color = 'var(--warning)';
                    }
                }

                // Render violations list if any
                if (breakdownDiv && data.violations) {
                    if (data.violations.length > 0) {
                        breakdownDiv.innerHTML = `
                            <div style="background: rgba(244, 63, 94, 0.08); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: var(--radius-md); padding: 14px 18px;">
                                <strong style="color: var(--danger)">Detected Safety Infractions:</strong>
                                <ul style="margin-top: 8px; padding-left: 20px; font-size: 0.88rem; color: var(--text-main);">
                                    ${data.violations.map(v => `<li>Worker #${v.worker_id}: <strong>${v.missing_ppe}</strong> (Confidence: ${(v.confidence * 100).toFixed(0)}%)</li>`).join('')}
                                </ul>
                            </div>
                        `;
                    } else {
                        breakdownDiv.innerHTML = `
                            <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: var(--radius-md); padding: 14px 18px; color: var(--success); font-size: 0.9rem;">
                                ✓ All detected workers are fully compliant with mandatory site safety equipment.
                            </div>
                        `;
                    }
                }

                // Scroll smoothly to results
                resultPanel.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert(data.message || 'Error processing image.');
                if (statusText) statusText.textContent = 'Processing failed.';
            }
        } catch (err) {
            alert('Failed to connect to AI server: ' + err.message);
            if (statusText) statusText.textContent = 'Network error.';
        } finally {
            if (btn) btn.disabled = false;
        }
    });
}

// 5. VIOLATION HISTORY MANAGEMENT
function initHistoryControls() {
    // Native form submission in history.html handles clearing reliably across all browsers
}
