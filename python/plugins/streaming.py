"""
Real-Time Streaming Inference for Breath Sounds.
Web Audio API + WebWorker + ring buffer + offline-first architecture.
"""
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
import threading
import queue
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming inference."""
    sample_rate: int = 16000
    window_duration: float = 2.0  # seconds
    hop_duration: float = 1.0  # seconds
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: Optional[str] = None


@dataclass
class StreamingResult:
    """Result from streaming inference."""
    timestamp: float
    prediction: str
    probabilities: Dict[str, float]
    confidence: float
    window_start: float
    window_end: float


class RingBuffer:
    """Thread-safe ring buffer for audio samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float32)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        """Write data to buffer. Returns number of samples written."""
        with self.lock:
            n = len(data)
            if n > self.capacity:
                data = data[-self.capacity:]
                n = self.capacity

            end = self.write_idx + n
            if end <= self.capacity:
                self.buffer[self.write_idx:end] = data
            else:
                first_part = self.capacity - self.write_idx
                self.buffer[self.write_idx:] = data[:first_part]
                self.buffer[:n - first_part] = data[first_part:]

            self.write_idx = end % self.capacity
            self.size = min(self.size + n, self.capacity)
            return n

    def read(self, n: int) -> np.ndarray:
        """Read n samples from buffer."""
        with self.lock:
            if n > self.size:
                n = self.size
            if n == 0:
                return np.array([], dtype=np.float32)

            end = self.read_idx + n
            if end <= self.capacity:
                data = self.buffer[self.read_idx:end].copy()
            else:
                first_part = self.capacity - self.read_idx
                data = np.concatenate([
                    self.buffer[self.read_idx:],
                    self.buffer[:n - first_part]
                ])

            self.read_idx = end % self.capacity
            self.size -= n
            return data

    def peek(self, n: int) -> np.ndarray:
        """Peek at n samples without consuming."""
        with self.lock:
            if n > self.size:
                n = self.size
            if n == 0:
                return np.array([], dtype=np.float32)

            end = self.read_idx + n
            if end <= self.capacity:
                return self.buffer[self.read_idx:end].copy()
            else:
                first_part = self.capacity - self.read_idx
                return np.concatenate([
                    self.buffer[self.read_idx:],
                    self.buffer[:n - first_part]
                ])

    def available(self) -> int:
        with self.lock:
            return self.size

    def clear(self) -> None:
        with self.lock:
            self.write_idx = 0
            self.read_idx = 0
            self.size = 0


class BreathStreamingInference:
    """Real-time streaming inference for breath sounds."""

    CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Load model
        from plugins.breath import BreathCNN
        self.model = BreathCNN(num_classes=len(self.CLASS_NAMES)).to(self.device)
        self.model.eval()

        if config.model_path and Path(config.model_path).exists():
            checkpoint = torch.load(config.model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded streaming model from {config.model_path}")
        else:
            logger.warning("No model checkpoint for streaming, using random weights")

        # Audio transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        ).to(self.device)
        self.db_transform = T.AmplitudeToDB(top_db=80.0).to(self.device)

        # Ring buffer for audio
        window_samples = int(config.sample_rate * config.window_duration)
        self.buffer = RingBuffer(window_samples * 2)  # 2x for overlap

        # Streaming state
        self.running = False
        self.callback: Optional[Callable[[StreamingResult], None]] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.window_counter = 0
        self.last_inference_time = 0.0

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute normalized log-mel spectrogram."""
        spec = self.mel_transform(waveform)
        spec = self.db_transform(spec)
        spec = (spec + 80.0) / 80.0
        spec = spec.clamp(0.0, 1.0)
        return spec

    def _infer_window(self, audio_window: np.ndarray) -> StreamingResult:
        """Run inference on a single audio window."""
        window_samples = int(self.config.sample_rate * self.config.window_duration)

        # Pad or trim
        if len(audio_window) < window_samples:
            audio_window = np.pad(audio_window, (0, window_samples - len(audio_window)))
        else:
            audio_window = audio_window[:window_samples]

        waveform = torch.from_numpy(audio_window).unsqueeze(0).to(self.device)
        spec = self._compute_spectrogram(waveform).unsqueeze(0)  # [1, 1, n_mels, time]

        with torch.no_grad():
            logits = self.model(spec)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        prediction = self.CLASS_NAMES[pred_idx]

        return StreamingResult(
            timestamp=time.time(),
            prediction=prediction,
            probabilities={name: float(prob) for name, prob in zip(self.CLASS_NAMES, probs)},
            confidence=confidence,
            window_start=self.window_counter * self.config.hop_duration,
            window_end=(self.window_counter + 1) * self.config.hop_duration
        )

    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to ring buffer (called from audio callback)."""
        # Ensure mono
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.mean(axis=0)
        self.buffer.write(audio_chunk.astype(np.float32))

    def start(self, callback: Callable[[StreamingResult], None]) -> None:
        """Start streaming inference."""
        self.callback = callback
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Streaming inference started")

    def stop(self) -> None:
        """Stop streaming inference."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        logger.info("Streaming inference stopped")

    def _processing_loop(self) -> None:
        """Main processing loop."""
        window_samples = int(self.config.sample_rate * self.config.window_duration)
        hop_samples = int(self.config.sample_rate * self.config.hop_duration)

        while self.running:
            available = self.buffer.available()
            if available >= window_samples:
                # Read window
                window = self.buffer.peek(window_samples)
                result = self._infer_window(window)
                self.window_counter += 1

                # Advance buffer by hop
                self.buffer.read(hop_samples)

                # Call callback
                if self.callback:
                    try:
                        self.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            # Sleep to maintain real-time rate
            elapsed = time.time() - self.last_inference_time
            target_interval = self.config.hop_duration
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
            self.last_inference_time = time.time()

    def get_latest_result(self) -> Optional[StreamingResult]:
        """Get latest inference result without callback."""
        available = self.buffer.available()
        window_samples = int(self.config.sample_rate * self.config.window_duration)
        if available >= window_samples:
            window = self.buffer.peek(window_samples)
            return self._infer_window(window)
        return None


# Offline-first Web Audio integration (for frontend reference)
STREAMING_WORKER_JS = r"""
// streaming-worker.js - Runs in WebWorker for offline real-time inference
// This is a reference implementation for the frontend

class StreamingProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(16000 * 2); // 2s at 16kHz
        this.writePos = 0;
        this.readPos = 0;
        this.windowSize = 16000 * 2; // 2s
        this.hopSize = 16000 * 1; // 1s
        this.windowCount = 0;
        this.port.onmessage = (e) => {
            if (e.data.type === 'config') {
                this.windowSize = e.data.windowSize;
                this.hopSize = e.data.hopSize;
                this.buffer = new Float32Array(this.windowSize * 2);
            }
        };
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const channel = input[0];
            // Write to ring buffer
            for (let i = 0; i < channel.length; i++) {
                this.buffer[this.writePos] = channel[i];
                this.writePos = (this.writePos + 1) % this.buffer.length;
            }

            // Check if we have enough for a window
            const available = (this.writePos - this.readPos + this.buffer.length) % this.buffer.length;
            if (available >= this.windowSize) {
                // Extract window
                const window = new Float32Array(this.windowSize);
                for (let i = 0; i < this.windowSize; i++) {
                    window[i] = this.buffer[(this.readPos + i) % this.buffer.length];
                }

                // Send to main thread for inference (or run WASM model here)
                this.port.postMessage({
                    type: 'window',
                    data: window,
                    windowIndex: this.windowCount++,
                    timestamp: currentTime
                });

                // Advance by hop
                this.readPos = (this.readPos + this.hopSize) % this.buffer.length;
            }
        }
        return true;
    }
}

registerProcessor('streaming-processor', StreamingProcessor);
"""


# IndexedDB for offline model storage (frontend reference)
INDEXEDDB_MODEL_STORAGE = r"""
// model-store.js - IndexedDB wrapper for offline model caching

class ModelStore {
    constructor(dbName = 'stethos-models', version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
    }

    async open() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => { this.db = request.result; resolve(); };
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('models')) {
                    db.createObjectStore('models', { keyPath: 'name' });
                }
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata', { keyPath: 'key' });
                }
            };
        });
    }

    async putModel(name, modelBlob, metadata = {}) {
        const tx = this.db.transaction(['models', 'metadata'], 'readwrite');
        await Promise.all([
            new Promise((resolve, reject) => {
                const req = tx.objectStore('models').put({ name, blob: modelBlob, timestamp: Date.now() });
                req.onsuccess = resolve; req.onerror = reject;
            }),
            new Promise((resolve, reject) => {
                const req = tx.objectStore('metadata').put({ key: name, ...metadata });
                req.onsuccess = resolve; req.onerror = reject;
            })
        ]);
    }

    async getModel(name) {
        return new Promise((resolve, reject) => {
            const req = this.db.transaction('models').objectStore('models').get(name);
            req.onsuccess = () => resolve(req.result?.blob);
            req.onerror = reject;
        });
    }

    async getAllModels() {
        return new Promise((resolve, reject) => {
            const req = this.db.transaction('models').objectStore('models').getAll();
            req.onsuccess = () => resolve(req.result);
            req.onerror = reject;
        });
    }

    async deleteModel(name) {
        const tx = this.db.transaction(['models', 'metadata'], 'readwrite');
        await Promise.all([
            new Promise((resolve, reject) => {
                const req = tx.objectStore('models').delete(name);
                req.onsuccess = resolve; req.onerror = reject;
            }),
            new Promise((resolve, reject) => {
                const req = tx.objectStore('metadata').delete(name);
                req.onsuccess = resolve; req.onerror = reject;
            })
        ]);
    }
}
"""


# Service Worker for offline-first PWA (frontend reference)
SERVICE_WORKER_JS = r"""
// sw.js - Service Worker for Stethos PWA

const CACHE_NAME = 'stethos-v1';
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
    // Models will be cached dynamically
];

const MODEL_CACHE = 'stethos-models-v1';

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
    );
    self.skipWaiting();
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((names) => 
            Promise.all(names.filter(n => n !== CACHE_NAME && n !== MODEL_CACHE).map(n => caches.delete(n)))
        )
    );
    self.clients.claim();
});

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Model files - cache first, then network
    if (url.pathname.startsWith('/models/')) {
        event.respondWith(
            caches.open(MODEL_CACHE).then(async (cache) => {
                const cached = await cache.match(event.request);
                if (cached) return cached;

                try {
                    const response = await fetch(event.request);
                    if (response.ok) cache.put(event.request, response.clone());
                    return response;
                } catch {
                    return new Response('Offline', { status: 503 });
                }
            })
        );
        return;
    }

    // Static assets - stale while revalidate
    event.respondWith(
        caches.open(CACHE_NAME).then(async (cache) => {
            const cached = await cache.match(event.request);
            const fetchPromise = fetch(event.request).then((response) => {
                if (response.ok) cache.put(event.request, response.clone());
                return response;
            }).catch(() => cached);
            return cached || fetchPromise;
        })
    );
});

// Background sync for inference results
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-inference') {
        event.waitUntil(syncInferenceResults());
    }
});

async function syncInferenceResults() {
    const db = await openResultsDB();
    const pending = await db.getAll('pending');
    for (const result of pending) {
        try {
            await fetch('/api/inference/sync', {
                method: 'POST',
                body: JSON.stringify(result),
                headers: { 'Content-Type': 'application/json' }
            });
            await db.delete('pending', result.id);
        } catch (e) {
            console.log('Sync failed, will retry', e);
        }
    }
}
"""


def get_streaming_worker_js() -> str:
    """Get the streaming worker JavaScript code."""
    return STREAMING_WORKER_JS


def get_indexeddb_model_storage_js() -> str:
    """Get the IndexedDB model storage JavaScript code."""
    return INDEXEDDB_MODEL_STORAGE


def get_service_worker_js() -> str:
    """Get the Service Worker JavaScript code."""
    return SERVICE_WORKER_JS