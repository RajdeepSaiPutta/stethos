// Service Worker for Stethos PWA
// Provides offline-first caching for static assets and ML models

const CACHE_NAME = 'stethos-v1';
const MODEL_CACHE = 'stethos-models-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/favicon.ico'
];

// Install - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter(n => n !== CACHE_NAME && n !== MODEL_CACHE).map(n => caches.delete(n)))
    )
  );
  self.clients.claim();
});

// Fetch - cache-first for models, stale-while-revalidate for assets
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Model files - cache first, then network
  if (url.pathname.startsWith('/models/') || url.pathname.endsWith('.pt') || url.pathname.endsWith('.onnx')) {
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
  // Implementation would use IndexedDB to store pending results
  // and sync when online
  console.log('Syncing inference results...');
}

// Push notification support (optional)
self.addEventListener('push', (event) => {
  if (event.data) {
    const data = event.data.json();
    event.waitUntil(
      self.registration.showNotification(data.title, {
        body: data.body,
        icon: '/icon-192.png',
        badge: '/badge-72.png',
        data: data.url
      })
    );
  }
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  if (event.notification.data) {
    event.waitUntil(clients.openWindow(event.notification.data));
  }
});