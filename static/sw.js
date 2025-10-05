// Service Worker for BRI Dashboard PWA
const CACHE_NAME = 'bri-dashboard-v1.0.0';
const STATIC_CACHE = 'bri-static-v1.0.0';
const DYNAMIC_CACHE = 'bri-dynamic-v1.0.0';

// Files to cache for offline functionality
const STATIC_FILES = [
  '/',
  '/static/manifest.json',
  '/static/css/style.css',
  '/static/js/app.js',
  'https://cdn.plot.ly/plotly-2.35.2.min.js',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
  '/api/summary',
  '/api/bri_chart',
  '/api/candlestick_chart',
  '/api/box_plots',
  '/api/correlation_heatmap'
];

// Install event - cache static files
self.addEventListener('install', event => {
  console.log('Service Worker: Installing...');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('Service Worker: Caching static files');
        return cache.addAll(STATIC_FILES);
      })
      .then(() => {
        console.log('Service Worker: Static files cached');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('Service Worker: Error caching static files', error);
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  console.log('Service Worker: Activating...');
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames.map(cacheName => {
            if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
              console.log('Service Worker: Deleting old cache', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log('Service Worker: Activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache or network
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle API requests
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      caches.open(DYNAMIC_CACHE)
        .then(cache => {
          return cache.match(request)
            .then(response => {
              if (response) {
                // Return cached response and update in background
                fetch(request)
                  .then(fetchResponse => {
                    if (fetchResponse.ok) {
                      cache.put(request, fetchResponse.clone());
                    }
                  })
                  .catch(() => {
                    // Network error, keep using cached response
                  });
                return response;
              } else {
                // No cached response, fetch from network
                return fetch(request)
                  .then(fetchResponse => {
                    if (fetchResponse.ok) {
                      cache.put(request, fetchResponse.clone());
                    }
                    return fetchResponse;
                  })
                  .catch(error => {
                    console.error('Service Worker: Network error', error);
                    // Return offline response for API calls
                    return new Response(
                      JSON.stringify({
                        error: 'Offline - No cached data available',
                        offline: true
                      }),
                      {
                        status: 503,
                        statusText: 'Service Unavailable',
                        headers: { 'Content-Type': 'application/json' }
                      }
                    );
                  });
              }
            });
        })
    );
    return;
  }

  // Handle static file requests
  if (request.method === 'GET') {
    event.respondWith(
      caches.match(request)
        .then(response => {
          if (response) {
            return response;
          }
          
          return fetch(request)
            .then(fetchResponse => {
              // Don't cache non-successful responses
              if (!fetchResponse || fetchResponse.status !== 200 || fetchResponse.type !== 'basic') {
                return fetchResponse;
              }

              // Cache successful responses
              const responseToCache = fetchResponse.clone();
              caches.open(DYNAMIC_CACHE)
                .then(cache => {
                  cache.put(request, responseToCache);
                });

              return fetchResponse;
            })
            .catch(error => {
              console.error('Service Worker: Fetch error', error);
              
              // Return offline page for navigation requests
              if (request.mode === 'navigate') {
                return caches.match('/offline.html');
              }
              
              throw error;
            });
        })
    );
  }
});

// Background sync for data updates
self.addEventListener('sync', event => {
  if (event.tag === 'background-sync') {
    console.log('Service Worker: Background sync triggered');
    event.waitUntil(
      // Update cached data in background
      updateCachedData()
    );
  }
});

// Push notifications for BRI alerts
self.addEventListener('push', event => {
  console.log('Service Worker: Push notification received');
  
  const options = {
    body: event.data ? event.data.text() : 'BRI Dashboard Update',
    icon: '/static/icons/icon-192x192.png',
    badge: '/static/icons/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Dashboard',
        icon: '/static/icons/action-view.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/static/icons/action-close.png'
      }
    ]
  };

  event.waitUntil(
    self.registration.showNotification('BRI Dashboard Alert', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('Service Worker: Notification clicked');
  event.notification.close();

  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  } else if (event.action === 'close') {
    // Just close the notification
    return;
  } else {
    // Default action - open dashboard
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Helper function to update cached data
async function updateCachedData() {
  try {
    const cache = await caches.open(DYNAMIC_CACHE);
    
    // Update summary data
    const summaryResponse = await fetch('/api/summary');
    if (summaryResponse.ok) {
      await cache.put('/api/summary', summaryResponse.clone());
    }
    
    // Update BRI chart data
    const chartResponse = await fetch('/api/bri_chart');
    if (chartResponse.ok) {
      await cache.put('/api/bri_chart', chartResponse.clone());
    }
    
    console.log('Service Worker: Cached data updated');
  } catch (error) {
    console.error('Service Worker: Error updating cached data', error);
  }
}

// Handle messages from main thread
self.addEventListener('message', event => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'GET_VERSION') {
    event.ports[0].postMessage({ version: CACHE_NAME });
  }
});

// Periodic background sync (if supported)
self.addEventListener('periodicsync', event => {
  if (event.tag === 'data-sync') {
    console.log('Service Worker: Periodic sync triggered');
    event.waitUntil(updateCachedData());
  }
});
