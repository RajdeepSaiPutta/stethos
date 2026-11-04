import React from 'react'
import ReactDOM from 'react-dom/client'
import { App as RespiratoryApp } from './App'
import { initI18n } from './lib/i18n'
import './index.css'

// Initialize i18n
initI18n().then(() => {
  // Register service worker for PWA
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').then(
        (registration) => console.log('SW registered:', registration.scope),
        (error) => console.log('SW registration failed:', error)
      );
    });
  }

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <RespiratoryApp />
    </React.StrictMode>
  )
})