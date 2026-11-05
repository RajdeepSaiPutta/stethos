import React from 'react'
import ReactDOM from 'react-dom/client'
import { App as RespiratoryApp } from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { initI18n } from './lib/i18n'
import './index.css'

// Initialize i18n, but never let it hold the UI hostage: a failed init used to
// leave the window blank because render() only ran inside .then().
initI18n().catch((error) => {
  console.error('[i18n] init failed, continuing with default strings:', error)
})

// Register service worker for PWA
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').then(
      (registration) => console.log('SW registered:', registration.scope),
      (error) => console.log('SW registration failed:', error)
    )
  })
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <RespiratoryApp />
    </ErrorBoundary>
  </React.StrictMode>
)
