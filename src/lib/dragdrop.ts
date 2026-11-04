import { getCurrentWebview, type DragDropEvent } from '@tauri-apps/api/webview'
import type { Event } from '@tauri-apps/api/event'

// Tauri shell detection. The shell is required for both invoke() bridges and
// the native drag-drop events below; browser preview (vite) has neither.
export const INSIDE_TAURI =
  typeof window !== 'undefined' &&
  ('__TAURI_INTERNALS__' in window || '__TAURI__' in window)

// On WebKitGTK (linux) the injected File.path is not present on HTML5 drop
// events, and both the DOM drop handler and the native drag-drop event can
// fire for the same gesture. Dedupe: whichever fires first marks the drop and
// the other skips for a short window.
const DEDUPE_WINDOW_MS = 2000
let lastHandledDrop: { name: string; at: number } = { name: '', at: 0 }

export function markDropHandled(name: string) {
  lastHandledDrop = { name, at: Date.now() }
}

export function isDuplicateDrop(name: string): boolean {
  return (
    !!name &&
    lastHandledDrop.name === name &&
    Date.now() - lastHandledDrop.at < DEDUPE_WINDOW_MS
  )
}

export function basenameOfPath(path: string): string {
  const i = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'))
  return i >= 0 ? path.slice(i + 1) : path
}

/**
 * Subscribe to the webview's native drag-drop events, which carry real file
 * system paths on every platform (unlike DOM dataTransfer on WebKitGTK).
 *
 * Returns an unlisten function. No-op outside Tauri.
 */
export function subscribeTauriDragDrop(
  onDrop: (paths: string[]) => void,
  onHighlight?: (active: boolean) => void,
): () => void {
  if (!INSIDE_TAURI) return () => {}
  let unlisten: (() => void) | undefined
  let cancelled = false
  getCurrentWebview()
    .onDragDropEvent((event: Event<DragDropEvent>) => {
      if (event.payload.type === 'enter' || event.payload.type === 'over') {
        onHighlight?.(true)
      } else if (event.payload.type === 'leave') {
        onHighlight?.(false)
      } else if (event.payload.type === 'drop') {
        onHighlight?.(false)
        const paths = event.payload.paths || []
        const name = paths.length > 0 ? basenameOfPath(paths[0]) : ''
        if (paths.length > 0 && !isDuplicateDrop(name)) {
          markDropHandled(name)
          onDrop(paths)
        }
      }
    })
    .then((u) => {
      if (cancelled) u()
      else unlisten = u
    })
    .catch((e) => {
      // eslint-disable-next-line no-console
      console.warn('[dragdrop] drag-drop subscription failed:', e)
    })
  return () => {
    cancelled = true
    unlisten?.()
  }
}