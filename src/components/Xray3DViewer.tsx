import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RotateCcw, Play, Pause, Camera, Box, Grid3x3, CircleDot } from 'lucide-react'

interface Xray3DViewerProps {
  /** grayscale radiograph as data-url (from the scan result) */
  src: string
  height?: number
}

type RenderMode = 'surface' | 'wireframe' | 'points'
type Colormap = 'radiograph' | 'bone' | 'thermal'

const GRID = 160 // heightmap resolution

function lum(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255
}

// colormaps: tissue density 0..1 (1 = most radiodense) -> rgb 0..1
function colorRadiograph(v: number): [number, number, number] {
  const t = 1 - v // dark/dense -> bright cool; air -> dim
  return [
    Math.min(1, 0.06 + 0.82 * Math.pow(t, 1.6)),
    Math.min(1, 0.10 + 0.86 * Math.pow(t, 1.25)),
    Math.min(1, 0.16 + 0.90 * Math.pow(t, 0.95)),
  ]
}
function colorBone(v: number): [number, number, number] {
  const t = 1 - v
  return [0.08 + 0.88 * t, 0.10 + 0.80 * t, 0.14 + 0.70 * Math.pow(t, 1.3)]
}
function colorThermal(v: number): [number, number, number] {
  // inferno-like: dense tissue glows hot
  const t = Math.min(1, Math.max(0, 1 - v))
  const stops: Array<[number, number, number]> = [
    [0.06, 0.03, 0.17], [0.46, 0.05, 0.38], [0.74, 0.19, 0.34],
    [0.96, 0.56, 0.10], [0.99, 0.94, 0.56],
  ]
  const s = t * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(s))
  const f = s - i
  return [0, 1, 2].map(k => stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f) as [number, number, number]
}

function sampleImage(src: string): Promise<Float32Array | null> {
  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = GRID
      c.height = GRID
      const ctx = c.getContext('2d', { willReadFrequently: true })
      if (!ctx) return resolve(null)
      ctx.drawImage(img, 0, 0, GRID, GRID)
      const data = ctx.getImageData(0, 0, GRID, GRID).data
      const heights = new Float32Array(GRID * GRID)
      for (let y = 0; y < GRID; y++) {
        for (let x = 0; x < GRID; x++) {
          const i = (y * GRID + x) * 4
          heights[y * GRID + x] = lum(data[i], data[i + 1], data[i + 2])
        }
      }
      // light 3x3 box smooth to suppress sensor noise on the relief
      const sm = new Float32Array(GRID * GRID)
      for (let y = 1; y < GRID - 1; y++) {
        for (let x = 1; x < GRID - 1; x++) {
          let sum = 0
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) sum += heights[(y + dy) * GRID + (x + dx)]
          sm[y * GRID + x] = sum / 9
        }
      }
      resolve(sm)
    }
    img.onerror = () => resolve(null)
    img.src = src
  })
}

export function Xray3DViewer({ src, height = 380 }: Xray3DViewerProps) {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const dataRef = useRef<Float32Array | null>(null)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const [mode, setMode] = useState<RenderMode>('surface')
  const [colormap, setColormap] = useState<Colormap>('radiograph')
  const [depth, setDepth] = useState(0.55)
  const [autoRotate, setAutoRotate] = useState(true)

  // load + sample the radiograph whenever src changes
  useEffect(() => {
    let cancelled = false
    setReady(false)
    setFailed(false)
    sampleImage(src).then((data) => {
      if (cancelled) return
      if (!data) { setFailed(true); return }
      dataRef.current = data
      setReady(true)
    })
    return () => { cancelled = true }
  }, [src])

  // build the three.js scene (runs once per loaded image)
  useEffect(() => {
    if (!ready || !canvasHostRef.current || !dataRef.current) return
    const host = canvasHostRef.current

    let renderer: THREE.WebGLRenderer
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true })
    } catch (e) {
      console.error('[Xray3DViewer] WebGL init failed:', e)
      setFailed(true)
      return
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x05070b, 1)
    host.appendChild(renderer.domElement)
    renderer.domElement.style.width = '100%'
    renderer.domElement.style.height = '100%'
    renderer.domElement.style.display = 'block'
    renderer.domElement.style.touchAction = 'none'

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100)
    const HOME = { pos: new THREE.Vector3(0, -2.4, 2.2), target: new THREE.Vector3(0, 0, 0) }
    camera.position.copy(HOME.pos)
    camera.lookAt(HOME.target)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1.2
    controls.maxDistance = 8
    controls.autoRotateSpeed = 1.1

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.15)
    keyLight.position.set(1.5, -1.5, 2)
    scene.add(keyLight)
    const fillLight = new THREE.DirectionalLight(0xbfd4ff, 0.4)
    fillLight.position.set(-2, 1, 1)
    scene.add(fillLight)
    scene.add(new THREE.AmbientLight(0xffffff, 0.35))

    // ---- heightmap geometry ----
    const heights = dataRef.current
    const segs = GRID - 1
    const geometry = new THREE.PlaneGeometry(2, 2, segs, segs)
    geometry.rotateX(Math.PI / 2) // plane in XZ, up = +Y
    const pos = geometry.attributes.position as THREE.BufferAttribute
    const colorAttr = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
    geometry.setAttribute('color', colorAttr)

    const applyState = (depthScale: number, colormap: Colormap) => {
      const cmapFn = colormap === 'radiograph' ? colorRadiograph : colormap === 'bone' ? colorBone : colorThermal
      for (let i = 0; i < pos.count; i++) {
        const v = Math.pow(heights[i], 0.9)
        pos.setY(i, v * depthScale * 1.6 - depthScale * 0.8)
        const rgb = cmapFn(v)
        colorAttr.setXYZ(i, rgb[0], rgb[1], rgb[2])
      }
      pos.needsUpdate = true
      colorAttr.needsUpdate = true
      geometry.computeVertexNormals()
    }
    applyState(depthBuildRef.current, colormapBuildRef.current)

    const solidMat: THREE.MeshStandardMaterial | THREE.MeshBasicMaterial = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      metalness: 0.15,
      roughness: 0.65,
    })
    const wireMat = new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true, transparent: true, opacity: 0.7 })
    const pointMat = new THREE.PointsMaterial({ vertexColors: true, size: 0.014 })

    const mesh = new THREE.Mesh<THREE.BufferGeometry, THREE.Material>(geometry, solidMat)
    scene.add(mesh)
    const points = new THREE.Points(geometry, pointMat)
    points.visible = false
    scene.add(points)

    const setMode = (m: RenderMode) => {
      mesh.visible = m !== 'points'
      mesh.material = m === 'wireframe' ? wireMat : solidMat
      points.visible = m === 'points'
    }


    const resize = () => {
      const w = host.clientWidth, h = host.clientHeight
      if (!w || !h) return
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(host)
    resize()

    // imperative hooks for the react handlers
    ;(host as any).__stethos3d = {
      applyState,
      setMode,
      setAutoRotate: (on: boolean) => { controls.autoRotate = on },
      reset: () => {
        camera.position.copy(HOME.pos)
        controls.target.copy(HOME.target)
        controls.update()
      },
      snapshot: () => {
        renderer.render(scene, camera)
        const a = document.createElement('a')
        a.href = renderer.domElement.toDataURL('image/png')
        a.download = 'stethos-xray-3d.png'
        a.click()
      },
    }

    let raf = 0
    const tick = () => {
      raf = requestAnimationFrame(tick)
      controls.update()
      renderer.render(scene, camera)
    }
    tick()

    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      solidMat.dispose()
      wireMat.dispose()
      pointMat.dispose()
      geometry.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === host) host.removeChild(renderer.domElement)
      delete (host as any).__stethos3d
    }
  }, [ready])

  // refs so the scene-builder picks up the current controls at build time
  const depthBuildRef = useRef(depth)
  depthBuildRef.current = depth
  const colormapBuildRef = useRef(colormap)
  colormapBuildRef.current = colormap

  // imperative wiring — push mode/depth/colormap/autorotate into the live scene
  useEffect(() => {
    const api = (canvasHostRef.current as any)?.__stethos3d
    if (api) api.applyState(depth, colormap)
  }, [depth, colormap, ready])
  useEffect(() => {
    const api = (canvasHostRef.current as any)?.__stethos3d
    if (api) api.setMode(mode)
  }, [mode, ready])
  useEffect(() => {
    const api = (canvasHostRef.current as any)?.__stethos3d
    if (api) api.setAutoRotate(autoRotate)
  }, [autoRotate, ready])

  if (!ready && !failed) {
    return (
      <div className="inset-well flex items-center justify-center" style={{ height, borderRadius: 4 }}>
        <p className="mono text-[10.5px] text-[var(--text-faint)]">building 3d volume…</p>
      </div>
    )
  }
  if (failed) {
    return (
      <div className="inset-well flex items-center justify-center" style={{ height, borderRadius: 4 }}>
        <p className="mono text-[10.5px] text-[var(--text-faint)]">3d view unavailable · re-run scan to enable</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden" style={{ border: '1px solid var(--border)', borderRadius: 4, background: '#05070b' }}>
        <div ref={canvasHostRef} style={{ height, width: '100%' }} />
        <div className="absolute top-2 right-2 bg-black/70 px-2 py-0.5" style={{ borderRadius: 3 }}>
          <p className="mono text-[9.5px] text-gray-400">3d relief · {GRID}² samples</p>
        </div>
        <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-0.5" style={{ borderRadius: 3 }}>
          <p className="mono text-[9.5px] text-gray-400">drag rotate · scroll zoom</p>
        </div>
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        {/* render mode */}
        <div className="flex" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {([
            ['surface', Box], ['wireframe', Grid3x3], ['points', CircleDot],
          ] as Array<[RenderMode, React.ComponentType<{ className?: string }>]>).map(([m, Icon]) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className="px-2 py-1 flex items-center gap-1 mono text-[10px]"
              style={{
                background: mode === m ? 'var(--accent-dim)' : 'transparent',
                color: mode === m ? 'var(--accent)' : 'var(--text-faint)',
              }}
            >
              <Icon className="w-3 h-3" /> {m}
            </button>
          ))}
        </div>
        {/* colormap */}
        <div className="flex mono text-[10px]" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {(['radiograph', 'bone', 'thermal'] as Colormap[]).map(c => (
            <button
              key={c}
              onClick={() => setColormap(c)}
              className="px-2 py-1"
              style={{
                background: colormap === c ? 'var(--accent-dim)' : 'transparent',
                color: colormap === c ? 'var(--accent)' : 'var(--text-faint)',
              }}
            >
              {c}
            </button>
          ))}
        </div>
        {/* depth */}
        <label className="flex items-center gap-1.5 mono text-[10px] text-[var(--text-faint)]">
          depth
          <input
            type="range" min={0.1} max={1.4} step={0.05} value={depth}
            onChange={(e) => setDepth(parseFloat(e.target.value))}
            className="w-20"
          />
        </label>
        {/* auto rotate */}
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: autoRotate ? 'var(--accent)' : 'var(--text-faint)' }}
        >
          {autoRotate ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />} {autoRotate ? 'spin on' : 'spin off'}
        </button>
        {/* reset */}
        <button
          onClick={() => { const api = (canvasHostRef.current as any)?.__stethos3d; if (api) api.reset() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          <RotateCcw className="w-3 h-3" /> reset view
        </button>
        {/* snapshot */}
        <button
          onClick={() => { const api = (canvasHostRef.current as any)?.__stethos3d; if (api) api.snapshot() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          <Camera className="w-3 h-3" /> save png
        </button>
      </div>
    </div>
  )
}
