import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  RotateCcw, Play, Pause, Camera, Box, Grid3x3, CircleDot,
  Contrast, Sun, Maximize2, Minimize2, Crosshair, Ruler,
  HelpCircle, Eye
} from 'lucide-react'

interface Xray3DViewerProps {
  /** grayscale radiograph as data-url (from the scan result) */
  src: string
  height?: number
}

type RenderMode = 'surface' | 'wireframe' | 'points'
type Colormap = 'radiograph' | 'bone' | 'thermal' | 'hot-iron' | 'mammography' | 'inverted'

const GRID = 160 // heightmap resolution

function lum(r: number, g: number, b: number): number {
  return (0.299 * r + 0.587 * g + 0.114 * b) / 255
}

// colormaps: tissue density 0..1 (1 = most radiodense) -> rgb 0..1
function colorRadiograph(v: number): [number, number, number] {
  const t = 1 - v
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
function colorHotIron(v: number): [number, number, number] {
  const t = Math.min(1, Math.max(0, 1 - v))
  if (t < 0.25) return [4 * t, 0, 0]
  if (t < 0.5) return [1, 4 * (t - 0.25), 0]
  if (t < 0.75) return [1, 1, 4 * (t - 0.5)]
  return [1, 1, 1]
}
function colorMammography(v: number): [number, number, number] {
  const t = 1 - v
  const enhanced = Math.pow(t, 0.7)
  return [enhanced, enhanced, enhanced]
}
function colorInverted(v: number): [number, number, number] {
  return [v, v, v]
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

interface Measurement {
  id: string
  type: 'distance' | 'angle' | 'roi'
  points: Array<{x: number, y: number, z: number}>
  value: number
  unit: string
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
  
  // Window/Level (Brightness/Contrast) controls
  const [windowCenter, setWindowCenter] = useState(0.5)
  const [windowWidth, setWindowWidth] = useState(1.0)
  const [inverted, setInverted] = useState(false)
  
  // Measurement tools
  const [measurements, setMeasurements] = useState<Array<{
    id: string
    type: 'distance' | 'angle' | 'roi'
    points: Array<{x: number, y: number, z: number}>
    value: number
    unit: string
  }>>([])
  const [activeTool, setActiveTool] = useState<'none' | 'distance' | 'angle' | 'roi'>('none')
  
  // View state
  const [showHistogram, setShowHistogram] = useState(false)
  const [showCrosshair, setShowCrosshair] = useState(false)
  const [crosshairPos, setCrosshairPos] = useState({x: 0, y: 0})
  const [showFullscreen, setShowFullscreen] = useState(false)
  
  // Presets
  const presets = [
    { name: 'Lung', center: 0.3, width: 0.6 },
    { name: 'Mediastinum', center: 0.5, width: 0.4 },
    { name: 'Bone', center: 0.7, width: 0.3 },
    { name: 'Soft Tissue', center: 0.5, width: 0.8 },
  ]

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
    geometry.rotateX(Math.PI / 2)
    const pos = geometry.attributes.position as THREE.BufferAttribute
    const colorAttr = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
    geometry.setAttribute('color', colorAttr)

    const applyState = (depthScale: number, colormap: Colormap, wCenter: number, wWidth: number, inv: boolean) => {
      let cmapFn: (v: number) => [number, number, number]
      switch (colormap) {
        case 'radiograph': cmapFn = colorRadiograph; break
        case 'bone': cmapFn = colorBone; break
        case 'thermal': cmapFn = colorThermal; break
        case 'hot-iron': cmapFn = colorHotIron; break
        case 'mammography': cmapFn = colorMammography; break
        case 'inverted': cmapFn = colorInverted; break
        default: cmapFn = colorRadiograph
      }
      
      const halfWidth = wWidth / 2
      const minV = wCenter - halfWidth
      const maxV = wCenter + halfWidth
      
      for (let i = 0; i < pos.count; i++) {
        const rawV = Math.pow(heights[i], 0.9)
        
        let v: number
        if (wWidth >= 1.0) {
          v = rawV
        } else {
          v = (rawV - minV) / (maxV - minV)
          v = Math.max(0, Math.min(1, v))
        }
        
        if (inv) v = 1 - v
        
        pos.setY(i, v * depthScale * 1.6 - depthScale * 0.8)
        const rgb = cmapFn(v)
        colorAttr.setXYZ(i, rgb[0], rgb[1], rgb[2])
      }
      pos.needsUpdate = true
      colorAttr.needsUpdate = true
      geometry.computeVertexNormals()
    }
    applyState(depth, colormap, windowCenter, windowWidth, inverted)

    const solidMat = new THREE.MeshStandardMaterial({
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

    const crosshairGeometry = new THREE.BufferGeometry()
    const crosshairMaterial = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.8 })
    const crosshairSize = 0.15
    const crosshairPoints = new Float32Array([
      -crosshairSize, 0, 0, crosshairSize, 0, 0,
      0, -crosshairSize, 0, 0, crosshairSize, 0,
      0, 0, -crosshairSize, 0, 0, crosshairSize
    ])
    crosshairGeometry.setAttribute('position', new THREE.Float32BufferAttribute(crosshairPoints, 3))
    const crosshair = new THREE.LineSegments(crosshairGeometry, crosshairMaterial)
    crosshair.visible = false
    scene.add(crosshair)

    const measurementLines = new THREE.Group()
    scene.add(measurementLines)

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
      setWindowLevel: (center: number, width: number) => { setWindowCenter(center); setWindowWidth(width) },
      setInverted: (inv: boolean) => { setInverted(inv) },
      setCrosshair: (visible: boolean, pos?: {x: number, y: number}) => {
        crosshair.visible = visible
        if (pos) {
          crosshair.position.set(pos.x, pos.y, 0)
        }
      },
      addMeasurement: (type: 'distance' | 'angle' | 'roi', points: Array<{x: number, y: number, z: number}>) => {
        const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
        let value = 0, unit = ''
        if (type === 'distance' && points.length >= 2) {
          const dx = points[1].x - points[0].x
          const dy = points[1].y - points[0].y
          const dz = points[1].z - points[0].z
          value = Math.sqrt(dx*dx + dy*dy + dz*dz) * 100
          unit = 'mm'
        } else if (type === 'angle' && points.length >= 3) {
          const v1 = new THREE.Vector3(points[0].x - points[1].x, points[0].y - points[1].y, points[0].z - points[1].z)
          const v2 = new THREE.Vector3(points[2].x - points[1].x, points[2].y - points[1].y, points[2].z - points[1].z)
          value = v1.angleTo(v2) * 180 / Math.PI
          unit = '°'
        }
        setMeasurements((prev: Measurement[]) => [...prev, { id, type, points, value, unit }])
      },
      clearMeasurements: () => {
        setMeasurements([])
        measurementLines.clear()
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
      crosshairGeometry.dispose()
      crosshairMaterial.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === host) host.removeChild(renderer.domElement)
      delete (host as any).__stethos3d
    }
  }, [ready])

  const depthBuildRef = useRef(depth)
  depthBuildRef.current = depth
  const colormapBuildRef = useRef(colormap)
  colormapBuildRef.current = colormap

  useEffect(() => {
    const api = (canvasHostRef.current as any)?.__stethos3d
    if (api) api.applyState(depth, colormap, windowCenter, windowWidth, inverted)
  }, [depth, colormap, windowCenter, windowWidth, inverted, ready])
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

  const histogramData = useRef<Array<number>>([])
  if (dataRef.current && histogramData.current.length === 0) {
    const hist = new Array(256).fill(0)
    for (let i = 0; i < dataRef.current.length; i++) {
      const v = Math.floor(dataRef.current[i] * 255)
      hist[Math.min(255, Math.max(0, v))]++
    }
    histogramData.current = hist
  }

  const maxHist = Math.max(...histogramData.current)

  return (
    <div className="space-y-2">
      <div className="relative overflow-hidden" style={{ border: '1px solid var(--border)', borderRadius: 4, background: '#05070b' }}>
        <div ref={canvasHostRef} style={{ height, width: '100%' }} 
             onMouseMove={(e) => {
               const rect = e.currentTarget.getBoundingClientRect()
               setCrosshairPos({ x: e.clientX - rect.left, y: e.clientY - rect.top })
             }}
        >
          {showCrosshair && (
            <div className="absolute pointer-events-none" style={{
              left: crosshairPos.x - 10, top: crosshairPos.y - 10,
              width: 20, height: 20, border: '1px solid #00ffff',
              borderRadius: '50%', pointerEvents: 'none', zIndex: 10
            }}>
              <div style={{position: 'absolute', left: '50%', top: 0, width: '1px', height: '100%', background: '#00ffff', transform: 'translateX(-50%)'}} />
              <div style={{position: 'absolute', top: '50%', left: 0, width: '100%', height: '1px', background: '#00ffff', transform: 'translateY(-50%)'}} />
            </div>
          )}
        </div>
        <div className="absolute top-2 right-2 bg-black/70 px-2 py-0.5" style={{ borderRadius: 3 }}>
          <p className="mono text-[9.5px] text-gray-400">3d relief · {GRID}² samples</p>
        </div>
        <div className="absolute bottom-2 left-2 bg-black/70 px-2 py-0.5" style={{ borderRadius: 3 }}>
          <p className="mono text-[9.5px] text-gray-400">drag rotate · scroll zoom · right-click pan</p>
        </div>
      </div>
      
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
        {/* Window/Level Presets */}
        <div className="flex items-center gap-1.5">
          <span className="mono text-[10px] text-[var(--text-faint)]">W/L:</span>
          {presets.map(p => (
            <button
              key={p.name}
              onClick={() => { setWindowCenter(p.center); setWindowWidth(p.width) }}
              className="px-2 py-1 text-[9px] mono rounded-[3px] transition-colors"
              style={{
                background: Math.abs(windowCenter - p.center) < 0.01 ? 'var(--accent-dim)' : 'transparent',
                color: Math.abs(windowCenter - p.center) < 0.01 ? 'var(--accent)' : 'var(--text-faint)',
                border: '1px solid var(--border)',
                borderRadius: 3,
              }}
              title={p.name}
            >
              {p.name}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Window/Level Sliders */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1 mono text-[10px] text-[var(--text-faint)]">
            <Contrast className="w-3 h-3" /> WC
            <input
              type="range" min={0} max={1} step={0.01} value={windowCenter}
              onChange={(e) => setWindowCenter(parseFloat(e.target.value))}
              className="w-24"
            />
          </label>
          <label className="flex items-center gap-1 mono text-[10px] text-[var(--text-faint)]">
            WW
            <input
              type="range" min={0.1} max={1} step={0.01} value={windowWidth}
              onChange={(e) => setWindowWidth(parseFloat(e.target.value))}
              className="w-24"
            />
          </label>
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Invert */}
        <button
          onClick={() => setInverted(!inverted)}
          className={`px-2 py-1 text-[10px] mono flex items-center gap-1 rounded-[3px] transition-colors ${inverted ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}
        >
          {inverted ? <Sun className="w-3 h-3" /> : <Contrast className="w-3 h-3" />} {inverted ? 'normal' : 'invert'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Colormap */}
        <div className="flex mono text-[10px]" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {(['radiograph', 'bone', 'thermal', 'hot-iron', 'mammography', 'inverted'] as Colormap[]).map(c => (
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

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Render Mode */}
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

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Depth */}
        <label className="flex items-center gap-1.5 mono text-[10px] text-[var(--text-faint)]">
          depth
          <input
            type="range" min={0.1} max={1.4} step={0.05} value={depth}
            onChange={(e) => setDepth(parseFloat(e.target.value))}
            className="w-20"
          />
        </label>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Auto Rotate */}
        <button
          onClick={() => setAutoRotate(!autoRotate)}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: autoRotate ? 'var(--accent)' : 'var(--text-faint)' }}
        >
          {autoRotate ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />} {autoRotate ? 'spin on' : 'spin off'}
        </button>

        {/* Reset View */}
        <button
          onClick={() => { const api = (canvasHostRef.current as any)?.__stethos3d; if (api) api.reset() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          <RotateCcw className="w-3 h-3" /> reset
        </button>

        {/* Snapshot */}
        <button
          onClick={() => { const api = (canvasHostRef.current as any)?.__stethos3d; if (api) api.snapshot() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          <Camera className="w-3 h-3" /> save
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Measurement Tools */}
        <div className="flex items-center gap-1" style={{ border: '1px solid var(--border)', borderRadius: 3, background: activeTool !== 'none' ? 'var(--accent-dim)' : 'transparent' }}>
          {(['distance', 'angle', 'roi'] as Array<'distance' | 'angle' | 'roi'>).map(t => (
            <button
              key={t}
              onClick={() => setActiveTool(activeTool === t ? 'none' : t)}
              className="px-2 py-1 flex items-center gap-1 mono text-[10px] rounded-l"
              style={{
                background: activeTool === t ? 'var(--accent)' : 'transparent',
                color: activeTool === t ? 'white' : 'var(--text-faint)',
                borderRight: t !== 'roi' ? '1px solid var(--border)' : 'none',
                borderRadius: t === 'distance' ? '3px 0 0 3px' : t === 'roi' ? '0 3px 3px 0' : 0,
              }}
              title={t}
            >
              {t === 'distance' && <Ruler className="w-3 h-3" />}
              {t === 'angle' && <Maximize2 className="w-3 h-3" />}
              {t === 'roi' && <CircleDot className="w-3 h-3" />}
              <span className="hidden sm:inline">{t}</span>
            </button>
          ))}
        </div>

        {activeTool !== 'none' && (
          <button
            onClick={() => { setActiveTool('none') }}
            className="px-2 py-1 mono text-[10px] flex items-center gap-1 text-[var(--bad)]"
            style={{ border: '1px solid var(--bad)', borderRadius: 3 }}
          >
            <Eye className="w-3 h-3" /> cancel
          </button>
        )}

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Crosshair Toggle */}
        <button
          onClick={() => setShowCrosshair(!showCrosshair)}
          className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${showCrosshair ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}
        >
          <Crosshair className="w-3 h-3" /> {showCrosshair ? 'on' : 'off'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Fullscreen */}
        <button
          onClick={() => setShowFullscreen(!showFullscreen)}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          {showFullscreen ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />} {showFullscreen ? 'exit fs' : 'fullscreen'}
        </button>

        {/* Help */}
        <button
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}
        >
          <HelpCircle className="w-3 h-3" /> help
        </button>
      </div>

      {/* Histogram */}
      {showHistogram && (
        <div className="p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
          <div className="flex items-center justify-between mb-2">
            <span className="mono text-[10px] text-[var(--text-faint)]">Histogram</span>
            <button onClick={() => setShowHistogram(false)} className="text-[var(--text-faint)] hover:text-[var(--text)]">×</button>
          </div>
          <div className="flex items-end h-24 gap-[1px]" style={{ background: 'var(--inset)', borderRadius: 3 }}>
            {histogramData.current.map((count, i) => (
              <div key={i} className="flex-1" style={{
                height: maxHist > 0 ? `${(count / maxHist) * 100}%` : 0,
                background: `hsl(\${200 + i * 0.5}, 70%, 50%)`,
                transition: 'height 0.1s',
              }} />
            ))}
          </div>
        </div>
      )}

      {/* Measurements List */}
      {measurements.length > 0 && (
        <div className="p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
          <div className="flex items-center justify-between mb-2">
            <span className="mono text-[10px] text-[var(--text-faint)]">Measurements ({measurements.length})</span>
            <button onClick={() => { const api = (canvasHostRef.current as any)?.__stethos3d; if (api) api.clearMeasurements() }} className="text-[var(--text-faint)] hover:text-[var(--text)] text-xs">clear all</button>
          </div>
          <div className="space-y-1 max-h-32 overflow-auto">
            {measurements.map(m => (
              <div key={m.id} className="flex items-center justify-between p-1.5 bg-[var(--inset)] rounded-[3px] text-[10px] mono">
                <span>{m.type}:</span>
                <span className="text-[var(--accent)]">{m.value.toFixed(1)} {m.unit}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
