import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  RotateCcw, Play, Pause, Camera, Box, Grid3x3, CircleDot,
  Contrast, Sun, Maximize2, Minimize2, Crosshair, Ruler,
  HelpCircle, Eye, Layout, Link, Navigation, Activity
} from 'lucide-react'

interface Xray3DViewerProps {
  src: string
  height?: number
  segmentationOverlay?: string
  onMeasurement?: (m: { type: string; value: number; unit: string }) => void
}

type RenderMode = 'surface' | 'wireframe' | 'points' | 'volume'
type Colormap = 'radiograph' | 'bone' | 'thermal' | 'hot-iron' | 'mammography' | 'inverted'
type ViewMode = 'quad' | '3d' | 'axial' | 'sagittal' | 'coronal'

const GRID = 128 // volume resolution

// one render layer per viewport so each camera only draws its own geometry
const LAYER = { AXIAL: 1, SAGITTAL: 2, CORONAL: 3, VOLUME: 4 } as const

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
      // Build 3D volume by extruding with noise variation (simulated slices)
      const volume = new Float32Array(GRID * GRID * GRID)
      for (let z = 0; z < GRID; z++) {
        const sliceVariation = Math.sin(z / GRID * Math.PI * 2) * 0.02
        for (let y = 0; y < GRID; y++) {
          for (let x = 0; x < GRID; x++) {
            const base = heights[y * GRID + x]
            volume[z * GRID * GRID + y * GRID + x] = Math.max(0, Math.min(1, base + sliceVariation))
          }
        }
      }
      resolve(volume)
    }
    img.onerror = () => resolve(null)
    img.src = src
  })
}

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl') || canvas.getContext('experimental-webgl')
    if (!gl) return false
    const lose = (gl as WebGLRenderingContext).getExtension('WEBGL_lose_context')
    if (lose) lose.loseContext()
    return true
  } catch {
    return false
  }
}

interface Viewport {
  id: string
  label: string
  type: 'axial' | 'sagittal' | 'coronal' | '3d'
  camera: THREE.OrthographicCamera | THREE.PerspectiveCamera
  sliceIndex: number
  mesh?: THREE.Mesh
  crosshair?: THREE.LineSegments
  orientationCube?: THREE.Group
}

export function Xray3DViewer({ 
  src, 
  height = 600, 
  onMeasurement 
}: Xray3DViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const volumeRef = useRef<Float32Array | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('3d')
  const [mode, setMode] = useState<RenderMode>('volume')
  const [colormap, setColormap] = useState<Colormap>('radiograph')
  const [depth, setDepth] = useState(0.55)
  const [autoRotate, setAutoRotate] = useState(true)
  
  // Window/Level
  const [windowCenter, setWindowCenter] = useState(0.5)
  const [windowWidth, setWindowWidth] = useState(1.0)
  const [inverted, setInverted] = useState(false)
  
  // MPR slice indices
  const [axialSlice, setAxialSlice] = useState(Math.floor(GRID / 2))
  const [sagittalSlice, setSagittalSlice] = useState(Math.floor(GRID / 2))
  const [coronalSlice, setCoronalSlice] = useState(Math.floor(GRID / 2))
  
  // Tools
  const [activeTool, setActiveTool] = useState<'none' | 'distance' | 'angle' | 'roi' | 'window' | 'pan' | 'zoom'>('none')
  const [measurements, setMeasurements] = useState<Array<{
    id: string
    type: 'distance' | 'angle' | 'roi'
    points: Array<{x: number, y: number, z: number}>
    value: number
    unit: string
    viewport: string
  }>>([])
  
  // UI state
  const [showHistogram, setShowHistogram] = useState(false)
  const [showCrosshair, setShowCrosshair] = useState(true)
  const [showOrientation, setShowOrientation] = useState(true)
  const [showFullscreen, setShowFullscreen] = useState(false)
  const [linkedCursor, setLinkedCursor] = useState(true)

  const presets = [
    { name: 'Lung', center: 0.3, width: 0.6 },
    { name: 'Mediastinum', center: 0.5, width: 0.4 },
    { name: 'Bone', center: 0.7, width: 0.3 },
    { name: 'Soft Tissue', center: 0.5, width: 0.8 },
  ]

  // Histogram
  const histogramData = useMemo(() => {
    const hist = new Array(256).fill(0)
    const data = volumeRef.current
    if (data) {
      for (let i = 0; i < data.length; i++) {
        const v = Math.floor(data[i] * 255)
        hist[Math.min(255, Math.max(0, v))]++
      }
    }
    return hist
  }, [ready, src])

  // Load volume
  useEffect(() => {
    let cancelled = false
    setReady(false)
    setFailed(false)
    sampleImage(src).then((data) => {
      if (cancelled) return
      if (!data) { setFailed(true); return }
      volumeRef.current = data
      setReady(true)
    })
    return () => { cancelled = true }
  }, [src])

  // Initialize viewports
  useEffect(() => {
    if (!ready || !containerRef.current) return

    const container = containerRef.current
    if (!webglAvailable()) {
      setFailed(true)
      return
    }

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
      failIfMajorPerformanceCaveat: false,
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x05070b, 1)
    renderer.domElement.style.position = 'absolute'
    renderer.domElement.style.top = '0'
    renderer.domElement.style.left = '0'
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Create 4 viewports for quad view
    const viewports: Viewport[] = [
      {
        id: 'axial',
        label: 'Axial',
        type: 'axial',
        camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10),
        sliceIndex: axialSlice,
      },
      {
        id: 'sagittal',
        label: 'Sagittal',
        type: 'sagittal',
        camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10),
        sliceIndex: sagittalSlice,
      },
      {
        id: 'coronal',
        label: 'Coronal',
        type: 'coronal',
        camera: new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10),
        sliceIndex: coronalSlice,
      },
      {
        id: '3d',
        label: '3D Volume',
        type: '3d',
        camera: new THREE.PerspectiveCamera(45, 1, 0.01, 100),
        sliceIndex: Math.floor(GRID / 2),
      },
    ]

    // Setup cameras
    viewports[0].camera.position.set(0, 2, 0.001)
    viewports[0].camera.lookAt(0, 0, 0)
    viewports[0].camera.up.set(0, 0, -1)

    viewports[1].camera.position.set(2, 0, 0)
    viewports[1].camera.lookAt(0, 0, 0)
    viewports[1].camera.up.set(0, 1, 0)

    viewports[2].camera.position.set(0, 0, 2)
    viewports[2].camera.lookAt(0, 0, 0)
    viewports[2].camera.up.set(0, 1, 0)

    viewports[3].camera.position.set(2.0, 3.4, 2.8)
    viewports[3].camera.lookAt(0, 0, 0)

    // Each camera draws only its own render layer (prevents every viewport
    // from drawing every object, which turned the quad view into soup).
    viewports[0].camera.layers.set(LAYER.AXIAL)
    viewports[1].camera.layers.set(LAYER.SAGITTAL)
    viewports[2].camera.layers.set(LAYER.CORONAL)
    viewports[3].camera.layers.set(LAYER.VOLUME)

    const scene = new THREE.Scene()
    
    // Lights (visible to every viewport layer)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.15)
    keyLight.position.set(1.5, -1.5, 2)
    keyLight.layers.enableAll()
    scene.add(keyLight)
    const fillLight = new THREE.DirectionalLight(0xbfd4ff, 0.4)
    fillLight.position.set(-2, 1, 1)
    fillLight.layers.enableAll()
    scene.add(fillLight)
    const ambient = new THREE.AmbientLight(0xffffff, 0.35)
    ambient.layers.enableAll()
    scene.add(ambient)

    // Create slice geometry for MPR views
    const createSliceGeometry = (plane: 'axial' | 'sagittal' | 'coronal') => {
      const geom = new THREE.PlaneGeometry(2, 2, 1, 1)
      if (plane === 'axial') {
        geom.rotateX(-Math.PI / 2)   // horizontal (XZ), normal +Y
      } else if (plane === 'sagittal') {
        geom.rotateY(Math.PI / 2)    // vertical (YZ), normal +X
      }                              // coronal stays XY, normal +Z
      return geom
    }

    // Create slice meshes for each viewport
    const sliceMeshes: Map<string, THREE.Mesh> = new Map()
    const sliceMaterials: Map<string, THREE.MeshBasicMaterial> = new Map()

    const colormapFns: Record<Colormap, (v: number) => [number, number, number]> = {
      radiograph: colorRadiograph,
      bone: colorBone,
      thermal: colorThermal,
      'hot-iron': colorHotIron,
      mammography: colorMammography,
      inverted: colorInverted,
    }

    const applySliceTexture = (viewportId: string, sliceIdx: number) => {
      const material = sliceMaterials.get(viewportId)
      if (!material || !volumeRef.current) return

      const canvas = document.createElement('canvas')
      canvas.width = GRID
      canvas.height = GRID
      const ctx = canvas.getContext('2d', { willReadFrequently: true })
      if (!ctx) return

      const imageData = ctx.createImageData(GRID, GRID)
      const data = volumeRef.current
      const halfWidth = windowWidth / 2
      const minV = windowCenter - halfWidth
      const maxV = windowCenter + halfWidth
      const cmapFn = colormapFns[colormap]

      let sliceData: Float32Array
      if (viewportId === 'axial') {
        sliceData = new Float32Array(GRID * GRID)
        for (let y = 0; y < GRID; y++) {
          for (let x = 0; x < GRID; x++) {
            sliceData[y * GRID + x] = data![sliceIdx * GRID * GRID + y * GRID + x]
          }
        }
      } else if (viewportId === 'sagittal') {
        sliceData = new Float32Array(GRID * GRID)
        for (let z = 0; z < GRID; z++) {
          for (let y = 0; y < GRID; y++) {
            sliceData[y * GRID + z] = data![z * GRID * GRID + y * GRID + sliceIdx]
          }
        }
      } else {
        sliceData = new Float32Array(GRID * GRID)
        for (let z = 0; z < GRID; z++) {
          for (let x = 0; x < GRID; x++) {
            sliceData[x * GRID + z] = data![z * GRID * GRID + sliceIdx * GRID + x]
          }
        }
      }

      for (let i = 0; i < sliceData.length; i++) {
        let v = sliceData[i]
        if (windowWidth < 1.0) {
          v = (v - minV) / (maxV - minV)
          v = Math.max(0, Math.min(1, v))
        }
        if (inverted) v = 1 - v
        const rgb = cmapFn(v)
        imageData.data[i * 4] = Math.round(rgb[0] * 255)
        imageData.data[i * 4 + 1] = Math.round(rgb[1] * 255)
        imageData.data[i * 4 + 2] = Math.round(rgb[2] * 255)
        imageData.data[i * 4 + 3] = 255
      }
      ctx.putImageData(imageData, 0, 0)
      material.map?.dispose()
      material.map = new THREE.CanvasTexture(canvas)
      material.map.needsUpdate = true
    }

    // Create slice meshes
    viewports.slice(0, 3).forEach((vp, idx) => {
      const planes = ['axial', 'sagittal', 'coronal'] as const
      const plane = planes[idx]
      const geom = createSliceGeometry(plane)
      const material = new THREE.MeshBasicMaterial({ 
        transparent: true, 
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      sliceMaterials.set(vp.id, material)
      const mesh = new THREE.Mesh(geom, material)
      mesh.visible = viewMode === 'quad' || viewMode === vp.id
      mesh.layers.set(plane === 'axial' ? LAYER.AXIAL : plane === 'sagittal' ? LAYER.SAGITTAL : LAYER.CORONAL)
      scene.add(mesh)
      sliceMeshes.set(vp.id, mesh)
      vp.mesh = mesh
      // Initial texture
      if (plane === 'axial') applySliceTexture(vp.id, axialSlice)
      else if (plane === 'sagittal') applySliceTexture(vp.id, sagittalSlice)
      else applySliceTexture(vp.id, coronalSlice)
    })

    // 3D Volume rendering
    let volumeMesh: THREE.Mesh | null = null
    let volumePoints: THREE.Points | null = null
    let wireframeMesh: THREE.Mesh | null = null

    if (viewMode === '3d' || viewMode === 'quad') {
      const geometry = new THREE.PlaneGeometry(2, 2, GRID - 1, GRID - 1)
      geometry.rotateX(-Math.PI / 2)
      const pos = geometry.attributes.position as THREE.BufferAttribute
      const colorAttr = new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3)
      geometry.setAttribute('color', colorAttr)

      const applyVolumeState = (depthScale: number, colormap: Colormap, wCenter: number, wWidth: number, inv: boolean) => {
        const heights = volumeRef.current
        if (!heights) return
        const cmapFn = colormapFns[colormap]
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
      applyVolumeState(depth, colormap, windowCenter, windowWidth, inverted)

      const solidMat = new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, metalness: 0.15, roughness: 0.65 })
      const wireMat = new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true, transparent: true, opacity: 0.7 })
      const pointMat = new THREE.PointsMaterial({ vertexColors: true, size: 0.014 })

      volumeMesh = new THREE.Mesh(geometry, solidMat)
      volumeMesh.visible = false
      volumeMesh.layers.set(LAYER.VOLUME)
      scene.add(volumeMesh)

      wireframeMesh = new THREE.Mesh(geometry, wireMat)
      wireframeMesh.visible = false
      wireframeMesh.layers.set(LAYER.VOLUME)
      scene.add(wireframeMesh)

      volumePoints = new THREE.Points(geometry, pointMat)
      volumePoints.visible = false
      volumePoints.layers.set(LAYER.VOLUME)
      scene.add(volumePoints)
    }

    // ── Volumetric rendering ───────────────────────────────────────────────
    // A stack of additive slices through the body. This is what makes the
    // scan read as a glowing 3-D volume you can orbit around, rather than a
    // extruded relief map. Cheap enough to run on integrated graphics.
    const VOLUME_SLICES = 72
    const volumeGroup = new THREE.Group()
    volumeGroup.layers.set(LAYER.VOLUME)
    volumeGroup.visible = false
    const volumeCanvas = document.createElement('canvas')
    volumeCanvas.width = GRID
    volumeCanvas.height = GRID
    const volumeCtx = volumeCanvas.getContext('2d', { willReadFrequently: true })
    const volumeTexture = new THREE.CanvasTexture(volumeCanvas)
    volumeTexture.minFilter = THREE.LinearFilter
    volumeTexture.magFilter = THREE.LinearFilter
    const volumeSliceMat = new THREE.MeshBasicMaterial({
      map: volumeTexture,
      transparent: true,
      opacity: 0.018,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    })
    const volumeSliceGeom = new THREE.PlaneGeometry(2, 2)
    volumeSliceGeom.rotateX(-Math.PI / 2)
    for (let i = 0; i < VOLUME_SLICES; i++) {
      const plane = new THREE.Mesh(volumeSliceGeom, volumeSliceMat)
      plane.position.y = -0.75 + (i / (VOLUME_SLICES - 1)) * 1.5
      plane.layers.set(LAYER.VOLUME)
      volumeGroup.add(plane)
    }
    scene.add(volumeGroup)

    const paintVolumeTexture = () => {
      if (!volumeCtx || !volumeRef.current) return
      const data = volumeRef.current
      const cmapFn = colormapFns[colormap]
      const halfWidth = windowWidth / 2
      const minV = windowCenter - halfWidth
      const maxV = windowCenter + halfWidth
      const image = volumeCtx.createImageData(GRID, GRID)
      for (let i = 0; i < GRID * GRID; i++) {
        const rawV = Math.pow(data[i], 0.85)
        let v = windowWidth >= 1.0 ? rawV : (rawV - minV) / (maxV - minV)
        v = Math.max(0, Math.min(1, v))
        if (inverted) v = 1 - v
        // emissive colour: dense tissue glows, air stays dark
        const rgb = cmapFn(1 - v)
        const density = v < 0.10 ? 0 : Math.pow((v - 0.10) / 0.90, 1.9)
        image.data[i * 4] = Math.round(rgb[0] * 255 * density)
        image.data[i * 4 + 1] = Math.round(rgb[1] * 255 * density)
        image.data[i * 4 + 2] = Math.round(rgb[2] * 255 * density)
        image.data[i * 4 + 3] = 255
      }
      volumeCtx.putImageData(image, 0, 0)
      volumeTexture.needsUpdate = true
    }
    paintVolumeTexture()

    // Crosshair lines
    const crosshairGroup = new THREE.Group()
    const crosshairSize = 1.15
    const crosshairMat = new THREE.LineBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.8 })
    const axes = [
      [-crosshairSize, 0, 0, crosshairSize, 0, 0],
      [0, -crosshairSize, 0, 0, crosshairSize, 0],
      [0, 0, -crosshairSize, 0, 0, crosshairSize],
    ]
    axes.forEach(pts => {
      const geom = new THREE.BufferGeometry()
      geom.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(pts), 3))
      crosshairGroup.add(new THREE.LineSegments(geom, crosshairMat))
    })
    crosshairGroup.visible = showCrosshair
    crosshairGroup.traverse(o => o.layers.set(LAYER.VOLUME))
    scene.add(crosshairGroup)

    // Measurement lines
    const measurementGroup = new THREE.Group()
    measurementGroup.layers.set(LAYER.VOLUME)
    scene.add(measurementGroup)

    // Viewport dimensions (updated on resize)
    let vpW = viewMode === 'quad' ? container.clientWidth / 2 : container.clientWidth
    let vpH = viewMode === 'quad' ? container.clientHeight / 2 : container.clientHeight

    // Resize handler
    const resize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (!w || !h) return
      renderer.setSize(w, h)
      vpW = viewMode === 'quad' ? w / 2 : w
      vpH = viewMode === 'quad' ? h / 2 : h
      
      viewports.forEach((vp, i) => {
        if (viewMode === 'quad') {
          // GL origin is bottom-left; flip rows so axial sits top-left
          const col = i % 2
          const row = Math.floor(i / 2)
          const x = col * vpW
          const y = (1 - row) * vpH
          renderer.setViewport(x, y, vpW, vpH)
          renderer.setScissor(x, y, vpW, vpH)
          renderer.setScissorTest(true)
        }
        
        if (vp.type !== '3d') {
          const orthoCam = vp.camera as THREE.OrthographicCamera
          orthoCam.left = -vpW / vpH
          orthoCam.right = vpW / vpH
          orthoCam.top = 1
          orthoCam.bottom = -1
          orthoCam.updateProjectionMatrix()
        } else {
          const perspCam = vp.camera as THREE.PerspectiveCamera
          perspCam.aspect = vpW / vpH
          perspCam.updateProjectionMatrix()
        }
      })
    }
    
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    resize()

    let raf = 0
    let disposed = false
    let lastSliceKey = ''
    let lastVolKey = ''
    const controls = new OrbitControls(viewports[3].camera as THREE.PerspectiveCamera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 1.2
    controls.maxDistance = 8

    const tick = () => {
      if (disposed) return
      raf = requestAnimationFrame(tick)
      
      try {
        // Update slice textures only when something actually changed —
        // rebuilding 3 canvases every frame was melting the CPU.
        const sliceKey = [axialSlice, sagittalSlice, coronalSlice, windowCenter, windowWidth, inverted, colormap, viewMode].join('|')
        if (sliceKey !== lastSliceKey) {
          lastSliceKey = sliceKey
          if (viewMode === 'quad' || viewMode === 'axial') applySliceTexture('axial', axialSlice)
          if (viewMode === 'quad' || viewMode === 'sagittal') applySliceTexture('sagittal', sagittalSlice)
          if (viewMode === 'quad' || viewMode === 'coronal') applySliceTexture('coronal', coronalSlice)
        }

        // Update 3D volume only when its parameters changed
        const volKey = [depth, windowCenter, windowWidth, inverted, colormap].join('|')
        if (volumeMesh && volKey !== lastVolKey) {
          lastVolKey = volKey
          const cmapFn = colormapFns[colormap]
          const heights = volumeRef.current
          if (heights) {
            const pos = volumeMesh.geometry.attributes.position as THREE.BufferAttribute
            const colorAttr = volumeMesh.geometry.attributes.color as THREE.BufferAttribute
            const halfWidth = windowWidth / 2
            const minV = windowCenter - halfWidth
            const maxV = windowCenter + halfWidth
            for (let j = 0; j < pos.count; j++) {
              const rawV = Math.pow(heights[j], 0.9)
              let v: number
              if (windowWidth >= 1.0) v = rawV
              else {
                v = (rawV - minV) / (maxV - minV)
                v = Math.max(0, Math.min(1, v))
              }
              if (inverted) v = 1 - v
              pos.setY(j, v * depth * 1.6 - depth * 0.8)
              const rgb = cmapFn(v)
              colorAttr.setXYZ(j, rgb[0], rgb[1], rgb[2])
            }
            pos.needsUpdate = true
            colorAttr.needsUpdate = true
            volumeMesh.geometry.computeVertexNormals()
            paintVolumeTexture()
          }
        }

        // Render each viewport
        viewports.forEach((vp, i) => {
          if (viewMode === 'quad') {
            // GL origin is bottom-left; flip rows so axial sits top-left
            const col = i % 2
            const row = Math.floor(i / 2)
            const x = col * vpW
            const y = (1 - row) * vpH
            renderer.setViewport(x, y, vpW, vpH)
            renderer.setScissor(x, y, vpW, vpH)
          } else {
            renderer.setViewport(0, 0, vpW, vpH)
            renderer.setScissor(0, 0, vpW, vpH)
          }
          
          // Show/hide meshes based on view mode
          sliceMeshes.forEach((mesh, id) => {
            mesh.visible = viewMode === 'quad' || viewMode === id
            // Position slice mesh at current slice index
            const sliceIdx = id === 'axial' ? axialSlice : id === 'sagittal' ? sagittalSlice : coronalSlice
            const normalized = (sliceIdx / GRID) * 2 - 1
            if (id === 'axial') mesh.position.y = -normalized
            else if (id === 'sagittal') mesh.position.x = -normalized
            else mesh.position.z = -normalized
          })
          
          const volumeVis = viewMode === '3d' || viewMode === 'quad'
          if (volumeMesh) {
            volumeMesh.visible = volumeVis && mode === 'surface'
          }
          volumeGroup.visible = volumeVis && mode === 'volume'
          if (wireframeMesh) {
            wireframeMesh.visible = volumeVis && mode === 'wireframe'
          }
          if (volumePoints) {
            volumePoints.visible = volumeVis && mode === 'points'
          }
          crosshairGroup.visible = showCrosshair && (viewMode === '3d' || viewMode === 'quad')
          
          renderer.render(scene, vp.camera)
        })
        
        controls.update()
      } catch (e) {
        console.error('[Xray3DViewer] render failed:', e)
      }
    }
    tick()

    // Store refs for cleanup
    ;(container as any).__stethos3d = {
      renderer,
      scene,
      viewports,
      sliceMeshes,
      volumeMesh,
      wireframeMesh,
      volumePoints,
      controls,
      crosshairGroup,
      measurementGroup,
      applySliceTexture,
      setMode: (m: RenderMode) => { setMode(m) },
      setAutoRotate: (on: boolean) => { controls.autoRotate = on },
      reset: () => {
        viewports[3].camera.position.set(2.0, 3.4, 2.8)
        controls.target.set(0, 0, 0)
        controls.update()
      },
      snapshot: () => {
        renderer.render(scene, viewports[3].camera)
        const a = document.createElement('a')
        a.href = renderer.domElement.toDataURL('image/png')
        a.download = 'stethos-xray-3d.png'
        a.click()
      },
      setWindowLevel: (center: number, width: number) => { setWindowCenter(center); setWindowWidth(width) },
      setInverted: (inv: boolean) => { setInverted(inv) },
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
        const m = { id, type, points, value, unit, viewport: '3d' }
        setMeasurements(prev => [...prev, m])
        onMeasurement?.(m)
      },
      clearMeasurements: () => {
        setMeasurements([])
        measurementGroup.clear()
      },
    }

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      volumeTexture.dispose()
      volumeSliceGeom.dispose()
      volumeSliceMat.dispose()
      renderer.dispose()
      if (renderer.domElement.parentElement === container) container.removeChild(renderer.domElement)
      delete (container as any).__stethos3d
    }
  }, [ready, viewMode])

  // Update slice textures when slice indices change
  useEffect(() => {
    const api = (containerRef.current as any)?.__stethos3d
    if (api && volumeRef.current) {
      api.applySliceTexture('axial', axialSlice)
      api.applySliceTexture('sagittal', sagittalSlice)
      api.applySliceTexture('coronal', coronalSlice)
    }
  }, [axialSlice, sagittalSlice, coronalSlice, windowCenter, windowWidth, inverted, colormap, ready])

  // Volume state updates
  useEffect(() => {
    const api = (containerRef.current as any)?.__stethos3d
    if (api) api.setMode(mode)
  }, [mode, ready])

  useEffect(() => {
    const api = (containerRef.current as any)?.__stethos3d
    if (api) api.setAutoRotate(autoRotate)
  }, [autoRotate, ready])

  // Slice navigation with keyboard
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      const step = e.shiftKey ? 10 : 1
      switch (e.key) {
        case 'ArrowUp':
          e.preventDefault()
          setAxialSlice(Math.min(GRID - 1, axialSlice + step))
          break
        case 'ArrowDown':
          e.preventDefault()
          setAxialSlice(Math.max(0, axialSlice - step))
          break
        case 'ArrowRight':
          e.preventDefault()
          setSagittalSlice(Math.min(GRID - 1, sagittalSlice + step))
          break
        case 'ArrowLeft':
          e.preventDefault()
          setSagittalSlice(Math.max(0, sagittalSlice - step))
          break
        case 'PageUp':
          e.preventDefault()
          setCoronalSlice(Math.min(GRID - 1, coronalSlice + step))
          break
        case 'PageDown':
          e.preventDefault()
          setCoronalSlice(Math.max(0, coronalSlice - step))
          break
        case 'r':
          e.preventDefault()
          const api = (containerRef.current as any)?.__stethos3d
          if (api) api.reset()
          break
        case ' ':
          e.preventDefault()
          setAutoRotate(!autoRotate)
          break
        case '1':
          setViewMode('3d')
          break
        case '2':
          setViewMode('quad')
          break
        case '3':
          setViewMode('axial')
          break
        case '4':
          setViewMode('sagittal')
          break
        case '5':
          setViewMode('coronal')
          break
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [axialSlice, sagittalSlice, coronalSlice, autoRotate])

  if (!ready && !failed) {
    return (
      <div className="inset-well flex items-center justify-center" style={{ height, borderRadius: 4 }}>
        <div className="flex flex-col items-center gap-2">
          <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
          <p className="mono text-[10.5px] text-[var(--text-faint)]">building 3d volume…</p>
        </div>
      </div>
    )
  }
  if (failed) {
    return (
      <div className="inset-well flex items-center justify-center" style={{ height, borderRadius: 4 }}>
        <p className="mono text-[10.5px] text-[var(--text-faint)] text-center px-4">
          3d view unavailable (no WebGL) · use 2d view instead
        </p>
      </div>
    )
  }

  const maxHist = Math.max(...histogramData)

  return (
    <div className="space-y-2" style={{ display: 'flex', flexDirection: 'column', height }}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
        {/* View Mode */}
        <div className="flex items-center gap-1" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {(['quad', 'axial', 'sagittal', 'coronal', '3d'] as ViewMode[]).map(v => (
            <button
              key={v}
              onClick={() => setViewMode(v)}
              className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${viewMode === v ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-faint)]'}`}
              style={{ border: 'none', background: 'transparent' }}
            >
              {v === 'quad' && <Layout className="w-3 h-3" />}
              {v === 'axial' && <Box className="w-3 h-3" />}
              {v === 'sagittal' && <Grid3x3 className="w-3 h-3" />}
              {v === 'coronal' && <CircleDot className="w-3 h-3" />}
              {v === '3d' && <Box className="w-3 h-3" />}
              <span className="hidden sm:inline">{v}</span>
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

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
            <input type="range" min={0} max={1} step={0.01} value={windowCenter}
              onChange={e => setWindowCenter(parseFloat(e.target.value))} className="w-24" />
          </label>
          <label className="flex items-center gap-1 mono text-[10px] text-[var(--text-faint)]">
            WW
            <input type="range" min={0.1} max={1} step={0.01} value={windowWidth}
              onChange={e => setWindowWidth(parseFloat(e.target.value))} className="w-24" />
          </label>
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Invert */}
        <button onClick={() => setInverted(!inverted)}
          className={`px-2 py-1 text-[10px] mono flex items-center gap-1 rounded-[3px] ${inverted ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {inverted ? <Sun className="w-3 h-3" /> : <Contrast className="w-3 h-3" />} {inverted ? 'normal' : 'invert'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Colormap */}
        <div className="flex" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {(['radiograph', 'bone', 'thermal', 'hot-iron', 'mammography', 'inverted'] as Colormap[]).map(c => (
            <button key={c} onClick={() => setColormap(c)}
              className="px-2 py-1 mono text-[10px]"
              style={{ background: colormap === c ? 'var(--accent-dim)' : 'transparent', color: colormap === c ? 'var(--accent)' : 'var(--text-faint)', border: 'none' }}>
              {c}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Render Mode */}
        <div className="flex" style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          {([
            ['surface', Box], ['wireframe', Grid3x3], ['points', CircleDot], ['volume', Maximize2],
          ] as Array<[RenderMode, React.ComponentType<{ className?: string }>]>).map(([m, Icon]) => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-2 py-1 flex items-center gap-1 mono text-[10px] ${mode === m ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-faint)]'}`}
              style={{ border: 'none', background: 'transparent' }}>
              <Icon className="w-3 h-3" /> {m}
            </button>
          ))}
        </div>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Depth (3D only) */}
        {viewMode === '3d' && (
          <>
            <label className="flex items-center gap-1.5 mono text-[10px] text-[var(--text-faint)]">
              depth
              <input type="range" min={0.1} max={1.4} step={0.05} value={depth}
                onChange={e => setDepth(parseFloat(e.target.value))} className="w-20" />
            </label>
            <div className="w-px h-6 bg-[var(--border)]" />
          </>
        )}

        {/* Auto Rotate (3D only) */}
        {viewMode === '3d' && (
          <>
            <button onClick={() => setAutoRotate(!autoRotate)}
              className="px-2 py-1 mono text-[10px] flex items-center gap-1"
              style={{ border: '1px solid var(--border)', borderRadius: 3, color: autoRotate ? 'var(--accent)' : 'var(--text-faint)' }}>
              {autoRotate ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />} {autoRotate ? 'spin on' : 'spin off'}
            </button>
            <div className="w-px h-6 bg-[var(--border)]" />
          </>
        )}

        {/* Slice Navigation (MPR only) */}
        {viewMode !== '3d' && (
          <>
            <div className="flex items-center gap-1.5 mono text-[10px] text-[var(--text-faint)]">
              <label className="flex items-center gap-1">
                <span className="w-10 text-right">Axial:</span>
                <input type="range" min={0} max={GRID-1} step={1} value={axialSlice}
                  onChange={e => setAxialSlice(parseInt(e.target.value))} className="w-28" />
                <span className="w-8">{axialSlice}</span>
              </label>
              <label className="flex items-center gap-1">
                <span className="w-10 text-right">Sag:</span>
                <input type="range" min={0} max={GRID-1} step={1} value={sagittalSlice}
                  onChange={e => setSagittalSlice(parseInt(e.target.value))} className="w-28" />
                <span className="w-8">{sagittalSlice}</span>
              </label>
              <label className="flex items-center gap-1">
                <span className="w-10 text-right">Cor:</span>
                <input type="range" min={0} max={GRID-1} step={1} value={coronalSlice}
                  onChange={e => setCoronalSlice(parseInt(e.target.value))} className="w-28" />
                <span className="w-8">{coronalSlice}</span>
              </label>
            </div>
            <div className="w-px h-6 bg-[var(--border)]" />
          </>
        )}

        {/* Reset View */}
        <button onClick={() => { const api = (containerRef.current as any)?.__stethos3d; if (api) api.reset() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}>
          <RotateCcw className="w-3 h-3" /> reset
        </button>

        {/* Snapshot */}
        <button onClick={() => { const api = (containerRef.current as any)?.__stethos3d; if (api) api.snapshot() }}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}>
          <Camera className="w-3 h-3" /> save
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Measurement Tools */}
        <div className="flex items-center gap-1" style={{ border: '1px solid var(--border)', borderRadius: 3, background: activeTool !== 'none' ? 'var(--accent-dim)' : 'transparent' }}>
          {(['distance', 'angle', 'roi'] as Array<'distance' | 'angle' | 'roi'>).map(t => (
            <button key={t} onClick={() => setActiveTool(activeTool === t ? 'none' : t)}
              className={`px-2 py-1 flex items-center gap-1 mono text-[10px] rounded-l ${activeTool === t ? 'bg-[var(--accent)] text-white' : 'text-[var(--text-faint)]'}`}
              style={{ borderRight: t !== 'roi' ? '1px solid var(--border)' : 'none', borderRadius: t === 'distance' ? '3px 0 0 3px' : t === 'roi' ? '0 3px 3px 0' : 0 }}>
              {t === 'distance' && <Ruler className="w-3 h-3" />}
              {t === 'angle' && <Maximize2 className="w-3 h-3" />}
              {t === 'roi' && <CircleDot className="w-3 h-3" />}
              <span className="hidden sm:inline">{t}</span>
            </button>
          ))}
        </div>

        {activeTool !== 'none' && (
          <button onClick={() => { setActiveTool('none') }}
            className="px-2 py-1 mono text-[10px] flex items-center gap-1 text-[var(--bad)]"
            style={{ border: '1px solid var(--bad)', borderRadius: 3 }}>
            <Eye className="w-3 h-3" /> cancel
          </button>
        )}

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Crosshair Toggle */}
        <button onClick={() => setShowCrosshair(!showCrosshair)}
          className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${showCrosshair ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          <Crosshair className="w-3 h-3" /> {showCrosshair ? 'on' : 'off'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Linked Cursor */}
        <button onClick={() => setLinkedCursor(!linkedCursor)}
          className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${linkedCursor ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          <Link className="w-3 h-3" /> {linkedCursor ? 'linked' : 'unlinked'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Orientation */}
        <button onClick={() => setShowOrientation(!showOrientation)}
          className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${showOrientation ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          <Navigation className="w-3 h-3" /> {showOrientation ? 'on' : 'off'}
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Fullscreen */}
        <button onClick={() => setShowFullscreen(!showFullscreen)}
          className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}>
          {showFullscreen ? <Minimize2 className="w-3 h-3" /> : <Maximize2 className="w-3 h-3" />} {showFullscreen ? 'exit fs' : 'fullscreen'}
        </button>

        {/* Histogram Toggle */}
        <button onClick={() => setShowHistogram(!showHistogram)}
          className={`px-2 py-1 mono text-[10px] flex items-center gap-1 ${showHistogram ? 'bg-[var(--accent-dim)] text-[var(--accent)]' : 'text-[var(--text-faint)]'}`}
          style={{ border: '1px solid var(--border)', borderRadius: 3 }}>
          <Activity className="w-3 h-3" /> hist
        </button>

        <div className="w-px h-6 bg-[var(--border)]" />

        {/* Help */}
        <button className="px-2 py-1 mono text-[10px] flex items-center gap-1"
          style={{ border: '1px solid var(--border)', borderRadius: 3, color: 'var(--text-faint)' }}>
          <HelpCircle className="w-3 h-3" /> help
        </button>
      </div>

      {/* Main Viewport Area */}
      <div className="flex-1 relative" style={{ minHeight: 0 }}>
        <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
          {viewMode === 'quad' && (
            <div className="absolute inset-0 grid grid-cols-2 grid-rows-2 gap-1 p-1" style={{ pointerEvents: 'none' }}>
              {[
                { label: 'Axial', value: axialSlice },
                { label: 'Sagittal', value: sagittalSlice },
                { label: 'Coronal', value: coronalSlice },
                { label: '3D Volume', value: mode },
              ].map(cell => (
                <div key={cell.label} className="relative">
                  <div className="absolute top-1 left-1 px-1.5 py-0.5 mono text-[9px] text-[var(--text-faint)] bg-black/70 rounded">{cell.label}</div>
                  <div className="absolute top-1 right-1 px-1.5 py-0.5 mono text-[9px] text-[var(--text-faint)] bg-black/70 rounded">{cell.value}</div>
                </div>
              ))}
            </div>
          )}
          {(viewMode === '3d' || viewMode === 'quad') && (
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 px-2.5 py-1 mono text-[9.5px] text-gray-300 bg-black/70 rounded" style={{ pointerEvents: 'none', whiteSpace: 'nowrap' }}>
              drag to orbit · scroll to zoom · right-drag to pan
            </div>
          )}
        </div>
      </div>

      {/* Histogram */}
      {showHistogram && (
        <div className="p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
          <div className="flex items-center justify-between mb-2">
            <span className="mono text-[10px] text-[var(--text-faint)]">Histogram</span>
            <button onClick={() => setShowHistogram(false)} className="text-[var(--text-faint)] hover:text-[var(--text)]">×</button>
          </div>
          <div className="flex items-end h-24 gap-[1px]" style={{ background: 'var(--inset)', borderRadius: 3 }}>
            {histogramData.map((count, i) => (
              <div key={i} className="flex-1" style={{
                height: maxHist > 0 ? `${(count / maxHist) * 100}%` : 0,
                background: `hsl(${200 + i * 0.5}, 70%, 50%)`,
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
            <button onClick={() => { const api = (containerRef.current as any)?.__stethos3d; if (api) api.clearMeasurements() }} className="text-[var(--text-faint)] hover:text-[var(--text)] text-xs">clear all</button>
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

      {/* Keyboard Shortcuts Help */}
      <details className="mt-2 p-2 bg-[var(--chrome)] border border-[var(--border)] rounded-[4px]">
        <summary className="mono text-[10px] text-[var(--text-faint)] cursor-pointer">keyboard shortcuts</summary>
        <div className="mt-2 grid grid-cols-2 gap-1 text-[9px] mono text-[var(--text-faint)]">
          <span>↑/↓</span><span>axial slice ±1</span>
          <span>←/→</span><span>sagittal slice ±1</span>
          <span>PgUp/PgDn</span><span>coronal slice ±1</span>
          <span>Shift+arrows</span><span>±10 slices</span>
          <span>Space</span><span>toggle auto-rotate</span>
          <span>R</span><span>reset 3d view</span>
          <span>1-5</span><span>switch view mode</span>
        </div>
      </details>
    </div>
  )
}