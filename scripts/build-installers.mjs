#!/usr/bin/env node
/**
 * Build Stethos installers for the current platform and collect them in
 * `release/<platform>/`.
 *
 *   node scripts/build-installers.mjs                 # installers + sidecar
 *   node scripts/build-installers.mjs --no-sidecar    # shell-only (fast)
 *   node scripts/build-installers.mjs --bundles deb,rpm,appimage
 *
 * The Python sidecar is a single self-contained executable. If it has not been
 * built yet this script runs PyInstaller with the current interpreter (set
 * PYTHON=/path/to/python to override).
 */
import { execFileSync, execSync } from 'node:child_process'
import { createReadStream, copyFileSync, existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { basename, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)))
const TAURI_DIR = join(ROOT, 'src-tauri')
const PYTHON_DIR = join(ROOT, 'python')
const BIN_DIR = join(TAURI_DIR, 'binaries')

const args = process.argv.slice(2)
const flag = (name) => args.includes(name)
const option = (name, fallback) => {
  const i = args.indexOf(name)
  return i !== -1 && args[i + 1] && !args[i + 1].startsWith('-') ? args[i + 1] : fallback
}

const PLATFORM = process.platform === 'win32' ? 'windows' : process.platform === 'darwin' ? 'macos' : 'linux'
const NPM = PLATFORM === 'windows' ? 'npm.cmd' : 'npm'
const NPX = PLATFORM === 'windows' ? 'npx.cmd' : 'npx'
const DEFAULT_BUNDLES = {
  linux: 'deb,rpm,appimage',
  windows: 'nsis,msi',
  macos: 'app,dmg',
}
const bundles = option('--bundles', DEFAULT_BUNDLES[PLATFORM])
const withSidecar = !flag('--no-sidecar')

function run(command, cmdArgs, cwd = ROOT) {
  console.log(`\n$ ${command} ${cmdArgs.join(' ')}`)
  // Windows resolves npm/npx through .cmd shims, which Node refuses to spawn
  // without a shell.
  execFileSync(command, cmdArgs, {
    cwd,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  })
}

function rustHostTriple() {
  const out = execSync('rustc -vV', { encoding: 'utf8' })
  const match = out.match(/^host:\s*(\S+)$/m)
  if (!match) throw new Error('could not determine the Rust host triple')
  return match[1]
}

function buildSidecar(triple) {
  const exeName = PLATFORM === 'windows' ? 'stethos-sidecar.exe' : 'stethos-sidecar'
  const built = join(PYTHON_DIR, 'dist', exeName)
  if (!existsSync(built)) {
    const python = process.env.PYTHON || (PLATFORM === 'windows' ? 'python' : 'python3')
    console.log('\nSidecar binary not found - building it with PyInstaller...')
    run(python, ['-m', 'PyInstaller', '--clean', '-y', 'sidecar.spec'], PYTHON_DIR)
  }
  if (!existsSync(built)) {
    throw new Error(`sidecar build did not produce ${built}`)
  }
  mkdirSync(BIN_DIR, { recursive: true })
  const target = join(BIN_DIR, `stethos-sidecar-${triple}${PLATFORM === 'windows' ? '.exe' : ''}`)
  copyFileSync(built, target)
  console.log(`sidecar -> ${target} (${(statSync(target).size / 1e6).toFixed(0)} MB)`)
  return target
}

function sha256(file) {
  return new Promise((resolve, reject) => {
    const hash = createHash('sha256')
    createReadStream(file)
      .on('data', (chunk) => hash.update(chunk))
      .on('error', reject)
      .on('end', () => resolve(hash.digest('hex')))
  })
}

async function collectArtifacts() {
  const outDir = join(ROOT, 'release', PLATFORM)
  mkdirSync(outDir, { recursive: true })
  const bundleRoot = join(TAURI_DIR, 'target', 'release', 'bundle')
  const wanted = new Set(bundles.split(',').map((b) => b.trim().toLowerCase()))
  const dirFor = {
    deb: 'deb',
    rpm: 'rpm',
    appimage: 'appimage',
    nsis: 'nsis',
    msi: 'msi',
    app: 'macos',
    dmg: 'dmg',
  }

  const copied = []
  for (const bundle of wanted) {
    const dir = join(bundleRoot, dirFor[bundle] || bundle)
    if (!existsSync(dir)) continue
    for (const entry of readdirSync(dir)) {
      const file = join(dir, entry)
      if (!statSync(file).isFile()) continue
      if (entry.endsWith('.sig') || entry.endsWith('.json')) continue
      const dest = join(outDir, entry)
      copyFileSync(file, dest)
      copied.push(dest)
    }
  }

  if (!copied.length) {
    throw new Error(`no installers were produced under ${bundleRoot}`)
  }

  // Hash everything currently in the folder, so running the script once per
  // bundle target still leaves one complete checksum manifest.
  const allArtifacts = readdirSync(outDir)
    .map((entry) => join(outDir, entry))
    .filter((file) => statSync(file).isFile() && basename(file) !== 'SHA256SUMS.txt')
    .sort()

  const sums = (
    await Promise.all(
      allArtifacts.map(async (file) => `${await sha256(file)}  ${basename(file)}`),
    )
  ).join('\n')
  const sumsPath = join(outDir, 'SHA256SUMS.txt')
  writeFileSync(sumsPath, sums + '\n')

  console.log(`\nInstallers in release/${PLATFORM}:`)
  for (const file of allArtifacts) console.log(`  ${basename(file)}`)
  console.log(`  SHA256SUMS.txt`)
}

async function main() {
  const triple = rustHostTriple()
  console.log(`Building Stethos for ${PLATFORM} (${triple}) -> ${bundles}`)

  if (withSidecar) buildSidecar(triple)
  else console.log('\nSkipping sidecar (shell-only build)')

  // Always rebuild the web assets so the installer cannot pick up a stale UI.
  run(NPM, ['run', 'build'])

  const tauriArgs = ['tauri', 'build', '--bundles', bundles]
  if (withSidecar) {
    tauriArgs.push('--config', 'src-tauri/tauri.bundle.conf.json')
  }
  run(NPX, tauriArgs)

  await collectArtifacts()
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
