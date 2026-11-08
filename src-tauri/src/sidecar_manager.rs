use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command as AsyncCommand};
use anyhow::Result;

static SIDECAR_PROCESS: OnceLock<Arc<Mutex<Option<Child>>>> = OnceLock::new();
static SIDECAR_PORT: u16 = 8765;

pub struct SidecarManager;

impl SidecarManager {
    fn get_process_handle() -> &'static Arc<Mutex<Option<Child>>> {
        SIDECAR_PROCESS.get_or_init(|| Arc::new(Mutex::new(None)))
    }

    fn sidecar_path(app: &AppHandle) -> Result<std::path::PathBuf> {
        // In production: sidecar is bundled next to the app binary
        // In dev: use the one in src-tauri/binaries/
        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| anyhow::anyhow!("Failed to get resource dir: {}", e))?;

        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_default();

        let mut candidates = vec![
            // Production: bundled as a Tauri resource at resources/binaries/
            resource_dir.join("binaries").join(Self::sidecar_name()),
            // Production: bundled as a plain resource at the resource root
            resource_dir.join(Self::sidecar_name()),
            // Windows/Linux: copied next to the app binary
            exe_dir.join(Self::sidecar_name()),
            // macOS .app: Contents/MacOS/stethos -> Contents/Resources/binaries/
            exe_dir.join("..").join("Resources").join("binaries").join(Self::sidecar_name()),
            exe_dir.join("..").join("Resources").join(Self::sidecar_name()),
            // Dev fallback
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("binaries")
                .join(Self::sidecar_name()),
        ];
        candidates.dedup();

        for path in &candidates {
            if path.exists() {
                return Ok(path.clone());
            }
        }
        Err(anyhow::anyhow!("Sidecar binary not found. Tried: {:?}", candidates))
    }

    #[cfg(target_os = "windows")]
    fn sidecar_name() -> &'static str {
        "stethos-sidecar.exe"
    }

    #[cfg(not(target_os = "windows"))]
    fn sidecar_name() -> &'static str {
        "stethos-sidecar"
    }

    pub async fn start(app: &AppHandle) -> Result<String> {
        // Fast path: a healthy sidecar is already answering on the port.
        // Adopt it instead of spawning a second one that would die on bind.
        if Self::healthy_wait(300).await {
            tracing::info!("Sidecar already healthy on port {}, adopting it", SIDECAR_PORT);
            return Ok("Sidecar already running".to_string());
        }

        // Quick check without holding lock across await
        {
            let guard = Self::get_process_handle().lock().unwrap();
            if guard.is_some() {
                return Ok("Sidecar already running".to_string());
            }
        }

        let sidecar_path = Self::sidecar_path(app)?;
        tracing::info!("Starting sidecar at: {:?}", sidecar_path);

        // Make executable on Unix - skip if read-only filesystem (e.g., AppImage)
        #[cfg(not(target_os = "windows"))]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(&sidecar_path) {
                let mut perms = metadata.permissions();
                if perms.mode() & 0o111 == 0 {
                    // Not executable, try to make it executable
                    perms.set_mode(0o755);
                    if let Err(e) = std::fs::set_permissions(&sidecar_path, perms) {
                        tracing::warn!("Could not set executable bit on sidecar (read-only fs?): {}", e);
                        // Continue anyway - the file might already be executable from the bundle
                    }
                }
            }
        }

        let mut child = AsyncCommand::new(&sidecar_path)
            .current_dir(sidecar_path.parent().unwrap_or_else(|| std::path::Path::new(".")))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHON_PORT", SIDECAR_PORT.to_string())
            .spawn()?;

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::info!("[Sidecar] {}", line);
            }
        });
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::error!("[Sidecar ERR] {}", line);
            }
        });

        // Store process handle in tight scope
        {
            let mut guard = Self::get_process_handle().lock().unwrap();
            *guard = Some(child);
        }

        // Wait for sidecar to be ready. Model loading happens during startup
        // and can take a while on slow disk / cold cache — the bundled
        // one-file build unpacks PyTorch first, so allow 120s.
        for _ in 0..240 {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            if let Ok(resp) = reqwest::get(format!("http://127.0.0.1:{}/health", SIDECAR_PORT)).await {
                if resp.status().is_success() {
                    return Ok(format!("Sidecar started on port {}", SIDECAR_PORT));
                }
            }
        }
        Err(anyhow::anyhow!("Sidecar failed to start within 120 seconds"))
    }

    /// True if a sidecar answers /health on the port within `tries` * 100ms.
    async fn healthy_wait(tries: u32) -> bool {
        for _ in 0..tries {
            if let Ok(resp) = reqwest::get(format!("http://127.0.0.1:{}/health", SIDECAR_PORT)).await {
                if resp.status().is_success() {
                    return true;
                }
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        false
    }

    pub async fn stop() -> Result<String> {
        let child = {
            let mut guard = Self::get_process_handle().lock().unwrap();
            guard.take()
        };
        if let Some(mut child) = child {
            child.kill().await?;
            child.wait().await?;
            Ok("Sidecar stopped".to_string())
        } else {
            Ok("Sidecar was not running".to_string())
        }
    }
}