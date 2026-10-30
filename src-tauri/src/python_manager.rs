use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{AppHandle, Manager};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Child;
use anyhow::Result;

static PYTHON_PROCESS: OnceLock<Arc<Mutex<Option<Child>>>> = OnceLock::new();
static PYTHON_PORT: u16 = 8765;

pub struct PythonManager;

impl PythonManager {
    fn get_process_handle() -> &'static Arc<Mutex<Option<Child>>> {
        PYTHON_PROCESS.get_or_init(|| Arc::new(Mutex::new(None)))
    }

    pub async fn start(app: &AppHandle) -> Result<String> {
        let mut guard = Self::get_process_handle().lock().unwrap();
        if guard.is_some() {
            return Ok("Python backend already running".to_string());
        }

        let app_dir = app
            .path()
            .app_data_dir()
            .map_err(|e| anyhow::anyhow!("Failed to get app data dir: {}", e))?;

        let python_dir = app_dir.join("python");
        std::fs::create_dir_all(&python_dir)?;

        let resource_dir = app
            .path()
            .resource_dir()
            .map_err(|e| anyhow::anyhow!("Failed to get resource dir: {}", e))?;

        let backend_src = resource_dir.join("python");
        if backend_src.exists() {
            Self::copy_dir(&backend_src, &python_dir)?;
        }

        let python_exe = Self::find_python()?;
        let venv_python = python_dir.join("venv").join(if cfg!(windows) { "Scripts/python.exe" } else { "bin/python" });

        if !venv_python.exists() {
            tracing::info!("Creating virtual environment at {:?}", python_dir.join("venv"));
            let output = Command::new(&python_exe)
                .args(["-m", "venv", "venv"])
                .current_dir(&python_dir)
                .output()?;
            if !output.status.success() {
                return Err(anyhow::anyhow!(
                    "Failed to create venv: {}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }

        let pip = python_dir.join("venv").join(if cfg!(windows) { "Scripts/pip.exe" } else { "bin/pip" });
        let requirements = python_dir.join("requirements.txt");
        if requirements.exists() {
            tracing::info!("Installing Python dependencies");
            let output = Command::new(&pip)
                .args(["install", "-r", "requirements.txt", "--quiet"])
                .current_dir(&python_dir)
                .output()?;
            if !output.status.success() {
                tracing::warn!("pip install had issues: {}", String::from_utf8_lossy(&output.stderr));
            }
        }

        let main_py = python_dir.join("api").join("main.py");
        let mut child = Command::new(&venv_python)
            .arg(&main_py)
            .current_dir(&python_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::info!("[Python] {}", line);
            }
        });
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                tracing::error!("[Python ERR] {}", line);
            }
        });

        *guard = Some(child);

        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        Ok(format!("Python backend started on port {}", PYTHON_PORT))
    }

    pub async fn stop() -> Result<String> {
        let mut guard = Self::get_process_handle().lock().unwrap();
        if let Some(mut child) = guard.take() {
            child.kill().await?;
            child.wait().await?;
            Ok("Python backend stopped".to_string())
        } else {
            Ok("Python backend was not running".to_string())
        }
    }

    pub async fn health_check() -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://localhost:{}/health", PYTHON_PORT))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json)
    }

    pub async fn run_inference(
        modality: String,
        input_path: String,
        options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://localhost:{}/inference", PYTHON_PORT))
            .json(&serde_json::json!({
                "modality": modality,
                "input_path": input_path,
                "options": options.unwrap_or(serde_json::json!({}))
            }))
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json)
    }

    pub async fn run_fusion_inference(inputs: serde_json::Value) -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://localhost:{}/fusion", PYTHON_PORT))
            .json(&inputs)
            .timeout(std::time::Duration::from_secs(180))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json)
    }

    pub async fn export_report(
        format: String,
        data: serde_json::Value,
        output_path: String,
    ) -> Result<String> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://localhost:{}/export", PYTHON_PORT))
            .json(&serde_json::json!({
                "format": format,
                "data": data,
                "output_path": output_path
            }))
            .timeout(std::time::Duration::from_secs(60))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string())
    }

    pub async fn list_models() -> Result<serde_json::Value> {
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("http://localhost:{}/models", PYTHON_PORT))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json)
    }

    pub async fn download_model(model_name: String) -> Result<String> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://localhost:{}/models/download", PYTHON_PORT))
            .json(&serde_json::json!({ "model": model_name }))
            .timeout(std::time::Duration::from_secs(600))
            .send()
            .await?;
        let json: serde_json::Value = resp.json().await?;
        Ok(json.get("message").and_then(|v| v.as_str()).unwrap_or("Done").to_string())
    }

    fn find_python() -> Result<std::path::PathBuf> {
        let candidates = if cfg!(windows) {
            vec!["python.exe", "python3.exe", "py.exe"]
        } else {
            vec!["python3", "python3.11", "python3.10", "python"]
        };
        for cmd in candidates {
            if let Ok(output) = Command::new(cmd).arg("--version").output() {
                if output.status.success() {
                    return Ok(std::path::PathBuf::from(cmd));
                }
            }
        }
        Err(anyhow::anyhow!("Python not found in PATH"))
    }

    fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dst)?;
        for entry in std::fs::read_dir(src)? {
            let entry = entry?;
            let path = entry.path();
            let dest = dst.join(entry.file_name());
            if path.is_dir() {
                Self::copy_dir(&path, &dest)?;
            } else {
                std::fs::copy(&path, &dest)?;
            }
        }
        Ok(())
    }
}