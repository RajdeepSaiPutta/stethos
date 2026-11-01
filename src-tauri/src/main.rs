#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{command, AppHandle};

mod sidecar_manager;
use sidecar_manager::SidecarManager;

#[command]
async fn start_sidecar(app: AppHandle) -> Result<String, String> {
    SidecarManager::start(&app).await.map_err(|e| e.to_string())
}

#[command]
async fn stop_sidecar() -> Result<String, String> {
    SidecarManager::stop().await.map_err(|e| e.to_string())
}

#[command]
async fn sidecar_health_check() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://127.0.0.1:8765/health")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json)
}

#[command]
async fn run_inference(
    modality: String,
    input_path: String,
    options: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/inference")
        .json(&serde_json::json!({
            "modality": modality,
            "input_path": input_path,
            "options": options.unwrap_or_default(),
        }))
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json)
}

#[command]
async fn run_fusion_inference(
    inputs: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/fusion")
        .json(&inputs)
        .timeout(std::time::Duration::from_secs(180))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json)
}

#[command]
async fn export_report(
    format: String,
    data: serde_json::Value,
    output_path: String,
) -> Result<String, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/export")
        .json(&serde_json::json!({
            "format": format,
            "data": data,
            "output_path": output_path
        }))
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string())
}

#[command]
async fn list_models() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .get("http://127.0.0.1:8765/models")
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json)
}

#[command]
async fn set_device(device: String) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/device")
        .json(&serde_json::json!({ "device": device }))
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json)
}

#[command]
async fn download_model(model_name: String) -> Result<String, String> {
    let client = reqwest::Client::new();
    let resp = client
        .post("http://127.0.0.1:8765/models/download")
        .json(&serde_json::json!({ "model": model_name }))
        .timeout(std::time::Duration::from_secs(600))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json.get("message").and_then(|v| v.as_str()).unwrap_or("Done").to_string())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Wayland sessions can kill the webkit webview with a Gdk protocol error
    // before the window ever paints. XWayland is present on every Linux
    // desktop that ships Wayland, so route the app through it.
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("GDK_BACKEND").is_none() {
            std::env::set_var("GDK_BACKEND", "x11");
        }
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_sidecar,
            stop_sidecar,
            sidecar_health_check,
            run_inference,
            run_fusion_inference,
            export_report,
            list_models,
            set_device,
            download_model
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if let Err(e) = SidecarManager::start(&handle).await {
                    tracing::error!("Failed to start sidecar: {}", e);
                }
            });
            Ok(())
        })
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                tauri::async_runtime::block_on(async {
                    let _ = SidecarManager::stop().await;
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}