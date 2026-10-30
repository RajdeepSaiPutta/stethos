#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{command, AppHandle};

mod python_manager;
use python_manager::PythonManager;

#[command]
async fn start_python_backend(app: AppHandle) -> Result<String, String> {
    PythonManager::start(&app).await.map_err(|e| e.to_string())
}

#[command]
async fn stop_python_backend() -> Result<String, String> {
    PythonManager::stop().await.map_err(|e| e.to_string())
}

#[command]
async fn python_health_check() -> Result<serde_json::Value, String> {
    PythonManager::health_check().await.map_err(|e| e.to_string())
}

#[command]
async fn run_inference(
    modality: String,
    input_path: String,
    options: Option<serde_json::Value>,
) -> Result<serde_json::Value, String> {
    PythonManager::run_inference(modality, input_path, options).await.map_err(|e| e.to_string())
}

#[command]
async fn run_fusion_inference(
    inputs: serde_json::Value,
) -> Result<serde_json::Value, String> {
    PythonManager::run_fusion_inference(inputs).await.map_err(|e| e.to_string())
}

#[command]
async fn export_report(
    format: String,
    data: serde_json::Value,
    output_path: String,
) -> Result<String, String> {
    PythonManager::export_report(format, data, output_path).await.map_err(|e| e.to_string())
}

#[command]
async fn list_models() -> Result<serde_json::Value, String> {
    PythonManager::list_models().await.map_err(|e| e.to_string())
}

#[command]
async fn set_device(device: String) -> Result<serde_json::Value, String> {
    PythonManager::set_device(device).await.map_err(|e| e.to_string())
}

#[command]
async fn download_model(model_name: String) -> Result<String, String> {
    PythonManager::download_model(model_name).await.map_err(|e| e.to_string())
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            start_python_backend,
            stop_python_backend,
            python_health_check,
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
                if let Err(e) = PythonManager::start(&handle).await {
                    tracing::error!("Failed to start Python backend: {}", e);
                }
            });
            Ok(())
        })
        .on_window_event(|_window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                tauri::async_runtime::block_on(async {
                    let _ = PythonManager::stop().await;
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}