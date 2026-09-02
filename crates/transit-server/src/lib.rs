//! Minimal inference service surface.

use anyhow::Result;
use axum::{extract::State, routing::get, Json, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use transit_inference::PredictionFile;

#[derive(Clone)]
pub struct AppState {
    pub predictions: Arc<PredictionFile>,
}

pub fn app(prediction_file: PredictionFile) -> Router {
    let state = AppState {
        predictions: Arc::new(prediction_file),
    };
    Router::new()
        .route("/health", get(health))
        .route("/predictions", get(get_predictions))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn get_predictions(State(state): State<AppState>) -> Json<PredictionFile> {
    Json((*state.predictions).clone())
}

pub async fn serve(predictions: PredictionFile, address: SocketAddr) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, app(predictions)).await?;
    Ok(())
}
