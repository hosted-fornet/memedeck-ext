use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum PredictRequest {
    Run,
    Ping,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum PredictResponse {
    Run,
    Ping(bool),
}
