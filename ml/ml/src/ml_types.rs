use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub enum MlLibrary {
    PyTorch,
    TensorFlow,
    Keras,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MlDataType {
    Float16,
    BFloat16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    //...
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MlRequestPayload {
    pub library: MlLibrary,
    pub data_shape: Vec<u64>,
    pub data_type: MlDataType,
    pub model: Model,
    pub data_bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Model {
    Bytes(Vec<u8>),
    Name(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MlResponsePayload {
    pub library: MlLibrary,
    pub data_shape: Vec<u64>,
    pub data_type: MlDataType,
    pub data_bytes: Vec<u8>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MlRequest {
    Run,
    Ping,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MlResponse {
    Run,
    Ping(bool),
}
