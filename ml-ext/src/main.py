import argparse
import io
import os
import tempfile

import asyncio
import msgpack
import torch
import websockets

import numpy as np
import tensorflow as tf


def bytes_to_tensors(bytes_data, single_tensor_shape, dtype=torch.float32):
    element_size = torch.tensor([], dtype=dtype).element_size()
    tensor_size = torch.Size(single_tensor_shape).numel() * element_size
    num_tensors = len(bytes_data) // tensor_size

    tensors = []
    for i in range(num_tensors):
        start = i * tensor_size
        end = start + tensor_size
        buffer = io.BytesIO(bytes_data[start:end])
        tensor = torch.load(buffer).reshape(single_tensor_shape).to(dtype)
        tensors.append(tensor)

    return tensors

# # Example usage
# bytes_data = ...  # your bytes data from WebSocket
# single_tensor_shape = (3, 224, 224)  # replace with your model's input shape
#
# tensors = bytes_to_tensors(bytes_data, single_tensor_shape)
#
# # Now, `tensors` is a list of tensors, each ready to be fed into your PyTorch model


def bytes_to_tensors_tf(bytes_data, single_tensor_shape, dtype=tf.float32):
    element_size = np.dtype(dtype.as_numpy_dtype).itemsize
    tensor_size = np.prod(single_tensor_shape) * element_size
    num_tensors = len(bytes_data) // tensor_size

    tensors = []
    for i in range(num_tensors):
        start = i * tensor_size
        end = start + tensor_size
        tensor_data = np.frombuffer(bytes_data[start:end], dtype=dtype.as_numpy_dtype)
        tensor = tf.convert_to_tensor(tensor_data).reshape(single_tensor_shape)
        tensors.append(tensor)

    return tensors

# # Example usage
# bytes_data = ...  # your bytes data from WebSocket
# single_tensor_shape = (3, 224, 224)  # replace with your model's input shape
#
# tensors = bytes_to_tensors_tf(bytes_data, single_tensor_shape)
#
# # Now, `tensors` is a list of TensorFlow tensors, each ready to be used in your model


def load_model_from_bytes(model_bytes):
    """
    Load a PyTorch model from bytes.

    :param model_bytes: Bytes of the serialized TorchScript model.
    :return: Loaded PyTorch model.
    """
    buffer = io.BytesIO(model_bytes)
    model = torch.jit.load(buffer)
    return model


def evaluate_model(model, input_list):
    """
    Evaluate a list of inputs using the provided PyTorch model.

    :param model: Loaded PyTorch model.
    :param input_list: List of inputs to evaluate. Each input should be a tensor.
    :return: List of model outputs.
    """
    outputs = []
    for input_tensor in input_list:
        output = model(input_tensor)
        outputs.append(output)
    return outputs

# # Assume `model_bytes` is your TorchScript model in bytes form
# # and `input_list` is your list of input tensors.
#
# # Load the model
# loaded_model = load_model_from_bytes(model_bytes)
#
# # Evaluate the model with the list of inputs
# results = evaluate_model(loaded_model, input_list)
#
# # `results` will contain the outputs from the model for each input


def load_model_from_bytes_tf(model_bytes):
    """
    Load a TensorFlow model from bytes.

    :param model_bytes: Bytes of the serialized TensorFlow SavedModel.
    :return: Loaded TensorFlow model.
    """
    # Create a temporary directory to write the model bytes
    temp_dir = tempfile.mkdtemp()
    model_path = os.path.join(temp_dir, 'saved_model')

    # Write the model bytes to the directory
    with open(model_path, 'wb') as file:
        file.write(model_bytes)

    # Load the model from the temporary directory
    model = tf.saved_model.load(model_path)

    # Cleanup the temporary directory if needed
    # shutil.rmtree(temp_dir)

    return model

def evaluate_model(model, input_list):
    """
    Evaluate a list of inputs using the provided TensorFlow model.

    :param model: Loaded TensorFlow model.
    :param input_list: List of inputs to evaluate. Each input should be a tensor or a compatible type.
    :return: List of model outputs.
    """
    outputs = []
    for input_tensor in input_list:
        output = model(input_tensor)
        outputs.append(output)
    return outputs

# # Assume `model_bytes` is your TensorFlow SavedModel in bytes form
# # and `input_list` is your list of input tensors.
#
# # Load the model
# loaded_model = load_model_from_bytes(model_bytes)
#
# # Evaluate the model with the list of inputs
# results = evaluate_model(loaded_model, input_list)
#
# # `results` will contain the outputs from the model for each input


# import torch
#
# # Check if GPU is available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # Load your model (assuming it's already loaded into 'model')
# model.to(device)
#
# # When evaluating, move your input data to the same device
# # Assume 'inputs' is your data for evaluation
# inputs = inputs.to(device)
#
# # Now perform your model evaluation
# with torch.no_grad():
#     outputs = model(inputs)
#     # ... additional operations
#
# # Move outputs back to CPU if necessary for further CPU-bound processing
# outputs = outputs.to('cpu')


# Enum mappings
MESSAGE_TYPE_MAP = {
    0: "Request",
    1: "Response"
}

KINODE_ML_LIBRARY_MAP = {
    0: "PyTorch",
    1: "TensorFlow"
}

KINODE_ML_DATA_TYPE_MAP = {
    0: "Float16",
    1: "BFloat16",
    2: "Float32",
    3: "Float64",
    4: "Int8",
    # ... other data types
}

# Reverse mapping for serialization
REVERSE_MESSAGE_TYPE_MAP = {v: k for k, v in MESSAGE_TYPE_MAP.items()}
REVERSE_KINODE_ML_LIBRARY_MAP = {v: k for k, v in KINODE_ML_LIBRARY_MAP.items()}
REVERSE_KINODE_ML_DATA_TYPE_MAP = {v: k for k, v in KINODE_ML_DATA_TYPE_MAP.items()}


def deserialize_kinode_ml_request(encoded_blob):
    """Deserialize the blob into KinodeMlRequest structure."""
    blob = msgpack.unpackb(encoded_blob, raw=False)
    blob['library'] = KINODE_ML_LIBRARY_MAP[blob['library']]
    blob['data_type'] = KINODE_ML_DATA_TYPE_MAP[blob['data_type']]
    return blob


def deserialize_message(encoded_message):
    """Deserialize MessagePack-encoded KinodeExtWSMessage to a Python dictionary."""
    message = msgpack.unpackb(encoded_message, raw=False)
    message["message_type"] = MESSAGE_TYPE_MAP[message["message_type"]]
    message["blob"] = deserialize_kinode_ml_request(message["blob"])
    return message


def serialize_kinode_ml_request(library, model_bytes_length, data_shape, data_type):
    """Serialize KinodeMlRequest structure to MessagePack bytes."""
    request = {
        "library": REVERSE_KINODE_ML_LIBRARY_MAP[library],
        "data_shape": data_shape,
        "data_type": REVERSE_KINODE_ML_DATA_TYPE_MAP[data_type],
        "model_bytes": ,
        "data_bytes": ,
    }
    return msgpack.packb(request, use_bin_type=True)


def serialize_message(id, message_type, kinode_ml_request):
    """Serialize data to MessagePack format."""
    message = {
        "id": id,
        "message_type": REVERSE_MESSAGE_TYPE_MAP[message_type],
        "blob": serialize_kinode_ml_request(kinode_ml_request),
    }
    return msgpack.packb(message, use_bin_type=True)


async def run(port, process="ml:ml:sys"):
    uri = f"ws://localhost:{port}/{process}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            message = deserialize_message(message)
            print(f"Got message with id {message.id}")


def main():
    parser = argparse.ArgumentParser(
        description="Connect to a WebSocket server on a specified port.",
    )
    parser.add_argument(
        "port",
        type=int,
        help="Port number of the WebSocket server to connect to.",
    )
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(run(args.port))

if __name__ == "__main__":
    main()
