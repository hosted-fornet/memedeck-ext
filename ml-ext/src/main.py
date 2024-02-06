#!/usr/bin/env python3

import argparse
from datetime import datetime
import io
import json
import logging
import os
import tempfile

import asyncio
import msgpack
import torch
import websockets

import numpy as np
import tensorflow as tf


def bytes_to_tensors_torch(bytes_data, single_tensor_shape, dtype=torch.float32):
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


#def bytes_to_tensors_tf(bytes_data, single_tensor_shape, dtype=tf.float32):
#    print(bytes_data, single_tensor_shape, dtype)
#    element_size = np.dtype(dtype.as_numpy_dtype).itemsize
#    tensor_size = np.prod(single_tensor_shape) * element_size
#    num_tensors = len(bytes_data) // tensor_size
#
#    tensors = []
#    for i in range(num_tensors):
#        start = i * tensor_size
#        end = start + tensor_size
#        tensor_data = np.frombuffer(bytes_data[start:end], dtype=dtype.as_numpy_dtype)
#        tensor = tf.convert_to_tensor(tensor_data).reshape(single_tensor_shape)
#        tensors.append(tensor)
#
#    return tensors

# # Example usage
# bytes_data = ...  # your bytes data from WebSocket
# single_tensor_shape = (3, 224, 224)  # replace with your model's input shape
#
# tensors = bytes_to_tensors_tf(bytes_data, single_tensor_shape)
#
# # Now, `tensors` is a list of TensorFlow tensors, each ready to be used in your model


def load_model_from_bytes_torch(model_bytes):
    """
    Load a PyTorch model from bytes.

    :param model_bytes: Bytes of the serialized TorchScript model.
    :return: Loaded PyTorch model.
    """
    buffer = io.BytesIO(model_bytes)
    model = torch.jit.load(buffer)
    return model


def evaluate_model_torch(model, input_list):
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


def load_model_keras(model, models):
    """
    Load a TensorFlow Keras model.

    :param model_bytes: Bytes of the serialized TensorFlow SavedModel.
    :return: Loaded TensorFlow model.
    """
    model_bytes = model.get("Bytes")
    if model_bytes is not None:
        logging.debug(f"Loading Keras model from bytes (length {len(model_bytes)})...")
        # Create a temporary directory to write the model bytes
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'saved_model')

        # Write the model bytes to the directory
        with open(model_path, 'wb') as file:
            file.write(bytes(model_bytes))

        # Load the model from the temporary directory
        model = tf.keras.saving.load_model(model_path)

        # Cleanup the temporary directory if needed
        # shutil.rmtree(temp_dir)
        logging.debug("Done loading Keras model from bytes")

        return model

    model_name = model.get("Name")
    if model_name is not None:
        logging.debug(f"Loading Keras model from name ({model_name})...")
        model_path = models.get(model_name)
        logging.debug(f"Found model name {model_name} at path {model_path}...")
        if model_path is None:
            raise Exception(f"no such model {model_name} amongst {models}")
        model = tf.keras.saving.load_model(model_path)
        logging.debug(f"Done loading Keras model from name ({model_name})")

        return model

    raise Exception(f"unexpected model type: {model}")


def evaluate_model_tf(model, input_list):
    """
    Evaluate a list of inputs using the provided TensorFlow model.

    :param model: Loaded TensorFlow model.
    :param input_list: List of inputs to evaluate. Each input should be a tensor or a compatible type.
    :return: List of model outputs.
    """
    logging.debug(f"Evaluating TF model on {len(input_list)} tensors...")
    outputs = []
    for input_tensor in input_list:
        output = model(input_tensor)
        outputs.append(output)
    logging.debug(f"Done evaluating TF model on {len(input_list)} tensors")
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
KINODE_ML_DATA_TYPE_TO_PYTORCH_MAP = {
    "Float16": torch.float16,
    #"BFloat16": torch.bfloat16,
    "Float32": torch.float32,
    "Float64": torch.float64,
    "Int8": torch.int8,
    # ... other data types
}

KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP = {
    "Float16": tf.float16,
    #"BFloat16": tf.bfloat16,
    "Float32": tf.float32,
    "Float64": tf.float64,
    "Int8": tf.int8,
    # ... other data types
}

KINODE_ML_DATA_TYPE_TO_NUMPY_MAP = {
    "Float16": np.float16,
    #"BFloat16": np.bfloat16,
    "Float32": np.float32,
    "Float64": np.float64,
    "Int8": np.int8,
    # ... other data types
}

# Reverse mapping for serialization
REVERSE_KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP = {v: k for k, v in KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP.items()}
REVERSE_KINODE_ML_DATA_TYPE_TO_NUMPY_MAP = {v: k for k, v in KINODE_ML_DATA_TYPE_TO_NUMPY_MAP.items()}


def deserialize_message(encoded_message):
    """Deserialize MessagePack-encoded KinodeExtWSMessage to a Python dictionary."""
    message = msgpack.unpackb(encoded_message, raw=False)
    message = message["WebSocketExtPushData"]
    message["blob"] = msgpack.unpackb(bytes(message["blob"]), raw=False)
    return message


def serialize_kinode_ml_response(library, data_tensor):
    """Serialize KinodeMlResponse structure to MessagePack bytes."""
    if library == "PyTorch":
        pass
    elif library == "TensorFlow":
        pass
    elif library == "Keras":
        data_shape, data_type, data_bytes = serialize_tensor_data_tf(data_tensor)
    else:
        pass
    request = {
        "library": library,
        "data_shape": data_shape,
        "data_type": data_type,
        "data_bytes": data_bytes,
    }
    return list(msgpack.packb(request, use_bin_type=True))


def serialize_message(id, message_type, library, data_tensor):
    """Serialize data to MessagePack format."""
    message = {
        "WebSocketExtPushData": {
            "id": id,
            "kinode_message_type": message_type,
            "blob": serialize_kinode_ml_response(library, data_tensor),
        }
    }
    return msgpack.packb(message, use_bin_type=True)


def serialize_tensor_data_tf(tensors):
    dtype = REVERSE_KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP[tensors[0].dtype]
    # Ensure all tensors are on CPU memory
    tensors = [tensor.numpy() for tensor in tensors]

    # Check if tensors list is not empty and retrieve the dtype and shape from the first tensor
    if not tensors:
        raise ValueError("Tensors list must not be empty")

    shape = tensors[0].shape
    np_dtype = KINODE_ML_DATA_TYPE_TO_NUMPY_MAP[dtype]

    # Convert each tensor to the desired numpy dtype and flatten
    flattened_arrays = [tensor.astype(np_dtype).ravel() for tensor in tensors]

    # Concatenate all numpy arrays into one
    concatenated_array = np.concatenate(flattened_arrays)

    # Convert the concatenated array to bytes
    bytes_data = list(concatenated_array.tobytes())

    return shape, dtype, bytes_data


def deserialize_tensor_data_tf(bytes_list, shape, dtype):
    logging.debug("Deserializing TF data...")
    # Determine the numpy and TensorFlow data types from the mappings
    np_dtype = KINODE_ML_DATA_TYPE_TO_NUMPY_MAP[dtype]
    tf_dtype = KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP[dtype]

    # Calculate the size of a single tensor's data in bytes
    single_tensor_byte_size = np.prod(shape) * np.dtype(np_dtype).itemsize

    # Ensure the byte_list is divisible by the size of a single tensor
    if len(bytes_list) % single_tensor_byte_size != 0:
        raise ValueError(f"byte_list size ({len(bytes_list)}) is not a multiple of shape product and dtype size ({single_tensor_byte_size})")

    # Calculate how many tensors are represented by the byte_list
    num_tensors = len(bytes_list) // single_tensor_byte_size

    # Convert the entire bytes_list to a numpy array first
    full_array = np.frombuffer(bytes(bytes_list), dtype=np_dtype)

    # Split the array into multiple arrays, each representing a tensor, and convert each to a TensorFlow tensor
    tensors = [tf.convert_to_tensor(full_array[i * np.prod(shape):(i + 1) * np.prod(shape)].reshape(shape), dtype=tf_dtype)
               for i in range(num_tensors)]

    logging.debug("Done deserializing TF data")
    return tensors


async def run(port, models, process="ml:ml:sys"):
    uri = f"ws://localhost:{port}/{process}"
    async with websockets.connect(uri, ping_interval=None, max_size=100 * 1024 * 1024) as websocket:
        while True:
            try:
                message = await websocket.recv()
                message = deserialize_message(message)
                logging.info(f"Got message with id {message['id']}")
                request = message["blob"]

                if request["library"] == "PyTorch":
                    logging.error("TODO")
                    pass
                    # # load model
                    # model = load_model_from_bytes_torch(request["model_bytes"])
                    # # load data
                    # dtype = KINODE_ML_DATA_TYPE_TO_PYTORCH_MAP[request["data_type"]]
                    # data = bytes_to_tensors_torch(
                    #     request["data_bytes"],
                    #     request["data_shape"],
                    #     dtype=dtype,
                    # )
                    # # evaluate model
                    # outputs = evaluate_model_torch(model, data)
                    # # serialize response
                    # pass
                elif request["library"] == "Keras":
                    # load model
                    try:
                        model = load_model_keras(request["model"], models)
                    except Exception as e:
                        logging.error(f"Failed to load Keras model: {e}")
                        continue

                    # load data
                    dtype = KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP[request["data_type"]]
                    request['data_bytes'] = deserialize_tensor_data_tf(
                        request['data_bytes'],
                        request['data_shape'],
                        request['data_type'],
                    )

                    # evaluate model
                    outputs = evaluate_model_tf(model, request["data_bytes"])

                    # serialize response
                    logging.debug(f"Sending back Kinode Message of type {message['kinode_message_type']}")
                    response = serialize_message(
                        message["id"],
                        message["kinode_message_type"],
                        request["library"],
                        outputs,
                    )
                elif request["library"] == "TensorFlow":
                    # load model
                    model = load_model_from_bytes_tf(request["model_bytes"])

                    # load data
                    dtype = KINODE_ML_DATA_TYPE_TO_TENSORFLOW_MAP[request["data_type"]]
                    try:
                        request['data_bytes'] = deserialize_tensor_data_tf(
                            request['data_bytes'],
                            request['data_shape'],
                            request['data_type'],
                        )
                    except ValueError as e:
                        logging.error(f"Failed to deserialize TF tensor: {e}")
                        continue


                    # evaluate model
                    outputs = evaluate_model_tf(model, request["data_bytes"])

                    # serialize response
                    response = serialize_message(
                        message["id"],
                        message["kinode_message_type"],
                        request["library"],
                        outputs,
                    )
                else:
                    # print error
                    logging.error(f"Don't recognize library {request['library']}; try PyTorch or TensorFlow")
                    # send error over ws
                    pass

                await websocket.send(response)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")


def validate_paths(paths_dict):
    """Validate if the paths exist, error out if not."""
    invalid_paths = []
    for name, path in paths_dict.items():
        if not os.path.exists(path):
            invalid_paths.append((name, path))

    if invalid_paths:
        for name, path in invalid_paths:
            logging.error(f"Error: The path for '{name}' does not exist: {path}")
        exit(1)
    logging.debug("Given paths successfully validated")


def setup_logging(console_level, file_level):
    """
    Configures logging to both the terminal and a file with different levels and formats.
    """
    console_level = getattr(logging, console_level.upper())
    file_level = getattr(logging, file_level.upper())

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of messages

    # Create console handler and set level to console_level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler and set level to file_level
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_handler = logging.FileHandler(f"{date_str}-ml-ext.log")
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(
        description="Connect to a WebSocket server on a specified port.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number of the WebSocket server to connect to.",
        required=True,
    )
    parser.add_argument(
        "--models",
        type=str,
        help="JSON string with name and path pairs",
        required=False,
    )
    parser.add_argument(
        '--console-log',
        default='INFO',
        help='Set the logging level for the console (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
    )
    parser.add_argument(
        '--file-log',
        default='DEBUG',
        help='Set the logging level for the file (DEBUG, INFO, WARNING, ERROR, CRITICAL)',
    )
    args = parser.parse_args()
    setup_logging(args.console_log, args.file_log)
    logging.info(f"Got args: {args}")
    if args.models is not None:
        try:
            models = json.loads(args.models)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            exit(1)
    else:
        models = {}
    validate_paths(models)

    asyncio.get_event_loop().run_until_complete(run(args.port, models))

if __name__ == "__main__":
    main()
