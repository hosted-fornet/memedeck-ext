#!/usr/bin/env python3

import argparse
from datetime import datetime
import json
import logging
import os
import pickle
import time

import asyncio
import msgpack
import websockets

import numpy as np

from scipy.spatial.distance import cdist
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model


def predict_template(image, kmeans_model, pca_model, cluster_to_template_ids, vgg16_model):
    """
    This function takes the path of the image, kmeans model, pca model, cluster_to_template_ids dictionary and vgg16 model
    and returns the top 3 meme template ids with their probabilities as a json string.

    Args:
        image_path (str): The path of the image
        kmeans_path (str): The path of the kmeans model
        pca_path (str): The path of the pca model
        cluster_to_template_ids_path (str): The path of the cluster_to_template_ids dictionary
        vgg16_path (str): The path of the vgg16 model
    """
    times = {}
    start = time.time()
    # load the meme image
    image_path = "/tmp/image"
    with open(image_path, "wb") as file:
        file.write(bytes(image))
    image = load_img(image_path, target_size=(224,224))
    image = np.array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    times["load"] = time.time() - start

    start = time.time()
    # get the features
    feature = vgg16_model.predict(image, use_multiprocessing=True, verbose=0)
    times["vgg16"] = time.time() - start
    start = time.time()
    feature = pca_model.transform(feature)
    times["pca"] = time.time() - start

    start = time.time()
    # predict the cluster
    prediction = kmeans_model.predict(feature)
    # get the cluster centers
    cluster_centers = kmeans_model.cluster_centers_
    # calculate the distance of the input image from each cluster center
    distances = cdist(feature, cluster_centers, 'euclidean')
    # calculate the probability of the input image belonging to each cluster
    max_distance = np.max(distances)
    probabilities = (max_distance - distances) / max_distance
    # get the probabilities
    probabilities = list(probabilities[0])
    times["kmeans"] = time.time() - start
    start = time.time()
    # get the template_ids
    template_ids = list(cluster_to_template_ids[prediction[0]])
    # get the probabilities with template_ids
    probabilities_with_template_ids = list(zip(template_ids, probabilities))
    # sort the probabilities
    probabilities_with_template_ids.sort(key=lambda x: x[1], reverse=True)
    # return the top 3 template_ids
    logging.info(f"Probabilities: {probabilities_with_template_ids}")
    prob_json = [{'template': template_id, 'prob': prob} for template_id, prob in probabilities_with_template_ids[:3]]
    # return as json string with json.dumps
    times["rest"] = time.time() - start
    logging.debug(f"run time of each predict step: {times}")
    return json.dumps(prob_json, indent=2)


def serialize_message(id, message_type, blob):
    """Serialize data to MessagePack format."""
    message = {
        "WebSocketExtPushData": {
            "id": id,
            "kinode_message_type": message_type,
            "blob": blob,
        }
    }
    return msgpack.packb(message, use_bin_type=True)


def deserialize_message(encoded_message):
    """Deserialize MessagePack-encoded WebSocketExtPushdata to a Python dictionary."""
    message = msgpack.unpackb(encoded_message, raw=False)
    message = message["WebSocketExtPushData"]
    return message


async def run(port, kmeans_path, pca_path, cluster_to_template_ids_path, vgg16_path, process="predict:memedeck:holium.os"):
    kmeans_model = pickle.load(open(kmeans_path, 'rb'))
    pca_model = pickle.load(open(pca_path, 'rb'))
    cluster_to_template_ids = pickle.load(open(cluster_to_template_ids_path, 'rb'))
    vgg16_model = load_model(vgg16_path)

    uri = f"ws://localhost:{port}/{process}"
    async with websockets.connect(uri, ping_interval=None, max_size=100 * 1024 * 1024) as websocket:
        while True:
            try:
                message = await websocket.recv()
                message = deserialize_message(message)
                logging.info(f"Got message with id {message['id']}")
                image = message["blob"]

                result = predict_template(
                    image,
                    kmeans_model,
                    pca_model,
                    cluster_to_template_ids,
                    vgg16_model,
                )
                response = serialize_message(
                    message["id"],
                    message["kinode_message_type"],
                    list(result.encode("utf-8")),
                )
                await websocket.send(response)
            except Exception as e:
                logging.error(f"Unexpected error: {e}")


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
        "--kmeans",
        type=str,
        help="Path to kmeans model file",
        required=True,
    )
    parser.add_argument(
        "--pca",
        type=str,
        help="Path to PCA model file",
        required=True,
    )
    parser.add_argument(
        "--cluster-to-template-ids",
        type=str,
        help="Path to cluster_to_template_ids file",
        required=True,
    )
    parser.add_argument(
        "--vgg16",
        type=str,
        help="Path to vgg16 model file",
        required=True,
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

    asyncio.get_event_loop().run_until_complete(run(
        args.port,
        args.kmeans,
        args.pca,
        args.cluster_to_template_ids,
        args.vgg16,
    ))

if __name__ == "__main__":
    main()
