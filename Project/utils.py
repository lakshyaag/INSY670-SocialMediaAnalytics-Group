#!/usr/bin/env python

"""
This module contains utility functions for the project. 
author: @lakshyaag
"""

import logging
import os

import httpx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pandas.api.typing import JsonReader
from tqdm import tqdm

tqdm.pandas()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataLoader")

BASE_URL = "https://the-eye.eu/redarcs/files/"


def get_input_files(
    subreddit: str, type: str = "submissions", chunk_size: int = 10000
) -> JsonReader:
    """
    Fetches and downloads input files from the-eye.eu, subsequently returning the data as a pandas DataFrame.

    Parameters:
    - subreddit (str): The specific subreddit for which to download data.
    - type (str): The category of data to download, options include "submissions" or "comments".
    - chunk_size (int): The size of chunks for reading the data, enhancing performance for large files.

    Returns:
    - pd.DataFrame genrator: A generator object containing the data in chunks.
    """
    # Create data directory if it doesn't exist
    os.makedirs("./data", exist_ok=True)

    file_name = f"{subreddit}_{type}.zst"
    file_path = f"./data/{file_name}"
    json_read_params = {
        "compression": {
            "method": "zstd",
            "max_window_size": 2**31,
        },
        "lines": True,
    }

    if not os.path.exists(file_path):
        logger.info(f"Downloading {type} for {subreddit}")
        url = BASE_URL + file_name
        with httpx.stream("GET", url) as r:
            r.raise_for_status()

            total_size_in_bytes = int(r.headers.get("content-length", 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(file_path, "wb") as f:
                for data in r.iter_bytes(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logger.error("ERROR, something went wrong")

            logger.info(f"Downloaded and saved {type} for {subreddit}")

    else:
        logger.info(f"File {file_name} already exists")

    return pd.read_json(file_path, **json_read_params, chunksize=chunk_size)


def view_sample_graph(graph: nx.Graph, n: int = 50) -> nx.Graph:
    """
    Displays a sample graph with a specified number of nodes.

    Parameters:
    - graph (nx.Graph): The graph to sample from.
    - n (int): The number of nodes to sample.

    Returns:
    - nx.Graph: The sampled graph.
    """

    sample_nodes = np.random.choice(list(graph.nodes), n)
    sample_graph = graph.subgraph(sample_nodes)

    pos = nx.spring_layout(sample_graph)

    plt.figure(figsize=(10, 10))
    nx.draw(sample_graph, pos, with_labels=True, node_size=300, font_size=10)
    plt.axis("off")
    plt.title(f"Sample Graph with {n} Nodes")
    plt.show()

    return sample_graph
