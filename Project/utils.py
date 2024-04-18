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
logger = logging.getLogger("UTILS")

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

    plt.figure(figsize=(20, 20))

    pos = nx.spring_layout(sample_graph)
    nx.draw(
        sample_graph,
        pos,
        with_labels=True,
        node_size=300,
        font_size=10,
    )
    plt.axis("off")
    plt.title(f"Sample Graph with {n} Nodes")
    plt.show()

    return sample_graph


# Assign submission to topic
def assign_topic(lda_model, corpus, topic_interpretation):
    """
    Assigns a topic to each submission based on the topic with the highest probability

    Parameters:
    - lda_model (gensim.models.LdaMulticore): The trained LDA model.
    - corpus (list): The corpus of documents.
    - topic_interpretation (dict): A dictionary mapping topic numbers to their interpretations.
    """
    doc_lda = lda_model[corpus]
    topics = [max(doc, key=lambda x: x[1])[0] for doc in doc_lda]
    return [topic_interpretation[topic] for topic in topics]


def get_topic_submissions(SELECTED_TOPIC, submissions, comments):
    """
    Retrieves submissions and their corresponding comments for a given topic

    Parameters:
    - SELECTED_TOPIC (str): The selected topic.
    - submissions (pd.DataFrame): The DataFrame containing submissions.
    - comments (pd.DataFrame): The DataFrame containing comments.
    """
    submission_t = submissions[submissions["topic"] == SELECTED_TOPIC]
    comments_t = comments[comments["submission_id"].isin(submission_t["id"])]

    return submission_t, comments_t


def create_bipartite_graph(df: pd.DataFrame):
    """
    Creates a bipartite graph from a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to create the graph from.
    """

    B = nx.Graph()

    users = df["author"].unique()
    posts = df["submission_id"].unique()

    B.add_nodes_from(users, bipartite=0)
    B.add_nodes_from(posts, bipartite=1)

    edges = [
        (row["author"], row["submission_id"], row["score"])
        for index, row in df.iterrows()
    ]

    B.add_weighted_edges_from(edges)

    if nx.is_bipartite(B):
        user_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
        post_nodes = set(B) - user_nodes

        M = nx.bipartite.biadjacency_matrix(B, user_nodes, post_nodes)

    return user_nodes, post_nodes, B, M


def project_bipartite_graph(M, nodes):
    """
    Projects a bipartite graph onto one of its node sets.

    Parameters:
    - M (scipy.sparse.csr_matrix): The bipartite adjacency matrix.
    - nodes (list): The list of nodes to project onto.

    """
    G_projection = nx.Graph()
    logger.info("Adding nodes...")
    G_projection.add_nodes_from(nodes)

    ("Adding edges...")
    G_projection.add_edges_from([(nodes[i], nodes[j]) for i, j in zip(*M.nonzero())])

    return G_projection


def create_topic_network(
    submission_t, comments_t, filter_posts=False, filter_threshold=None
):
    """
    Creates a network of users who have commented on submissions related to a given topic.

    Parameters:
    - submission_t (pd.DataFrame): The DataFrame containing submissions for a given topic.
    - comments_t (pd.DataFrame): The DataFrame containing comments for a given topic.
    - filter_posts (bool): Whether to filter posts based on a threshold.
    - filter_threshold (int): The threshold to filter posts by.
    """

    if not filter_posts:
        assert filter_threshold is None

    if filter_posts:
        authors_with_threshold_posts = (
            submission_t.groupby("author")
            .filter(lambda x: len(x) > filter_threshold)["author"]
            .unique()
        )

        submission_t = submission_t[
            submission_t["author"].isin(authors_with_threshold_posts)
        ]
        comments_t = comments_t[comments_t["submission_id"].isin(submission_t["id"])]

    # Calculate user weights
    user_weights = (
        comments_t.groupby("author")
        .agg(
            average_score=("score", "mean"),
            num_comments=("body", "count"),
        )
        .sort_values(by=["num_comments", "average_score"], ascending=False)
    )

    user_nodes_t, post_nodes_t, B_t, M_t = create_bipartite_graph(comments_t)

    G_user_t = project_bipartite_graph(M_t, list(user_nodes_t))

    # G_post_t = project_bipartite_graph(M_t.T, list(post_nodes_t))

    for attribute in user_weights.columns:
        nx.set_node_attributes(
            G_user_t,
            pd.Series(user_weights[attribute], index=user_weights.index).to_dict(),
            attribute,
        )

    return G_user_t, user_weights, B_t


def get_largest_component(G):
    """
    Retrieves the largest connected component of a graph.

    Parameters:
    - G (nx.Graph): The input graph.
    """

    largest_cc_t = max(nx.connected_components(G), key=len)
    G_largest_cc_t = G.subgraph(largest_cc_t)

    (
        f"Largest component has {len(G_largest_cc_t.nodes)} nodes and {len(G_largest_cc_t.edges)} edges."
    )

    return G_largest_cc_t


def calculate_centralities(G, betweenness_k=50):
    """
    Calculates centrality scores for a given graph.

    Parameters:
    - G (nx.Graph): The input graph.
    - betweenness_k (int): The number of nodes to sample for betweenness centrality. Default is 50.
    """

    degree_centrality = pd.DataFrame(
        nx.degree_centrality(G).items(),
        columns=["author", "degree_centrality"],
    ).sort_values(by="degree_centrality", ascending=False)

    betweenness_centrality = pd.DataFrame(
        nx.betweenness_centrality(G, k=betweenness_k, seed=42, weight="weight").items(),
        columns=["author", "betweenness_centrality"],
    ).sort_values(by="betweenness_centrality", ascending=False)

    centrality_scores = pd.concat(
        [
            degree_centrality.set_index("author"),
            betweenness_centrality.set_index("author"),
        ],
        axis=1,
    )

    return centrality_scores


def build_node_data(G):
    """
    Builds node data for a given graph.

    Parameters:
    - G (nx.Graph): The input graph.
    """

    centrality_scores = calculate_centralities(G)

    filtered_node_data = pd.DataFrame(G.nodes(data=True), columns=["author", "attrs"])

    filtered_node_data = (
        pd.concat(
            [
                filtered_node_data["author"],
                pd.json_normalize(filtered_node_data["attrs"]),
            ],
            axis=1,
        )
        .set_index("author")
        .merge(centrality_scores, left_index=True, right_index=True)
    )

    filtered_node_data["influencer_score"] = (
        filtered_node_data["degree_centrality"]
        + filtered_node_data["betweenness_centrality"]
    ) * np.log1p(filtered_node_data["num_comments"])

    filtered_node_data = filtered_node_data.sort_values(
        by="influencer_score", ascending=False
    )

    return filtered_node_data


def get_communities(G, resolution=1):
    """
    Retrieves communities in a graph using the Louvain method.

    Parameters:
    - G (nx.Graph): The input graph.
    - resolution (float): The resolution parameter for the Louvain method. Default is 1.
    """

    communities = nx.community.louvain_communities(
        G, weight="weight", seed=42, resolution=resolution, backend="cugraph"
    )
    # Add community as node attribute to the graph
    for community_idx, community in enumerate(communities):
        for node in community:
            G.nodes[node]["community"] = community_idx

    community_lens = {i: len(x) for i, x in enumerate(communities)}

    return G, communities, community_lens


def plot_community_graph(
    G_filtered_community,
    community_lens,
    community_idx,
    SELECTED_TOPIC,
    ax=None,
):
    """
    Plots a graph for a given community.

    Parameters:
    - G_filtered_community (nx.Graph): The input graph.
    - communities (list): The list of communities.
    - community_lens (dict): A dictionary mapping community indices to their lengths.
    - community_idx (int): The index of the community to plot.
    - SELECTED_TOPIC (str): The selected topic.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 15))

    # Calculate node sizes based on influencer scores
    influencer_scores = [
        d.get("influencer_score", 0) for n, d in G_filtered_community.nodes(data=True)
    ]
    max_score = max(influencer_scores)
    node_size = [(score / max_score) * 1000 for score in influencer_scores]

    top_degree_nodes = sorted(
        G_filtered_community.nodes(data=True),
        key=lambda x: x[1].get("degree_centrality", 0),
    )[-15:]

    top_betweenness_nodes = sorted(
        G_filtered_community.nodes(data=True),
        key=lambda x: x[1].get("betweenness_centrality", 0),
    )[-15:]

    nodes_label = top_degree_nodes + top_betweenness_nodes

    labels = {node[0]: node[0] for node in nodes_label}

    pos = nx.kamada_kawai_layout(G_filtered_community, scale=2, weight="weight")

    nx.draw_networkx_nodes(
        G_filtered_community,
        pos,
        node_size=node_size,
        node_color="red",
        ax=ax,
    )

    nx.draw_networkx_edges(
        G_filtered_community,
        pos,
        alpha=0.3,
        width=0.75,
        edge_color="grey",
        ax=ax,
    )

    nx.draw_networkx_labels(G_filtered_community, pos, labels, font_size=12, ax=ax)

    ax.set_title(
        f"Network plot for Community {community_idx} with {community_lens[community_idx]} nodes\nTopic: {SELECTED_TOPIC}\nNode size based on influencer score"
    )

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth("1")

    ax.set_axis_off()

    return G_filtered_community


def get_top_users(G, n=10):
    """
    Retrieves the top users in a graph based on their influencer scores.

    Parameters:
    - G (nx.Graph): The input graph.
    - n (int): The number of top users to retrieve. Default is 10.
    """
    top_users = sorted(
        G.nodes(data=True), key=lambda x: x[1]["influencer_score"], reverse=True
    )[:n]

    return top_users


def get_filtered_topic_graph(
    SELECTED_TOPIC,
    submission_t,
    comments_t,
    filter_posts=False,
    filter_threshold=None,
    largest_component=True,
):
    """
    Retrieves a filtered graph for a given topic.

    Parameters:
    - SELECTED_TOPIC (str): The selected topic.
    - submission_t (pd.DataFrame): The DataFrame containing submissions for a given topic.
    - comments_t (pd.DataFrame): The DataFrame containing comments for a given topic.
    - filter_posts (bool): Whether to filter posts based on a threshold.
    - filter_threshold (int): The threshold to filter posts by.
    - largest_component (bool): Whether to filter for the largest connected component.
    """

    G_user_t, user_weights, B_t = create_topic_network(
        submission_t, comments_t, filter_posts, filter_threshold
    )

    logger.info(f"Selected topic: {SELECTED_TOPIC}")

    logger.info(
        f"The graph has {len(G_user_t.nodes)} nodes and {len(G_user_t.edges)} edges."
    )

    if largest_component:
        G_largest_cc_t = get_largest_component(G_user_t)
    else:
        G_largest_cc_t = G_user_t

    # view_sample_graph(G_largest_cc_t, 100)

    # Filter nodes based on the number of comments
    nodes_with_more_than_one_comment = [
        node
        for node, attr in G_largest_cc_t.nodes(data=True)
        if attr.get("num_comments", 0) > 1
    ]

    G_filtered = G_largest_cc_t.subgraph(nodes_with_more_than_one_comment)

    logger.info(
        f"After filtering for nodes with more than one comment, the graph has {len(G_filtered.nodes)} nodes and {len(G_filtered.edges)} edges."
    )

    filtered_node_data = build_node_data(G_filtered)

    # Add centrality scores and influencer score as node attributes
    attributes = filtered_node_data[
        [
            "degree_centrality",
            "betweenness_centrality",
            "influencer_score",
        ]
    ].to_dict("index")

    nx.set_node_attributes(G_filtered, attributes)

    return G_filtered, filtered_node_data, B_t


def get_community_dataframe(G):
    """
    Retrieves a DataFrame of nodes in a community.

    Parameters:
    - G (nx.Graph): The input graph.
    """

    community_nodes_df = pd.DataFrame(G.nodes(data=True), columns=["author", "attrs"])
    community_nodes_df = (
        pd.concat(
            [
                community_nodes_df["author"],
                pd.json_normalize(community_nodes_df["attrs"]),
            ],
            axis=1,
        )
    ).sort_values(by="degree_centrality", ascending=False)

    return community_nodes_df
