# %%
import logging
import random as rng
import re

import networkx as nx
import numpy as np
import pandas as pd
from ipysigma import Sigma
from matplotlib import pyplot as plt
from rich import print
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# %load_ext rich

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RedditNetwork")

SEED = 42
random = np.random.RandomState(SEED)

tqdm.pandas()


# %% [markdown]
# ### Documentation:
# - https://praw.readthedocs.io/en/stable/code_overview/models/submission.html
# - https://praw.readthedocs.io/en/stable/code_overview/models/comment.html

# %% [markdown]
# ## Load the data

# %%
subreddit = "IndiaInvestments"

submissions = pd.read_json(
    f"./data/{subreddit}_submissions.zst",
    compression={
        "method": "zstd",
        "max_window_size": 2**31,
    },
    lines=True,
)

comments = pd.read_json(
    f"./data/{subreddit}_comments.zst",
    compression={
        "method": "zstd",
        "max_window_size": 2**31,
    },
    lines=True,
)

# %%
submissions.head()

# %%
comments.head()

# %% [markdown]
# ## Transform the data

# %%
submissions = submissions[~submissions['author'].isin(["[deleted]"])]

# %%
comments = comments.rename(
    columns={
        "link_id": "submission_id",
        "id": "comment_id",
    }
)

# Remove prefix "t3_" from submission_id as it is not required
comments["submission_id"] = comments["submission_id"].str.replace("t3_", "")

# %%
comments["link_type"] = comments["parent_id"].apply(
    lambda x: "respond_comment" if x.startswith("t1_") else "respond_submission"
)

comments

# %%
comments["submission_author"] = comments["submission_id"].map(
    submissions.set_index("id")["author"]
)

# %%
comments

# %%
comments_graph = comments[
    [
        "comment_id",
        "parent_id",
        "submission_id",
        "link_type",
        "score",
        "author",
        "submission_author",
    ]
].copy()

comments_graph["parent_id"] = comments_graph["parent_id"].apply(
    lambda x: re.sub(r"t[13]_", "", x)
)

comments_graph = comments_graph[
    ~comments_graph["author"].isin(["[deleted]"])
    & ~comments_graph["submission_author"].isin(["[deleted]"])
]

comments_graph

# %% [markdown]
# #### Get parent comment author

# %%
def get_parent_author(row, parent_lookup):
    if row["link_type"] == "respond_submission":
        return row["submission_author"]

    else:
        return parent_lookup.get(row["parent_id"], None)


parent_lookup = comments_graph.set_index("comment_id")["author"].to_dict()
comments_graph["parent_author"] = comments_graph.progress_apply(
    get_parent_author, axis=1, parent_lookup=parent_lookup
)


# %%
# Dropping comments with deleted `parent_author`
comments_graph = comments_graph.dropna(subset=['parent_author'])

comments_graph

# %% [markdown]
# ## Create a graph

# %%
# Convert edge list to networkx graph

G_comments_graph = nx.from_pandas_edgelist(
    comments_graph,
    source="author",
    target="parent_author",
    edge_attr=["comment_id", "submission_id", "link_type", "score", "parent_id"],
    create_using=nx.DiGraph,
)

nodes = list(G_comments_graph.nodes())


print(f"Number of nodes: {G_comments_graph.number_of_nodes()}")
print(f"Number of edges: {G_comments_graph.number_of_edges()}")

# %% [markdown]
# ### Sampling from the edges

# %%
# Since we have a large number of edges, take a sample of 50000 edges for faster processing

rng.seed(SEED)
G_sample = nx.DiGraph()
G_sample.add_edges_from(
    rng.sample(list(G_comments_graph.edges), 50000),
)

print(f"Sample Number of nodes: {G_sample.number_of_nodes()}")
print(f"Sample Number of edges: {G_sample.number_of_edges()}")


# %%
G_comments_graph = G_sample.copy()

# %%
# These are authors with no replies
comments_graph[
    comments_graph["author"].isin(
        [node for node, in_degree in G_comments_graph.in_degree() if in_degree == 0]
    )
]

# %% [markdown]
# ### Degree

# %%
# Calculating the degree centrality

degree_centrality = pd.DataFrame(
    nx.degree_centrality(G_comments_graph).items(),
    columns=["node", "degree_centrality"],
).sort_values("degree_centrality", ascending=False)

degree_centrality

# %%
# Calculating the betweenness centrality
betweenness_centrality = pd.DataFrame(
    nx.betweenness_centrality(G_comments_graph, k=500, seed=SEED).items(),
    columns=["node", "betweenness_centrality"],
).sort_values("betweenness_centrality", ascending=False)


betweenness_centrality

# %%
# Calculating the closeness centrality
closeness_centrality = pd.DataFrame(
    nx.closeness_centrality(G_sample).items(),
    columns=["node", "closeness_centrality"],
).sort_values("closeness_centrality", ascending=False)

closeness_centrality


# %%
centrality = (
    degree_centrality.merge(betweenness_centrality, on="node")
    .merge(closeness_centrality, on="node")
    .set_index("node")
)

centrality

# %% [markdown]
# ## Identify the most influential nodes

# %%
number_of_submissions = (
    submissions["author"]
    .value_counts()
    .reset_index()
    .rename(columns={"author": "node", "count": "number_of_submissions"})
    .set_index("node")
)

number_of_comments = (
    comments_graph["author"]
    .value_counts()
    .reset_index()
    .rename(columns={"author": "node", "count": "number_of_comments"})
    .set_index("node")
)

c_to_s_c = comments_graph.groupby(["author", "link_type"]).size().unstack().fillna(0)

centrality.shape, number_of_submissions.shape, number_of_comments.shape, c_to_s_c.shape

# %%
scores = (
    centrality.merge(
        number_of_submissions, how="left", left_index=True, right_index=True
    )
    .merge(number_of_comments, how="left", left_index=True, right_index=True)
    .merge(c_to_s_c, how="left", left_index=True, right_index=True)
).fillna(0)

scores

# %% [markdown]
# ## Perform PCA

# %%
pca = PCA(n_components=5)

scaler = StandardScaler()


scores_scaled = scaler.fit_transform(
    SimpleImputer(strategy="constant", fill_value=0).fit_transform(
        scores[
            [
                "closeness_centrality",
                "number_of_submissions",
                "number_of_comments",
                "respond_comment",
                "respond_submission",
            ]
        ]
    )
)


scores_pca = pca.fit_transform(scores_scaled)


# %%
plt.figure(figsize=(10, 7))
plt.plot(range(1, 6), pca.explained_variance_ratio_, marker="o", linestyle="--")
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.xticks(range(1, 6))
plt.show()

print(f'First component weights: {pca.components_[0]}')

# %%
fc1_weights = np.abs(pca.components_[0]) / np.sum(np.abs(pca.components_[0]))
fc1_weights

# %%
weights = [0.2, 0.10, 0.25, 0.25, 0.2]

# %%
scores_df = pd.DataFrame(
    scores_scaled,
    columns=[
        "closeness_centrality",
        "number_of_submissions",
        "number_of_comments",
        "respond_comment",
        "respond_submission",
    ],
)

scores_df["total_score"] = np.dot(scores_scaled, weights)
scores_df_sorted = scores_df.sort_values(by="total_score", ascending=False)
scores_df_sorted.set_index(centrality.index, inplace=True)

scores_df_sorted


# %% [markdown]
# ## Plot top 100 nodes

# %%
top_graph = G_comments_graph.subgraph(scores_df_sorted.index[:100])

plt.figure(figsize=(12, 8))
nx.draw(
    top_graph,
    with_labels=True,
    node_size=100,
    alpha=0.3,
    arrows=True,
    pos=nx.fruchterman_reingold_layout(top_graph),
)

plt.title("Top 100 Authors")
plt.savefig("top_100_authors.png", dpi=300)
plt.show()


# %%
sigma = Sigma(top_graph, default_node_color="lightblue", node_size=top_graph.degree, height=800)

sigma


# %%
sigma.to_html("top_100_authors.html")


