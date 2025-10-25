from collections import defaultdict
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
from datetime import datetime
import dateparser  # type: ignore
from tqdm import tqdm  # type: ignore
import json
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import pickle
from multiprocessing import Pool
from functools import partial
from random import sample


class Node:
    def __init__(self, case_id: str, date: datetime, text: str) -> None:
        self.case_id = case_id
        self.date = date
        self.text = text


def days_diff(node1: Node, node2: Node):
    delta = node1.date - node2.date
    return abs(delta.days)


def common_out_neighbors(g: nx.DiGraph, i, j):
    return set(g.successors(i)).intersection(g.successors(j))


def common_in_neighbors(g: nx.DiGraph, i, j):
    return set(g.predecessors(i)).intersection(g.predecessors(j))


def compute_features(
    graph: nx.Graph,
    digraph: nx.DiGraph,
    vectorizer,
    node1: Node,
    node2: Node,
    remove_edge=False,
):
    # Compute the set of features for node1 and node2 in the given graph
    days_difference = days_diff(node1, node2)
    assert days_difference > 0
    if graph.has_edge(node1.case_id, node2.case_id) and remove_edge:
        graph.remove_edge(node1.case_id, node2.case_id)
    adamic_adar = list(nx.adamic_adar_index(graph, [(node1.case_id, node2.case_id)]))[
        0
    ][-1]
    pref_attach = list(
        nx.preferential_attachment(graph, [(node1.case_id, node2.case_id)])
    )[0][-1]
    common_neigh = len(list(nx.common_neighbors(graph, node1.case_id, node2.case_id)))
    common_in = len(list(common_in_neighbors(digraph, node1.case_id, node2.case_id)))
    common_out = len(list(common_out_neighbors(digraph, node1.case_id, node2.case_id)))
    if (
        adamic_adar or pref_attach or common_neigh or common_in or common_out
    ):  # if at least one of the structural values is greater than one
        cos_sim = cosine_similarity(vectorizer.transform([node1.text, node2.text]))[0][
            1
        ]
        worth = True
    else:
        worth = False
    if remove_edge:  # DO this only when creating training data with positive examples
        graph.add_edge(node1.case_id, node2.case_id)
    if worth:
        return {
            "days_diff": days_difference,
            "adamic_adar": adamic_adar,
            "pref_attach": pref_attach,
            "common_neigh": common_neigh,
            "common_in_neigh": common_in,
            "common_out_neigh": common_out,
            "cos_sim": cos_sim,
        }


def compute_features_wrapper(
    node_pair: tuple[Node, Node],
    graph: nx.Graph,
    digraph: nx.DiGraph,
    vectorizer,
    label: int,
) -> dict | None:
    """Wrapper function for parallel processing"""
    node1, node2 = node_pair
    features = compute_features(graph, digraph, vectorizer, node1, node2)
    if features is not None:
        features["label"] = label
        return features
    return None


def main():

    workers = 4

    # Load judgments from JSON file
    print("Loading judgments from JSON...")
    with open("data/judgments.json", "r", encoding="utf-8") as f:
        judgments = json.load(f)

    cases_text = defaultdict(str)
    corpus = list()

    for judgment in judgments:
        celex_id = judgment["celex_id"]
        paragraphs = judgment["paragraphs"]

        # Combine all paragraphs into a single text
        case_text = list()
        for number in sorted(paragraphs.keys(), key=int):
            text = paragraphs[number]
            if isinstance(text, str):
                case_text.append(text)

        case_text = " ".join(case_text)
        cases_text[celex_id] = case_text
        corpus.append(case_text)

    print(f"Loaded {len(corpus)} cases from judgments.json")

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    with open("artifacts/tfidf.pkl", "wb") as fhandle:
        pickle.dump(vectorizer, fhandle)
    vectorizer = pickle.load(open("artifacts/tfidf.pkl", "rb"))

    print("Loading data...")
    paragraphs_df = pd.read_excel("data/par-to-par-2.xlsx")
    print("Rows", len(paragraphs_df))
    paragraphs_df = paragraphs_df.dropna()
    print(paragraphs_df.columns)
    train_idx = paragraphs_df["DATE_FROM"].map(lambda x: int(x.split("-")[0]) < 2018)
    train_df = paragraphs_df[train_idx]

    # Build edges from par-to-par dataset
    grpd_case_df = train_df.groupby("CELEX_FROM")
    edges = list()
    for name, group in grpd_case_df:
        for cited_case in set(group["CELEX_TO"].tolist()):
            edges.append((name, cited_case))

    print(f"Generated {len(edges)} case-to-case edges from par-to-par dataset")

    # Build node objects
    nodes_obj = dict()
    for i, row in train_df.iterrows():
        celex_from = row["CELEX_FROM"]
        if celex_from not in nodes_obj:
            text = cases_text[celex_from]
            node1 = Node(row["CELEX_FROM"], dateparser.parse(row["DATE_FROM"]), text)
            nodes_obj[celex_from] = node1
        celex_to = row["CELEX_TO"]
        if celex_to not in nodes_obj:
            text = cases_text[celex_to]
            node2 = Node(row["CELEX_TO"], dateparser.parse(row["DATE_TO"]), text)
            nodes_obj[celex_to] = node2

    # Generate graphs directly from edges
    print("Building undirected graph...")
    graph = nx.Graph()
    graph.add_edges_from(edges)

    print("Building directed graph...")
    digraph = nx.DiGraph()
    digraph.add_edges_from(edges)

    payloads = list()
    for node in tqdm(list(graph.nodes)):
        edges = graph.edges(node)
        if (
            len(edges) > 1
        ):  # we consider only cases for which there's more than one citation
            for e in edges:
                node1 = nodes_obj[e[0]]
                node2 = nodes_obj[e[1]]
                if node1.date > node2.date:
                    payloads.append((node1, node2))

    print(f"Processing {len(payloads)} positive pairs...")

    # Note: For positive examples we don't use multiprocessing because remove_edge=True
    # requires modifying the shared graph, which is not thread-safe
    positive = list()
    positive_pairs = set()
    for node1, node2 in tqdm(payloads):
        features = compute_features(
            graph, digraph, vectorizer, node1, node2, remove_edge=True
        )
        if features:
            features["label"] = 1
            positive.append(features)
            positive_pairs.add(f"{node1.case_id}-{node2.case_id}")

    print(f"Generated {len(positive)} positive examples")

    negative_payloads = list()
    for n1 in tqdm(list(graph.nodes)):
        for n2 in list(graph.nodes):
            if n1 != n2:
                node1 = nodes_obj[n1]
                node2 = nodes_obj[n2]
                if (
                    node2.date < node1.date
                    and f"{node1.case_id}-{node2.case_id}" not in positive_pairs
                ):
                    negative_payloads.append((node1, node2))

    print(f"Total negative pairs: {len(negative_payloads)}")

    # Subsample to balance dataset (10x negatives to positives is common)
    max_negatives = min(len(negative_payloads), len(positive) * 100)
    subsample = (
        sample(negative_payloads, max_negatives)
        if len(negative_payloads) > max_negatives
        else negative_payloads
    )
    print(f"Processing {len(subsample)} negative pairs with {workers} cores...")

    # Parallel processing for negative examples
    compute_partial = partial(
        compute_features_wrapper,
        graph=graph,
        digraph=digraph,
        vectorizer=vectorizer,
        label=0,
    )

    with Pool(workers) as pool:
        negative = list(
            filter(
                None,
                tqdm(
                    pool.imap_unordered(compute_partial, subsample, chunksize=100),
                    total=len(subsample),
                ),
            )
        )

    print(f"Generated {len(negative)} negative examples")

    examples = positive + negative
    df = pd.DataFrame.from_dict(examples)
    df.to_csv("data/case_to_case_training2.csv")


if __name__ == "__main__":
    main()
