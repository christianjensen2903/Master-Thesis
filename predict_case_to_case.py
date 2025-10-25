from __future__ import annotations

import json
import os
from collections import defaultdict

import joblib  # type: ignore
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
import pickle
from datetime import datetime
from tqdm import tqdm  # type: ignore
import dateparser  # type: ignore


class Node:
    def __init__(self, case_id: str, date: datetime, text: str) -> None:
        self.case_id = case_id
        self.date = date
        self.text = text


def days_diff(node1: Node, node2: Node) -> int:
    delta = node1.date - node2.date
    return abs(delta.days)


def common_out_neighbors(g: nx.DiGraph, i: str, j: str) -> set:
    if not g.has_node(i) or not g.has_node(j):
        return set()
    return set(g.successors(i)).intersection(g.successors(j))


def common_in_neighbors(g: nx.DiGraph, i: str, j: str) -> set:
    if not g.has_node(i) or not g.has_node(j):
        return set()
    return set(g.predecessors(i)).intersection(g.predecessors(j))


def compute_features(
    graph: nx.Graph,
    digraph: nx.DiGraph,
    vectorizer,
    node1: Node,
    node2: Node,
    remove_edge: bool = False,
) -> dict | None:
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

    days_difference = days_diff(node1, node2)
    assert days_difference > 0

    # Check if both nodes exist in the graph
    if not graph.has_node(node1.case_id) or not graph.has_node(node2.case_id):
        return None

    if graph.has_edge(node1.case_id, node2.case_id) and remove_edge:
        graph.remove_edge(node1.case_id, node2.case_id)

    try:
        adamic_adar = list(
            nx.adamic_adar_index(graph, [(node1.case_id, node2.case_id)])
        )[0][-1]
        pref_attach = list(
            nx.preferential_attachment(graph, [(node1.case_id, node2.case_id)])
        )[0][-1]
        common_neigh = len(
            list(nx.common_neighbors(graph, node1.case_id, node2.case_id))
        )
        common_in = len(
            list(common_in_neighbors(digraph, node1.case_id, node2.case_id))
        )
        common_out = len(
            list(common_out_neighbors(digraph, node1.case_id, node2.case_id))
        )
    except (nx.NodeNotFound, KeyError) as e:
        print(f"Warning: Node not found in graph: {e}")
        return None

    if adamic_adar or pref_attach or common_neigh or common_in or common_out:
        cos_sim = cosine_similarity(vectorizer.transform([node1.text, node2.text]))[0][
            1
        ]
        worth = True
    else:
        worth = False

    if remove_edge:
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
    return None


def main() -> None:
    artifacts_dir = "artifacts"
    output_dir = "artifacts/predictions_case_to_case"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model and artifacts
    print("Loading trained model and artifacts...")
    model = joblib.load(
        os.path.join(artifacts_dir, "random_forest_case_to_case.joblib")
    )
    vectorizer = pickle.load(open(os.path.join(artifacts_dir, "tfidf.pkl"), "rb"))
    with open(os.path.join(artifacts_dir, "random_forest_features.json"), "r") as f:
        feature_names = json.load(f)

    print(f"Loaded model with features: {feature_names}")

    # Load judgments from JSON
    print("Loading judgments from JSON...")
    with open("data/judgments.json", "r", encoding="utf-8") as f:
        judgments = json.load(f)

    cases_text = defaultdict(str)
    for celex_id, judgment in judgments.items():
        paragraphs = judgment["paragraphs"]

        case_text = []
        for number in sorted(paragraphs.keys(), key=int):
            text = paragraphs[number]
            if isinstance(text, str):
                case_text.append(text)

        cases_text[celex_id] = " ".join(case_text)

    print(f"Loaded {len(cases_text)} cases from judgments.json")

    # Load paragraph data
    print("Loading paragraph data...")
    paragraphs_df = pd.read_excel("data/par-to-par-2.xlsx")
    print(f"Total rows: {len(paragraphs_df)}")
    paragraphs_df = paragraphs_df.dropna()
    print(f"After dropna: {len(paragraphs_df)}")

    # Split by year
    train_idx = paragraphs_df["DATE_FROM"].map(lambda x: int(x.split("-")[0]) < 2018)
    test_idx = paragraphs_df["DATE_FROM"].map(lambda x: int(x.split("-")[0]) >= 2018)
    train_df = paragraphs_df[train_idx]
    test_df = paragraphs_df[test_idx]

    print(f"Train cases: {len(train_df)}, Test cases: {len(test_df)}")

    # Build edges from training data only
    print("Building graphs from training data...")
    grpd_case_df = train_df.groupby("CELEX_FROM")
    edges = []
    for name, group in grpd_case_df:
        for cited_case in set(group["CELEX_TO"].tolist()):
            edges.append((name, cited_case))

    print(f"Generated {len(edges)} case-to-case edges")

    # Build graphs
    graph = nx.Graph()
    graph.add_edges_from(edges)

    digraph = nx.DiGraph()
    digraph.add_edges_from(edges)

    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )
    print(
        f"Digraph has {digraph.number_of_nodes()} nodes and {digraph.number_of_edges()} edges"
    )

    # Build node objects from all data
    print("Building node objects...")
    nodes_obj = {}
    for i, row in pd.concat([train_df, test_df]).iterrows():
        celex_from = row["CELEX_FROM"]
        if celex_from not in nodes_obj and celex_from in cases_text:
            text = cases_text[celex_from]
            node1 = Node(celex_from, dateparser.parse(row["DATE_FROM"]), text)
            nodes_obj[celex_from] = node1

        celex_to = row["CELEX_TO"]
        if celex_to not in nodes_obj and celex_to in cases_text:
            text = cases_text[celex_to]
            node2 = Node(celex_to, dateparser.parse(row["DATE_TO"]), text)
            nodes_obj[celex_to] = node2

    print(f"Created {len(nodes_obj)} node objects")

    # Create test cases mapping
    print("Preparing test cases...")
    testing_cases = defaultdict(set)
    for i, row in test_df.iterrows():
        celex_from = row["CELEX_FROM"]
        celex_to = row["CELEX_TO"]
        testing_cases[celex_from].add(celex_to)

    print(f"Unique test cases: {len(testing_cases)}")

    # Predict for each test case
    print("Generating predictions...")
    for celex_from, celex_tos in tqdm(
        testing_cases.items(), desc="Processing test cases"
    ):
        output_file = os.path.join(output_dir, f"{celex_from}.json")

        if os.path.exists(output_file):
            print(f"Skipping {celex_from} (already exists)")
            continue

        if celex_from not in nodes_obj:
            print(f"Skipping {celex_from} (no node object)")
            continue

        all_probas = {}
        all_features = {}
        node_from = nodes_obj[celex_from]

        # Iterate over all nodes in graph as candidates
        for node in graph.nodes:
            if node not in nodes_obj:
                continue

            candidate_node = nodes_obj[node]

            # Only consider cases before the source case
            if candidate_node.date >= node_from.date:
                continue

            # Skip self-reference
            if node == celex_from:
                continue

            # Check if both nodes exist in the graph before computing features
            if not graph.has_node(celex_from) or not graph.has_node(node):
                print(f"Warning: Skipping {celex_from} -> {node} (nodes not in graph)")
                continue

            # Determine if we should remove edge (for true positives)
            remove_edge = node in celex_tos

            # Compute features
            features = compute_features(
                graph,
                digraph,
                vectorizer,
                node_from,
                candidate_node,
                remove_edge=remove_edge,
            )

            if features:
                # Store features in the order expected by the model
                feature_vector = [features[feat] for feat in feature_names]
                all_features[node] = feature_vector

        # Predict probabilities for all candidates
        if all_features:
            feature_matrix = list(all_features.values())
            probas = model.predict_proba(feature_matrix)[:, 1]

            for key, proba in zip(all_features.keys(), probas):
                all_probas[key] = float(proba)

            # Save predictions
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(all_probas, f, indent=2, sort_keys=True)
        else:
            print(f"No valid features for {celex_from}")

    print(f"Predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
