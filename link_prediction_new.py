from datetime import datetime as dt
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import vstack


# ----------------------------
# Data structures
# ----------------------------
class Paragraph:
    def __init__(self, case_id, number, date, text, citations):
        self.case_id = case_id
        self.number = number
        self.date = dt.strptime(date, "%Y-%m-%d") if isinstance(date, str) else date
        self.text = text
        self.citations = citations or []

    def get_id(self):
        return f"{self.case_id}-{self.number}"


# ----------------------------
# Helpers
# ----------------------------
def generate_train_test(metadata):
    # Training set will be paragraphs from < 2018, test set >= 2018
    train, test = [], []
    for case_id, m in metadata.items():
        m["case_id"] = case_id
        date = m["meta"]["date"]
        year = int(date.split("-")[0])
        (train if year < 2018 else test).append(m)
    return train, test


def build_paragraph_objects(paragraphs_df, train_celex, test_celex):
    """
    Build Paragraph objects matching the original link_prediction_clf.py logic.
    """
    train_paragraphs_obj, test_paragraphs_obj = [], []
    texts = set()

    # Group all sources (FROM) by (celex, number) - matching original logic
    grp_by_celex_df = paragraphs_df.groupby(["CELEX_FROM", "NUMBER_FROM"])
    for (celex_from, number_from), subset_df in tqdm(
        grp_by_celex_df, desc="Build objects"
    ):
        paragraph = subset_df["TEXT_FROM"].tolist()[0]
        date = subset_df["DATE_FROM"].tolist()[0]
        citations_text = subset_df["TEXT_TO"].tolist()
        citations_celex = subset_df["CELEX_TO"].tolist()
        citations = list(zip(citations_celex, citations_text))
        obj = Paragraph(celex_from, number_from, date, paragraph, citations)

        if celex_from in train_celex:
            train_paragraphs_obj.append(obj)
            texts.add(paragraph)
        elif celex_from in test_celex:
            test_paragraphs_obj.append(obj)
            texts.add(paragraph)
        else:
            print("oups")

    # Add TO paragraphs (targets that never appeared as a FROM paragraph text)
    paragraphs_to_obj = []
    for _, row in tqdm(paragraphs_df.iterrows(), desc="Add TO-only objects"):
        paragraph = row["TEXT_TO"]
        if paragraph not in texts:
            celex_to = row["CELEX_TO"]
            number_to = row["NUMBER_TO"]
            date = row["DATE_TO"]
            citations = None
            obj = Paragraph(celex_to, number_to, date, paragraph, citations)
            paragraphs_to_obj.append(obj)
            texts.add(paragraph)

    return train_paragraphs_obj, test_paragraphs_obj, paragraphs_to_obj


def retrieve_candidate_paragraphs(paragraphs, date):
    # Given a paragraph's date of publication,
    # all the previous paragraphs can be considered as citation candidates
    candidates = set()
    for p in paragraphs:
        if p.date < date:
            candidates.add(p)
    return candidates


def concat_sparse_matrix(vectors_by_par, paragraphs):
    vectors = [vectors_by_par[p].reshape(1, -1) for p in paragraphs]
    matrix = vstack(vectors)
    return matrix


def compute_precision_original_style(
    all_paragraphs_obj, vectors_by_par, paragraph, k=10, verbose=False
):
    candidates = retrieve_candidate_paragraphs(all_paragraphs_obj, paragraph.date)
    candidates_texts = list({p.text for p in candidates if p.text in vectors_by_par})
    citations_to_find = {t for c, t in paragraph.citations if t in vectors_by_par}
    num_citations = len(citations_to_find)
    if num_citations:
        candidates_vectors = concat_sparse_matrix(vectors_by_par, candidates_texts)
        source_vector = vectors_by_par[paragraph.text]
        sims = cosine_similarity(
            candidates_vectors, source_vector.reshape(1, -1)
        ).reshape(-1)
        indices = np.argsort(sims)[::-1]

        results = defaultdict(lambda: defaultdict(dict))
        num_good = 0
        precisions = list()
        ranks = list()
        for i, candidate_index in enumerate(indices):
            candidate_sim = sims[candidate_index]
            candidate_text = candidates_texts[candidate_index]
            if verbose and i < 11:
                ranks.append((candidate_text, float(candidate_sim)))
            if candidate_text in citations_to_find:
                num_good += 1
                precision = num_good / (i + 1)
                precisions.append(precision)
                citations_to_find.remove(candidate_text)
        results["average_precision"] = sum(precisions) / num_citations

        return dict(results)
    else:
        return None


def main():
    print("Loading data...")
    paragraphs_df = pd.read_excel("data/par-to-par-2.xlsx")
    print("Rows", len(paragraphs_df))
    paragraphs_df = paragraphs_df.dropna()
    print("Rows", len(paragraphs_df))

    metadata = json.load(open("data/par-to-par.json"))
    train, test = generate_train_test(metadata)
    train_celex = {m["case_id"] for m in train}
    test_celex = {m["case_id"] for m in test}

    # Texts for fitting the vectorizer (like original)
    train_paragraphs_from = list(
        set(
            paragraphs_df[paragraphs_df["CELEX_FROM"].isin(train_celex)][
                "TEXT_FROM"
            ].tolist()
        )
    )
    train_paragraphs_to = list(
        set(
            paragraphs_df[paragraphs_df["CELEX_TO"].isin(train_celex)][
                "TEXT_TO"
            ].tolist()
        )
    )
    # filter to valid strings
    train_paragraphs = [
        p for p in train_paragraphs_from + train_paragraphs_to if isinstance(p, str)
    ]

    # Build objects (sources with citations + to-only objects)
    (
        train_paragraphs_obj,
        test_paragraphs_obj,
        paragraphs_to_obj,
    ) = build_paragraph_objects(paragraphs_df, train_celex, test_celex)

    print(len(train_paragraphs_obj), len(test_paragraphs_obj), len(paragraphs_to_obj))

    # Unique set of ALL paragraph texts - matching original logic
    p_from = set(paragraphs_df["TEXT_FROM"].tolist())
    p_to = set(paragraphs_df["TEXT_TO"].tolist())
    all_paragraphs = list(p_from.union(p_to))
    all_paragraphs = [p for p in all_paragraphs if type(p) is str]
    print(len(p_from), len(p_to), len(all_paragraphs))

    # Build all paragraph objects
    all_paragraphs_obj = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj

    # Fit TF-IDF - matching original logic
    print("Fitting tf-idf...")
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="ascii")
    X = vectorizer.fit_transform(train_paragraphs)
    vectors = vectorizer.transform(all_paragraphs)
    vectors_by_par = dict()
    for p, v in zip(all_paragraphs, vectors):
        vectors_by_par[p] = v

    # Test paragraphs with citations
    test_pars_with_citations = [p for p in test_paragraphs_obj if len(p.citations)]
    train_pars_with_citations = [p for p in train_paragraphs_obj if len(p.citations)]

    print("Computing precisions single thread...")
    results = list()
    logs = list()
    pbar = tqdm(test_pars_with_citations)
    for i, p in enumerate(pbar):
        r = compute_precision_original_style(
            all_paragraphs_obj, vectors_by_par, p, verbose=True
        )
        if r is not None:
            results.append(r["average_precision"])
            logs.append(r)
        if i > 0 and i % 10 == 0:
            pbar.set_description(f"MAP: {np.mean(results)}")
    mean_average_precision = np.mean(results)
    print(mean_average_precision)


if __name__ == "__main__":
    main()
