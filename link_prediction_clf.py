from joblib import load
import json
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, vstack


def generate_train_test(metadata):
    # Training set will be paragraph from < 2018, test set >= 2018
    train = list()
    test = list()
    for case_id, m in metadata.items():
        m["case_id"] = case_id
        date = m["meta"]["date"]
        year = date.split("-")[0]
        if int(year) < 2018:
            train.append(m)
        else:
            test.append(m)
    return train, test


class Paragraph:
    def __init__(self, case_id, number, date, text, citations):
        self.case_id = case_id
        self.number = number
        self.date = dt.strptime(date, "%Y-%m-%d")
        self.text = text
        self.citations = citations

    def get_id(self):
        return f"{self.case_id}-{self.number}"


def retrieve_candidate_paragraphs(paragraphs, date):
    # Given a paragraph's date of publication,
    # all the previous paragraphs can be considered as citation candidates
    candidates = set()
    for p in paragraphs:
        if p.date < date:
            candidates.add(p)
    return candidates


def retrieve_candidates_ids(metadatas, date):
    # Given a paragraph's date of publication,
    # all the previous paragraphs can be considered as citation candidates
    candidates = set()
    for case_id, metadata in metadatas.items():
        p_date = dt.strptime(metadata["meta"]["date"], "%Y-%m-%d")
        if p_date < date:
            candidates.add(case_id)
    return candidates


def tf_idf(tf_idf, text1, text2):
    vector1 = tf_idf.transform([text1])
    vector2 = tf_idf.transform([text2])
    similarity = cosine_similarity(vector1, vector2)
    return similarity[0]


def concat_sparse_matrix(vectors_by_par, paragraphs):
    vectors = [vectors_by_par[p].reshape(1, -1) for p in paragraphs]
    matrix = vstack(vectors)
    return matrix


def concat_par_vectors(vectors_by_par, paragraphs):
    vectors = [vectors_by_par[p].reshape(1, -1) for p in paragraphs]
    matrix = np.vstack(vectors)
    return matrix


def compute_precision(
    concat_func, all_paragraphs_obj, vectors_by_par, paragraph, k=10, verbose=False
):
    candidates = retrieve_candidate_paragraphs(all_paragraphs_obj, paragraph.date)
    candidates_texts = list({p.text for p in candidates if p.text in vectors_by_par})
    citations_to_find = {t for c, t in paragraph.citations if t in vectors_by_par}
    num_citations = len(citations_to_find)
    if num_citations:
        candidates_vectors = concat_func(vectors_by_par, candidates_texts)
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
                # results[paragraph.text][candidate_text]['rank'] = i
                # results[paragraph.text][candidate_text]['precision'] = precision
                # results[paragraph.text][candidate_text]['similarity'] = float(candidate_sim)
        results["average_precision"] = sum(precisions) / num_citations

        return dict(results)
    else:
        return None


def find_paragraph_by_text(paragraphs, text):
    for p in paragraphs:
        if p.text == text:
            return p


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
    train_paragraphs = [
        p for p in train_paragraphs_from + train_paragraphs_to if type(p) is str
    ]
    test_paragraphs_from = list(
        set(
            paragraphs_df[paragraphs_df["CELEX_FROM"].isin(test_celex)][
                "TEXT_FROM"
            ].tolist()
        )
    )
    test_paragraphs_to = list(
        set(
            paragraphs_df[paragraphs_df["CELEX_TO"].isin(test_celex)][
                "TEXT_TO"
            ].tolist()
        )
    )
    test_paragraphs = test_paragraphs_from + test_paragraphs_to

    print("Building objects...")
    train_paragraphs_obj = list()
    test_paragraphs_obj = list()
    texts = set()
    grp_by_celex_df = paragraphs_df.groupby(["CELEX_FROM", "NUMBER_FROM"])
    for (celex_from, number_from), subset_df in tqdm(grp_by_celex_df):
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

    paragraphs_to_obj = list()
    for _, row in tqdm(paragraphs_df.iterrows()):
        paragraph = row["TEXT_TO"]
        if paragraph not in texts:
            celex_to = row["CELEX_TO"]
            number_to = row["NUMBER_TO"]
            date = row["DATE_TO"]
            citations = None
            obj = Paragraph(celex_to, number_to, date, paragraph, citations)
            paragraphs_to_obj.append(obj)
            texts.add(paragraph)

    print(len(train_paragraphs_obj), len(test_paragraphs_obj), len(paragraphs_to_obj))

    p_from = set(paragraphs_df["TEXT_FROM"].tolist())
    p_to = set(paragraphs_df["TEXT_TO"].tolist())
    all_paragraphs = list(p_from.union(p_to))
    all_paragraphs = [p for p in all_paragraphs if type(p) is str]
    print(len(p_from), len(p_to), len(all_paragraphs))

    test_pars_with_citations = [p for p in test_paragraphs_obj if len(p.citations)]
    train_pars_with_citations = [p for p in train_paragraphs_obj if len(p.citations)]
    all_paragraphs_obj = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj

    print("Fitting tf-idf...")
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="ascii")
    X = vectorizer.fit_transform(train_paragraphs)
    vectors = vectorizer.transform(all_paragraphs)
    vectors_by_par = dict()
    for p, v in zip(all_paragraphs, vectors):
        vectors_by_par[p] = v
    concat_func = concat_sparse_matrix

    test_pars_with_citations = [p for p in test_paragraphs_obj if len(p.citations)]
    train_pars_with_citations = [p for p in train_paragraphs_obj if len(p.citations)]
    # all_paragraphs_obj = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj
    all_paragraphs_obj = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj

    print("Computing precisions single thread...")
    results = list()
    logs = list()
    pbar = tqdm(test_pars_with_citations)
    for i, p in enumerate(pbar):
        # ordered_rankings = {k: v for k, v in sorted(rankings.items(), key=lambda item: item[1], reverse=True)}
        # ordered_keys = [k for k in ordered_rankings.keys()][:1000]
        # candidates = [p for p in all_paragraphs_obj if p.case_id in ordered_keys]
        r = compute_precision(
            concat_func, all_paragraphs_obj, vectors_by_par, p, verbose=True
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
