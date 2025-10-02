from datetime import datetime as dt
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # efficient on normalized TF-IDF

# from scipy.sparse import csr_matrix  # (not strictly needed, kept for reference)


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
    Build Paragraph objects for FROM paragraphs (sources with citations),
    and synthetic Paragraph objects for TO paragraphs we haven't seen as FROM.
    Also returns a dict mapping paragraph text -> earliest publication date across all occurrences.
    """
    train_paragraphs_obj, test_paragraphs_obj = [], []
    seen_texts = set()

    # Group all sources (FROM) by (celex, number)
    grp = paragraphs_df.groupby(["CELEX_FROM", "NUMBER_FROM"], sort=False)
    for (celex_from, number_from), subset_df in tqdm(grp, desc="Build FROM objects"):
        paragraph_text = subset_df["TEXT_FROM"].iloc[0]
        date_from = subset_df["DATE_FROM"].iloc[0]

        # Targets/citations for this source
        citations_text = subset_df["TEXT_TO"].tolist()
        citations_celex = subset_df["CELEX_TO"].tolist()
        citations = list(zip(citations_celex, citations_text))

        obj = Paragraph(celex_from, number_from, date_from, paragraph_text, citations)

        if celex_from in train_celex:
            train_paragraphs_obj.append(obj)
            seen_texts.add(paragraph_text)
        elif celex_from in test_celex:
            test_paragraphs_obj.append(obj)
            seen_texts.add(paragraph_text)

    # Add TO paragraphs (targets that never appeared as a FROM paragraph text)
    paragraphs_to_obj = []
    for row in tqdm(
        paragraphs_df.itertuples(index=False),
        total=len(paragraphs_df),
        desc="Add TO-only objects",
    ):
        paragraph_text = row.TEXT_TO
        if isinstance(paragraph_text, str) and paragraph_text not in seen_texts:
            celex_to = row.CELEX_TO
            number_to = row.NUMBER_TO
            date_to = row.DATE_TO
            obj = Paragraph(
                celex_to, number_to, date_to, paragraph_text, citations=None
            )
            paragraphs_to_obj.append(obj)
            seen_texts.add(paragraph_text)

    # Build earliest date per text across all objects
    all_objs = train_paragraphs_obj + test_paragraphs_obj + paragraphs_to_obj
    text_earliest_date = {}
    for o in all_objs:
        if not isinstance(o.text, str):
            continue
        d = o.date
        if o.text not in text_earliest_date or d < text_earliest_date[o.text]:
            text_earliest_date[o.text] = d

    return (
        train_paragraphs_obj,
        test_paragraphs_obj,
        paragraphs_to_obj,
        text_earliest_date,
    )


def compute_average_precision(ranked_global_idxs, relevant_global_idxs):
    """
    ranked_global_idxs: np.array of global indices sorted by descending similarity
    relevant_global_idxs: set of indices that are relevant (cited targets)
    Returns AP or None if no relevant items exist.
    """
    if not relevant_global_idxs:
        return None
    num_good = 0
    precisions = []
    for i, gidx in enumerate(ranked_global_idxs, 1):
        if gidx in relevant_global_idxs:
            num_good += 1
            precisions.append(num_good / i)
            if num_good == len(relevant_global_idxs):
                # Early stop once we've seen all relevant items
                break
    return float(np.mean(precisions)) if precisions else None


def main():
    print("Loading data...")
    paragraphs_df = pd.read_csv("data/clean_data.csv")
    print("Rows before dropna:", len(paragraphs_df))
    paragraphs_df = paragraphs_df.dropna(
        subset=["TEXT_FROM", "TEXT_TO", "DATE_FROM", "DATE_TO"]
    )
    print("Rows after dropna:", len(paragraphs_df))

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

    # Build objects (sources with citations + to-only objects) and earliest date per text
    (
        train_paragraphs_obj,
        test_paragraphs_obj,
        paragraphs_to_obj,
        text_earliest_date,
    ) = build_paragraph_objects(paragraphs_df, train_celex, test_celex)

    print(
        "Object counts:",
        len(train_paragraphs_obj),
        len(test_paragraphs_obj),
        len(paragraphs_to_obj),
    )

    # Unique set of ALL paragraph texts
    p_from = set(paragraphs_df["TEXT_FROM"].tolist())
    p_to = set(paragraphs_df["TEXT_TO"].tolist())
    all_paragraphs_texts = [p for p in list(p_from.union(p_to)) if isinstance(p, str)]
    print(
        "Counts | FROM:",
        len(p_from),
        "TO:",
        len(p_to),
        "UNIQUE:",
        len(all_paragraphs_texts),
    )

    # Keep only those with a known earliest date (safety)
    all_paragraphs_texts = [t for t in all_paragraphs_texts if t in text_earliest_date]

    # Build a stable ordering for indexing
    text2row = {t: i for i, t in enumerate(all_paragraphs_texts)}

    # Dates array aligned to all_paragraphs_texts (use earliest date per text)
    dates_np = np.array(
        [np.datetime64(text_earliest_date[t].date()) for t in all_paragraphs_texts],
        dtype="datetime64[D]",
    )

    # Fit TF-IDF on training texts only; transform all texts once
    print("Fitting TF-IDF and transforming...")
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="ascii", norm="l2")
    _ = vectorizer.fit_transform(train_paragraphs)  # we discard this matrix; fit only

    X_all = vectorizer.transform(all_paragraphs_texts)  # CSR [n_texts, n_features]

    # Candidates: any paragraph published strictly before the source's date
    # We'll evaluate only test sources that have citations
    test_sources = [p for p in test_paragraphs_obj if p.citations]

    print("Computing MAP (fast path)...")
    ap_values = []
    pbar = tqdm(test_sources, desc="AP")

    for i, src in enumerate(pbar):
        src_idx = text2row.get(src.text, None)
        if src_idx is None:
            continue

        # candidate mask by date
        src_date64 = np.datetime64(src.date.date())
        cand_mask = dates_np < src_date64
        cand_idxs = np.flatnonzero(cand_mask)
        if cand_idxs.size == 0:
            continue

        # Similarities: dot product == cosine for L2-normalized TF-IDF
        # (Use linear_kernel to avoid densification & keep it efficient)
        sims = linear_kernel(X_all[cand_idxs], X_all[src_idx]).ravel()

        # Rank all candidates (you can argpartition+refine if extremely large)
        order = np.argsort(-sims)
        ranked_global = cand_idxs[order]

        # Relevant set: cited target texts that we have vectors for
        cited_targets = {
            text2row[t]
            for (_, t) in src.citations
            if isinstance(t, str) and (t in text2row)
        }
        if not cited_targets:
            continue

        ap = compute_average_precision(ranked_global, cited_targets)
        if ap is not None:
            ap_values.append(ap)

        if i > 0 and i % 10 == 0 and ap_values:
            pbar.set_description(f"MAP: {np.mean(ap_values):.6f}")

    mean_average_precision = float(np.mean(ap_values)) if ap_values else float("nan")
    print("Mean Average Precision:", mean_average_precision)


if __name__ == "__main__":
    main()
