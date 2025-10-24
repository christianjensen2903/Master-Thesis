import json
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_train_test(metadata):
    train = []
    test = []
    for case_id, m in metadata.items():
        m["case_id"] = case_id
        year = int(m["meta"]["date"].split("-")[0])
        (train if year < 2018 else test).append(m)
    return train, test


def mean_average_precision(all_ap):
    if not all_ap:
        return 0.0
    return float(np.mean(all_ap))


# -----------------
# Main pipeline
# -----------------


def main():
    print("Loading data...")
    df = pd.read_excel("data/par-to-par-2.xlsx")
    print("Rows (raw)", len(df))
    df = df.dropna()
    print("Rows (dropna)", len(df))

    metadata = json.load(open("data/par-to-par.json"))
    train_meta, test_meta = generate_train_test(metadata)

    train_celex = {m["case_id"] for m in train_meta}
    test_celex = {m["case_id"] for m in test_meta}

    # First collect all unique paragraph texts
    all_texts = pd.unique(
        pd.concat([df["TEXT_FROM"], df["TEXT_TO"]], ignore_index=True)
    )
    # Filter out non-strings defensively
    all_texts = [t for t in all_texts if isinstance(t, str)]

    # pid <-> text
    text_to_pid = {t: i for i, t in enumerate(all_texts)}
    pid_to_text = np.array(all_texts, dtype=object)

    n_par = len(pid_to_text)
    print("Unique paragraphs:", n_par)

    tmp_info = {
        pid: {
            "date": None,  # datetime
            "celex": None,  # str
            "number": None,  # paragraph number
            "set_type": None,  # "train" / "test" / None
        }
        for pid in range(n_par)
    }

    # pass 1: fill from TEXT_FROM rows (they have citations out)
    for (celex_from, number_from), sub in tqdm(
        df.groupby(["CELEX_FROM", "NUMBER_FROM"]), desc="pass1_from"
    ):
        paragraph_text = sub["TEXT_FROM"].iloc[0]
        date_str = sub["DATE_FROM"].iloc[0]
        if not isinstance(paragraph_text, str):
            continue
        pid = text_to_pid[paragraph_text]
        d = dt.strptime(date_str, "%Y-%m-%d")

        info = tmp_info[pid]
        if info["date"] is None or d < info["date"]:
            info["date"] = d
            info["celex"] = celex_from
            info["number"] = number_from
            info["set_type"] = (
                "train"
                if celex_from in train_celex
                else ("test" if celex_from in test_celex else info["set_type"])
            )

    # pass 2: fill from TEXT_TO rows (catch paragraphs that only appear as targets)
    for (celex_to, number_to), sub in tqdm(
        df.groupby(["CELEX_TO", "NUMBER_TO"]), desc="pass2_to"
    ):
        paragraph_text = sub["TEXT_TO"].iloc[0]
        date_str = sub["DATE_TO"].iloc[0]
        if not isinstance(paragraph_text, str):
            continue
        pid = text_to_pid[paragraph_text]
        d = dt.strptime(date_str, "%Y-%m-%d")

        info = tmp_info[pid]
        if info["date"] is None or d < info["date"]:
            info["date"] = d
            info["celex"] = celex_to
            info["number"] = number_to
            # only set set_type if not known
            if info["set_type"] is None:
                info["set_type"] = (
                    "train"
                    if celex_to in train_celex
                    else ("test" if celex_to in test_celex else None)
                )

    # Now build arrays
    paragraph_dates = np.array(
        [tmp_info[pid]["date"] for pid in range(n_par)],
        dtype="datetime64[ns]",
    )
    paragraph_celex = np.array(
        [tmp_info[pid]["celex"] for pid in range(n_par)],
        dtype=object,
    )
    paragraph_set = np.array(
        [tmp_info[pid]["set_type"] for pid in range(n_par)],
        dtype=object,
    )

    # -----------------
    # Build citation graph (source pid -> list of cited pid)
    # -----------------
    # We'll aggregate once using pid.
    cited_by_pid = defaultdict(set)

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="citations"):
        src_txt = row["TEXT_FROM"]
        tgt_txt = row["TEXT_TO"]
        if not isinstance(src_txt, str) or not isinstance(tgt_txt, str):
            continue
        src_pid = text_to_pid[src_txt]
        tgt_pid = text_to_pid[tgt_txt]
        cited_by_pid[src_pid].add(tgt_pid)

    # Turn sets into sorted lists for determinism
    for k in list(cited_by_pid.keys()):
        cited_by_pid[k] = sorted(cited_by_pid[k])

    # -----------------
    # Build train corpus for TF-IDF (only paragraphs from train celex)
    # -----------------
    train_mask = paragraph_set == "train"
    train_texts = pid_to_text[train_mask]

    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        strip_accents="ascii",
        norm="l2",  # default, makes cosine = dot
    )
    vectorizer.fit(train_texts)

    print("Vectorizing all paragraphs...")
    tfidf_matrix = vectorizer.transform(pid_to_text)  # csr_matrix [n_par, vocab]

    # -----------------
    # Pre-sort paragraph IDs by date ASC once.
    # We'll use this to quickly get "all candidates published before this paragraph".
    # -----------------
    sort_idx = np.argsort(paragraph_dates)  # oldest -> newest
    sorted_dates = paragraph_dates[sort_idx]

    # -----------------
    # Evaluation: MAP on *test* paragraphs that have citations
    # We'll iterate only pids with set_type=="test" and at least 1 citation.
    # -----------------
    print("Computing MAP...")
    test_source_pids = [
        pid
        for pid in range(n_par)
        if paragraph_set[pid] == "test" and len(cited_by_pid.get(pid, [])) > 0
    ]

    avg_precs = []

    for src_pid in tqdm(test_source_pids, desc="eval"):
        src_date = paragraph_dates[src_pid]

        # Candidates = all paragraphs strictly older than src_date
        # Binary search boundary using np.searchsorted
        # searchsorted gives first index where src_date could be inserted
        # Since sorted_dates is ascending, everything < src_date is before that index
        cutoff = np.searchsorted(sorted_dates, src_date, side="left")
        cand_pids = sort_idx[:cutoff]

        if len(cand_pids) == 0:
            continue  # nothing to retrieve against, skip

        # Ground truth relevant docs for this source paragraph
        relevant = set(cited_by_pid[src_pid])
        # filter out any target that is not in cand_pids (can't cite the future)
        relevant = relevant.intersection(set(cand_pids))
        num_rel = len(relevant)
        if num_rel == 0:
            continue

        # Build similarity scores:
        # sim = C * v, where C is (num_cand x dim), v is (dim x 1)
        src_vec = tfidf_matrix[src_pid]  # (1, dim)
        cand_mat = tfidf_matrix[cand_pids]  # (num_cand, dim)
        sims = cand_mat.dot(src_vec.T).toarray().ravel()  # dense 1D

        # Rank candidates by sim desc
        rank_order = np.argsort(-sims)  # descending

        # Compute Average Precision for this src_pid
        good = 0
        precisions = []
        for rank_pos, local_idx in enumerate(rank_order, start=1):
            pid_candidate = cand_pids[local_idx]
            if pid_candidate in relevant:
                good += 1
                precisions.append(good / rank_pos)

                # optional early exit: if we already found them all
                if good == num_rel:
                    break

        ap = np.mean(precisions) if precisions else 0.0
        avg_precs.append(ap)

    map_score = mean_average_precision(avg_precs)
    print("MAP:", map_score)


if __name__ == "__main__":
    main()
