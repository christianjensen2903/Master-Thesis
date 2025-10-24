import json
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer


class Evaluator:
    def __init__(
        self,
        excel_path="data/par-to-par-2.xlsx",
        metadata_path="data/par-to-par.json",
        train_cutoff_year=2018,
        tfidf_kwargs=None,
    ):
        """
        excel_path: path to the paragraph-to-paragraph citation xlsx
        metadata_path: path to the json metadata with CELEX + date
        train_cutoff_year: years < this go to train, else test
        tfidf_kwargs: optional dict passed to TfidfVectorizer
        """

        self.excel_path = excel_path
        self.metadata_path = metadata_path
        self.train_cutoff_year = train_cutoff_year

        # populated later
        self.df = None
        self.metadata = None
        self.train_meta = None
        self.test_meta = None

        self.pid_to_text = None  # np.array[str]
        self.text_to_pid = None  # dict[str -> int]
        self.paragraph_dates = None  # np.array[datetime64[ns]]
        self.paragraph_celex = None  # np.array[object]
        self.paragraph_set = None  # np.array[object] ("train"/"test"/None)

        self.cited_by_pid = None  # dict[int -> list[int]]
        self.vectorizer = None
        self.tfidf_matrix = None  # csr_matrix [n_par, vocab]

        self.sort_idx = None  # np.argsort(paragraph_dates)
        self.sorted_dates = None  # paragraph_dates[sort_idx]

        self.map_score = None  # final MAP float

        if tfidf_kwargs is None:
            tfidf_kwargs = {
                "stop_words": "english",
                "strip_accents": "ascii",
                "norm": "l2",  # cosine ~ dot
            }
        self.tfidf_kwargs = tfidf_kwargs

    # -----------------
    # Static helpers
    # -----------------

    @staticmethod
    def _split_train_test(metadata, cutoff_year):
        """Return (train_meta_list, test_meta_list) based on date year < cutoff_year."""
        train = []
        test = []
        for case_id, m in metadata.items():
            m = dict(m)  # shallow copy so we don't mutate caller
            m["case_id"] = case_id
            year = int(m["meta"]["date"].split("-")[0])
            (train if year < cutoff_year else test).append(m)
        return train, test

    @staticmethod
    def _mean_average_precision(all_ap):
        if not all_ap:
            return 0.0
        return float(np.mean(all_ap))

    # -----------------
    # Pipeline steps
    # -----------------

    def load_data(self):
        """Load the Excel pairs and metadata JSON, drop NA rows, split train/test."""
        df = pd.read_excel(self.excel_path)
        df = df.dropna()

        metadata = json.load(open(self.metadata_path))
        train_meta, test_meta = self._split_train_test(metadata, self.train_cutoff_year)

        # persist
        self.df = df
        self.metadata = metadata
        self.train_meta = train_meta
        self.test_meta = test_meta

    def build_paragraph_index(self):
        """
        Build:
        - pid_to_text / text_to_pid (unique paragraph text -> integer ID)
        - paragraph_dates / paragraph_celex / paragraph_set arrays
        """
        df = self.df
        train_celex = {m["case_id"] for m in self.train_meta}
        test_celex = {m["case_id"] for m in self.test_meta}

        # collect all unique paragraph texts
        all_texts = pd.unique(
            pd.concat([df["TEXT_FROM"], df["TEXT_TO"]], ignore_index=True)
        )
        all_texts = [t for t in all_texts if isinstance(t, str)]

        text_to_pid = {t: i for i, t in enumerate(all_texts)}
        pid_to_text = np.array(all_texts, dtype=object)
        n_par = len(pid_to_text)

        # temp storage to resolve earliest date + metadata per paragraph
        tmp_info = {
            pid: {
                "date": None,
                "celex": None,
                "number": None,
                "set_type": None,  # "train"/"test"/None
            }
            for pid in range(n_par)
        }

        # pass 1: fill from TEXT_FROM rows
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

        # pass 2: fill from TEXT_TO rows
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
                if info["set_type"] is None:
                    info["set_type"] = (
                        "train"
                        if celex_to in train_celex
                        else ("test" if celex_to in test_celex else None)
                    )

        # finalize arrays
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

        # persist
        self.text_to_pid = text_to_pid
        self.pid_to_text = pid_to_text
        self.paragraph_dates = paragraph_dates
        self.paragraph_celex = paragraph_celex
        self.paragraph_set = paragraph_set

    def build_citation_graph(self):
        """
        Build mapping: source pid -> sorted list of cited target pids.
        """
        df = self.df
        text_to_pid = self.text_to_pid
        cited_by_pid = defaultdict(set)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="citations"):
            src_txt = row["TEXT_FROM"]
            tgt_txt = row["TEXT_TO"]
            if not isinstance(src_txt, str) or not isinstance(tgt_txt, str):
                continue
            src_pid = text_to_pid[src_txt]
            tgt_pid = text_to_pid[tgt_txt]
            cited_by_pid[src_pid].add(tgt_pid)

        # make deterministic
        cited_by_pid = {k: sorted(v) for k, v in cited_by_pid.items()}

        self.cited_by_pid = cited_by_pid

    def build_tfidf(self):
        """
        Fit TF-IDF on train paragraphs, then transform all paragraphs.
        """
        train_mask = self.paragraph_set == "train"
        train_texts = self.pid_to_text[train_mask]

        vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        vectorizer.fit(train_texts)

        tfidf_matrix = vectorizer.transform(self.pid_to_text)

        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix

    def prepare_temporal_index(self):
        """
        Pre-sort paragraph IDs by date ascending for temporal candidate filtering.
        """
        sort_idx = np.argsort(self.paragraph_dates)  # oldest -> newest
        sorted_dates = self.paragraph_dates[sort_idx]

        self.sort_idx = sort_idx
        self.sorted_dates = sorted_dates

    def evaluate_map(self):
        """
        Compute MAP over test paragraphs that actually cite something.
        Only considers candidate targets strictly older than the source paragraph.
        """
        paragraph_set = self.paragraph_set
        paragraph_dates = self.paragraph_dates
        cited_by_pid = self.cited_by_pid
        sort_idx = self.sort_idx
        sorted_dates = self.sorted_dates
        tfidf_matrix = self.tfidf_matrix

        # source paragraphs: test set + has >=1 citation
        test_source_pids = [
            pid
            for pid in range(len(self.pid_to_text))
            if paragraph_set[pid] == "test" and len(cited_by_pid.get(pid, [])) > 0
        ]

        avg_precs = []

        for src_pid in tqdm(test_source_pids, desc="eval"):
            src_date = paragraph_dates[src_pid]

            # all candidate paragraphs strictly older
            cutoff = np.searchsorted(sorted_dates, src_date, side="left")
            cand_pids = sort_idx[:cutoff]

            if len(cand_pids) == 0:
                continue

            # ground truth relevant = cited paragraphs that are also older
            relevant = set(cited_by_pid[src_pid]).intersection(set(cand_pids))
            num_rel = len(relevant)
            if num_rel == 0:
                continue

            # cosine ~ dot because tf-idf is l2-normalized
            src_vec = tfidf_matrix[src_pid]  # (1, dim)
            cand_mat = tfidf_matrix[cand_pids]  # (C, dim)
            sims = cand_mat.dot(src_vec.T).toarray().ravel()

            rank_order = np.argsort(-sims)  # high -> low

            good = 0
            precisions = []
            for rank_pos, local_idx in enumerate(rank_order, start=1):
                pid_candidate = cand_pids[local_idx]
                if pid_candidate in relevant:
                    good += 1
                    precisions.append(good / rank_pos)
                    if good == num_rel:
                        break

            ap = np.mean(precisions) if precisions else 0.0
            avg_precs.append(ap)

        self.map_score = self._mean_average_precision(avg_precs)
        return self.map_score

    # -----------------
    # Orchestration
    # -----------------

    def run(self, verbose=True):
        """
        Full pipeline:
        1. load data
        2. build paragraph index + metadata
        3. build citation graph
        4. fit + apply TF-IDF
        5. build temporal index
        6. evaluate MAP
        """
        if verbose:
            print("Loading data...")
        self.load_data()
        if verbose:
            print(f"Rows (after dropna): {len(self.df)}")

        if verbose:
            print("Indexing paragraphs...")
        self.build_paragraph_index()
        if verbose:
            print("Unique paragraphs:", len(self.pid_to_text))

        if verbose:
            print("Building citation graph...")
        self.build_citation_graph()

        if verbose:
            print("Fitting and transforming TF-IDF...")
        self.build_tfidf()

        if verbose:
            print("Preparing temporal index...")
        self.prepare_temporal_index()

        if verbose:
            print("Computing MAP...")
        score = self.evaluate_map()
        if verbose:
            print("MAP:", score)

        return score


if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run(verbose=True)
