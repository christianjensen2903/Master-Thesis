import json
import re
from datetime import datetime as dt
from typing import Any
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class LTRFeatureExtractor:
    """Extracts features for Learning-to-Rank from judgment metadata."""

    def __init__(self, judgments_path: str = "data/judgments_cleaned.json"):
        self.judgments_path = judgments_path
        self.judgments: dict[str, dict[str, Any]] = {}
        self.celex_to_metadata: dict[str, dict[str, Any]] = {}
        self.celex_to_par_count: dict[str, int] = {}

    def load(self) -> None:
        """Load judgments and build indices."""
        print(f"Loading judgments from {self.judgments_path}...")
        with open(self.judgments_path) as f:
            self.judgments = json.load(f)

        print("Building metadata indices...")
        for celex, judgment in tqdm(self.judgments.items(), desc="Indexing"):
            meta = judgment.get("meta", {}).get("meta", {})
            self.celex_to_metadata[celex] = meta

            # Count paragraphs
            paragraphs = judgment.get("paragraphs", {})
            self.celex_to_par_count[celex] = len(paragraphs)

        print(f"Loaded {len(self.judgments)} judgments")

    def extract_features(
        self,
        query_celex: str,
        query_par_num: int,
        cand_celex: str,
        cand_par_num: int,
        dense_similarity: float,
        query_date: np.datetime64 | None = None,
        cand_date: np.datetime64 | None = None,
    ) -> dict[str, float]:
        """
        Extract features for a query-candidate pair.

        Returns dict of feature_name -> value
        """
        features: dict[str, float] = {}

        # Dense similarity
        features["dense_similarity"] = dense_similarity

        # Get metadata
        query_meta = self.celex_to_metadata.get(query_celex, {})
        cand_meta = self.celex_to_metadata.get(cand_celex, {})

        # Time difference (in days)
        if query_date is not None and cand_date is not None:
            time_diff = (query_date - cand_date) / np.timedelta64(1, "D")
            features["time_diff_days"] = float(time_diff)
            features["time_diff_years"] = float(time_diff) / 365.25
            features["log_time_diff_days"] = np.log1p(max(0, float(time_diff)))
        else:
            features["time_diff_days"] = 0.0
            features["time_diff_years"] = 0.0
            features["log_time_diff_days"] = 0.0

        # Authentic language features
        query_auth_lang = set(query_meta.get("authentic_language", []))
        cand_auth_lang = set(cand_meta.get("authentic_language", []))

        features["same_auth_lang"] = float(len(query_auth_lang & cand_auth_lang) > 0)
        features["num_shared_auth_langs"] = float(len(query_auth_lang & cand_auth_lang))

        # One-hot encode authentic languages (top languages)
        all_langs = query_auth_lang | cand_auth_lang
        for lang in ["fr", "de", "en", "it", "nl", "es"]:
            features[f"query_auth_lang_{lang}"] = float(
                lang.upper() in query_auth_lang or lang in query_auth_lang
            )
            features[f"cand_auth_lang_{lang}"] = float(
                lang.upper() in cand_auth_lang or lang in cand_auth_lang
            )

        # Advocate general
        query_ag = query_meta.get("advocate_general")
        cand_ag = cand_meta.get("advocate_general")
        features["same_advocate_general"] = float(
            query_ag is not None and cand_ag is not None and query_ag == cand_ag
        )
        features["has_advocate_general"] = float(
            query_ag is not None and cand_ag is not None
        )

        # Rapporteur
        query_rap = query_meta.get("rapporteur")
        cand_rap = cand_meta.get("rapporteur")
        features["same_rapporteur"] = float(
            query_rap is not None and cand_rap is not None and query_rap == cand_rap
        )
        features["has_rapporteur"] = float(
            query_rap is not None and cand_rap is not None
        )

        # Applicant/Defendant
        query_app = self._normalize_party(query_meta.get("applicant"))
        cand_app = self._normalize_party(cand_meta.get("applicant"))
        query_def = self._normalize_party(query_meta.get("defendant"))
        cand_def = self._normalize_party(cand_meta.get("defendant"))

        features["same_applicant"] = (
            1.0 if (query_app and cand_app and query_app == cand_app) else 0.0
        )
        features["same_defendant"] = (
            1.0 if (query_def and cand_def and query_def == cand_def) else 0.0
        )
        features["has_parties"] = (
            1.0
            if (bool(query_app or query_def) and bool(cand_app or cand_def))
            else 0.0
        )

        # Procedure type
        query_proc = query_meta.get("procedure_type")
        cand_proc = cand_meta.get("procedure_type")
        features["same_procedure_type"] = float(
            query_proc is not None and cand_proc is not None and query_proc == cand_proc
        )

        # Common procedure types
        for proc_type in [
            "reference for a preliminary ruling",
            "action for annulment",
            "appeal",
            "infringement procedure",
        ]:
            features[f"query_proc_{self._slugify(proc_type)}"] = (
                1.0
                if (query_proc and proc_type.lower() in str(query_proc).lower())
                else 0.0
            )
            features[f"cand_proc_{self._slugify(proc_type)}"] = (
                1.0
                if (cand_proc and proc_type.lower() in str(cand_proc).lower())
                else 0.0
            )

        # Subject matter overlap
        query_subjects = set(query_meta.get("subject_matter", []))
        cand_subjects = set(cand_meta.get("subject_matter", []))

        if query_subjects and cand_subjects:
            features["num_shared_subjects"] = float(len(query_subjects & cand_subjects))
            features["subject_jaccard"] = float(
                len(query_subjects & cand_subjects)
                / len(query_subjects | cand_subjects)
            )
        else:
            features["num_shared_subjects"] = 0.0
            features["subject_jaccard"] = 0.0

        # Case law about overlap
        query_case_law = self._extract_case_law_celexs(
            query_meta.get("case_law_about", {})
        )
        cand_case_law = self._extract_case_law_celexs(
            cand_meta.get("case_law_about", {})
        )

        if query_case_law and cand_case_law:
            features["num_shared_case_law"] = float(len(query_case_law & cand_case_law))
            features["case_law_jaccard"] = float(
                len(query_case_law & cand_case_law)
                / len(query_case_law | cand_case_law)
            )
        else:
            features["num_shared_case_law"] = 0.0
            features["case_law_jaccard"] = 0.0

        # Self-citation: does candidate cite the query case?
        features["cand_cites_query"] = float(query_celex in cand_case_law)

        # Relative paragraph position
        query_par_count = self.celex_to_par_count.get(query_celex, 0)
        cand_par_count = self.celex_to_par_count.get(cand_celex, 0)

        if query_par_count > 0:
            features["query_rel_position"] = float(query_par_num) / float(
                query_par_count
            )
        else:
            features["query_rel_position"] = 0.0

        if cand_par_count > 0:
            features["cand_rel_position"] = float(cand_par_num) / float(cand_par_count)
        else:
            features["cand_rel_position"] = 0.0

        features["same_rel_position_bin"] = float(
            abs(features["query_rel_position"] - features["cand_rel_position"]) < 0.2
        )

        # Absolute paragraph numbers
        features["query_par_num"] = float(query_par_num)
        features["cand_par_num"] = float(cand_par_num)
        features["par_num_diff"] = abs(float(query_par_num) - float(cand_par_num))

        # Document lengths
        features["query_doc_length"] = float(query_par_count)
        features["cand_doc_length"] = float(cand_par_count)

        return features

    def _normalize_party(self, party: Any) -> str | None:
        """Normalize party name for comparison."""
        if party is None:
            return None
        party_str = str(party).lower().strip()
        # Remove common prefixes/suffixes
        party_str = re.sub(r"\s+(v\.?|vs\.?|versus)\s+", " ", party_str)
        return party_str if party_str else None

    def _slugify(self, text: str) -> str:
        """Convert text to slug for feature names."""
        return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")

    def _extract_case_law_celexs(self, case_law_dict: dict[str, Any]) -> set[str]:
        """Extract CELEX IDs from case_law_about structure."""
        celexs = set()
        if not isinstance(case_law_dict, dict):
            return celexs

        for category, items in case_law_dict.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "celex" in item:
                        celexs.add(item["celex"])

        return celexs

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names in consistent order."""
        # Extract one pair to get all feature names
        # Use dummy values
        sample_features = self.extract_features(
            query_celex="dummy",
            query_par_num=1,
            cand_celex="dummy",
            cand_par_num=1,
            dense_similarity=0.5,
        )
        return sorted(sample_features.keys())
