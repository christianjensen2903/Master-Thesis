from __future__ import annotations

"""Streamlit UI to review and annotate sampled paragraph citation pairs.

The app loads JSON/JSONL exported by `sample_citation_pairs.py`, displays one
pair at a time, shows EUR-Lex links for source and target CELEX identifiers,
and lets you mark whether the citation is correct along with optional notes and
annotator. Changes can be saved back to disk.
"""

from pathlib import Path
from typing import Any
import json
import logging
import html
import re

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
from streamlit_shortcuts import shortcut_button  # type: ignore


LOGGER = logging.getLogger(__name__)


def build_eurlex_url(celex: str | None) -> str | None:
    """Return a EUR-Lex URL for a CELEX identifier.

    Parameters
    ----------
    celex
        The CELEX identifier (e.g., "62019CJ0311").

    Returns
    -------
    str | None
        EUR-Lex URL or None if `celex` is falsy.
    """

    if not celex:
        return None
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"


def read_records(path: Path) -> pd.DataFrame:
    """Load records from a JSON or JSONL file.

    Parameters
    ----------
    path
        Input file path.

    Returns
    -------
    pd.DataFrame
        DataFrame with the records.
    """

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try JSONL first for speed; fallback to JSON array
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        # Not JSON Lines; might be a JSON array
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)


def write_records(df: pd.DataFrame, path: Path) -> None:
    """Write DataFrame to JSONL with UTF-8 encoding.

    Parameters
    ----------
    df
        DataFrame to write.
    path
        Output file path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure datetimes are strings
    out_df = df.copy()
    for col in out_df.columns:
        if pd.api.types.is_datetime64_any_dtype(out_df[col]):
            out_df[col] = out_df[col].dt.strftime("%Y-%m-%d")
    # Write JSONL
    with path.open("w", encoding="utf-8") as f:
        for record in out_df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns for CELEX and text, if missing.

    - Derives `FROM_CELEX` and `TO_CELEX` from existing columns when possible.
    - Keeps original columns intact.
    """

    result = df.copy()

    def first_present(cols: list[str]) -> str | None:
        for c in cols:
            if c in result.columns:
                return c
        return None

    # CELEX ids
    from_celex_col = first_present(["CELEX_FROM", "from_celex", "FROM_CELEX"])
    to_celex_col = first_present(["CELEX_TO", "to_celex", "TO_CELEX"])

    if from_celex_col is None and "FROM_ID" in result.columns:
        # Expect format CELEX::NUMBER
        result["FROM_CELEX"] = result["FROM_ID"].astype(str).str.split("::").str[0]
    elif from_celex_col is not None and from_celex_col != "FROM_CELEX":
        result["FROM_CELEX"] = result[from_celex_col]

    if to_celex_col is None and "TO_ID" in result.columns:
        result["TO_CELEX"] = result["TO_ID"].astype(str).str.split("::").str[0]
    elif to_celex_col is not None and to_celex_col != "TO_CELEX":
        result["TO_CELEX"] = result[to_celex_col]

    # Text columns
    if "TEXT_FROM" not in result.columns:
        text_from_fallback = first_present(
            ["from_text", "FROM_TEXT", "source_text", "TEXT"]
        )
        if text_from_fallback is not None:
            result["TEXT_FROM"] = result[text_from_fallback]
    if "TEXT_TO" not in result.columns:
        text_to_fallback = first_present(["to_text", "TO_TEXT", "target_text", "TEXT"])
        if text_to_fallback is not None:
            result["TEXT_TO"] = result[text_to_fallback]

    # Title columns
    if "TITLE_FROM" not in result.columns:
        title_from_fallback = first_present(
            ["from_title", "FROM_TITLE", "source_title", "TITLE"]
        )
        if title_from_fallback is not None:
            result["TITLE_FROM"] = result[title_from_fallback]
    if "TITLE_TO" not in result.columns:
        title_to_fallback = first_present(
            ["to_title", "TO_TITLE", "target_title", "TITLE"]
        )
        if title_to_fallback is not None:
            result["TITLE_TO"] = result[title_to_fallback]

    # Number columns
    if "NUMBER_FROM" not in result.columns:
        num_from_fallback = first_present(
            ["number_from", "FROM_NUMBER", "source_number", "NUMBER"]
        )
        if num_from_fallback is not None:
            result["NUMBER_FROM"] = result[num_from_fallback]
    if "NUMBER_TO" not in result.columns:
        num_to_fallback = first_present(
            ["number_to", "TO_NUMBER", "target_number", "NUMBER"]
        )
        if num_to_fallback is not None:
            result["NUMBER_TO"] = result[num_to_fallback]

    # Date columns
    if "DATE_FROM" in result.columns:
        result["DATE_FROM"] = pd.to_datetime(result["DATE_FROM"], errors="coerce")
    if "DATE_TO" in result.columns:
        result["DATE_TO"] = pd.to_datetime(result["DATE_TO"], errors="coerce")

    # Annotation columns
    for col in ["is_correct", "annotator", "notes"]:
        if col not in result.columns:
            result[col] = None

    # Ensure stable dtypes for annotation columns to avoid pandas warnings
    if "annotator" in result.columns:
        result["annotator"] = result["annotator"].astype("object")
    if "notes" in result.columns:
        result["notes"] = result["notes"].astype("object")

    # Split column
    if "split" not in result.columns and "DATE_FROM" in result.columns:
        cutoff = pd.Timestamp("2015-01-01")
        result["split"] = (result["DATE_FROM"] < cutoff).map(
            {True: "pre_cutoff", False: "post_cutoff"}
        )

    return result


def _extract_highlight_terms(
    title: str | None, number: str | None, date_value: Any | None = None
) -> list[str]:
    """Build a list of tokens to highlight based on title and number strings.

    Parameters
    ----------
    title
        Title string to tokenize.
    number
        Number string to tokenize.

    Returns
    -------
    list[str]
        Distinct tokens to highlight, longest-first for regex stability.
    """

    tokens: set[str] = set()
    if title:
        # Split on non-word characters; keep alphanumeric words length >= 3
        for tok in re.split(r"\W+", str(title)):
            tok_clean = tok.strip()
            if len(tok_clean) >= 3:
                tokens.add(tok_clean)
    if number:
        # Keep compact tokens composed of word chars and separators like '-' and '/'
        # Also split to capture subparts like numbers
        raw = str(number)
        complex_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/]*", raw)
        for tok in complex_tokens:
            tokens.add(tok)
        for tok in re.split(r"\W+", raw):
            if len(tok) >= 2:
                tokens.add(tok)

    # Years from date (e.g., 2020). Accept any 4-digit sequence in the date.
    if date_value is not None and str(date_value).strip():
        raw_date = str(date_value)
        for year in re.findall(r"\b\d{4}\b", raw_date):
            tokens.add(year)

    # Order by length desc to avoid partial shadowing in alternation
    return sorted(tokens, key=len, reverse=True)


def _highlight_text(text: str, terms: list[str]) -> str:
    """Return HTML with highlighted terms within the provided text.

    Uses case-insensitive matching with word boundaries when appropriate.
    """

    if not text or not terms:
        return f"<div class=\"para-box\">{html.escape(str(text or ''))}</div>"

    safe_text = html.escape(str(text))

    # Only match standalone tokens: not part of longer alphanumeric sequences.
    escaped_terms = [re.escape(t) for t in terms]
    if not escaped_terms:
        return f'<div class="para-box">{safe_text}</div>'
    combined_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(escaped_terms) + r")(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        return f'<mark class="hl">{match.group(0)}</mark>'

    highlighted = combined_pattern.sub(repl, safe_text)
    return f'<div class="para-box">{highlighted}</div>'


def _row_key(row: pd.Series) -> str:
    """Return a stable key string for a row to use in session state."""

    parts: list[str] = []
    if "id" in row.index and pd.notna(row.get("id")):
        return f"id:{row.get('id')}"
    for c in ["FROM_ID", "TO_ID", "FROM_CELEX", "TO_CELEX", "NUMBER_FROM", "NUMBER_TO"]:
        if c in row.index and pd.notna(row.get(c)):
            parts.append(f"{c}:{row.get(c)}")
    if parts:
        return ";".join(parts)
    # Fallback to position (will change with filters)
    return f"pos:{int(st.session_state.get('idx', 0))}"


def _update_memory(
    df: pd.DataFrame,
    row: pd.Series,
    is_correct: bool | None,
) -> None:
    """Update both the filtered view row and the original df with new annotation values.

    Parameters
    ----------
    df
        The original dataframe loaded from disk (normalized columns).
    row
        The current row (from the filtered view).
    is_correct
        Current correctness annotation.
    """

    # Reflect changes back to the main df via index alignment
    try:
        key_cols = [c for c in ["id", "FROM_ID", "TO_ID"] if c in df.columns]
        if key_cols:
            mask = pd.Series([True] * len(df))
            for c in key_cols:
                mask &= df[c].astype(str) == str(row.get(c, ""))
            idx_list = df[mask].index.tolist()
            if idx_list:
                idx0 = idx_list[0]
                df.at[idx0, "is_correct"] = is_correct
        else:
            # Fallback by position (less reliable)
            idx0 = int(st.session_state.get("idx", 0))
            if "is_correct" in df.columns:
                df.iat[idx0, df.columns.get_loc("is_correct")] = is_correct
    except Exception as e:  # pragma: no cover - UI feedback only
        st.error(f"Failed to update record in memory: {e}")


def render_pair(row: pd.Series) -> None:
    """Render a single pair with EUR-Lex links and metadata.

    Parameters
    ----------
    row
        Current record as a pandas Series.
    """

    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("Source paragraph")
        st.write(f"CELEX: {row.get('FROM_CELEX', row.get('CELEX_FROM', '—'))}")
        url_from = build_eurlex_url(
            str(row.get("FROM_CELEX", row.get("CELEX_FROM", ""))) or None
        )
        if url_from:
            st.link_button("Open on EUR-Lex", url_from, use_container_width=True)
        st.write(f"Number: {row.get('NUMBER_FROM', '—')}")
        st.write(f"Title: {row.get('TITLE_FROM', '—')}")
        st.write(f"Date: {row.get('DATE_FROM', '—')}")
        st.caption("Text")
        # Highlight words from target title/number in source text
        terms = _extract_highlight_terms(
            str(row.get("TITLE_TO", "") or ""),
            str(row.get("NUMBER_TO", "") or ""),
            row.get("DATE_TO", None),
        )
        st.markdown(
            _highlight_text(str(row.get("TEXT_FROM", "") or ""), terms),
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Target paragraph")
        st.write(f"CELEX: {row.get('TO_CELEX', row.get('CELEX_TO', '—'))}")
        url_to = build_eurlex_url(
            str(row.get("TO_CELEX", row.get("CELEX_TO", ""))) or None
        )
        if url_to:
            st.link_button(
                "Open on EUR-Lex", url_to, use_container_width=True, type="secondary"
            )
        st.write(f"Number: {row.get('NUMBER_TO', '—')}")
        st.write(f"Title: {row.get('TITLE_TO', '—')}")
        st.write(f"Date: {row.get('DATE_TO', '—')}")
        st.caption("Text")
        st.markdown(
            _highlight_text(str(row.get("TEXT_TO", "") or ""), terms),
            unsafe_allow_html=True,
        )


def main() -> None:
    """Run the Streamlit application."""

    st.title("Review Citation Pairs")

    with st.sidebar:
        st.header("Data")
        default_path = Path("data/sampled_par_pairs.jsonl")
        input_path_str = st.text_input(
            "Input file (JSONL/JSON)", value=str(default_path)
        )
        input_path = Path(input_path_str)
        save_path_str = st.text_input("Save to", value=str(input_path))
        save_path = Path(save_path_str)

        st.divider()
        st.header("Filters")
        split_filter = st.selectbox(
            "Split", options=["all", "pre_cutoff", "post_cutoff"], index=0
        )

    try:
        df = read_records(input_path)
    except Exception as e:
        st.error(f"Failed to load records: {e}")
        st.stop()

    df = normalize_columns(df)

    # Apply split filter
    if split_filter != "all" and "split" in df.columns:
        view_df = df[df["split"] == split_filter].reset_index(drop=True)
    else:
        view_df = df.reset_index(drop=True)

    if len(view_df) == 0:
        st.info("No records to display with current filter.")
        st.stop()

    # Session state for index
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    def clamp_index(idx: int) -> int:
        return max(0, min(idx, len(view_df) - 1))

    # Navigation controls (defer index change until after saving current row)
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 3, 1])
    with col_nav1:
        if st.button("⏮ First", key="btn_first"):
            st.session_state["pending_nav"] = "first"
    with col_nav2:
        if shortcut_button("◀ Prev", ["arrowleft", "a"], key="btn_prev", hint=False):
            st.session_state["pending_nav"] = "prev"
    with col_nav3:
        st.write("")
        st.write(f"Record {st.session_state.idx + 1} / {len(view_df)}")
    with col_nav4:
        if shortcut_button("Next ▶", ["arrowright", "d"], key="btn_next", hint=False):
            st.session_state["pending_nav"] = "next"

    row = view_df.iloc[st.session_state.idx]
    render_pair(row)

    st.divider()
    st.subheader("Label this citation pair")

    # Show current status
    row_key = _row_key(row)
    current_status = row.get("is_correct", None)
    if current_status is True:
        st.info("✅ Currently marked as: **Correct**")
    elif current_status is False:
        st.warning("❌ Currently marked as: **Incorrect**")
    else:
        st.caption("No label yet")

    # Two buttons for labeling with keyboard shortcuts
    col_label1, col_label2 = st.columns(2)
    label_correct_clicked = False
    label_incorrect_clicked = False

    with col_label1:
        label_correct_clicked = shortcut_button(
            "✅ Correct (q)",
            "q",
            key=f"btn_correct_{row_key}",
            type="primary" if current_status is True else "secondary",
            use_container_width=True,
        )
    with col_label2:
        label_incorrect_clicked = shortcut_button(
            "❌ Incorrect (w)",
            "w",
            key=f"btn_incorrect_{row_key}",
            type="primary" if current_status is False else "secondary",
            use_container_width=True,
        )

    # Helper to persist current row into memory and optionally to disk
    def persist_current(save_disk: bool, is_correct_value: bool | None) -> None:
        view_df.at[st.session_state.idx, "is_correct"] = is_correct_value
        _update_memory(
            df,
            row,
            is_correct=is_correct_value,
        )
        if save_disk:
            try:
                write_records(df, save_path)
                st.toast(f"Saved to {save_path}")
            except Exception as e:  # pragma: no cover - UI feedback only
                st.error(f"Failed to write records: {e}")

    # Handle labeling - save to disk and advance to next
    if label_correct_clicked:
        persist_current(save_disk=True, is_correct_value=True)
        st.session_state.idx = clamp_index(st.session_state.idx + 1)
        st.rerun()
    elif label_incorrect_clicked:
        persist_current(save_disk=True, is_correct_value=False)
        st.session_state.idx = clamp_index(st.session_state.idx + 1)
        st.rerun()

    # Handle deferred navigation
    pending = st.session_state.pop("pending_nav", None)
    if pending is not None:
        if pending == "first":
            st.session_state.idx = 0
        elif pending == "prev":
            st.session_state.idx = clamp_index(st.session_state.idx - 1)
        elif pending == "next":
            st.session_state.idx = clamp_index(st.session_state.idx + 1)
        # Force a rerun so the new index renders immediately
        try:
            st.rerun()
        except Exception:
            pass

    # Minimal CSS for highlighted text blocks
    st.markdown(
        """
        <style>
        .para-box { white-space: pre-wrap; border: 1px solid #eee; padding: 0.75rem; border-radius: 6px; max-height: 240px; overflow-y: auto; }
        mark.hl { background-color: #fff3bf; padding: 0 2px; border-radius: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
