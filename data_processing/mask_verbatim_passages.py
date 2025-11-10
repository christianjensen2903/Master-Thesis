import csv
import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm  # type: ignore


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for matching."""
    return " ".join(text.split())


def _expand_match(
    query_words: list[str],
    target_words: list[str],
    match_start_word: int,
    match_end_word: int,
) -> tuple[int, int]:
    """
    Expand a match as much as possible by extending in both directions.

    Args:
        query_words: Query text split into words
        target_words: Target text split into words
        match_start_word: Starting word index of current match in query
        match_end_word: Ending word index (exclusive) of current match in query

    Returns:
        (expanded_start_word, expanded_end_word) - expanded word indices
    """
    # Find where this match appears in target
    match_text = " ".join(query_words[match_start_word:match_end_word])
    target_match_start = -1
    target_match_end = -1

    # Find the match position in target
    for i in range(len(target_words) - (match_end_word - match_start_word) + 1):
        target_window = " ".join(
            target_words[i : i + (match_end_word - match_start_word)]
        )
        if target_window == match_text:
            target_match_start = i
            target_match_end = i + (match_end_word - match_start_word)
            break

    if target_match_start == -1:
        # Match not found in target (shouldn't happen), return original
        return match_start_word, match_end_word

    # Try to expand backward
    expanded_start = match_start_word
    while expanded_start > 0 and target_match_start > 0:
        # Check if we can add one more word from both query and target
        if query_words[expanded_start - 1] == target_words[target_match_start - 1]:
            expanded_start -= 1
            target_match_start -= 1
        else:
            break

    # Try to expand forward
    expanded_end = match_end_word
    while expanded_end < len(query_words) and target_match_end < len(target_words):
        # Check if we can add one more word from both query and target
        if query_words[expanded_end] == target_words[target_match_end]:
            expanded_end += 1
            target_match_end += 1
        else:
            break

    return expanded_start, expanded_end


def find_verbatim_passages(
    query_text: str, target_texts: list[str], min_length: int = 50
) -> list[tuple[int, int]]:
    """
    Find all verbatim passages in query_text that match any target_text.
    Expands matches as much as possible to consume maximum verbatim text.

    Returns list of (start, end) positions in original query_text that should be masked.

    Args:
        query_text: The query paragraph text (original)
        target_texts: List of target passage texts to match against
        min_length: Minimum length of verbatim passage to mask (in characters)

    Returns:
        List of (start, end) tuples for passages to mask
    """
    if not query_text or not target_texts:
        return []

    # Normalize whitespace for matching
    query_normalized = normalize_whitespace(query_text)
    query_words = query_normalized.split()

    # Collect all matches from all targets
    matches: list[tuple[int, int]] = []

    for target_text in target_texts:
        if not target_text:
            continue

        target_normalized = normalize_whitespace(target_text)
        target_words = target_normalized.split()

        if len(target_normalized) < min_length or len(target_words) < 3:
            continue

        # Strategy: Find all possible word sequence matches, then expand each one
        # Try all possible word sequences from target that could match in query
        min_words = max(3, min_length // 20)  # Minimum words to match
        max_words = min(len(target_words), 100)  # Reasonable upper limit

        # Try sequences of decreasing length to find longest matches first
        for seq_len in range(max_words, min_words - 1, -1):
            for target_start in range(len(target_words) - seq_len + 1):
                target_seq = target_words[target_start : target_start + seq_len]
                target_seq_text = " ".join(target_seq)

                if len(target_seq_text) < min_length:
                    continue

                # Find all occurrences of this sequence in query
                if target_seq_text in query_normalized:
                    query_start = 0
                    while True:
                        query_start = query_normalized.find(
                            target_seq_text, query_start
                        )
                        if query_start == -1:
                            break

                        # Find which words in query this corresponds to
                        query_word_start = 0
                        char_count = 0
                        match_start_word = 0
                        match_end_word = 0

                        for i, word in enumerate(query_words):
                            if char_count <= query_start < char_count + len(word):
                                match_start_word = i
                            if (
                                char_count
                                <= query_start + len(target_seq_text)
                                <= char_count + len(word)
                            ):
                                match_end_word = i + 1
                                break
                            char_count += len(word) + 1

                        if match_end_word > match_start_word:
                            # Expand this match as much as possible
                            expanded_start, expanded_end = _expand_match(
                                query_words,
                                target_words,
                                match_start_word,
                                match_end_word,
                            )

                            # Map expanded word indices back to character positions
                            orig_pos = _map_word_indices_to_char_positions(
                                query_text, query_words, expanded_start, expanded_end
                            )
                            if orig_pos:
                                start, end = orig_pos
                                if end - start >= min_length:
                                    matches.append((start, end))

                        query_start += 1

    if not matches:
        return []

    # Sort matches by start position
    matches.sort(key=lambda x: x[0])

    # Merge overlapping matches (with larger merge distance to catch adjacent matches)
    merged: list[tuple[int, int]] = []
    for start, end in matches:
        if not merged:
            merged.append((start, end))
        else:
            last_start, last_end = merged[-1]
            # If overlapping or very close (within 20 chars), merge
            if start <= last_end + 20:
                merged[-1] = (min(last_start, start), max(last_end, end))
            else:
                merged.append((start, end))

    return merged


def _map_word_indices_to_char_positions(
    original_text: str,
    normalized_words: list[str],
    start_word_idx: int,
    end_word_idx: int,
) -> tuple[int, int] | None:
    """
    Map word indices to character positions in original text.

    Returns (start_char, end_char) or None if mapping fails.
    """
    if start_word_idx >= end_word_idx or end_word_idx > len(normalized_words):
        return None

    original_words = original_text.split()

    # If word counts match, direct mapping
    if len(original_words) == len(normalized_words):
        char_pos = 0
        start_char = 0
        end_char = len(original_text)

        for i, word in enumerate(original_words):
            if i == start_word_idx:
                start_char = char_pos
            if i == end_word_idx - 1:
                end_char = char_pos + len(word)
                break
            char_pos += len(word) + 1

        return (start_char, end_char)
    else:
        # Word counts differ - try to find the word sequence in original
        target_words = normalized_words[start_word_idx:end_word_idx]
        target_phrase = " ".join(target_words)

        # Try finding with different whitespace
        for sep in [" ", "  ", "\t"]:
            target_with_sep = sep.join(target_words)
            start_char = original_text.find(target_with_sep)
            if start_char != -1:
                end_char = start_char + len(target_with_sep)
                return (start_char, end_char)

        return None


def mask_verbatim_in_text(
    text: str, passages: list[tuple[int, int]], mask_token: str = "<VERBATIM>"
) -> str:
    """
    Mask verbatim passages in text.

    Args:
        text: Original text
        passages: List of (start, end) positions to mask
        mask_token: Token to replace verbatim passages with

    Returns:
        Text with verbatim passages masked
    """
    if not passages:
        return text

    # Sort by start position (descending) to mask from end to start
    passages_sorted = sorted(passages, key=lambda x: x[0], reverse=True)

    # Apply masks from end to start to preserve positions
    masked_text = text
    for start, end in passages_sorted:
        # Replace the substring with mask token
        before = masked_text[:start]
        after = masked_text[end:]
        masked_text = before + mask_token + after

    # Clean up multiple consecutive mask tokens
    while mask_token + " " + mask_token in masked_text:
        masked_text = masked_text.replace(mask_token + " " + mask_token, mask_token)
    while mask_token + mask_token in masked_text:
        masked_text = masked_text.replace(mask_token + mask_token, mask_token)

    # Normalize whitespace around mask tokens
    masked_text = masked_text.replace("  ", " ")  # Remove double spaces

    return masked_text


def mask_verbatim_passages(
    queries_path: str | Path,
    qrel_path: str | Path,
    judgments_path: str | Path,
    output_path: str | Path,
    min_verbatim_length: int = 50,
    mask_token: str = "<VERBATIM>",
) -> None:
    """
    Create a version of queries_cleaned.tsv where verbatim passages from relevant documents (qrels) are masked.

    Args:
        queries_path: Path to input queries_cleaned.tsv file
        qrel_path: Path to qrel.txt file
        judgments_path: Path to judgments_cleaned.json file
        output_path: Path to output TSV file with masked verbatim passages
        min_verbatim_length: Minimum length of verbatim passage to mask (in characters)
        mask_token: Token to replace verbatim passages with
    """
    # Load judgments to get paragraph texts
    print("Loading judgments...")
    with open(judgments_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)

    # Build paragraph index: (celex, paragraph_number) -> text
    paragraph_texts: dict[tuple[str, int], str] = {}
    for celex, judgment in tqdm(judgments.items(), desc="Indexing paragraphs"):
        for par_num_str, text in judgment.get("paragraphs", {}).items():
            try:
                par_num = int(par_num_str)
                paragraph_texts[(celex, par_num)] = text
            except ValueError:
                continue

    print(f"Indexed {len(paragraph_texts)} paragraphs")

    # Load queries
    print("\nLoading queries...")
    queries: list[tuple[str, int, str]] = []  # (celex, paragraph_number, query_text)
    query_key_to_text: dict[tuple[str, int], str] = {}

    with open(queries_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile, delimiter="\t")
        next(reader)  # Skip header

        for row in tqdm(reader, desc="Loading queries"):
            if len(row) < 3:
                continue
            celex = row[0].strip()
            par_num_str = row[1].strip()
            query_text = row[2].strip()

            try:
                par_num = int(par_num_str)
            except ValueError:
                continue

            query_key = (celex, par_num)
            queries.append((celex, par_num, query_text))
            query_key_to_text[query_key] = query_text

    print(f"Loaded {len(queries)} queries")

    # Load qrel to map queries to relevant documents
    print("\nLoading qrel...")
    query_to_targets: dict[tuple[str, int], list[str]] = defaultdict(list)

    with open(qrel_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading qrel"):
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            query_id = parts[0]
            doc_id = parts[2]

            # Parse celex_paragraph_number format
            try:
                celex_q, par_num_q_str = query_id.rsplit("_", 1)
                celex_d, par_num_d_str = doc_id.rsplit("_", 1)
                par_num_q = int(par_num_q_str)
                par_num_d = int(par_num_d_str)
            except (ValueError, IndexError):
                continue

            query_key = (celex_q, par_num_q)
            doc_key = (celex_d, par_num_d)

            # Get target text from paragraph index
            if doc_key in paragraph_texts:
                target_text = paragraph_texts[doc_key]
                query_to_targets[query_key].append(target_text)

    print(f"Loaded qrel mappings for {len(query_to_targets)} queries")

    # Mask verbatim passages for each query
    print("\nFinding and masking verbatim passages...")
    masked_queries: dict[tuple[str, int], str] = {}
    modified_count = 0

    for query_key, query_text in tqdm(
        query_key_to_text.items(), desc="Masking verbatim passages"
    ):
        target_texts = query_to_targets.get(query_key, [])

        if not target_texts:
            # No relevant documents, keep original
            masked_queries[query_key] = query_text
            continue

        # Remove duplicates from target_texts
        unique_targets = list(dict.fromkeys(target_texts))

        # Find verbatim passages
        passages = find_verbatim_passages(
            query_text, unique_targets, min_verbatim_length
        )

        if passages:
            # Mask the passages
            masked_query = mask_verbatim_in_text(query_text, passages, mask_token)
            masked_queries[query_key] = masked_query
            modified_count += 1
        else:
            # No verbatim passages found, keep original
            masked_queries[query_key] = query_text

    print(f"\nModified {modified_count} out of {len(masked_queries)} queries")

    # Write output TSV
    print(f"\nSaving masked queries to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        writer.writerow(["celex", "paragraph_number", "query"])

        for celex, par_num, _ in queries:
            query_key = (celex, par_num)
            masked_query = masked_queries.get(
                query_key, query_key_to_text.get(query_key, "")
            )
            writer.writerow([celex, par_num, masked_query])

    print(f"\nComplete!")
    print(f"Processed {len(queries)} queries")
    print(f"Masked verbatim passages in {modified_count} queries")


def mask_verbatim_passages_par_to_par(
    input_path: str | Path,
    output_path: str | Path,
    min_verbatim_length: int = 50,
    mask_token: str = "<VERBATIM>",
) -> None:
    """
    Create a version of par-to-par-cleaned.csv where verbatim passages from TEXT_TO are masked in TEXT_FROM.

    Args:
        input_path: Path to input par-to-par-cleaned.csv file
        output_path: Path to output CSV file with masked verbatim passages
        min_verbatim_length: Minimum length of verbatim passage to mask (in characters)
        mask_token: Token to replace verbatim passages with
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load par-to-par data
    print(f"Loading data from {input_path}...")
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []

    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])
        rows = list(tqdm(reader, desc="Loading rows"))

    print(f"Loaded {len(rows)} rows")

    if "TEXT_FROM" not in fieldnames or "TEXT_TO" not in fieldnames:
        raise ValueError("CSV must contain TEXT_FROM and TEXT_TO columns")

    # Mask verbatim passages for each row
    print("\nFinding and masking verbatim passages...")
    modified_count = 0

    for row in tqdm(rows, desc="Masking verbatim passages"):
        text_from = row.get("TEXT_FROM", "").strip()
        text_to = row.get("TEXT_TO", "").strip()

        if not text_from or not text_to:
            # Skip rows with empty text
            continue

        # Find verbatim passages in TEXT_FROM that match TEXT_TO
        passages = find_verbatim_passages(text_from, [text_to], min_verbatim_length)

        if passages:
            # Mask the passages
            masked_text = mask_verbatim_in_text(text_from, passages, mask_token)
            row["TEXT_FROM"] = masked_text
            modified_count += 1

    print(f"\nModified {modified_count} out of {len(rows)} rows")

    # Write output CSV
    print(f"\nSaving masked data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComplete!")
    print(f"Processed {len(rows)} rows")
    print(f"Masked verbatim passages in {modified_count} rows")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    queries_path = base_dir / "data" / "evaluation" / "queries_cleaned.tsv"
    qrel_path = base_dir / "data" / "evaluation" / "qrel.txt"
    judgments_path = base_dir / "data" / "judgments_cleaned.json"
    output_path = base_dir / "data" / "evaluation" / "queries_cleaned_masked.tsv"

    # Run masking
    # mask_verbatim_passages(
    #     queries_path=queries_path,
    #     qrel_path=qrel_path,
    #     judgments_path=judgments_path,
    #     output_path=output_path,
    #     min_verbatim_length=50,
    #     mask_token="<VERBATIM>",
    # )
    mask_verbatim_passages_par_to_par(
        input_path=base_dir / "data" / "par-to-par-cleaned.csv",
        output_path=base_dir / "data" / "par-to-par-cleaned-masked.csv",
        min_verbatim_length=50,
        mask_token="<VERBATIM>",
    )
