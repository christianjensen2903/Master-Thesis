import csv
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


def _map_normalized_to_original(
    original_text: str, normalized_text: str, norm_start: int, norm_end: int
) -> tuple[int, int] | None:
    """
    Map normalized text positions back to original text positions.

    Returns (start_char, end_char) in original_text or None if mapping fails.
    """
    # Get the substring from normalized text
    norm_substring = normalized_text[norm_start:norm_end]

    # Find this substring in original text
    # Since normalization only changes whitespace, we can find word sequences
    norm_words = norm_substring.split()
    if not norm_words:
        return None

    # Try to find the word sequence in original text
    original_words = original_text.split()
    normalized_all_words = normalized_text.split()

    # Find word indices in normalized text
    char_count = 0
    norm_start_idx = 0
    norm_end_idx = len(normalized_all_words)

    for i, word in enumerate(normalized_all_words):
        if char_count <= norm_start < char_count + len(word):
            norm_start_idx = i
        if char_count <= norm_end <= char_count + len(word):
            norm_end_idx = i + 1
            break
        char_count += len(word) + 1

    if norm_start_idx >= norm_end_idx:
        return None

    # Map to original words
    if len(original_words) == len(normalized_all_words):
        # Word counts match - direct mapping
        char_pos = 0
        start_char = 0
        end_char = len(original_text)

        for i, word in enumerate(original_words):
            if i == norm_start_idx:
                start_char = char_pos
            if i == norm_end_idx - 1:
                end_char = char_pos + len(word)
                break
            char_pos += len(word) + 1

        return (start_char, end_char)
    else:
        # Word counts differ - try to find substring directly
        # Try finding the phrase with different whitespace patterns
        for sep in [" ", "  ", "\t"]:
            target_phrase = sep.join(norm_words)
            start_char = original_text.find(target_phrase)
            if start_char != -1:
                end_char = start_char + len(target_phrase)
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
    input_path: str | Path,
    output_path: str | Path,
    min_verbatim_length: int = 50,
    min_word_count: int = 10,
    mask_token: str = "<VERBATIM>",
) -> None:
    """
    Create a version of par-to-par-cleaned where verbatim passages from targets are masked in queries.

    Args:
        input_path: Path to input par-to-par-cleaned.csv file
        output_path: Path to output CSV file with masked verbatim passages
        min_verbatim_length: Minimum length of verbatim passage to mask (in characters)
        min_word_count: Minimum word count for queries to process
        mask_token: Token to replace verbatim passages with
    """
    # First pass: group rows by TEXT_FROM to collect all targets for each query
    print("Pass 1: Grouping rows by query paragraph...")
    query_to_targets: dict[str, list[str]] = defaultdict(list)
    all_rows: list[dict] = []
    removed_count = 0

    with open(input_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError("CSV file appears to be empty or malformed")

        for row in tqdm(reader, desc="Grouping queries"):
            text_from = row.get("TEXT_FROM", "").strip()
            text_to = row.get("TEXT_TO", "").strip()

            # Skip queries with less than min_word_count words
            if text_from:
                word_count = len(text_from.split())
                if word_count < min_word_count:
                    removed_count += 1
                    continue

            if text_from and text_to:
                query_to_targets[text_from].append(text_to)

            all_rows.append(row)

    print(f"Found {len(query_to_targets)} unique query paragraphs")
    print(f"Total rows: {len(all_rows)}")
    print(f"Removed {removed_count} rows with queries below {min_word_count} words")

    # Second pass: mask verbatim passages for each unique query
    print("\nPass 2: Finding and masking verbatim passages...")
    masked_queries: dict[str, str] = {}

    for query_text, target_texts in tqdm(
        query_to_targets.items(), desc="Masking verbatim passages"
    ):
        # Remove duplicates from target_texts
        unique_targets = list(dict.fromkeys(target_texts))

        # Find verbatim passages
        passages = find_verbatim_passages(
            query_text, unique_targets, min_verbatim_length
        )

        if passages:
            # Mask the passages
            masked_query = mask_verbatim_in_text(query_text, passages, mask_token)
            masked_queries[query_text] = masked_query
        else:
            # No verbatim passages found, keep original
            masked_queries[query_text] = query_text

    # Count how many queries were modified
    modified_count = sum(1 for k, v in masked_queries.items() if k != v)
    print(f"\nModified {modified_count} out of {len(masked_queries)} unique queries")

    # Third pass: update rows with masked queries
    print("\nPass 3: Updating rows with masked queries...")
    output_rows = []

    for row in tqdm(all_rows, desc="Updating rows"):
        output_row = row.copy()
        text_from = row.get("TEXT_FROM", "").strip()

        if text_from and text_from in masked_queries:
            output_row["TEXT_FROM"] = masked_queries[text_from]

        output_rows.append(output_row)

    # Save output CSV
    print(f"\nSaving masked par-to-par data to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nComplete!")
    print(f"Processed {len(output_rows)} rows")
    print(f"Masked verbatim passages in {modified_count} unique queries")


if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "par-to-par-cleaned.csv"
    output_path = base_dir / "data" / "par-to-par-cleaned-masked.csv"

    # Run masking
    mask_verbatim_passages(
        input_path=input_path,
        output_path=output_path,
        min_verbatim_length=50,
        min_word_count=10,
        mask_token="<VERBATIM>",
    )
