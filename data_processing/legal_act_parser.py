from bs4 import BeautifulSoup, Tag, NavigableString
import re
import json
from pathlib import Path
from tqdm import tqdm  # type: ignore


class SimpleLegalActParser:

    def parse(self, html_path: Path) -> dict:
        """Parse legal act and return structured data with articles only"""
        with open(html_path, "r") as file:
            content = file.read()
        self.soup = BeautifulSoup(content, "lxml")

        articles = self._parse_articles()

        return {
            "articles": articles,
        }

    def _parse_articles(self) -> list[dict]:
        """Find articles by looking for p tags with 'Article {number}'"""
        articles = []

        # Find all p tags in the document
        all_p_tags = self.soup.find_all("p")

        i = 0
        while i < len(all_p_tags):
            p_tag = all_p_tags[i]
            text = p_tag.get_text().strip()

            # Check if this is a "Done at ..." tag - stop parsing
            if re.match(r"Done at\s+", text, re.IGNORECASE):
                break

            # Check if this is an article header
            article_match = re.match(r"Article\s+(\d+)", text, re.IGNORECASE)
            if article_match:
                article_number = article_match.group(1)
                article_text = ""

                # Start collecting text from the next element
                j = i + 1
                while j < len(all_p_tags):
                    next_p_tag = all_p_tags[j]
                    next_text = next_p_tag.get_text().strip()

                    # Stop if we find "Done at ..."
                    if re.match(r"Done at\s+", next_text, re.IGNORECASE):
                        break

                    # Stop if we find a new article
                    if re.match(r"Article\s+\d+", next_text, re.IGNORECASE):
                        break

                    # Stop if we find section or subsection in p or span tags
                    if self._has_section_or_subsection(next_p_tag):
                        break

                    # Add this paragraph text
                    if next_text:
                        article_text += next_text + "\n"

                    j += 1

                # Clean up the article text
                article_text = article_text.strip().replace("\u00a0", " ")

                articles.append(
                    {
                        "id": f"Article {article_number}",
                        "text": article_text,
                    }
                )

                # Move to the position where we stopped
                i = j
            else:
                i += 1

        return articles

    def _has_section_or_subsection(self, tag: Tag) -> bool:
        """Check if a tag or its children contain 'section' or 'subsection'"""
        # Check the tag's own text
        text = tag.get_text().strip().lower()
        if re.match(r"(sub)?section", text, re.IGNORECASE):
            return True

        # Check all span children
        for span in tag.find_all("span"):
            span_text = span.get_text().strip().lower()
            if re.match(r"(sub)?section", span_text, re.IGNORECASE):
                return True

        return False


if __name__ == "__main__":
    parser = SimpleLegalActParser()
    legal_acts_dir = Path("legal_acts")
    output_file = Path("data/legal_acts_simple.json")

    all_legal_acts = {}
    html_files = list(legal_acts_dir.glob("*.html"))

    print(f"Found {len(html_files)} HTML files to process")

    for html_file in tqdm(
        html_files, desc="Processing legal acts", total=len(html_files)
    ):
        try:
            celex_id = html_file.stem
            data = parser.parse(html_file)
            all_legal_acts[celex_id] = data

        except Exception as e:
            print(f"Error processing {html_file.name}: {e}")
            continue

    print(f"\nWriting {len(all_legal_acts)} legal acts to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_legal_acts, f, indent=2, ensure_ascii=False)

    print(f"Done! Saved to {output_file}")
