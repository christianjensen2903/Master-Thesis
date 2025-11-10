from collections import OrderedDict
from bs4 import BeautifulSoup, Tag
import re
import json
from urllib.parse import urljoin
from pathlib import Path
from typing import Any
from tqdm import tqdm  # type: ignore


class LegalActParser:
    BASE_URL = "https://eur-lex.europa.eu/"

    def parse(self, html_path: Path) -> dict:
        """Parse legal act and return structured data"""
        with open(html_path, "r") as file:
            content = file.read()
        self.soup = BeautifulSoup(content, "lxml")

        modifies_documents = self._extract_related_documents("relatedDocsTbMS")
        modified_by_documents = self._extract_related_documents("relatedDocsTb")

        preamble = self._parse_preamble()
        articles = self._parse_articles()
        article_notes = [note for article in articles for note in article["notes"]]
        article_references = [
            ref for article in articles for ref in article["references"]
        ]

        return {
            "title": self._parse_title(),
            "preamble": preamble,
            "articles": articles,
            "final_part": self._parse_final_part(),
            "notes": preamble["notes"] + article_notes,
            "references": list(
                dict.fromkeys(preamble["references"] + article_references)
            ),
            "annexes": self._parse_annexes(),
            "summary": self._get_summary(),
            "related_documents": {
                "modifies": modifies_documents,
                "modified_by": modified_by_documents,
            },
        }

    def _parse_title(self) -> str:
        tit_1_div = self.soup.find("div", id="tit_1")
        if tit_1_div:
            title_text = tit_1_div.text
            return title_text.replace("\u00a0", " ").strip()
        return ""

    def _parse_final_part(self) -> str:
        fnp_1_div = self.soup.find("div", id="fnp_1")
        if fnp_1_div:
            lines = [line for line in fnp_1_div.text.split("\n") if line.strip()]
            text = "\n".join(lines)
            fnp_text = re.sub(r"(\(\d+\))\n", r"\1 ", text)
            return fnp_text.replace("\u00a0", " ")
        return ""

    def _parse_preamble(self) -> dict:
        pbl_1_div = self.soup.find("div", id="pbl_1")
        pbl_text = ""
        if pbl_1_div:
            lines = [line for line in pbl_1_div.text.split("\n") if line.strip()]
            text = "\n".join(lines)
            pbl_text = re.sub(r"(\(\d+\))\n", r"\1 ", text)
            pbl_text = pbl_text.replace("\u00a0", " ")

        notes = self._extract_notes(pbl_1_div)

        return {
            "text": pbl_text,
            "notes": notes,
            "references": self._extract_directives_and_regulations(pbl_text),
        }

    def _parse_articles(self) -> list[dict]:
        articles = []
        divs_with_art_id = self.soup.find_all(
            "div", class_="eli-subdivision", id=lambda x: x and x.startswith("art")
        )

        for div in divs_with_art_id:
            notes = self._extract_notes(div)
            article_id = ""
            article_title = ""
            article_text = ""

            for c in div.children:
                if not isinstance(c, Tag):
                    continue

                if c.name == "p" and "ti-art" in str(c.get("class")):
                    article_id = c.text.replace("\n", "").replace("\u00a0", " ")
                elif c.name == "div" and c.get("class") == ["eli-title"]:
                    article_title = c.text.replace("\n", "")
                else:
                    article_text += self._clean_text(c.text)

            article_text = (
                article_text.lstrip("\n").rstrip("\n").replace("\n\n\n", "\n")
            )

            parent_info = self._find_parent_title(div.find_parent("div"))
            parent_info = OrderedDict(reversed(list(parent_info.items())))

            articles.append(
                {
                    "id": article_id,
                    "title": article_title,
                    "text": article_text,
                    "metadata": parent_info,
                    "notes": notes,
                    "references": self._extract_directives_and_regulations(
                        article_text
                    ),
                }
            )

        return articles

    def _parse_annexes(self) -> list[dict]:
        annexes = []
        divs_with_anx_id = self.soup.find_all(
            "div", class_="eli-container", id=lambda x: x and x.startswith("anx")
        )

        for div in divs_with_anx_id:
            annex_id = ""
            annex_title = ""
            annex_text = ""
            annex_table = ""

            for c in div.children:
                if not isinstance(c, Tag):
                    continue

                if c.name == "p" and "doc-ti" in str(c.get("class")):
                    annex_id = c.text.strip()
                elif (
                    c.name == "p"
                    and "ti-grseq-1" in str(c.get("class"))
                    and not annex_title
                ):
                    annex_title = c.text.strip()
                elif c.name == "table" and "table" in str(c.get("class")):
                    annex_table = self._html_table_to_markdown(str(c))
                else:
                    annex_text += self._clean_text(c.text)

            annex_text = annex_text.lstrip("\n").rstrip("\n").replace("\n\n\n", "\n")

            annexes.append(
                {
                    "id": annex_id,
                    "title": annex_title,
                    "text": annex_text,
                    "table": annex_table,
                    "references": self._extract_directives_and_regulations(annex_text),
                }
            )

        return annexes

    def _get_summary(self) -> dict:
        """Extract summary information supporting multiple languages"""
        title_h1 = self.soup.find("h1", class_="ti-main")
        title_text = title_h1.text if title_h1 else ""

        lastmod_div = self.soup.find("p", class_="lastmod")
        last_modified = lastmod_div.text.strip() if lastmod_div else ""

        chapter_contents = {}
        chapters = self.soup.find_all("h2", class_="ti-chapter")

        for chapter in chapters:
            chapter_title = chapter.text.strip()
            content = []

            for sibling in chapter.find_next_siblings():
                sibling_classes = sibling.get("class")
                if sibling_classes and isinstance(sibling_classes, list):
                    if sibling.name == "h2" and (
                        "ti-chapter" in sibling_classes or "lastmod" in sibling_classes
                    ):
                        break
                elif sibling.name == "h2":
                    break

                if sibling.name == "ul":
                    list_items = sibling.find_all("li")
                    for item in list_items:
                        text = "- " + item.get_text().strip().replace("\xa0", "")
                        content.append(text)
                else:
                    content.append(sibling.get_text().replace("\xa0", ""))

            chapter_contents[chapter_title] = "\n".join(content)

        return {
            "title": title_text,
            "chapters": chapter_contents,
            "last_modified": last_modified,
        }

    def _extract_related_documents(self, table_id: str) -> list[dict]:
        table = self.soup.find("table", id=table_id)
        if not table:
            return []

        thead = table.find("thead")
        if not thead:
            return []

        headers = [header.get_text(strip=True) for header in thead.find_all("th")]

        tbody = table.find("tbody")
        if not tbody:
            return []

        data_list = []
        for row in tbody.find_all("tr"):
            columns = row.find_all("td")
            data_dict: dict[str, Any] = {}

            for i, key in enumerate(headers):
                if i >= len(columns):
                    continue

                if key == "Act":
                    a_tag = columns[i].find("a")
                    if a_tag:
                        href = a_tag.get("href")
                        relative_url = href if isinstance(href, str) else ""
                        absolute_url = urljoin(self.BASE_URL, relative_url)
                        data_dict[key] = {
                            "celex": a_tag.get_text(strip=True),
                            "url": absolute_url,
                        }
                    else:
                        data_dict[key] = {}
                else:
                    data_dict[key] = columns[i].get_text(strip=True)

            data_list.append(data_dict)

        return data_list

    def _extract_notes(self, div: Tag | None) -> list[dict]:
        if not div:
            return []

        note_tags = div.find_all("span", class_="oj-super oj-note-tag")
        notes = []

        for note in note_tags:
            note_dic = {}

            parent_a = note.find_parent("a")
            foot_note_id = (
                parent_a["href"][1:] if parent_a and "href" in parent_a.attrs else None
            )
            foot_note = self.soup.find("a", id=foot_note_id) if foot_note_id else None

            note_dic["id"] = note.text
            note_text = ""
            parent_p = None

            if foot_note:
                parent_p = foot_note.find_parent("p")
                if parent_p:
                    note_text = parent_p.text

            cleaned_note_text = self._extract_note_text(note_text)
            note_dic["text"] = cleaned_note_text

            url = ""
            if foot_note and parent_p:
                a_tags = parent_p.find_all("a")
                if len(a_tags) >= 2:
                    second_a_tag = a_tags[1]
                    href = second_a_tag.get("href", "")
                    if href and isinstance(href, str):
                        index = href.find("legal-content")
                        url = (
                            "https://eur-lex.europa.eu/" + href[index:]
                            if index != -1
                            else ""
                        )

            note_dic["url"] = url
            note_dic["reference"] = self._extract_directive_at_beginning(
                cleaned_note_text
            )
            notes.append(note_dic)

        return notes

    @staticmethod
    def _extract_note_text(text: str) -> str:
        cleaned_text = text.strip().replace("\u00a0", " ")
        cleaned_text = re.sub(r"^\(\d+\)\s+", "", cleaned_text)
        cleaned_text = re.sub(r"^\(\d+\)", "", cleaned_text)
        cleaned_text = re.sub(r"^\(\*\d+\)", "", cleaned_text)
        return cleaned_text.strip()

    def _find_parent_title(
        self, div: Tag | None, depth: int = 0, results: dict | None = None
    ) -> dict:
        if results is None:
            results = {}
        if div is None or depth > 10:
            return results

        key, value = None, None
        for d in div.children:
            if not isinstance(d, Tag):
                continue
            if d.name == "p" and d.get("class") == ["oj-ti-section-1"]:
                key = d.text.strip()
            elif d.name == "div" and d.get("class") == ["eli-title"]:
                value = d.text.strip()

        if key and value:
            results[key] = value
        elif key:
            results[key] = ""

        parent_div = div.find_parent("div")
        return self._find_parent_title(parent_div, depth + 1, results)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"\n([a-z0-9]\))", r" \1", text)
        text = re.sub(r"\n(\d+\.\s+)", r" \1", text)
        text = re.sub(r"(\(\d+\))\n", r"\1 ", text)
        text = re.sub(r"(\s*\d+\.)\n", r"\1 ", text)
        text = re.sub(r"(\([a-z]\))\n", r"\1 ", text)
        text = re.sub(r"(\([IVXLCDM]+\))\n", r"\1 ", text)
        text = re.sub(r"(\([ivxlcdm]+\))\n", r"\1 ", text)
        text = text.replace("\n\n", "")
        text = text.replace("\u00a0", " ")
        return text

    @staticmethod
    def _extract_directive_at_beginning(text: str) -> str | None:
        """Extract directive/regulation at the beginning of text"""
        pattern = (
            r"(^\s*\(?\d{0,3}\)?\s*Directive \d+/\d+/\s?\w{2,3})|"
            r"(^\s*\(?\d{0,3}\)?\s*Directive \(\w{2,3}\) \d+/\d+)|"
            r"(^\s*\(?\d{0,3}\)?\s*Regulation \(\w{2,3}\) No \d+/\d+)|"
            r"(^\s*\(?\d{0,3}\)?\s*Council Regulation \(\w{2,3}\) No \d+/\d+)|"
            r"(^\s*\(?\d{0,3}\)?\s*Regulation \(\w{2,3}\) \d+/\d+)|"
            r"(^\s*\(?\d{0,3}\)?\s*Decision \d+/\d+/\w{2,3})|"
            r"(^\s*\(?\d{0,3}\)?\s*Commission Recommendation \d+/\d+/\w{2,3})|"
            r"(^\s*\(?\d{0,3}\)?\s*Regulation \d+/\d+)"
        )

        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            directive = match.group(0).strip()
            directive = re.sub(r"^\s*\(?\d{0,3}\)?\s*", "", directive)
            return directive
        return None

    @staticmethod
    def _extract_directives_and_regulations(text: str) -> list[str]:
        """Extract all directives and regulations from text"""
        pattern = (
            r"(Directive \d+/\d+/\s?\w{2,3})|"
            r"(Directive \(\w{2,3}\) \d+/\d+)|"
            r"(Regulation \(\w{2,3}\) No \d+/\d+)|"
            r"(Regulation \(\w{2,3}\) \d+/\d+)|"
            r"(Decision \d+/\d+/\w{2,3})|"
            r"(Commission Recommendation \d+/\d+/\w{2,3})|"
            r"(Regulation \d+/\d+)"
        )

        matches = re.findall(pattern, text, re.IGNORECASE)
        results = [match for group in matches for match in group if match]
        unique_results = list(dict.fromkeys(results))

        # Handle "Directives 2014/24/EU, 2014/25/EU or 2014/23/EU"
        directive_pattern = r"Directives?\s+((?:\d{4}/\d+/\w{2,3}\s*(?:, )?)+)\s*or\s+(\d{4}/\d+/\w{2,3})"
        directive_matches = re.findall(directive_pattern, text, re.IGNORECASE)

        if directive_matches:
            combined_directives = ", ".join(directive_matches[0])
            items = combined_directives.split(", ")
            directives_list = ["Directive " + item.strip() for item in items if item]
            unique_results.extend(directives_list)
            unique_results = list(dict.fromkeys(unique_results))

        # Handle "Directives 2014/24/EU or 2014/25/EU"
        directive_pattern = r"Directives? (\d{4}/\d+/\w{2,3})(?:, (\d{4}/\d+/\w{2,3}))* (?:and|or) (\d{4}/\d+/\w{2,3})"
        directive_matches = re.findall(directive_pattern, text, re.IGNORECASE)

        if directive_matches:
            all_matches = [
                match for sublist in directive_matches for match in sublist if match
            ]
            directives_list = ["Directive " + item.strip() for item in all_matches]
            unique_results.extend(directives_list)
            unique_results = list(dict.fromkeys(unique_results))

        # Handle "Regulations (EU) No 2016/679 or (EU) No 2016/680"
        regulation_pattern = r"Regulations? \(EU\) No (\d{3,4}/\d+)(?:, \(EU\) No (\d{3,4}/\d+))* (?:and|or) \(EU\) No (\d{3,4}/\d+)"
        regulation_matches = re.findall(regulation_pattern, text, re.IGNORECASE)

        if regulation_matches:
            all_matches = [
                match for sublist in regulation_matches for match in sublist if match
            ]
            regulations_list = [
                "Regulation (EU) No " + item.strip() for item in all_matches
            ]
            unique_results.extend(regulations_list)
            unique_results = list(dict.fromkeys(unique_results))

        return unique_results

    @staticmethod
    def _html_table_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")

        if not table:
            raise ValueError("No table found in the provided HTML")

        headers = []
        rows = []

        # Extract headers
        header_row = table.find("tr")
        if header_row:
            for th in header_row.find_all("th"):
                headers.append(th.get_text().strip())

        # Extract rows
        for tr in table.find_all("tr"):
            row = []
            for td in tr.find_all(["td", "th"]):
                row.append(td.get_text().strip())
            if row:
                rows.append(row)

        # Determine the number of columns
        num_columns = len(headers) if headers else max(len(row) for row in rows)

        # Ensure all rows have the correct number of columns
        for row in rows:
            while len(row) < num_columns:
                row.append("")

        # Determine the column widths
        column_widths = [0] * num_columns
        for i, header in enumerate(headers):
            column_widths[i] = len(header)
        for row in rows:
            for i, cell in enumerate(row):
                column_widths[i] = max(column_widths[i], len(cell))

        # Create the Markdown table
        markdown = []
        if headers:
            formatted_header = (
                "| "
                + " | ".join(
                    f"{cell:<{column_widths[i]}}" for i, cell in enumerate(headers)
                )
                + " |"
            )
            markdown.append(formatted_header)
        else:
            # first row as header
            formatted_header = (
                "| "
                + " | ".join(
                    f"{cell:<{column_widths[i]}}" for i, cell in enumerate(rows[0])
                )
                + " |"
            )
            markdown.append(formatted_header)

        markdown.append(
            "|" + "|".join("-" * (width + 2) for width in column_widths) + "|"
        )

        if not headers:
            rows = rows[1:]

        for row in rows:
            formatted_row = (
                "| "
                + " | ".join(
                    f"{cell:<{column_widths[i]}}" for i, cell in enumerate(row)
                )
                + " |"
            )
            markdown.append(formatted_row)

        return "\n".join(markdown)


if __name__ == "__main__":
    parser = LegalActParser()
    legal_acts_dir = Path("legal_acts")
    output_file = Path("data/legal_acts.json")

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
