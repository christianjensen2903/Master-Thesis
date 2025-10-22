import json
import sys
from pathlib import Path
from typing import Any
import logging
from pydantic import BaseModel, Field

# Add parent directory to path to import judgment_parser
sys.path.append(str(Path(__file__).parent.parent))
from judgment_parser import JudgementParser
from tqdm import tqdm  # type: ignore


class JudgmentData(BaseModel):
    """Main data structure for storing parsed judgment information."""

    celex_id: str = Field(..., description="Unique CELEX identifier for the judgment")
    paragraphs: dict[int, str] = Field(
        default_factory=dict, description="Paragraphs indexed by number"
    )
    meta: dict[str, Any] = Field(default_factory=dict, description="Metadata")


class ComprehensiveJudgmentParser:
    """Main parser for processing all CJ judgments."""

    def __init__(
        self,
        judgments_dir: str = "judgments",
        output_dir: str = "data",
        par_to_par_file: str = "data/par-to-par.json",
    ):
        self.judgments_dir = Path(judgments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.par_to_par_file = Path(par_to_par_file)

        self.parser = JudgementParser()
        with open(self.par_to_par_file, "r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

    def _find_metadata_for_celex_id(self, celex_id: str) -> dict[str, Any]:
        if celex_id not in self.metadata:
            return {}

        return self.metadata[celex_id].get("meta", {})

    def get_judgments(self) -> list[str]:
        judgments: list[str] = []

        if not self.judgments_dir.exists():
            print(f"Judgments directory not found: {self.judgments_dir}")
            return judgments

        for item in self.judgments_dir.iterdir():
            if item.is_dir() and "CJ" in item.name:
                judgments.append(item.name)

        return judgments

    def parse_single_judgment(self, celex_id: str) -> JudgmentData:
        try:
            paragraphs = self.parser.extract_paragraphs_from_celex(celex_id)
            meta = self._find_metadata_for_celex_id(celex_id)

            judgment_data = JudgmentData(
                celex_id=celex_id, paragraphs=paragraphs, meta=meta
            )

        except Exception as e:
            print(f"Error parsing {celex_id}: {e}")
            judgment_data = JudgmentData(celex_id=celex_id)

        return judgment_data

    def parse_judgments(self, celex_ids: list[str]) -> list[JudgmentData]:

        results = []
        for celex_id in tqdm(celex_ids, desc="Processing judgments"):
            try:
                result = self.parse_single_judgment(celex_id)
                results.append(result)

            except Exception as e:
                print(f"Error processing {celex_id}: {e}")

        return results

    def _save_json(self, results: list[JudgmentData]) -> None:
        complete_data = [result.model_dump() for result in results]
        complete_file = self.output_dir / "judgments.json"

        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)

    def run(self) -> None:
        judgments = self.get_judgments()
        results = self.parse_judgments(judgments)
        self._save_json(results)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse all CJ judgments")
    parser.add_argument(
        "--judgments-dir", default="judgments", help="Path to judgments directory"
    )
    parser.add_argument("--output-dir", default="data", help="Path to output directory")
    parser.add_argument(
        "--par-to-par-file",
        default="data/par-to-par.json",
        help="Path to par-to-par.json file",
    )

    args = parser.parse_args()

    # Create parser and run
    judgment_parser = ComprehensiveJudgmentParser(
        judgments_dir=args.judgments_dir,
        output_dir=args.output_dir,
        par_to_par_file=args.par_to_par_file,
    )

    judgment_parser.run()


if __name__ == "__main__":
    main()
