import json
import sys
from pathlib import Path
from typing import Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from judgment_parser import JudgementParser


class JudgmentData:
    """Data structure for storing parsed judgment information."""

    def __init__(self, celex_id: str):
        self.celex_id = celex_id
        self.paragraphs: dict[int, str] = {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "celex_id": self.celex_id,
            "paragraphs": self.paragraphs,
        }


class ComprehensiveJudgmentParser:
    """Main parser for processing all CJ judgments."""

    def __init__(self, judgments_dir: str = "judgments", output_dir: str = "data"):
        self.judgments_dir = Path(judgments_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.parser = JudgementParser()

        self._setup_logging()

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "total_paragraphs": 0,
            "total_text_length": 0,
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_file = self.output_dir / "parsing.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def get_judgments(self) -> list[str]:
        """Get all judgment CELEX IDs from the judgments directory."""
        judgments: list[str] = []

        if not self.judgments_dir.exists():
            self.logger.error(f"Judgments directory not found: {self.judgments_dir}")
            return judgments

        for item in self.judgments_dir.iterdir():
            if item.is_dir() and "CJ" in item.name:
                judgments.append(item.name)

        judgments.sort()
        self.logger.info(f"Found {len(judgments)} judgments")
        return judgments

    def parse_single_judgment(self, celex_id: str) -> JudgmentData:
        """Parse a single judgment and return structured data."""
        judgment_data = JudgmentData(celex_id)

        try:
            # Extract paragraphs using the existing parser
            paragraphs = self.parser.extract_paragraphs_from_celex(celex_id)
            judgment_data.paragraphs = paragraphs
            self.stats["total_paragraphs"] += len(paragraphs)
            self.stats["total_text_length"] += sum(
                len(text) for text in paragraphs.values()
            )
            self.stats["successful"] += 1

        except Exception as e:
            self.logger.error(f"Error parsing {celex_id}: {e}")
            self.stats["failed"] += 1

        return judgment_data

    def parse_judgments_batch(
        self, celex_ids: list[str], max_workers: int | None = None
    ) -> list[JudgmentData]:
        """Parse multiple judgments in parallel."""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers max

        self.logger.info(
            f"Starting batch parsing of {len(celex_ids)} judgments with {max_workers} workers"
        )

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_celex = {
                executor.submit(self.parse_single_judgment, celex_id): celex_id
                for celex_id in celex_ids
            }

            # Process completed tasks
            for future in as_completed(future_to_celex):
                celex_id = future_to_celex[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update statistics
                    self.stats["total_processed"] += 1
                    self.stats["total_paragraphs"] += len(result.paragraphs)
                    self.stats["total_text_length"] += sum(
                        len(text) for text in result.paragraphs.values()
                    )

                    # Log progress
                    if self.stats["total_processed"] % 100 == 0:
                        self.logger.info(
                            f"Processed {self.stats['total_processed']}/{len(celex_ids)} judgments"
                        )

                except Exception as e:
                    self.logger.error(f"Error processing {celex_id}: {e}")
                    self.stats["failed"] += 1

        return results

    def _save_json(self, results: list[JudgmentData]) -> None:
        """Save results as JSON files."""
        complete_data = [result.to_dict() for result in results]
        complete_file = self.output_dir / "judgments.json"

        with open(complete_file, "w", encoding="utf-8") as f:
            json.dump(complete_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved complete dataset to {complete_file}")

    def run(self) -> None:
        """Run the complete parsing process."""
        self.logger.info("Starting comprehensive CJ judgment parsing")

        judgments = self.get_judgments()

        if not judgments:
            self.logger.error("No judgments found")
            return

        results = self.parse_judgments_batch(judgments)

        self._save_json(results)

        self.logger.info("Parsing completed!")
        self.logger.info(f"Total processed: {self.stats['total_processed']}")


def main():
    """Main function to run the judgment parser."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse all CJ judgments")
    parser.add_argument(
        "--judgments-dir", default="judgments", help="Path to judgments directory"
    )
    parser.add_argument("--output-dir", default="data", help="Path to output directory")

    args = parser.parse_args()

    # Create parser and run
    judgment_parser = ComprehensiveJudgmentParser(
        judgments_dir=args.judgments_dir, output_dir=args.output_dir
    )

    judgment_parser.run()


if __name__ == "__main__":
    main()
