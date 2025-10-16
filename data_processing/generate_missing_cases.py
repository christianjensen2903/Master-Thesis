import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a JSON mapping of missing EUR-Lex HTML files "
            "(CJ/FJ/TJ) by comparing data/parsed_par_pairs.jsonl with the cases directory."
        )
    )
    p.add_argument(
        "--parsed-jsonl",
        type=Path,
        default=Path("data/parsed_par_pairs.jsonl"),
        help="Path to parsed_par_pairs.jsonl",
    )
    p.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("cases"),
        help="Directory containing downloaded HTML case files",
    )
    p.add_argument(
        "--types",
        type=str,
        default="CJ,FJ,TJ",
        help="Comma-separated CELEX case type filters (e.g., 'CJ,FJ,TJ' or 'CJ')",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/missing_cases.json"),
        help="Output JSON file path (mapping case_id -> {})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of missing cases to write (0 = no limit)",
    )
    p.add_argument(
        "--strict-exact",
        action="store_true",
        help=(
            "If set, require exact filename match (case_id.html). "
            "By default, a variant file like case_id(01).html also counts as present."
        ),
    )
    return p.parse_args()


def normalize_base_id(case_id: str) -> str:
    base = case_id
    if "(" in base:
        base = base.split("(", 1)[0]
    return base


def load_present_cases(cases_dir: Path) -> tuple[set[str], set[str]]:
    present_exact: set[str] = set()
    present_bases: set[str] = set()
    if not cases_dir.exists():
        return present_exact, present_bases
    for p in cases_dir.glob("*.html"):
        stem = p.stem
        present_exact.add(stem)
        present_bases.add(normalize_base_id(stem))
    return present_exact, present_bases


def case_is_present(
    case_id: str, present_exact: set[str], present_bases: set[str], strict_exact: bool
) -> bool:
    if case_id in present_exact:
        return True
    if strict_exact:
        return False
    return normalize_base_id(case_id) in present_bases


def main() -> None:
    args = parse_args()
    wanted_types = {t.strip().upper() for t in args.types.split(",") if t.strip()}

    present_exact, present_bases = load_present_cases(args.cases_dir)

    missing: dict[str, dict] = {}
    total_seen_by_type: dict[str, int] = {t: 0 for t in wanted_types}

    count_added = 0
    limit = args.limit if args.limit and args.limit > 0 else None

    with args.parsed_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            celex: str = obj.get("celex", "")
            if not isinstance(celex, str) or len(celex) < 5:
                continue

            case_type: str | None = None
            for t in wanted_types:
                if t in celex:
                    case_type = t
                    break
            if case_type is None:
                continue

            total_seen_by_type[case_type] = total_seen_by_type.get(case_type, 0) + 1

            if case_is_present(
                case_id=celex,
                present_exact=present_exact,
                present_bases=present_bases,
                strict_exact=args.strict_exact,
            ):
                continue

            missing[celex] = {}
            count_added += 1
            if limit is not None and count_added >= limit:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_missing = len(missing)
    summary = {
        "filters": sorted(wanted_types),
        "strict_exact": args.strict_exact,
        "seen_by_type": total_seen_by_type,
        "missing_total": total_missing,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
