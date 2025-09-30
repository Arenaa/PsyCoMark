import json
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple


def compute_basic_percentiles(values: List[int]) -> Tuple[int, int, int, int, int]:
    if not values:
        return 0, 0, 0, 0, 0
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> int:
        if n == 1:
            return int(s[0])
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        if f == c:
            return int(s[f])
        return int(s[f] + (s[c] - s[f]) * (k - f))

    min_v = int(s[0])
    p25 = pct(0.25)
    median = pct(0.5)
    p75 = pct(0.75)
    max_v = int(s[-1])
    return min_v, p25, median, p75, max_v


def run_eda(input_path: str) -> Dict[str, Any]:
    num_rows = 0
    label_counts: Counter = Counter()
    subreddit_counts: Counter = Counter()
    marker_type_counts: Counter = Counter()
    text_lengths: List[int] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            num_rows += 1

            label = obj.get("conspiracy")
            if label is not None:
                label_counts[label] += 1

            subreddit = obj.get("subreddit")
            if subreddit is not None:
                subreddit_counts[subreddit] += 1

            text = obj.get("text") or ""
            text_lengths.append(len(text))

            for m in obj.get("markers", []) or []:
                t = m.get("type")
                if t is not None:
                    marker_type_counts[t] += 1

    min_v, p25, median, p75, max_v = compute_basic_percentiles(text_lengths)

    summary: Dict[str, Any] = {
        "rows": num_rows,
        "labels": dict(label_counts),
        "top_subreddits": subreddit_counts.most_common(15),
        "marker_types": dict(marker_type_counts),
        "text_length_chars": {
            "min": min_v,
            "p25": p25,
            "median": median,
            "p75": p75,
            "max": max_v,
        },
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick EDA for PsyCoMark JSONL dataset")
    parser.add_argument(
        "--input",
        default="../data/train_rehydrated.jsonl",
        help="Path to input JSONL file (default: train_rehydrated.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="eda_summary.json",
        help="Where to save the EDA summary JSON (default: eda_summary.json)",
    )
    args = parser.parse_args()

    summary = run_eda(args.input)

    print("Rows:", summary["rows"])  # noqa: T201
    print("Labels:", summary["labels"])  # noqa: T201
    print("Top subreddits (15):", summary["top_subreddits"])  # noqa: T201
    print("Marker types:", summary["marker_types"])  # noqa: T201
    print("Text length (chars):", summary["text_length_chars"])  # noqa: T201

    with open(args.output, "w", encoding="utf-8") as w:
        json.dump(summary, w, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
