#!/usr/bin/env python3
"""Summarize LIBERO eval .txt logs into CSV rows."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("/mnt/nas/zhangyiming/last05_beta/experiments/logs")
EXCLUDED_TASK_ID = 5
FILENAME_TIME_RE = re.compile(r"(\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2})")


@dataclass
class ParsedLog:
    file: str
    run_time: str
    model_path: str = ""
    decosmos: str = ""
    cosmos_denoise_steps: str = ""
    cosmos_kv_cache_step: str = ""
    num_open_loop_steps: str = ""
    task_success_counts: str = ""
    total_successes: str = ""
    successes_excluding_task_5: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, str]:
        return {
            "file": self.file,
            "run_time": self.run_time,
            "model_path": self.model_path,
            "decosmos": self.decosmos,
            "cosmos_denoise_steps": self.cosmos_denoise_steps,
            "cosmos_kv_cache_step": self.cosmos_kv_cache_step,
            "num_open_loop_steps": self.num_open_loop_steps,
            "task_success_counts": self.task_success_counts,
            "total_successes": self.total_successes,
            "successes_excluding_task_5": self.successes_excluding_task_5,
            "warnings": "; ".join(self.warnings),
        }


def parse_since(value: str) -> datetime:
    normalized = value.strip()
    formats = (
        "%Y-%m-%d",
        "%Y_%m_%d",
        "%Y_%m_%d-%H_%M_%S",
    )
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        "--since must be one of YYYY-MM-DD, YYYY_MM_DD, or YYYY_MM_DD-HH_MM_SS"
    )


def parse_filename_time(path: Path) -> Optional[datetime]:
    matches = FILENAME_TIME_RE.findall(path.name)
    if not matches:
        return None
    return datetime.strptime(matches[-1], "%Y_%m_%d-%H_%M_%S")


def format_run_time(dt: datetime) -> str:
    return dt.strftime("%Y_%m_%d-%H_%M_%S")


def clean_value(value: str) -> str:
    return value.strip().strip("'\"")


def raw_first_token(value: str) -> str:
    return clean_value(value).split()[0] if clean_value(value) else ""


def normalize_bool(value: str) -> str:
    normalized = clean_value(value).lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return "true"
    if normalized in {"0", "false", "no", "n", "off"}:
        return "false"
    return clean_value(value)


def strip_log_prefix(line: str) -> str:
    """Remove common logging prefixes while preserving the message."""
    text = line.strip()
    info_match = re.search(r"\[(?:INFO|WARNING|ERROR)\]\s*(.*)$", text)
    if info_match:
        return info_match.group(1).strip()
    return text


def parse_key_value_message(message: str) -> Optional[tuple[str, str]]:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", message)
    if not match:
        return None
    return match.group(1), clean_value(match.group(2))


def infer_task_suite_name(lines: list[str]) -> str:
    for line in lines:
        message = strip_log_prefix(line)
        parsed = parse_key_value_message(message)
        if parsed and parsed[0] == "task_suite_name":
            return parsed[1]
        match = re.search(r"Task suite:\s*(\S+)", message)
        if match:
            return clean_value(match.group(1))
    return "libero_spatial"


@lru_cache(maxsize=None)
def load_task_description_map(task_suite_name: str) -> dict[str, int]:
    """Best-effort map from task language to LIBERO task id."""
    try:
        from libero.libero import benchmark  # type: ignore

        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite_name not in benchmark_dict:
            return {}
        task_suite = benchmark_dict[task_suite_name]()
        return {
            str(task_suite.get_task(task_id).language).strip(): task_id
            for task_id in range(task_suite.n_tasks)
        }
    except Exception as exc:  # pragma: no cover - depends on optional LIBERO install.
        print(
            f"[WARN] Could not import LIBERO benchmark for task mapping: {exc}",
            file=sys.stderr,
        )
        return {}


def task_id_for_description(
    description: str,
    task_description_to_id: dict[str, int],
    fallback_order: OrderedDict[str, int],
) -> int:
    if description in task_description_to_id:
        return task_description_to_id[description]
    if description not in fallback_order:
        fallback_order[description] = len(fallback_order)
    return fallback_order[description]


def parse_log(path: Path, run_time: datetime) -> ParsedLog:
    result = ParsedLog(file=str(path), run_time=format_run_time(run_time))
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        result.warnings.append(f"read_error={exc}")
        return result

    task_suite_name = infer_task_suite_name(lines)
    task_description_to_id = load_task_description_map(task_suite_name)
    fallback_task_order: OrderedDict[str, int] = OrderedDict()
    task_successes: OrderedDict[int, int] = OrderedDict()
    current_task_description: Optional[str] = None
    saw_task = False

    info_checkpoint: Optional[str] = None
    parsed_total_successes = ""
    saw_final_results = False
    saw_overall_success_rate = False

    for line in lines:
        message = strip_log_prefix(line)

        if re.search(r"(?:^|\]\s*)Final results:\s*$", line):
            saw_final_results = True
        if re.search(r"(?:^|\]\s*)Overall success rate:\s*", line):
            saw_overall_success_rate = True

        if "[INFO] checkpoint:" in line:
            info_checkpoint = clean_value(line.split("[INFO] checkpoint:", 1)[1])

        parsed = parse_key_value_message(message)
        if parsed:
            key, value = parsed
            if key == "pretrained_checkpoint":
                result.model_path = value
            elif key == "decosmos":
                result.decosmos = normalize_bool(value)
            elif key == "cosmos_denoise_steps":
                result.cosmos_denoise_steps = raw_first_token(value)
            elif key == "cosmos_kv_cache_step":
                result.cosmos_kv_cache_step = raw_first_token(value)
            elif key == "num_open_loop_steps":
                result.num_open_loop_steps = raw_first_token(value)

        total_match = re.search(r"(?:^|\]\s*)Total successes:\s*(\d+)\b", line)
        if total_match:
            parsed_total_successes = total_match.group(1)

        task_match = re.search(r"(?:^|\]\s*)Task:\s*(.*)$", line)
        if task_match:
            current_task_description = clean_value(task_match.group(1))
            saw_task = True
            task_id = task_id_for_description(
                current_task_description,
                task_description_to_id,
                fallback_task_order,
            )
            task_successes.setdefault(task_id, 0)
            continue

        success_match = re.search(r"(?:^|\]\s*)Success:\s*(True|False|true|false|1|0)\b", line)
        if success_match and current_task_description is not None:
            task_id = task_id_for_description(
                current_task_description,
                task_description_to_id,
                fallback_task_order,
            )
            task_successes.setdefault(task_id, 0)
            if normalize_bool(success_match.group(1)) == "true":
                task_successes[task_id] += 1

    if not result.model_path and info_checkpoint:
        result.model_path = info_checkpoint

    total_successes = sum(task_successes.values())
    result.total_successes = str(total_successes if saw_task else parsed_total_successes)
    result.successes_excluding_task_5 = str(
        sum(count for task_id, count in task_successes.items() if task_id != EXCLUDED_TASK_ID)
    )
    result.task_success_counts = ";".join(
        f"{task_id}={count}" for task_id, count in sorted(task_successes.items())
    )

    required_fields = (
        "model_path",
        "decosmos",
        "cosmos_denoise_steps",
        "cosmos_kv_cache_step",
        "num_open_loop_steps",
    )
    for field_name in required_fields:
        if not getattr(result, field_name):
            result.warnings.append(f"missing_{field_name}")
    if not saw_task:
        result.warnings.append("missing_task_success_entries")
    if not (saw_final_results and saw_overall_success_rate):
        result.warnings.append("incomplete_eval")

    return result


def iter_selected_logs(log_dir: Path, since: datetime) -> list[tuple[Path, datetime]]:
    selected: list[tuple[Path, datetime]] = []
    if not log_dir.exists():
        raise FileNotFoundError(f"log dir does not exist: {log_dir}")
    for path in sorted(log_dir.glob("*.txt")):
        run_time = parse_filename_time(path)
        if run_time is None:
            print(f"[WARN] Skipping file with no filename timestamp: {path}", file=sys.stderr)
            continue
        if run_time > since:
            selected.append((path, run_time))
    selected.sort(key=lambda item: (item[1], str(item[0])))
    return selected


def write_csv(rows: list[ParsedLog], output: Optional[Path]) -> None:
    fieldnames = [
        "file",
        "run_time",
        "model_path",
        "decosmos",
        "cosmos_denoise_steps",
        "cosmos_kv_cache_step",
        "num_open_loop_steps",
        "task_success_counts",
        "total_successes",
        "successes_excluding_task_5",
        "warnings",
    ]
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_row())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize LIBERO eval .txt logs into CSV."
    )
    parser.add_argument(
        "--since",
        required=True,
        type=parse_since,
        help="Only include logs whose filename timestamp is later than this date/time. "
        "Formats: YYYY-MM-DD, YYYY_MM_DD, YYYY_MM_DD-HH_MM_SS.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing eval .txt logs. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. Defaults to stdout.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        selected_logs = iter_selected_logs(args.log_dir, args.since)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    rows = []
    for path, run_time in selected_logs:
        row = parse_log(path, run_time)
        if "incomplete_eval" in row.warnings:
            print(f"[WARN] Skipping incomplete eval log: {path}", file=sys.stderr)
            continue
        rows.append(row)
    write_csv(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
