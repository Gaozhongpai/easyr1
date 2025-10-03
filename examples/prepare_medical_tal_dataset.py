"""Utility to convert Medical VideoChat TAL JSON dumps into EasyR1 RLHF format.

This script reads the JSON exports used by the original TRL-based pipeline
(`tal_train_data.json`, `tal_test_data.json`, …) and materialises them as
JSONL splits that EasyR1 can load via `data.train_files=/path/train.jsonl`.

Each emitted record contains:
  - `problem`: the conversational prompt with the `<video>` placeholder
  - `answer`: the original ground-truth answer string
  - `videos`: list of frame paths that describe the clip
  - rich metadata (qa_type, spans, dense segments, …) needed by the reward fn

Paths to frame folders can be rewritten on the fly to match the local layout
by supplying `--source-prefix` and `--target-prefix`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


def _resolve_paths(
    raw_list: Iterable[str] | None,
    source_prefix: str | None,
    target_prefix: str | None,
) -> list[str]:
    if not raw_list:
        return []

    resolved: list[str] = []
    for path in raw_list:
        if source_prefix and target_prefix and path.startswith(source_prefix):
            resolved.append(target_prefix + path[len(source_prefix) :])
        else:
            resolved.append(path)

    return resolved


def _ensure_serialisable(obj: Any) -> Any:
    """Recursively coerce values so `json.dump` never fails."""

    if isinstance(obj, dict):
        return {key: _ensure_serialisable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_ensure_serialisable(value) for value in obj]
    if isinstance(obj, (str, int, float)) or obj is None:
        return obj
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, tuple):
        return [_ensure_serialisable(value) for value in obj]
    # fall back to string representation for exotic objects
    return str(obj)


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_span(span: dict[str, Any]) -> dict[str, float]:
    start = _to_float(span.get("start"))
    end = _to_float(span.get("end"))
    if start is None or end is None:
        raise ValueError("Invalid span without numeric start/end")
    return {"start": start, "end": end}


def _sanitize_stg_entry(entry: dict[str, Any]) -> dict[str, Any]:
    bbox_dict = {}
    for key, value in (entry.get("bbox_dict") or {}).items():
        if not isinstance(value, (list, tuple)):
            continue
        bbox_dict[str(key)] = [int(val) for val in value]

    sanitized = {
        "object": entry.get("object"),
        "start": _to_float(entry.get("start")),
        "end": _to_float(entry.get("end")),
        "stride": int(entry.get("stride")) if entry.get("stride") is not None else None,
        "bbox_dict": bbox_dict,
    }
    video_input_id = entry.get("video_input_id")
    if isinstance(video_input_id, str):
        sanitized["video_input_id"] = video_input_id
    return sanitized


def _sanitize_cvs_scores(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    sanitized: dict[str, Any] = {}
    for key in ["two_structures", "cystic_plate", "hepatocystic_triangle", "total"]:
        value = _to_float(data.get(key))
        if value is not None:
            sanitized[key] = value
    flag = _to_bool(data.get("critical_view_achieved"))
    if flag is not None:
        sanitized["critical_view_achieved"] = flag
    return sanitized if sanitized else None


def _sanitize_skill_scores(data: Any) -> dict[str, float] | None:
    if not isinstance(data, dict):
        return None
    sanitized: dict[str, float] = {}
    for key, value in data.items():
        numeric = _to_float(value)
        if numeric is not None:
            sanitized[str(key)] = numeric
    return sanitized if sanitized else None


def _load_split(
    json_path: Path,
    source_prefix: str | None,
    target_prefix: str | None,
    dataset_name: str,
    shuffle: bool,
) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    examples: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_data):
        conversations = item.get("conversations", [])
        if len(conversations) < 2:
            continue  # malformed sample

        prompt = conversations[0].get("value", "").strip()
        answer = conversations[1].get("value", "").strip()

        # Normalise metadata and auxiliary fields.
        raw_metadata = item.get("metadata", {}) or {}
        metadata = {}
        if isinstance(raw_metadata, dict):
            video_id = raw_metadata.get("video_id")
            if isinstance(video_id, str):
                metadata["video_id"] = video_id

            fps_value = _to_float(raw_metadata.get("fps"))
            metadata["fps"] = fps_value

        qa_type = item.get("qa_type", "tal")
        raw_struc = item.get("struc_info", [])
        struc_info: list[dict[str, Any]] = []
        stg_tracks: list[dict[str, Any]] = []

        def _sanitize_struc_entry(entry: dict[str, Any]) -> dict[str, Any]:
            sanitized: dict[str, Any] = {}
            action = entry.get("action")
            if isinstance(action, str):
                sanitized["action"] = action

            spans_list = []
            for span in entry.get("spans", []) or []:
                try:
                    spans_list.append(_normalise_span(span))
                except ValueError:
                    continue
            if spans_list:
                sanitized["spans"] = spans_list

            phase = entry.get("phase")
            if isinstance(phase, str):
                sanitized["phase"] = phase

            phase_list = entry.get("phase_list")
            if isinstance(phase_list, list):
                sanitized["phase_list"] = [str(val) for val in phase_list]

            if not sanitized:
                return {}
            return sanitized

        if isinstance(raw_struc, list):
            for entry in raw_struc:
                if not isinstance(entry, dict):
                    continue
                if "bbox_dict" in entry:
                    stg_tracks.append(_sanitize_stg_entry(entry))
                    continue
                sanitized_entry = _sanitize_struc_entry(entry)
                if sanitized_entry:
                    struc_info.append(sanitized_entry)
        elif isinstance(raw_struc, dict):
            if "bbox_dict" in raw_struc:
                stg_tracks.append(_sanitize_stg_entry(raw_struc))
            else:
                sanitized_entry = _sanitize_struc_entry(raw_struc)
                if sanitized_entry:
                    struc_info.append(sanitized_entry)

        video_frames = _resolve_paths(item.get("video"), source_prefix, target_prefix)

        if len(video_frames) < 2:
            continue

        example: dict[str, Any] = {
            "uid": item.get("id", f"{dataset_name}-{idx}"),
            "problem": prompt,
            "answer": answer,
            "videos": [video_frames],
            "qa_type": qa_type,
            "metadata": metadata,
            "data_source": item.get("data_source"),
            "struc_info": struc_info,
        }

        # Task-specific payloads reused by the reward function.
        if qa_type == "tal":
            solution_spans = []
            for block in item.get("struc_info", []) or []:
                for span in block.get("spans", []) or []:
                    try:
                        solution_spans.append(_normalise_span(span))
                    except ValueError:
                        continue
            example["solution"] = solution_spans
        elif qa_type.startswith("dense_captioning"):
            segments: list[dict[str, Any]] = []
            for block in item.get("struc_info", []) or []:
                if isinstance(block, dict) and "spans" in block:
                    for span in block.get("spans", []) or []:
                        start = _to_float(span.get("start"))
                        end = _to_float(span.get("end"))
                        if start is None or end is None:
                            continue
                        segments.append(
                            {
                                "start": start,
                                "end": end,
                                "caption": span.get("caption", ""),
                                "action": block.get("action"),
                            }
                        )
                elif isinstance(block, list):
                    for seg in block:
                        start = _to_float(seg.get("start"))
                        end = _to_float(seg.get("end"))
                        if start is None or end is None:
                            continue
                        segments.append(
                            {
                                "start": start,
                                "end": end,
                                "caption": seg.get("caption", ""),
                                "action": seg.get("action"),
                            }
                        )
            example["dense_segments"] = json.dumps(segments, ensure_ascii=False)
            dense_solution = []
            for seg in segments:
                try:
                    dense_solution.append(_normalise_span(seg))
                except ValueError:
                    continue
            example["solution"] = dense_solution
        elif qa_type.startswith("video_summary"):
            example["summary_gt"] = answer
        elif qa_type == "next_action":
            next_action = None
            for info in item.get("struc_info", []) or []:
                next_action = info.get("next_action")
                if next_action:
                    break
            example["next_action"] = next_action
        elif qa_type == "cvs_assessment":
            cvs_scores = None
            for info in item.get("struc_info", []) or []:
                cvs_scores = _sanitize_cvs_scores(info.get("cvs_scores"))
                if cvs_scores:
                    break
            example["cvs_scores"] = json.dumps(cvs_scores, ensure_ascii=False) if cvs_scores else "null"
        elif qa_type == "skill_assessment":
            skill_scores = None
            for info in item.get("struc_info", []) or []:
                skill_scores = _sanitize_skill_scores(info.get("skill_scores"))
                if skill_scores:
                    break
            example["skill_scores"] = json.dumps(skill_scores, ensure_ascii=False) if skill_scores else "null"
        elif qa_type.startswith("region_caption"):
            region_entry = None
            question_text = None
            info_list = item.get("struc_info", []) or []
            if info_list:
                raw_region = info_list[0].get("struc_info")
                if isinstance(raw_region, dict):
                    region_entry = {
                        "start": _to_float(raw_region.get("start")),
                        "end": _to_float(raw_region.get("end")),
                        "object": raw_region.get("object"),
                    }
                question_text = info_list[0].get("question")
            example["region_info"] = json.dumps(region_entry, ensure_ascii=False) if region_entry else "null"
            example["region_question"] = question_text or ""
        elif qa_type == "stg":
            example["stg_tracks"] = json.dumps(stg_tracks, ensure_ascii=False)
            example["struc_info"] = []

        example.setdefault("solution", [])
        example.setdefault("dense_segments", "[]")
        example.setdefault("next_action", "")
        example.setdefault("cvs_scores", "null")
        example.setdefault("skill_scores", "null")
        example.setdefault("summary_gt", "")
        example.setdefault("stg_tracks", "[]")
        example.setdefault("region_info", "null")
        example.setdefault("region_question", "")

        examples.append(_ensure_serialisable(example))

    if shuffle:
        random.shuffle(examples)

    return examples


def _write_jsonl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True, help="Path to TAL training JSON file.")
    parser.add_argument("--eval-json", type=Path, required=True, help="Path to TAL evaluation JSON file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store the EasyR1-formatted splits (JSONL).",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default=None,
        help="If provided, strip this prefix from frame paths before writing output.",
    )
    parser.add_argument(
        "--target-prefix",
        type=str,
        default=None,
        help="If provided, prepend this prefix to frame paths (after removing source-prefix).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="medical_tal",
        help="Identifier stored in the `uid` field for bookkeeping.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable per-split shuffling before writing JSONL files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = _load_split(
        json_path=args.train_json,
        source_prefix=args.source_prefix,
        target_prefix=args.target_prefix,
        dataset_name=f"{args.dataset_name}-train",
        shuffle=not args.no_shuffle,
    )

    eval_examples = _load_split(
        json_path=args.eval_json,
        source_prefix=args.source_prefix,
        target_prefix=args.target_prefix,
        dataset_name=f"{args.dataset_name}-eval",
        shuffle=not args.no_shuffle,
    )

    _write_jsonl(output_dir / "train.jsonl", train_examples)
    _write_jsonl(output_dir / "validation.jsonl", eval_examples)

    manifest = {
        "train_file": str((output_dir / "train.jsonl").resolve()),
        "validation_file": str((output_dir / "validation.jsonl").resolve()),
        "num_train_examples": len(train_examples),
        "num_validation_examples": len(eval_examples),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Wrote EasyR1 splits:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
