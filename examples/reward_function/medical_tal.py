"""Rule-based reward for medical temporal action localisation tasks."""

from __future__ import annotations

import re
import json
from typing import Any


_time_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _extract_intervals(text: str) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for start, end in _time_pattern.findall(text or ""):
        try:
            s = float(start)
            e = float(end)
        except ValueError:
            continue
        if s <= e:
            intervals.append((s, e))
    return intervals


def _interval_iou(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    start = max(pred[0], gt[0])
    end = min(pred[1], gt[1])
    intersection = max(0.0, end - start)
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return intersection / union if union > 0 else 0.0


def _match_interval_sets(predicted: list[tuple[float, float]], gt: list[tuple[float, float]]) -> float:
    if not predicted or not gt:
        return 0.0

    used_gt: set[int] = set()
    total = 0.0
    for pred in predicted:
        best_iou = 0.0
        best_idx = None
        for idx, target in enumerate(gt):
            if idx in used_gt:
                continue
            iou = _interval_iou(pred, target)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None:
            used_gt.add(best_idx)
            total += best_iou
    return total / max(len(predicted), len(gt))


def _token_overlap(pred: str, ref: str) -> float:
    ref_tokens = set(re.findall(r"[a-zA-Z]+", (ref or "").lower()))
    if not ref_tokens:
        return 0.0
    pred_tokens = set(re.findall(r"[a-zA-Z]+", (pred or "").lower()))
    return len(ref_tokens & pred_tokens) / len(ref_tokens)

def _json_or_none(value: Any):
    if value in (None, "null", "NULL", "", []):
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def _dense_caption_reward(content: str, segments: list[dict[str, Any]] | None) -> float:
    if not segments:
        return 0.0

    gt_intervals = [
        (seg.get("start", 0.0), seg.get("end", 0.0))
        for seg in segments
        if seg.get("start") is not None and seg.get("end") is not None
    ]
    pred_intervals = _extract_intervals(content)
    interval_score = _match_interval_sets(pred_intervals, gt_intervals)

    caption_score = 0.0
    for seg in segments:
        caption = seg.get("caption", "")
        if not caption:
            continue
        caption_score += _token_overlap(content, caption)
    if segments:
        caption_score /= len(segments)

    return 0.7 * interval_score + 0.3 * caption_score


def _parse_tal_output(text: str) -> list[tuple[float, float]]:
    answer_blocks = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    content = answer_blocks[-1] if answer_blocks else text or ""
    matches = re.findall(r"(\d+\.?\d*)-(\d+\.?\d*)", content)
    if not matches:
        matches = re.findall(r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)", content, re.IGNORECASE)
    spans: list[tuple[float, float]] = []
    for start, end in matches:
        try:
            s = float(start)
            e = float(end)
        except ValueError:
            continue
        if s <= e:
            spans.append((s, e))
    return spans


def _tal_reward(content: str, spans: list[tuple[float, float]] | list[dict[str, Any]] | None) -> float:
    gt_spans: list[tuple[float, float]] = []
    if spans:
        for span in spans:
            if isinstance(span, dict):
                start = span.get("start")
                end = span.get("end")
            else:
                start, end = span
            if start is None or end is None:
                continue
            gt_spans.append((float(start), float(end)))

    pred_spans = _parse_tal_output(content)
    if not pred_spans:
        pred_spans = _extract_intervals(content)
    return _match_interval_sets(pred_spans, gt_spans)


def _summary_reward(content: str, reference: str | None) -> float:
    if not reference:
        return 0.0
    return _token_overlap(content, reference)


def _next_action_reward(content: str, target: str | None) -> float:
    if not target:
        return 0.0
    return 1.0 if target.lower() in (content or "").lower() else 0.0


def _cvs_reward(content: str, cvs_scores: dict | None) -> float:
    if not cvs_scores:
        return 0.0
    text = (content or "").lower()
    total = 0
    matched = 0
    for metric, value in cvs_scores.items():
        if metric in {"total", "video_input_id"}:
            continue
        total += 1
        metric_name = metric.replace("_", " ")
        token = f"{metric_name.lower()}: {str(value).lower()}"
        if token in text:
            matched += 1
    return matched / total if total else 0.0


def _skill_reward(content: str, scores: dict | None) -> float:
    if not scores:
        return 0.0
    text = (content or "").lower()
    total = 0
    matched = 0
    for metric, value in scores.items():
        total += 1
        needle = f"{metric.lower()}: {value}"
        if needle in text:
            matched += 1
    return matched / total if total else 0.0


def _region_reward(content: str, region_info: dict | None) -> float:
    if not region_info:
        return 0.0
    lower_text = (content or "").lower()
    name = region_info.get("object") if isinstance(region_info, dict) else None
    score = 0.0
    if name and name.lower() in lower_text:
        score += 0.6
    elif name:
        base = name.split("_")[0]
        if base.lower() in lower_text:
            score += 0.5
    if any(word in lower_text for word in ["move", "dissect", "retract", "grasp", "rotate", "pull", "push", "cut"]):
        score += 0.4
    return min(score, 1.0)


def _stg_reward(content: str, tracks: list | None) -> float:
    if isinstance(tracks, str):
        try:
            tracks = json.loads(tracks)
        except Exception:
            return 0.0
    if not tracks:
        return 0.0
    tokens = []
    for track in tracks:
        bbox_dict = track.get("bbox_dict", {}) if isinstance(track, dict) else {}
        for coords in bbox_dict.values():
            tokens.append(",".join(str(int(coord)) for coord in coords))
    if not tokens:
        return 0.0
    text = content or ""
    matches = sum(token in text for token in tokens)
    return matches / len(tokens)


def _format_reward(content: str, qa_type: str) -> float:
    lower = (content or "").lower()
    if qa_type == "tal":
        has_think = "<think>" in lower and "</think>" in lower
        has_answer = "<answer>" in lower and "</answer>" in lower
        has_times = bool(_extract_intervals(content))
        return (0.3 if has_think else 0.0) + (0.3 if has_answer else 0.0) + (0.4 if has_times else 0.0)
    if qa_type.startswith("dense_captioning"):
        lines = [ln for ln in (content or "").splitlines() if ln.strip()]
        time_lines = sum(1 for ln in lines if _extract_intervals(ln))
        return min(time_lines / max(len(lines), 1), 1.0)
    if qa_type.startswith("video_summary"):
        words = re.findall(r"[a-zA-Z]+", content or "")
        return min(len(words) / 80.0, 1.0)
    if qa_type == "next_action":
        tokens = ["cut", "tie", "sut", "clip"]
        return 1.0 if any(tok in lower for tok in tokens) else (0.5 if lower.strip() else 0.0)
    if qa_type == "cvs_assessment":
        required = ["two structures", "cystic plate", "hepatocystic triangle"]
        return sum(tok in lower for tok in required) / len(required)
    if qa_type == "skill_assessment":
        return min(lower.count(":"), 6) / 6.0
    if qa_type.startswith("region_caption"):
        keywords = ["object", "instrument", "tool", "hook", "grasper"]
        return 1.0 if "seconds" in lower and any(word in lower for word in keywords) else 0.0
    if qa_type == "stg":
        return 1.0 if "[" in content and "]" in content else 0.0
    return 0.0


def _task_reward(sample: dict[str, Any]) -> tuple[float, float]:
    response = sample.get("response", "")
    qa_type = sample.get("qa_type", "tal") or "tal"

    if qa_type == "tal":
        reward = _tal_reward(response, sample.get("solution") or sample.get("struc_info"))
    elif qa_type.startswith("dense_captioning"):
        reward = _dense_caption_reward(response, _json_or_none(sample.get("dense_segments")))
    elif qa_type.startswith("video_summary"):
        reward = _summary_reward(response, sample.get("summary_gt") or sample.get("ground_truth"))
    elif qa_type == "next_action":
        reward = _next_action_reward(response, sample.get("next_action"))
    elif qa_type == "cvs_assessment":
        reward = _cvs_reward(response, _json_or_none(sample.get("cvs_scores")))
    elif qa_type == "skill_assessment":
        reward = _skill_reward(response, _json_or_none(sample.get("skill_scores")))
    elif qa_type.startswith("region_caption"):
        reward = _region_reward(response, _json_or_none(sample.get("region_info")))
    elif qa_type == "stg":
        reward = _stg_reward(response, sample.get("stg_tracks"))
    else:
        reward = 0.0

    format_score = _format_reward(response, qa_type)
    return float(reward), float(format_score)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Batch reward compatible with EasyR1's `reward_type=batch` configuration."""

    if not isinstance(reward_inputs, list):
        raise ValueError("Medical TAL reward expects batched inputs; set `reward_type=batch`.")

    scores: list[dict[str, float]] = []
    for sample in reward_inputs:
        task_reward, format_reward = _task_reward(sample)
        scores.append(
            {
                "overall": task_reward + format_reward,
                "tal_task": task_reward,
                "tal_format": format_reward,
            }
        )

    return scores
