import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

INPUT_FILE = Path("私聊_王天予.json")
OUTPUT_DIR = Path("prepared_data")

TARGET_SPEAKER = "王天予"
TARGET_ALIASES = {"王天予", "王天予.", "王天予（手机）", "王天予(手机)"}

MAX_GAP_SECONDS = 6 * 3600
RAG_MAX_CHUNK_MESSAGES = 10
RAG_MAX_CHUNK_CHARS = 1600


def normalize_text(text: str) -> str:
    # 保留语气特征（例如“...”“[流泪]”），只做最小清洗
    text = str(text).replace("\u200b", "").strip()
    text = re.sub(r"[ \t]+", " ", text)
    return text


def parse_timestamp(msg: Dict[str, Any]) -> Dict[str, Any]:
    # 优先用 createTime（unix秒）
    ts = msg.get("createTime")
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(int(ts))
        return {
            "timestamp_unix": int(ts),
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
        }

    # 兜底用 formattedTime
    ft = str(msg.get("formattedTime", "")).strip()
    if ft:
        try:
            dt = datetime.strptime(ft, "%Y-%m-%d %H:%M:%S")
            return {
                "timestamp_unix": int(dt.timestamp()),
                "timestamp": ft,
            }
        except ValueError:
            pass

    return {"timestamp_unix": None, "timestamp": ""}


def resolve_role(msg: Dict[str, Any]) -> str:
    # isSend: 1=我发出，0=对方发出
    return "user" if int(msg.get("isSend", 0)) == 1 else "other"


def resolve_speaker(msg: Dict[str, Any]) -> str:
    name = str(msg.get("senderDisplayName", "")).strip()
    username = str(msg.get("senderUsername", "")).strip()

    if name in TARGET_ALIASES:
        return TARGET_SPEAKER

    # 如果显示名缺失，兜底用户名
    return name or username or "unknown"


def is_valid_text_message(msg: Dict[str, Any]) -> bool:
    # 过滤系统消息，保留文本消息
    if str(msg.get("type", "")).strip() != "文本消息":
        return False
    content = normalize_text(msg.get("content", ""))
    return bool(content)


def load_raw_messages(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    # 兼容两种结构：
    # 1) {"messages":[...]}
    # 2) [...]
    if isinstance(raw, dict):
        messages = raw.get("messages", [])
    elif isinstance(raw, list):
        messages = raw
    else:
        raise ValueError("raw_chat.json 格式不正确，应为对象或数组")

    if not isinstance(messages, list):
        raise ValueError("messages 字段必须是数组")

    return messages


def normalize_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    idx = 1

    for i, msg in enumerate(raw_messages):
        if not isinstance(msg, dict):
            continue
        if not is_valid_text_message(msg):
            continue

        ts = parse_timestamp(msg)
        speaker = resolve_speaker(msg)
        content = normalize_text(msg.get("content", ""))

        row = {
            "message_id": f"msg_{idx:08d}",
            "raw_index": i,
            "local_id": msg.get("localId"),
            "platform_message_id": msg.get("platformMessageId"),
            "timestamp_unix": ts["timestamp_unix"],
            "timestamp": ts["timestamp"],
            "speaker": speaker,
            "role": resolve_role(msg),
            "is_target": speaker == TARGET_SPEAKER,
            "sender_username": str(msg.get("senderUsername", "")).strip(),
            "sender_display_name": str(msg.get("senderDisplayName", "")).strip(),
            "content": content,
            "source_file": str(INPUT_FILE),
        }
        rows.append(row)
        idx += 1

    # 按时间排序（如果有）
    rows.sort(key=lambda x: (x["timestamp_unix"] is None, x["timestamp_unix"] or 0, x["raw_index"]))
    return rows


def segment_messages(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = []
    if not rows:
        return segments

    seg_idx = 1
    current = [rows[0]]

    for row in rows[1:]:
        prev = current[-1]
        gap = None
        if prev["timestamp_unix"] is not None and row["timestamp_unix"] is not None:
            gap = row["timestamp_unix"] - prev["timestamp_unix"]

        # 时间间隔过大则切段
        split = gap is not None and gap > MAX_GAP_SECONDS
        if split:
            segments.append(build_segment(current, seg_idx))
            seg_idx += 1
            current = [row]
        else:
            current.append(row)

    if current:
        segments.append(build_segment(current, seg_idx))

    return segments


def build_segment(segment_rows: List[Dict[str, Any]], seg_idx: int) -> Dict[str, Any]:
    speakers = sorted({r["speaker"] for r in segment_rows})
    target_count = sum(1 for r in segment_rows if r["is_target"])
    user_count = sum(1 for r in segment_rows if r["role"] == "user")

    return {
        "segment_id": f"seg_{seg_idx:06d}",
        "start_message_id": segment_rows[0]["message_id"],
        "end_message_id": segment_rows[-1]["message_id"],
        "start_time": segment_rows[0]["timestamp"],
        "end_time": segment_rows[-1]["timestamp"],
        "message_count": len(segment_rows),
        "speakers": speakers,
        "target_message_count": target_count,
        "user_message_count": user_count,
        "contains_target": target_count > 0,
        "message_ids": [r["message_id"] for r in segment_rows],
    }


def build_message_to_segment_map(segments: List[Dict[str, Any]]) -> Dict[str, str]:
    m = {}
    for seg in segments:
        for mid in seg["message_ids"]:
            m[mid] = seg["segment_id"]
    return m


def annotate_messages(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 轻量规则标注（后续可继续加）
    annotated = []
    for r in rows:
        text = r["content"]
        labels = {
            "dialogue_act": [],
            "emotion_hint": [],
            "intensity": "low",
        }

        if "?" in text or "？" in text:
            labels["dialogue_act"].append("question")
        if any(k in text for k in ["怎么办", "好累", "累", "难受"]):
            labels["emotion_hint"].append("stress_or_distress")
            labels["intensity"] = "medium"
        if any(k in text for k in ["哈哈", "太高兴", "开心"]):
            labels["emotion_hint"].append("positive")
            labels["intensity"] = "medium"
        if re.search(r"\[.*?\]", text):
            labels["dialogue_act"].append("emoji_text")

        annotated.append({
            "message_id": r["message_id"],
            "speaker": r["speaker"],
            "is_target": r["is_target"],
            "content": r["content"],
            "labels": labels
        })
    return annotated


def build_persona_evidence_candidates(
    rows: List[Dict[str, Any]],
    msg2seg: Dict[str, str]
) -> List[Dict[str, Any]]:
    out = []
    idx = 1
    for r in rows:
        if not r["is_target"]:
            continue

        claim = None
        text = r["content"]

        if re.search(r"\[.*?\]", text):
            claim = "使用表情文本进行情绪表达"
        elif len(text) <= 4:
            claim = "偏短句回应"
        elif any(k in text for k in ["太高兴", "开心"]):
            claim = "会直接表达正向情绪"
        elif any(k in text for k in ["怎么办", "累", "难受"]):
            claim = "在压力话题中关注情绪"

        if claim:
            out.append({
                "id": f"pe_{idx:08d}",
                "category": "language_or_emotion_style",
                "claim": claim,
                "evidence_text": text,
                "speaker": r["speaker"],
                "message_id": r["message_id"],
                "segment_id": msg2seg.get(r["message_id"], ""),
                "timestamp": r["timestamp"],
                "confidence": 0.72
            })
            idx += 1
    return out


def build_memory_candidates(
    rows: List[Dict[str, Any]],
    msg2seg: Dict[str, str]
) -> List[Dict[str, Any]]:
    out = []
    idx = 1

    # 简单关键词抽记忆候选（后续可升级为更强抽取器）
    topic_keywords = {
        "study_or_homework": ["作业", "模拟法庭", "cs"],
        "fatigue_or_stress": ["好累", "累", "怎么办", "忘记了"]
    }

    for topic, kws in topic_keywords.items():
        refs = []
        for r in rows:
            if any(k in r["content"] for k in kws):
                refs.append(r["message_id"])

        if refs:
            out.append({
                "id": f"mc_{idx:08d}",
                "type": "topic_pattern",
                "subject": "dialogue_context",
                "content": f"对话中反复出现主题: {topic}",
                "message_refs": refs[:50],
                "segment_refs": sorted({msg2seg.get(mid, "") for mid in refs if msg2seg.get(mid, "")}),
                "stability": "medium",
                "confidence": 0.68
            })
            idx += 1

    return out


def build_rag_chunks(rows: List[Dict[str, Any]], msg2seg: Dict[str, str]) -> List[Dict[str, Any]]:
    chunks = []
    chunk_rows = []
    chunk_chars = 0
    chunk_idx = 1

    def flush():
        nonlocal chunk_rows, chunk_chars, chunk_idx
        if not chunk_rows:
            return
        text_lines = [f"{r['speaker']}: {r['content']}" for r in chunk_rows]
        chunks.append({
            "chunk_id": f"chunk_{chunk_idx:08d}",
            "start_time": chunk_rows[0]["timestamp"],
            "end_time": chunk_rows[-1]["timestamp"],
            "message_ids": [r["message_id"] for r in chunk_rows],
            "segment_ids": sorted({msg2seg.get(r["message_id"], "") for r in chunk_rows}),
            "contains_target": any(r["is_target"] for r in chunk_rows),
            "speakers": sorted({r["speaker"] for r in chunk_rows}),
            "text": "\n".join(text_lines),
            "message_count": len(chunk_rows),
        })
        chunk_idx += 1
        chunk_rows = []
        chunk_chars = 0

    for r in rows:
        line = f"{r['speaker']}: {r['content']}"
        if chunk_rows and (
            len(chunk_rows) >= RAG_MAX_CHUNK_MESSAGES
            or chunk_chars + len(line) > RAG_MAX_CHUNK_CHARS
        ):
            flush()
        chunk_rows.append(r)
        chunk_chars += len(line)

    flush()
    return chunks


def build_target_only_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    idx = 1
    for r in rows:
        if not r["is_target"]:
            continue
        out.append({
            "id": f"target_msg_{idx:08d}",
            "message_id": r["message_id"],
            "timestamp": r["timestamp"],
            "speaker": TARGET_SPEAKER,
            "content": r["content"]
        })
        idx += 1
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    raw_messages = load_raw_messages(INPUT_FILE)
    normalized = normalize_messages(raw_messages)
    segments = segment_messages(normalized)
    msg2seg = build_message_to_segment_map(segments)

    annotated = annotate_messages(normalized)
    persona_candidates = build_persona_evidence_candidates(normalized, msg2seg)
    memory_candidates = build_memory_candidates(normalized, msg2seg)
    rag_chunks = build_rag_chunks(normalized, msg2seg)
    target_examples = build_target_only_examples(normalized)

    write_jsonl(OUTPUT_DIR / "normalized_messages.jsonl", normalized)
    write_jsonl(OUTPUT_DIR / "clean_messages.jsonl", normalized)  # 当前阶段与 normalized 一致
    write_jsonl(OUTPUT_DIR / "conversation_segments.jsonl", segments)
    write_jsonl(OUTPUT_DIR / "annotated_messages.jsonl", annotated)
    write_jsonl(OUTPUT_DIR / "persona_evidence_candidates.jsonl", persona_candidates)
    write_jsonl(OUTPUT_DIR / "memory_candidates.jsonl", memory_candidates)
    write_jsonl(OUTPUT_DIR / "rag_chunks.jsonl", rag_chunks)
    write_jsonl(OUTPUT_DIR / "target_only_examples.jsonl", target_examples)

    stats = {
        "input_file": str(INPUT_FILE),
        "raw_message_count": len(raw_messages),
        "normalized_message_count": len(normalized),
        "segment_count": len(segments),
        "rag_chunk_count": len(rag_chunks),
        "target_speaker": TARGET_SPEAKER,
        "target_message_count": sum(1 for r in normalized if r["is_target"]),
        "persona_evidence_candidate_count": len(persona_candidates),
        "memory_candidate_count": len(memory_candidates),
    }
    write_json(OUTPUT_DIR / "stats_report.json", stats)

    print("Done. Files written to prepared_data/")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
