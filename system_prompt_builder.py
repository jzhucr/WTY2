#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_DATA_DIR = Path("/Users/georgezhu/Documents/wty/prepared_data")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}", text.lower())


def _keyword_overlap_score(query: str, text: str) -> int:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0
    t_tokens = set(_tokenize(text))
    return len(q_tokens & t_tokens)


def _infer_user_state(user_message: str) -> List[str]:
    msg = user_message.lower()
    states = []
    if any(k in msg for k in ["累", "烦", "崩", "压力", "顶不住", "焦虑", "难受", "烦死", "怎么办"]):
        states.append("stressed")
    if any(k in msg for k in ["不知道", "怎么", "咋", "不会", "搞不懂", "confused", "why"]):
        states.append("confused")
    if any(k in msg for k in ["你还在吗", "你是不是", "你会不会", "想你", "在乎", "关系", "生气", "冷淡"]):
        states.append("seeking_closeness")
    if any(k in msg for k in ["误会", "吵", "争", "生气", "不舒服", "别这样"]):
        states.append("tense_or_conflict")
    return states


def _infer_topic(user_message: str) -> List[str]:
    msg = user_message.lower()
    topics = []
    if any(k in msg for k in ["作业", "考试", "ddl", "论文", "课程", "学习", "cs", "workload"]):
        topics.append("study")
    if any(k in msg for k in ["高铁", "飞机", "上海", "南通", "疫情", "核酸", "出发"]):
        topics.append("logistics")
    if any(k in msg for k in ["ipad", "电脑", "手机", "系统", "bug", "功能", "软件", "app"]):
        topics.append("tech_or_logistics")
    if any(k in msg for k in ["关系", "想你", "喜欢", "在乎", "分手", "和好", "吃醋"]):
        topics.append("relationship_check")
    if any(k in msg for k in ["压力", "崩", "累", "难受"]):
        topics.append("workload")
    return topics


def _select_memory_items(memory: List[Dict[str, Any]], user_message: str, top_k: int = 8) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, float, Dict[str, Any]]] = []
    for item in memory:
        content = str(item.get("content", ""))
        score = _keyword_overlap_score(user_message, content)
        conf = float(item.get("confidence", 0.0))
        # Even with no keyword overlap, keep high-confidence relationship memories as fallback.
        if score == 0 and item.get("type") not in {"relationship", "interaction_style", "trigger_pattern"}:
            continue
        scored.append((score, conf, item))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [x[2] for x in scored[:top_k]]


def _rule_matches(rule: Dict[str, Any], user_states: List[str], topics: List[str]) -> bool:
    trigger = rule.get("trigger", {})
    need_states = set(trigger.get("user_state", []))
    need_topics = set(trigger.get("topic", []))

    # If rule has declared states/topics, match on intersection.
    state_ok = True if not need_states else bool(need_states & set(user_states))
    topic_ok = True if not need_topics else bool(need_topics & set(topics))
    return state_ok and topic_ok


def _select_behavior_rules(behavior_rules: Dict[str, Any], user_message: str, top_k: int = 2) -> List[Dict[str, Any]]:
    user_states = _infer_user_state(user_message)
    topics = _infer_topic(user_message)
    rules = behavior_rules.get("rules", [])
    matched = [r for r in rules if _rule_matches(r, user_states, topics)]
    if matched:
        matched.sort(key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
        return matched[:top_k]

    # Fallback: choose strongest rules globally.
    ranked = sorted(rules, key=lambda r: float(r.get("confidence", 0.0)), reverse=True)
    return ranked[:top_k]


def _format_identity(persona_core: Dict[str, Any]) -> str:
    subject = persona_core.get("subject", {})
    identity = persona_core.get("identity", {})
    language_style = persona_core.get("language_style", {})
    values = persona_core.get("values", {})
    emotional_model = persona_core.get("emotional_model", {})
    runtime_guidance = persona_core.get("runtime_guidance", {})

    lines = []
    lines.append(f"你是 {subject.get('name', '目标角色')}。")
    lines.append(f"整体印象：{identity.get('self_presentation', {}).get('overall_impression', '')}")
    lines.append(f"基础语气：{', '.join(language_style.get('tone_profile', {}).get('baseline_tone', []))}")
    lines.append(f"表达节奏：{language_style.get('rhythm', {}).get('response_pacing', '')}；{language_style.get('rhythm', {}).get('topic_transition_style', '')}")
    lines.append(f"情绪姿态：{emotional_model.get('baseline_emotional_posture', '')}")
    lines.append(f"核心价值：{', '.join(values.get('core_values', []))}")
    lines.append(f"偏好回复长度：{runtime_guidance.get('response_length_preference', 'short_to_medium')}")
    if runtime_guidance.get("forbidden_styles"):
        lines.append(f"禁止风格：{', '.join(runtime_guidance['forbidden_styles'])}")
    return "\n".join(lines).strip()


def _format_relationship(relationship_model: Dict[str, Any]) -> str:
    frame = relationship_model.get("relationship_frame", {})
    patterns = relationship_model.get("interaction_patterns", {})
    emotional = relationship_model.get("emotional_dynamics", {})
    trust = relationship_model.get("trust_model", {})

    lines = [
        f"关系框架：{frame.get('current_interaction_type', '')}",
        f"关系稳定性：{frame.get('stability', '')}",
        f"用户常见需求：{', '.join(patterns.get('user_typical_needs', []))}",
        f"你的典型角色：{', '.join(patterns.get('target_typical_role', []))}",
        f"常见修复路径：{', '.join(patterns.get('repair_loops', []))}",
        f"安全感构建：{', '.join(trust.get('safety_builders', []))}",
        f"敏感区域：{', '.join(emotional.get('sensitivity_areas', []))}",
    ]
    return "\n".join(lines).strip()


def _format_memories(memory_items: List[Dict[str, Any]]) -> str:
    if not memory_items:
        return "无高相关长期记忆，按当前上下文谨慎回应。"
    lines = []
    for m in memory_items:
        lines.append(f"- [{m.get('type', 'memory')}] {m.get('content', '')} (conf={m.get('confidence', 0)})")
    return "\n".join(lines)


def _format_rules(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "无命中规则，按人设默认策略：先回应情绪，再给可执行建议。"
    lines = []
    for r in rules:
        strategy = r.get("response_strategy", {})
        lines.append(
            "- "
            + " -> ".join(strategy.get("sequence", []))
            + f" | tone={','.join(strategy.get('tone', []))}"
        )
    return "\n".join(lines)


def build_system_prompt(
    user_message: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    memory_top_k: int = 8,
    rules_top_k: int = 2,
) -> str:
    persona_core = _load_json(data_dir / "persona_core.json")
    relationship_model = _load_json(data_dir / "relationship_model.json")
    behavior_rules = _load_json(data_dir / "behavior_rules.json")
    memory = _load_json(data_dir / "memory.json")

    selected_memories = _select_memory_items(memory, user_message, top_k=memory_top_k)
    selected_rules = _select_behavior_rules(behavior_rules, user_message, top_k=rules_top_k)

    prompt = f"""[角色核心]
{_format_identity(persona_core)}

[关系上下文]
{_format_relationship(relationship_model)}

[长期记忆（检索）]
{_format_memories(selected_memories)}

[行为策略（本轮）]
{_format_rules(selected_rules)}

[执行约束]
1. 优先保持王天予风格：自然、口语化、短句优先。
2. 先回应用户情绪/诉求，再给一个可执行的下一步。
3. 不编造未发生事实，不伪装全知。
4. 在高压话题中避免说教和过度分析。
5. 回复长度以 short_to_medium 为主，必要时分两到三句发送。
"""
    return prompt.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build system prompt from persona/memory/rules.")
    parser.add_argument("--user-message", required=True, help="Latest user message")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing prepared json files")
    parser.add_argument("--memory-top-k", type=int, default=8)
    parser.add_argument("--rules-top-k", type=int, default=2)
    parser.add_argument(
        "--output-file",
        default="",
        help="Optional output file path. If omitted, print to stdout.",
    )
    args = parser.parse_args()

    prompt = build_system_prompt(
        user_message=args.user_message,
        data_dir=Path(args.data_dir),
        memory_top_k=args.memory_top_k,
        rules_top_k=args.rules_top_k,
    )

    if args.output_file:
        out = Path(args.output_file)
        out.write_text(prompt, encoding="utf-8")
        print(f"System prompt written to: {out}")
    else:
        print(prompt)


if __name__ == "__main__":
    main()
