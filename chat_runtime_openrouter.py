#!/usr/bin/env python3
import html
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlencode, urlparse, parse_qs, unquote
from urllib.request import Request, urlopen

from openai import OpenAI
from dotenv import load_dotenv

from system_prompt_builder import build_system_prompt


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "prepared_data"
DEFAULT_CLEAN_CHAT_PATH = DEFAULT_DATA_DIR / "clean_messages.jsonl"
DEFAULT_SESSION_MEMORY_PATH = DEFAULT_DATA_DIR / "session_memory.json"
DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"
_CLEAN_CHAT_CACHE: Dict[str, List[Dict[str, Any]]] = {}

# Load .env from project root, but do not override already-exported env vars.
load_dotenv(dotenv_path=DEFAULT_ENV_PATH, override=False)


def _create_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{1,4}", text.lower())


def _keyword_overlap_score(query: str, text: str) -> int:
    q_tokens = set(_tokenize(query))
    if not q_tokens:
        return 0
    t_tokens = set(_tokenize(text))
    return len(q_tokens & t_tokens)


def _load_clean_chat(path: Path) -> List[Dict[str, Any]]:
    cache_key = str(path.resolve())
    if cache_key in _CLEAN_CHAT_CACHE:
        return _CLEAN_CHAT_CACHE[cache_key]

    rows: List[Dict[str, Any]] = []
    if not path.exists():
        _CLEAN_CHAT_CACHE[cache_key] = rows
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    _CLEAN_CHAT_CACHE[cache_key] = rows
    return rows


def _load_session_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"summary": "", "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"summary": "", "history": []}

    summary = str(data.get("summary", "")).strip()
    raw_history = data.get("history", [])
    history: List[Dict[str, str]] = []
    if isinstance(raw_history, list):
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                history.append({"role": role, "content": content})
    return {"summary": summary, "history": history}


def _save_session_memory(path: Path, summary: str, history: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "history": history}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _line_for_summary(item: Dict[str, str]) -> str:
    role = "用户" if item["role"] == "user" else "王天予"
    text = re.sub(r"\s+", " ", item["content"]).strip()
    if len(text) > 60:
        text = text[:57] + "..."
    return f"{role}: {text}"


def _merge_summary(previous: str, to_compress: List[Dict[str, str]], max_lines: int = 14) -> str:
    lines: List[str] = []
    if previous:
        lines.extend([x.strip() for x in previous.splitlines() if x.strip()])
    lines.extend(_line_for_summary(item) for item in to_compress)
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return "\n".join(lines)


def _compact_history(
    summary: str,
    history: List[Dict[str, str]],
    compress_trigger_turns: int,
    keep_recent_turns: int,
) -> tuple[str, List[Dict[str, str]]]:
    turn_count = len(history) // 2
    if turn_count <= compress_trigger_turns:
        return summary, history

    keep_messages = max(2, keep_recent_turns * 2)
    if len(history) <= keep_messages:
        return summary, history

    to_compress = history[:-keep_messages]
    remaining = history[-keep_messages:]
    merged_summary = _merge_summary(summary, to_compress)
    return merged_summary, remaining


def _should_search(user_message: str) -> bool:
    mode = os.getenv("WEB_SEARCH_MODE", "auto").strip().lower()
    if mode == "off":
        return False
    if mode == "always":
        return True

    msg = user_message.lower()
    hints = [
        "最新", "现在", "今天", "刚刚", "新闻", "搜", "搜索", "查一下", "帮我查",
        "联网", "网页", "官网", "价格", "汇率", "天气", "股价", "hot", "latest",
        "news", "search", "web", "current", "today", "live",
    ]
    return any(k in msg for k in hints)


def _retrieve_clean_chat_memories(
    user_message: str,
    history: List[Dict[str, str]],
    clean_chat_path: Path,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    rows = _load_clean_chat(clean_chat_path)
    if not rows:
        return []

    history_text = " ".join(item["content"] for item in history[-4:] if item["role"] == "user")
    query = f"{history_text} {user_message}".strip()

    scored: List[tuple[int, float, Dict[str, Any]]] = []
    total = max(len(rows), 1)
    for idx, row in enumerate(rows):
        content = str(row.get("content", "")).strip()
        if not content:
            continue
        score = _keyword_overlap_score(query, content)
        if score <= 0:
            continue
        recency_bonus = idx / total
        scored.append((score, recency_bonus, row))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    seen = set()
    results: List[Dict[str, Any]] = []
    for _, _, row in scored:
        key = (row.get("timestamp", ""), row.get("speaker", ""), row.get("content", ""))
        if key in seen:
            continue
        seen.add(key)
        results.append(row)
        if len(results) >= top_k:
            break
    return results


def _format_retrieved_memories(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "[检索记忆]\n未命中历史聊天记忆。"

    lines = ["[检索记忆]", "以下内容来自老聊天记录 clean_messages.jsonl，只能作为参考，不要强行复述："]
    for i, item in enumerate(items, start=1):
        timestamp = item.get("timestamp", "")
        speaker = item.get("speaker", "")
        content = str(item.get("content", "")).strip()
        lines.append(f"{i}. {timestamp} | {speaker}: {content}")
    lines.append("使用要求：如果这些旧聊天与当前问题有关，可以自然地参考；如果无关，就忽略，不要硬提旧事。")
    return "\n".join(lines)


def _extract_duckduckgo_url(href: str) -> str:
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href

    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return href


def _strip_tags(text: str) -> str:
    clean = re.sub(r"<.*?>", "", html.unescape(text))
    return re.sub(r"\s+", " ", clean).strip()


def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urlencode({"q": query})
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    with urlopen(req, timeout=15) as resp:
        page = resp.read().decode("utf-8", "ignore")

    title_matches = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            page,
            re.S,
        )
    )
    snippet_matches = [
        _strip_tags(m.group(1) or m.group(2))
        for m in re.finditer(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|'
            r'<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
            page,
            re.S,
        )
    ]

    results: List[Dict[str, str]] = []
    for idx, match in enumerate(title_matches[:max_results]):
        title = _strip_tags(match.group(2))
        link = _extract_duckduckgo_url(match.group(1))
        snippet = snippet_matches[idx] if idx < len(snippet_matches) else ""
        if title and link:
            results.append({"title": title, "url": link, "snippet": snippet})
    return results


def _format_web_results(query: str, results: List[Dict[str, str]]) -> str:
    if not results:
        return f"[联网搜索]\n查询：{query}\n结果：未获得可用搜索结果。"

    lines = [f"[联网搜索]\n查询：{query}", "以下是 DuckDuckGo 搜索结果摘要："]
    for i, item in enumerate(results, start=1):
        lines.append(f"{i}. 标题：{item['title']}")
        lines.append(f"   链接：{item['url']}")
        if item["snippet"]:
            lines.append(f"   摘要：{item['snippet']}")
    lines.append("使用要求：如果引用了搜索信息，优先基于这些结果回答，并明确说明是联网搜索得到的信息；不要编造未看到的细节。")
    return "\n".join(lines)


def chat_once(
    client: OpenAI,
    user_message: str,
    history: List[Dict[str, str]],
    session_summary: str = "",
) -> str:
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    site_url = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
    site_name = os.getenv("OPENROUTER_SITE_NAME", "wty-agent")
    clean_chat_path = Path(os.getenv("CLEAN_CHAT_PATH", str(DEFAULT_CLEAN_CHAT_PATH)))
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    short_memory_turns = int(os.getenv("SHORT_MEMORY_TURNS", "6"))
    session_summary = session_summary.strip()

    system_prompt = build_system_prompt(user_message=user_message)
    retrieved_memories = _retrieve_clean_chat_memories(
        user_message=user_message,
        history=history,
        clean_chat_path=clean_chat_path,
        top_k=retrieval_top_k,
    )
    memory_note = _format_retrieved_memories(retrieved_memories)
    search_note = ""
    if _should_search(user_message):
        try:
            max_results = int(os.getenv("WEB_SEARCH_TOP_K", "5"))
            web_results = duckduckgo_search(user_message, max_results=max_results)
            search_note = _format_web_results(user_message, web_results)
        except Exception as exc:
            search_note = f"[联网搜索]\n查询：{user_message}\n结果：搜索失败（{exc}）。请如实说明未成功获取联网结果。"

    system_prompt = f"{system_prompt}\n\n{memory_note}"
    if session_summary:
        system_prompt = (
            f"{system_prompt}\n\n[短期记忆摘要]\n"
            f"{session_summary}\n"
            "使用要求：以上为本次会话早先内容的压缩摘要，仅在相关时自然参考。"
        )
    if search_note:
        system_prompt = f"{system_prompt}\n\n{search_note}"

    recent_history = history[-(short_memory_turns * 2):]
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": user_message})

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={
            "HTTP-Referer": site_url,
            "X-Title": site_name,
        },
    )
    return completion.choices[0].message.content or ""


def main() -> None:
    client = _create_client()
    session_memory_path = Path(os.getenv("SESSION_MEMORY_PATH", str(DEFAULT_SESSION_MEMORY_PATH)))
    compress_trigger_turns = int(os.getenv("MEMORY_COMPRESS_TRIGGER_TURNS", "8"))
    keep_recent_turns = int(os.getenv("MEMORY_KEEP_RECENT_TURNS", "4"))
    session = _load_session_memory(session_memory_path)
    history: List[Dict[str, str]] = session["history"]
    session_summary = session["summary"]

    print("王天予 Agent (OpenRouter) 已启动，输入 quit 退出。")
    print("联网搜索：DuckDuckGo（WEB_SEARCH_MODE=auto/off/always，默认 auto）")
    print("记忆：短期记忆 + 自动摘要压缩 + clean_messages.jsonl 检索记忆")

    while True:
        user_message = input("\n你: ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"quit", "exit", "q"}:
            break

        reply = chat_once(client, user_message, history, session_summary=session_summary)
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        session_summary, history = _compact_history(
            summary=session_summary,
            history=history,
            compress_trigger_turns=compress_trigger_turns,
            keep_recent_turns=keep_recent_turns,
        )
        _save_session_memory(session_memory_path, session_summary, history)
        print(f"\n王天予: {reply}")


if __name__ == "__main__":
    main()
