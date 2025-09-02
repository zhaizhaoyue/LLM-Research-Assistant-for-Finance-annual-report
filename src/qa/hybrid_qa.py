# src/qa/hybrid_qa.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import re

from .utils_citation import (
    top_citations_from_hits,
    ensure_citations,
    make_citation,
    merge_citations,
)

# 可选：如果暂时没有接 LLM，下面的兜底实现会返回 None
try:
    from .llm import llm_summarize
except Exception:
    def llm_summarize(prompt: str, **kwargs) -> Optional[str]:
        return None


def answer_textual_or_mixed(
    query: str,
    hits: List[Dict[str, Any]],
    filters: Dict[str, Any],
    use_llm: bool = True,
    k_ctx: int = 4,
    max_chars: int = 1600,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    文本/混合问题回答主流程：
    1) 选 Top-K 命中组装上下文（去重/清洗/截断）
    2) 生成引用（≥1 条兜底）
    3) 规则拼接答案（兜底）或 LLM 总结（可开关）
    返回: final_answer, reasoning, citations
    """
    # 1) 组装上下文
    chosen_hits = hits[: max(1, k_ctx)]
    ctx_blocks, cite_from_ctx = _build_context_blocks(chosen_hits, max_chars=max_chars)

    # 2) 引用（优先取命中前若干条）
    citations = ensure_citations(merge_citations(cite_from_ctx, top_citations_from_hits(hits, k=3)), hits)

    # 3) 规则拼接（兜底/可对照）
    rule_answer = _rule_based_answer(query, ctx_blocks)

    if not use_llm:
        return rule_answer, "基于 Top-K 片段的规则拼接。", citations

    # 4) LLM 总结（在限定上下文内）
    prompt = _build_llm_prompt(query, ctx_blocks)
    llm_text = None
    try:
        llm_text = llm_summarize(prompt, temperature=0.1, max_tokens=380)
    except Exception:
        llm_text = None

    final = (llm_text or "").strip() or rule_answer
    reasoning = "LLM 在限定上下文内做摘要整合（失败/空输出时回退到规则拼接）。"
    return final, reasoning, citations


# ---------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------

def _build_context_blocks(
    hits: List[Dict[str, Any]],
    max_chars: int = 1600,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    将若干命中片段清洗/截断为上下文块：
    - 去掉多余空白/重复
    - 控制总字数不超过 max_chars
    - 同时生成与上下文块对应的引用（便于追溯）
    """
    blocks: List[str] = []
    cites: List[Dict[str, Any]] = []
    used = set()
    total = 0

    for h in hits:
        raw = (h.get("snippet") or "").strip()
        meta = h.get("meta", {}) or {}
        if not raw:
            continue

        # 轻量清洗
        text = _clean_snippet(raw)
        if not text:
            continue

        # 去重（以文本做签名，避免近重复）
        sig = (text[:80], meta.get("source_path"), meta.get("page_no"))
        if sig in used:
            continue
        used.add(sig)

        # 截断（单块不超过 60% * max_chars/k）
        per_block_limit = max(180, int(max_chars * 0.6 / max(1, len(hits))))
        text = text[:per_block_limit]

        # 总长度控制
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain <= 40:
                break
            text = text[:remain]
        blocks.append(text)
        total += len(text)

        cites.append(make_citation(h))

        if total >= max_chars:
            break

    return blocks, cites


def _clean_snippet(s: str) -> str:
    """
    轻量清洗命中文本：
    - 合并多余空白
    - 去掉超长的标点/分隔
    - 简单去噪（表头/脚注常见符号）
    """
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = re.sub(r"[\u200b\u200c\u200d]+", "", s)  # 零宽字符
    s = re.sub(r"[│┃●■◆◇]+", " ", s)
    s = s.strip()
    return s


def _rule_based_answer(query: str, ctx_blocks: List[str]) -> str:
    """
    规则拼接：简单把上下文最关键的几行拼起来（优先首段）。
    """
    if not ctx_blocks:
        return f"根据当前可用的检索结果，无法给出充分的信息来回答：{query}。"

    # 取前 1–2 段，避免太长
    first = ctx_blocks[0]
    second = ctx_blocks[1] if len(ctx_blocks) > 1 else ""
    head = first.split("\n")[0].strip()
    tail = (second.split("\n")[0].strip() if second else "")
    if tail:
        ans = f"{head} …… {tail}"
    else:
        ans = head
    return f"{ans}"


def _build_llm_prompt(query: str, ctx_blocks: List[str]) -> str:
    """
    严格限定在给定上下文内回答，防止幻觉。
    """
    ctx = "\n\n---\n\n".join(ctx_blocks[:6])
    prompt = f"""You are a precise financial analyst.

Question:
{query}

You will receive only a few extracts from SEC filings (tables or text). Synthesize a concise, well-structured answer **strictly based on the provided context**:
- Do NOT invent facts or numbers outside the context.
- If multiple views exist, reconcile them; if insufficient, say so explicitly.
- Keep any figures exactly as shown; do not convert units or currencies.
- Output in Chinese if the question is Chinese; otherwise use English.
- Keep it within 5–8 concise sentences.
- End with a short list titled “要点 / Key Points:”, containing 2–4 bullets.

Context:
{ctx}
"""
    return prompt
