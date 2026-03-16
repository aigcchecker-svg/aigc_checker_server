from __future__ import annotations

import re


_SENTENCE_ENDINGS = set("。！？!?；;")
_CLOSERS = set("\"'”’）)]】}")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|[\d一二三四五六七八九十]+[\.、\)])\s+")


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_short_segment(text: str) -> bool:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    compact_len = len(re.sub(r"\s+", "", text))
    return zh_chars < 20 and en_words < 15 and compact_len < 35


def _split_sentences_with_spans(text: str) -> list[dict]:
    if not text:
        return []

    spans: list[dict] = []
    start = 0
    index = 0
    length = len(text)

    while index < length:
        ch = text[index]
        is_double_newline = ch == "\n" and index + 1 < length and text[index + 1] == "\n"
        is_sentence_end = ch in _SENTENCE_ENDINGS

        if is_sentence_end or is_double_newline:
            end = index + 1
            if is_sentence_end:
                while end < length and text[end] in _CLOSERS:
                    end += 1

            raw = text[start:end]
            left_trim = len(raw) - len(raw.lstrip())
            right_trim = len(raw) - len(raw.rstrip())
            actual_start = start + left_trim
            actual_end = end - right_trim
            segment = text[actual_start:actual_end]
            if segment.strip():
                spans.append({"start": actual_start, "end": actual_end, "text": segment.strip()})
            start = end
        index += 1

    if start < length:
        raw = text[start:length]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        actual_start = start + left_trim
        actual_end = length - right_trim
        segment = text[actual_start:actual_end]
        if segment.strip():
            spans.append({"start": actual_start, "end": actual_end, "text": segment.strip()})

    if not spans:
        return [{"start": 0, "end": len(text), "text": text}]

    merged: list[dict] = []
    cursor = 0
    while cursor < len(spans):
        current = spans[cursor]
        if _is_short_segment(current["text"]) and cursor + 1 < len(spans):
            nxt = spans[cursor + 1]
            merged.append(
                {
                    "start": current["start"],
                    "end": nxt["end"],
                    "text": text[current["start"] : nxt["end"]].strip(),
                }
            )
            cursor += 2
            continue

        if _is_short_segment(current["text"]) and merged:
            previous = merged.pop()
            merged.append(
                {
                    "start": previous["start"],
                    "end": current["end"],
                    "text": text[previous["start"] : current["end"]].strip(),
                }
            )
            cursor += 1
            continue

        merged.append(current)
        cursor += 1

    return merged


def split_sentences(text: str) -> list[str]:
    return [item["text"] for item in _split_sentences_with_spans(clean_text(text))]


def chunk_text(text: str, target_size: int = 220, min_size: int = 120) -> list[dict]:
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentence_spans = _split_sentences_with_spans(cleaned)
    if not sentence_spans:
        return [{"chunk_id": 0, "start": 0, "end": len(cleaned), "text": cleaned}]

    chunks: list[dict] = []
    current: list[dict] = []
    current_size = 0

    def flush() -> None:
        if not current:
            return
        chunk_id = len(chunks)
        start = current[0]["start"]
        end = current[-1]["end"]
        chunks.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "text": cleaned[start:end].strip(),
            }
        )

    for span in sentence_spans:
        sentence_size = len(re.sub(r"\s+", "", span["text"]))
        projected = current_size + sentence_size
        if current and projected > target_size and current_size >= min_size:
            flush()
            current = [span]
            current_size = sentence_size
            continue

        current.append(span)
        current_size = projected

    if current:
        if chunks and current_size < min_size:
            prev = chunks.pop()
            merged_start = prev["start"]
            merged_end = current[-1]["end"]
            chunks.append(
                {
                    "chunk_id": prev["chunk_id"],
                    "start": merged_start,
                    "end": merged_end,
                    "text": cleaned[merged_start:merged_end].strip(),
                }
            )
        else:
            flush()

    if not chunks:
        return [{"chunk_id": 0, "start": 0, "end": len(cleaned), "text": cleaned}]

    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = index
    return chunks


def detect_genre(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return "general"

    lowered = cleaned.lower()
    lines = [line for line in cleaned.split("\n") if line.strip()]
    line_count = max(len(lines), 1)
    list_hits = sum(1 for line in lines if _LIST_LINE_RE.match(line) or "|" in line or "\t" in line)
    if list_hits / line_count >= 0.4:
        return "list_or_table"

    academic_hits = sum(
        lowered.count(keyword)
        for keyword in ("abstract", "introduction", "methodology", "results", "conclusion", "参考文献", "摘要", "本文", "研究", "实验", "样本")
    )
    if academic_hits >= 3 or re.search(r"\[\d+\]", cleaned):
        return "academic"

    business_hits = sum(
        lowered.count(keyword)
        for keyword in ("prd", "kpi", "okr", "roadmap", "milestone", "owner", "deadline", "budget", "交付", "里程碑", "目标", "需求", "负责人", "验收")
    )
    if business_hits >= 3:
        return "business_doc"

    seo_hits = sum(
        lowered.count(keyword)
        for keyword in ("seo", "关键词", "长尾词", "搜索量", "流量", "转化率", "收录", "排名")
    )
    if seo_hits >= 2:
        return "seo_blog"

    marketing_hits = sum(
        lowered.count(keyword)
        for keyword in ("立即", "马上", "限时", "品牌", "增长", "欢迎", "点击", "免费", "exclusive", "offer", "campaign")
    )
    if marketing_hits >= 3:
        return "marketing"

    translation_hits = sum(
        lowered.count(keyword)
        for keyword in ("译文", "原文", "translation", "translated", "source text", "target text", "直译", "意译")
    )
    mixed_language = bool(re.search(r"[\u4e00-\u9fff]", cleaned) and re.search(r"[A-Za-z]{3,}", cleaned))
    if translation_hits >= 1 or (mixed_language and "：" in cleaned and "：" in cleaned[:60]):
        return "translation_like"

    return "general"

