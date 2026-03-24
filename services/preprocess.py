from __future__ import annotations

import re


_SENTENCE_ENDINGS = set("。！？!?；;")
_CLOSERS = set("\"'”’）)]】}")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|[\d一二三四五六七八九十]+[\.、\)])\s+")


def clean_text(text: str) -> str:
    """统一文本格式：规范化换行符、压缩多余空白、合并超过两个的连续空行。

    处理步骤：
    1. 统一换行符为 \\n，全角空格替换为半角
    2. 压缩行内多个空格/Tab 为单个空格
    3. 3 个以上连续空行合并为 2 个
    4. 去除每行首尾空白
    """
    # 统一换行符，兼容 Windows（\\r\\n）和旧 Mac（\\r）
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    # 行内多余空白压缩为单个空格
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    # 行 strip 后可能再次产生连续空行，再压缩一次
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_short_segment(text: str) -> bool:
    """判断文本段是否过短（三项同时满足）：中文 < 20 字、英文词 < 15 个、紧凑长度 < 35。

    过短的段落通常是标题或过渡句，单独评分可靠性低，会被合并到相邻段落。
    """
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
    compact_len = len(re.sub(r"\s+", "", text))
    return zh_chars < 20 and en_words < 15 and compact_len < 35


def _split_sentences_with_spans(text: str) -> list[dict]:
    """按句末标点或双换行切分文本为带位置信息的句子 span 列表。

    切分后会合并过短的段落（_is_short_segment），防止标题行单独成块导致评分失真。
    每个 span 包含 start、end（在原文中的字节偏移）和 text（去空白后的文本）。
    """
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
                # 跳过紧跟在句末标点后的关闭符号（引号、括号等）
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

    # 处理最后一个未以标点结尾的文本段
    if start < length:
        raw = text[start:length]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        actual_start = start + left_trim
        actual_end = length - right_trim
        segment = text[actual_start:actual_end]
        if segment.strip():
            spans.append({"start": actual_start, "end": actual_end, "text": segment.strip()})

    # 切分失败时将整个文本作为一个 span 返回
    if not spans:
        return [{"start": 0, "end": len(text), "text": text}]

    # 短段落合并：优先向后合并，否则向前合并到已处理的最后一段
    merged: list[dict] = []
    cursor = 0
    while cursor < len(spans):
        current = spans[cursor]
        if _is_short_segment(current["text"]) and cursor + 1 < len(spans):
            # 短段向后合并：将当前段与下一段拼为一段
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
            # 短段向前合并：追加到 merged 最后一段
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
    """返回清洗后文本的句子列表（仅文本内容，不含位置信息）。"""
    return [item["text"] for item in _split_sentences_with_spans(clean_text(text))]


def detect_language(text: str) -> str:
    """粗粒度语言识别：zh / en / mixed。"""
    cleaned = clean_text(text)
    if not cleaned:
        return "mixed"

    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
    en_words = len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", cleaned))
    latin_chars = len(re.findall(r"[A-Za-z]", cleaned))

    has_zh = zh_chars >= 8
    has_en = en_words >= 8 or latin_chars >= 40

    if has_zh and has_en:
        zh_weight = zh_chars
        en_weight = latin_chars
        ratio = min(zh_weight, en_weight) / max(zh_weight, en_weight, 1)
        return "mixed" if ratio >= 0.18 else ("zh" if zh_weight > en_weight else "en")
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "zh" if zh_chars >= latin_chars else "en"


def chunk_text(text: str, target_size: int = 320, min_size: int = 180) -> list[dict]:
    """将文本切分为用于逐块评分的分块列表。

    策略：累积句子直到字符数接近 target_size 时输出一块（flush）；
    若最后剩余的句子不足 min_size，则合并到上一块，避免孤立短块。
    每个分块包含 chunk_id、start/end（在清洗后文本的偏移）和 text。
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    sentence_spans = _split_sentences_with_spans(cleaned)
    if not sentence_spans:
        return [{"chunk_id": 0, "start": 0, "end": len(cleaned), "text": cleaned}]

    chunks: list[dict] = []
    current: list[dict] = []  # 当前正在积累的句子 span 列表
    current_size = 0

    def flush() -> None:
        """将 current 中积累的句子合并为一个 chunk 并追加到 chunks。"""
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
        # 加入当前句后超过目标大小，且已积累内容满足最小块要求，则先输出再开新块
        if current and projected > target_size and current_size >= min_size:
            flush()
            current = [span]
            current_size = sentence_size
            continue

        current.append(span)
        current_size = projected

    if current:
        # 最后剩余内容不足 min_size 时，合并到上一块，避免产生孤立短块
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

    # 合并操作可能导致 chunk_id 不连续，统一重新编号
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = index
    return chunks


def detect_genre(text: str) -> str:
    """通过关键词命中数和结构特征判断文本文体，决定后续评分策略的保守程度。

    优先级顺序（从高到低）：
    list_or_table → academic → business_doc → seo_blog → marketing → translation_like → general

    文体影响：academic/business_doc/list_or_table 触发保守策略（降低 AI 分）；
    marketing 触发激进策略（略微提高 AI 分）。
    """
    cleaned = clean_text(text)
    if not cleaned:
        return "general"

    lowered = cleaned.lower()
    lines = [line for line in cleaned.split("\n") if line.strip()]
    line_count = max(len(lines), 1)
    # 列表行：以 -/*/ 序号 开头，或包含表格分隔符 | 和制表符
    list_hits = sum(1 for line in lines if _LIST_LINE_RE.match(line) or "|" in line or "\t" in line)
    if list_hits / line_count >= 0.4:
        return "list_or_table"

    # 学术文：论文关键词或参考文献格式 [数字]
    academic_hits = sum(
        lowered.count(keyword)
        for keyword in ("abstract", "introduction", "methodology", "results", "conclusion", "参考文献", "摘要", "本文", "研究", "实验", "样本")
    )
    if academic_hits >= 3 or re.search(r"\[\d+\]", cleaned):
        return "academic"

    # 商业文档：PRD/OKR/里程碑等项目管理关键词
    business_hits = sum(
        lowered.count(keyword)
        for keyword in ("prd", "kpi", "okr", "roadmap", "milestone", "owner", "deadline", "budget", "交付", "里程碑", "目标", "需求", "负责人", "验收")
    )
    if business_hits >= 3:
        return "business_doc"

    # SEO 博客：搜索优化相关关键词
    seo_hits = sum(
        lowered.count(keyword)
        for keyword in ("seo", "关键词", "长尾词", "搜索量", "流量", "转化率", "收录", "排名")
    )
    if seo_hits >= 2:
        return "seo_blog"

    # 营销文案：促销/活动类关键词
    marketing_hits = sum(
        lowered.count(keyword)
        for keyword in ("立即", "马上", "限时", "品牌", "增长", "欢迎", "点击", "免费", "exclusive", "offer", "campaign")
    )
    if marketing_hits >= 3:
        return "marketing"

    # 翻译文本：含翻译标记词，或正文开头同时含中英文且有冒号
    translation_hits = sum(
        lowered.count(keyword)
        for keyword in ("译文", "原文", "translation", "translated", "source text", "target text", "直译", "意译")
    )
    mixed_language = bool(re.search(r"[\u4e00-\u9fff]", cleaned) and re.search(r"[A-Za-z]{3,}", cleaned))
    if translation_hits >= 1 or (mixed_language and "：" in cleaned and "：" in cleaned[:60]):
        return "translation_like"

    return "general"
