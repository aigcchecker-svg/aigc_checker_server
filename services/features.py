from __future__ import annotations

import re
import statistics
from collections import Counter

from services.preprocess import clean_text, detect_language, split_sentences


_PUNCTUATION_RE = re.compile(r"[，。！？；：、,.!?;:()\[\]{}\"'“”‘’《》<>/%-]")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[\u4e00-\u9fff]")
_ZH_CONNECTOR_PATTERNS = [
    r"因此",
    r"所以",
    r"但是",
    r"然而",
    r"此外",
    r"另外",
    r"同时",
    r"并且",
    r"首先",
    r"其次",
    r"最后",
    r"总之",
]
_EN_CONNECTOR_PATTERNS = [
    r"in addition",
    r"however",
    r"therefore",
    r"moreover",
    r"furthermore",
    r"firstly",
    r"secondly",
    r"finally",
    r"overall",
    r"in conclusion",
    r"in summary",
    r"as a result",
    r"for example",
    r"for instance",
    r"on the other hand",
    r"it is important to note(?: that)?",
    r"it should be noted(?: that)?",
    r"at the same time",
    r"more importantly",
    r"ultimately",
]
_COMMON_DETAIL_PATTERNS = [
    r"\b\d{4}[-/年]\d{1,2}(?:[-/月]\d{1,2}日?)?\b",
    r"\b\d{1,2}:\d{2}\b",
    r"\b\d+(?:\.\d+)?%\b",
    r"\b(?:USD|RMB|CNY|￥|\$)\s?\d+(?:\.\d+)?\b",
    r"(?:http|www\.)",
]
_ZH_DETAIL_PATTERNS = _COMMON_DETAIL_PATTERNS + [
    r"(?:截止|上线|验收|预算|负责人)",
    r"(?:邮箱|电话|微信|飞书)",
]
_EN_DETAIL_PATTERNS = _COMMON_DETAIL_PATTERNS + [
    r"\b(?:owner|deadline|milestone|sla|kpi|okr|jira|confluence|notion|slack)\b",
    r"\b[A-Z]{2,}-\d{2,}\b",
    r"\bv?\d+\.\d+(?:\.\d+)?\b",
    r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b",
    r"\b(?:ticket|issue|sprint|release)\s+#?\d+\b",
    r"\b[A-Z]{2,}(?:\d+)?\b",
]
_MIXED_DETAIL_PATTERNS = _ZH_DETAIL_PATTERNS + [
    r"(?:邮箱|电话|微信|Slack|飞书|Jira|Confluence|Notion)",
]
_STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "have",
    "will",
    "your",
    "我们",
    "你们",
    "他们",
    "以及",
    "并且",
    "如果",
    "一个",
    "一种",
    "进行",
    "可以",
    "需要",
    "为了",
    "其中",
}


def compute_pseudo_perplexity(text: str) -> float | None:
    """TODO: 后续可接语言模型打分，第一版仅保留扩展接口。"""
    return None


def _safe_mean(values: list[float]) -> float:
    """空列表时返回 0.0，避免 ZeroDivisionError。"""
    return sum(values) / len(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    """单元素或空列表时返回 0.0，使用总体标准差 pstdev。"""
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def _tokenize(text: str) -> list[str]:
    """提取英文单词、数字和中文单字，统一转小写，用于词汇多样性计算。"""
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _count_connectors(text: str, language: str) -> int:
    """按语言统计连接词。"""
    lowered = text.lower()
    patterns = _ZH_CONNECTOR_PATTERNS if language == "zh" else _EN_CONNECTOR_PATTERNS if language == "en" else _ZH_CONNECTOR_PATTERNS + _EN_CONNECTOR_PATTERNS
    return sum(len(re.findall(pattern, lowered)) for pattern in patterns)


def _detail_signal_count(text: str, language: str) -> int:
    """按语言统计具体细节信号，英文不再广泛把 Title Case 视为真人细节。"""
    if language == "zh":
        patterns = _ZH_DETAIL_PATTERNS
    elif language == "en":
        patterns = _EN_DETAIL_PATTERNS
    else:
        patterns = _MIXED_DETAIL_PATTERNS
    return sum(len(re.findall(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)


def _repeated_ngram_ratio(text: str, language: str, n: int = 4) -> float:
    """按语言计算重复 n-gram 比率。

    中文使用字符级 4-gram，英文使用词级 3-gram，mixed 取两者较高值。
    """
    if language == "en":
        tokens = [token.lower() for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]
        n_words = 3
        if len(tokens) < n_words:
            return 0.0
        grams = [tuple(tokens[index : index + n_words]) for index in range(len(tokens) - n_words + 1)]
        counts = Counter(grams)
        repeated = sum(count - 1 for count in counts.values() if count > 1)
        return repeated / len(grams) if grams else 0.0

    if language == "mixed":
        zh_ratio = _repeated_ngram_ratio(text, "zh", n=n)
        en_ratio = _repeated_ngram_ratio(text, "en", n=n)
        return max(zh_ratio, en_ratio)

    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return 0.0
    grams = [compact[index : index + n] for index in range(len(compact) - n + 1)]
    counts = Counter(grams)
    # 每个 gram 超出 1 次的部分都算作"重复"
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams) if grams else 0.0


def _top_terms(tokens: list[str], limit: int = 10) -> list[str]:
    """返回出现次数 > 1 的高频词（过滤停用词和单字符），用于识别可疑重复词汇。"""
    filtered = [token for token in tokens if token not in _STOPWORDS and len(token) > 1]
    counts = Counter(filtered)
    return [token for token, count in counts.most_common(limit) if count > 1]


def _extract_features(text: str, language: str | None = None) -> dict:
    """对文本进行全面的统计特征提取，返回供评分模型使用的特征字典。

    主要特征说明：
    - burstiness：句长标准差 / 均值，越小越平滑（AI 倾向）
    - lexical_diversity：不重复词数 / 总词数，越低词汇越单调（AI 倾向）
    - repeated_ngram_ratio：4-gram 重复率，越高越模板化（AI 倾向）
    - connector_density：连接词密度，过高表示结构过于规整（AI 倾向）
    - detail_signal_count：具体细节数量，越多越偏人类写作
    - list_line_ratio：列表行比例，高比例需降低 AI 判定权重
    """
    cleaned = clean_text(text)
    language = language or detect_language(cleaned)
    # 去除空白后的紧凑文本，用于计算字符数和数字密度
    compact = re.sub(r"\s+", "", cleaned)
    sentences = split_sentences(cleaned)
    # 每句的紧凑字符数，用于计算句长统计量
    sentence_lengths = [len(re.sub(r"\s+", "", sentence)) for sentence in sentences]
    paragraphs = [segment.strip() for segment in cleaned.split("\n\n") if segment.strip()]
    paragraph_lengths = [len(re.sub(r"\s+", "", paragraph)) for paragraph in paragraphs]
    tokens = _tokenize(cleaned)
    char_count = len(compact)
    token_count = len(tokens)
    connector_count = _count_connectors(cleaned, language)
    punctuation_count = len(_PUNCTUATION_RE.findall(cleaned))
    detail_signal_count = _detail_signal_count(cleaned, language)
    lines = [line for line in cleaned.split("\n") if line.strip()]
    # 统计以列表符号开头的行比例，高比例说明内容为结构化列表
    list_line_ratio = (
        sum(1 for line in lines if re.match(r"^\s*(?:[-*•]|[\d一二三四五六七八九十]+[\.、\)])", line)) / len(lines)
        if lines
        else 0.0
    )

    avg_sentence_length = _safe_mean(sentence_lengths)
    sentence_length_std = _safe_std(sentence_lengths)
    # burstiness = 变异系数，衡量句长波动程度
    burstiness = sentence_length_std / avg_sentence_length if avg_sentence_length else 0.0
    lexical_diversity = len(set(tokens)) / token_count if token_count else 0.0
    paragraph_length_variance = statistics.pvariance(paragraph_lengths) if len(paragraph_lengths) > 1 else 0.0

    return {
        "char_count": char_count,
        "sentence_count": len(sentences),
        "language": language,
        "avg_sentence_length": round(avg_sentence_length, 4),
        "sentence_length_std": round(sentence_length_std, 4),
        "burstiness": round(burstiness, 4),
        "lexical_diversity": round(lexical_diversity, 4),
        "repeated_ngram_ratio": round(_repeated_ngram_ratio(cleaned, language), 4),
        "punctuation_density": round(punctuation_count / char_count, 4) if char_count else 0.0,
        "connector_density": round(connector_count / max(token_count, 1), 4),
        "paragraph_length_variance": round(paragraph_length_variance, 4),
        "pseudo_perplexity": compute_pseudo_perplexity(cleaned),
        "token_count": token_count,
        "detail_signal_count": detail_signal_count,
        "digit_density": round(sum(ch.isdigit() for ch in compact) / char_count, 4) if char_count else 0.0,
        "list_line_ratio": round(list_line_ratio, 4),
        "suspicious_terms": _top_terms(tokens),  # 高频词，用于词汇层面辅助判断
    }


def extract_document_features(text: str, language: str | None = None) -> dict:
    """提取整篇文档的统计特征，供文档级聚合使用。"""
    return _extract_features(text, language=language)


def extract_chunk_features(chunk_text: str, language: str | None = None) -> dict:
    """提取单个分块的统计特征，供分块级评分使用。"""
    return _extract_features(chunk_text, language=language)
