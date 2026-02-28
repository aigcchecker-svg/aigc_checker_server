import json
import os
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

API_SOURCE = os.getenv("API_SOURCE", "azure").lower()

# --- Azure ---
AZURE_DEFAULT_MODEL = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
_azure_key = os.getenv("AZURE_API_KEY")
_azure_endpoint = os.getenv("AZURE_ENDPOINT")
_azure_client = (
    AsyncAzureOpenAI(
        api_key=_azure_key,
        azure_endpoint=_azure_endpoint,
        api_version=os.getenv("AZURE_API_VERSION", "2025-01-01-preview"),
    )
    if _azure_key and _azure_endpoint
    else None
)

# --- OpenRouter ---
OPENROUTER_DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
_openrouter_key = os.getenv("OPENROUTER_API_KEY")
_openrouter_client = (
    AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=_openrouter_key)
    if _openrouter_key
    else None
)

# 启动时校验默认 source 已配置
if API_SOURCE == "azure" and not _azure_client:
    raise RuntimeError("Azure 模式需要在 .env 中配置 AZURE_API_KEY 和 AZURE_ENDPOINT")
elif API_SOURCE == "openrouter" and not _openrouter_client:
    raise RuntimeError("OpenRouter 模式需要在 .env 中配置 OPENROUTER_API_KEY")
elif API_SOURCE not in ("azure", "openrouter"):
    raise RuntimeError(f"未知的 API_SOURCE='{API_SOURCE}'，只支持 'azure' 或 'openrouter'")

DEFAULT_MODEL = AZURE_DEFAULT_MODEL if API_SOURCE == "azure" else OPENROUTER_DEFAULT_MODEL


SYSTEM_PROMPT = """
你是一个顶尖的语言学专家和 AI 文本检测引擎的核心算法。
你的任务是分析用户提供的文本，并精准判断其由 AI（如 ChatGPT, Claude）生成的概率。

【防误判与特征识别指南（极度重要）】
1. 警惕“结构化业务文档”误判：用户可能会输入产品需求文档(PRD)、项目Brief、提纲、公文或技术规范。这类文本天生具有高度规范的格式（带有 1. 2. 3. 等序号）、客观正式的语气和行业术语。**绝对不要**仅仅因为文本结构工整、逻辑严密、没有语病就轻易判定为 AI 生成。
2. 寻找“人类业务细节”：着重分析文本中是否包含真实世界特有的业务细节。例如：提到具体的人名（如“Justin”）、特定的公司内部黑话、非常具体且琐碎的执行限制等。如果包含这些具有强烈“人类现实工作背景”的细节，即使格式再像 AI，也应大幅降低 AI 生成的概率预估，将其判定为 Human Written。
3. 词汇脱敏：当文本本身就是在讨论 AI 技术（如提到了“AI图生图”、“AI合成”）时，不要将这些行业专有名词算作“AI 写作高频词”。

除了常规的逐句打分和词汇提取，你还需要根据文本的特征，模拟计算出三个专业的学术指标（要求这些指标的值必须符合你最终得出的 AI 概率逻辑）：
1. Perplexity (困惑度): 衡量文本的不可预测性。AI 生成的文本极其顺畅，通常在 10-40 之间（越低越像 AI）；人类文本用词多变，通常在 60-120 之间。
2. Burstiness (突发性): 衡量句子长度和结构的变异程度。AI 文本结构均一，通常在 5-15 之间（越低越像 AI）；人类写作长短句交错起伏大，通常在 25-60 之间。
3. Binoculars Score (双筒望远镜分数): 衡量零样本模型差异率。AI 文本通常在 0.85-1.05 之间，人类文本通常在 1.10-1.30 之间。

【核心警告：绝对禁止修改原文】
在输出 `sentences` 数组的 `text` 字段时，必须 100% 逐字逐句复制用户原文中的句子（包括标点符号）。
绝不允许修改任何词汇、大小写、标点或修正错别字！如果不完全一致，前端的高亮匹配系统将会崩溃。

请严格按照以下 JSON 结构输出（不要包含 ```json 等 Markdown 标记，直接输出合法的 JSON 对象）：
{
  "summary": {
    "confidence_label": "Human Written", // 只能是: "AI Generated", "Human Written", 或 "Mixed"
    "percentages": {
      "ai": 10,    // 0-100 的整数，表示 AI 生成的总体比例
      "mixed": 20, // 0-100 的整数，表示混合生成的比例
      "human": 70  // 0-100 的整数，三个数字相加必须等于 100
    }
  },
  "metrics": {
    "perplexity": 85.4, // 保留一位小数的浮点数，请根据整体 AI 浓度给出合理估算值
    "burstiness": 32.1, // 保留一位小数的浮点数，请根据整体 AI 浓度给出合理估算值
    "binoculars": 1.18  // 保留两位小数的浮点数，请根据整体 AI 浓度给出合理估算值
  },
  "sentences": [
    {
      "text": "原文本中的完整句子1。", // 必须与原文完全一致
      "ai_probability": 15, // 0-100 的整数，该句为 AI 生成的概率
      "level": 2 // 1-10 的整数，1代表极度像人类，10代表极度像AI
    }
  ],
  "vocab": {
    "count": 3,
    "words": ["词汇1", "词汇2", "词汇3"] // 提取最多 10 个最能暴露 AI 痕迹的典型词汇，没有则为空数组，不要把专业术语当成AI词汇
  }
}
"""


REDUCE_SYSTEM_PROMPT = """
你是一位顶尖的文字润色专家，擅长将 AI 生成痕迹明显的文本改写为更具人类写作风格的内容。

【你的任务】
用户会提供一段文本，你需要：
1. 识别其中 AI 写作痕迹明显的句子或片段（如过于整齐的结构、典型 AI 高频词、过度书面化的表达等）
2. 对这些片段进行自然化改写，使其更像人类写作风格（可适当引入轻微不规整、口语化词汇、多样句式）
3. 未识别为 AI 痕迹明显的部分，保持原文不变
4. 记录每处修改的原始内容、改写内容和改写原因

【改写原则】
- 保留原文的核心语义和信息，不得改变事实
- 优先调整句式结构、替换 AI 高频词汇、打破过度规整的排比
- 改写幅度适中，不要过度口语化，保持文本的基本专业性
- 如果原文整体已经较像人类写作，可以只做少量调整

请严格按照以下 JSON 结构输出（不要包含 ```json 等 Markdown 标记，直接输出合法的 JSON 对象）：
{
  "original": "用户提供的原始完整文本（原样复制，不做任何修改）",
  "reduced": "经过改写后的完整文本",
  "changes": [
    {
      "original": "被修改的原始片段",
      "revised": "改写后的片段",
      "reason": "改写原因说明"
    }
  ]
}
"""


async def _call_llm(
    prompt: str,
    system: str,
    model: Optional[str] = None,
    api_source: Optional[str] = None,
) -> dict:
    source = (api_source or API_SOURCE).lower()

    if source == "azure":
        if not _azure_client:
            raise RuntimeError("Azure 未配置，请检查 AZURE_API_KEY 和 AZURE_ENDPOINT")
        actual_model = model or AZURE_DEFAULT_MODEL
        active_client = _azure_client
        extra_headers = {}
        print(f"Using Azure model: {actual_model}")
    else:
        if not _openrouter_client:
            raise RuntimeError("OpenRouter 未配置，请检查 OPENROUTER_API_KEY")
        actual_model = model or OPENROUTER_DEFAULT_MODEL
        active_client = _openrouter_client
        extra_headers = {
            "HTTP-Referer": "https://aigcchecker.com",
            "X-Title": "AIGC Checker Detection Engine",
        }
        print(f"Using OpenRouter model: {actual_model}")

    response = await active_client.chat.completions.create(
        model=actual_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        extra_headers=extra_headers,
    )

    raw_content = response.choices[0].message.content
    return json.loads(raw_content)


async def run_check(content: str, model: Optional[str] = None, api_source: Optional[str] = None) -> dict:
    return await _call_llm(content, SYSTEM_PROMPT, model=model, api_source=api_source)


async def run_reduce(content: str, model: Optional[str] = None, api_source: Optional[str] = None) -> dict:
    return await _call_llm(content, REDUCE_SYSTEM_PROMPT, model=model, api_source=api_source)
