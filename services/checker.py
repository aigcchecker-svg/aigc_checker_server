import json
import os
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

API_SOURCE = os.getenv("API_SOURCE", "azure").lower()

if API_SOURCE == "azure":
    _key = os.getenv("AZURE_API_KEY")
    _endpoint = os.getenv("AZURE_ENDPOINT")
    if not _key or not _endpoint:
        raise RuntimeError("Azure 模式需要在 .env 中配置 AZURE_API_KEY 和 AZURE_ENDPOINT")

    AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o-mini")
    AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

    client = AsyncAzureOpenAI(
        api_key=_key,
        azure_endpoint=_endpoint,
        api_version=AZURE_API_VERSION,
    )

elif API_SOURCE == "openrouter":
    _key = os.getenv("OPENROUTER_API_KEY")
    if not _key:
        raise RuntimeError("OpenRouter 模式需要在 .env 中配置 OPENROUTER_API_KEY")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=_key,
    )

else:
    raise RuntimeError(f"未知的 API_SOURCE='{API_SOURCE}'，只支持 'azure' 或 'openrouter'")


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


async def run_check(content: str, model: Optional[str] = None) -> dict:
    if API_SOURCE == "azure":
        actual_model = model or AZURE_DEPLOYMENT
        print(f"Using Azure model: {actual_model}")
        extra_headers = {}
    else:
        actual_model = model or "gpt-4o-mini"
        print(f"Using OpenRouter model: {actual_model}")
        extra_headers = {
            "HTTP-Referer": "https://aigcchecker.com",
            "X-Title": "AIGC Checker Detection Engine",
        }

    response = await client.chat.completions.create(
        model=actual_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        extra_headers=extra_headers,
    )

    raw_content = response.choices[0].message.content
    return json.loads(raw_content)
