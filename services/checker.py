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

【核心警告：绝对禁止修改原文】
在输出 `sentences` 数组的 `text` 字段时，必须 100% 逐字逐句复制用户原文中的句子（包括标点符号）。
绝不允许修改任何词汇、大小写、标点或修正错别字！如果不完全一致，前端的高亮匹配系统将会崩溃。

请严格按照以下 JSON 结构输出（不要包含 ```json 等 Markdown 标记，直接输出合法 JSON 对象）：
{
  "summary": {
    "confidence_label": "AI Generated", // 只能是: "AI Generated", "Human Written", 或 "Mixed"
    "percentages": {
      "ai": 85,    // 0-100 的整数，表示 AI 生成的总体比例
      "mixed": 10, // 0-100 的整数，表示混合生成的比例
      "human": 5   // 0-100 的整数，三个数字相加必须等于 100
    }
  },
  "sentences": [
    {
      "text": "原文本中的完整句子1。", // 必须与原文完全一致
      "ai_probability": 90, // 0-100 的整数，该句为 AI 生成的概率
      "level": 9 // 1-10 的整数，1代表极度像人类，10代表极度像AI
    }
  ],
  "vocab": {
    "count": 3,
    "words": ["词汇1", "词汇2", "词汇3"] // 提取最多 10 个最能暴露 AI 痕迹的典型词汇，没有则为空数组
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
