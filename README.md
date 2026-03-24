# AIGC Checker Engine

AI 内容检测与降 AI 改写服务，基于 FastAPI + 本地 LLM（Ollama）构建，支持混合多模型检测策略。

---

## 目录

- [架构概览](#架构概览)
- [业务处理流程](#业务处理流程)
  - [AI 检测流程 `/api/scan`](#ai-检测流程-apiscan)
  - [降 AI 改写流程 `/api/reduce`](#降-ai-改写流程-apiReduce)
- [会员与非会员区别](#会员与非会员区别)
  - [请求权限对比](#请求权限对比)
  - [返回字段对比](#返回字段对比)
- [API 接口文档](#api-接口文档)
- [评分体系说明](#评分体系说明)
- [环境配置](#环境配置)
- [启动方式](#启动方式)

---

## 架构概览

```
客户端请求
    │
    ▼
[api/detect.py]  Token 校验 → 权限上下文
    │
    ▼
[services/checker.py]  主编排层
    ├── preprocess.py   文本清洗 / 文体识别 / 分块
    ├── features.py     统计特征提取
    ├── judges.py       Qwen LLM 风格判断（Ollama）
    ├── aggregate.py    多路分数融合 → 文档级结论
    └── （Pro）远端二审  Azure / OpenRouter
```

---

## 业务处理流程

### AI 检测流程 `/api/scan`

```
原始文本
  │
  ▼ 1. 文本预处理（preprocess.clean_text）
     · 统一换行符、压缩多余空白、合并连续空行

  ▼ 2. 文体识别（preprocess.detect_genre）
     · 输出：list_or_table / academic / business_doc /
            seo_blog / marketing / translation_like / general
     · 影响后续所有评分的保守程度

  ▼ 3. 文档级特征提取（features.extract_document_features）
     · 字符数、句数、句长均值/标准差、burstiness
     · 词汇多样性、4-gram 重复率、连接词密度
     · 段落长度方差、具体细节信号数、可疑高频词

  ▼ 4. 文本分块（preprocess.chunk_text）
     · 目标块大小 ~220 字符，最小 120 字符
     · 过短的尾块自动合并到上一块

  ▼ 5. 逐块评分（对每个 chunk 并行执行以下三步）
     │
     ├── 5a. 分块特征提取（features.extract_chunk_features）
     │
     ├── 5b. Qwen LLM 风格判断（judges.judge_chunk_with_qwen）
     │       · 调用 Ollama 本地模型，返回：
     │         ai_score / confidence / signals / reasons / label
     │       · LLM 调用失败时自动降级为启发式规则（heuristic_fallback）
     │
     └── 5c. 分块综合评分（aggregate.score_chunk）
             · 权重融合：Qwen LLM 55% + 统计特征 30% + 风格信号 15%
             · 按文体执行降分惩罚（academic/business_doc: -8, list_or_table: -15）
             · 具体细节越多，分数越低（每个细节 -2.5，上限 -14）
             · 短文本（< 120 字符）向 50 收缩并降低置信度

  ▼ 6. 文档级聚合（aggregate.aggregate_document）
     · 按分块字符数加权平均得到文档 AI 概率
     · 识别高风险分块（ai_score ≥ 65）
     · 统计各 chunk reasons 出现频次，取 Top 5 作为摘要原因

  ▼ 7. 【仅 Pro】远端二审（checker._run_remote_review）
     · 满足以下任一条件时触发：
       - 本地置信度为 low
       - AI 概率处于边界区间 [40, 65]
       - 正式文体（academic / business_doc / list_or_table）且高风险块 ≤ 1
     · 调用 Azure 或 OpenRouter 做二次复核
     · 融合权重：本地 72% + 二审 28%（置信度 low 时变为 60% + 40%）

  ▼ 8. 统一补充套餐信息与 review 元数据
     · Free / Pro 均返回完整检测结果
     · 区别仅在是否允许远端二审，以及字数上限

  ▼ 最终 JSON 响应
```

---

### 降 AI 改写流程 `/api/reduce`

> 仅 Pro 套餐可用

```
原始文本
  │
  ▼ 1. 执行完整检测流程（run_check）
     · 获取 AI 概率 + 高风险分块列表

  ▼ 2. 构建改写 Prompt（checker._build_reduce_prompt）
     · 携带文体、摘要原因、高风险分块（最多 5 个）

  ▼ 3. 调用 LLM 改写（checker._rewrite_content）
     · Azure / OpenRouter：JSON 模式直接调用
     · Ollama：通过 JSON Schema 约束输出格式
     · 失败时降级返回原文，返回 `rewrite=false`，`quality_score` 保持 55

  ▼ 4. 对改写文本再次执行检测（run_check）
     · 获取改写后的 AI 概率

  ▼ 5. 计算改写质量分（checker._quality_score）
     · 内容保留度（60%）= 词汇重叠（70%）+ 长度比（30%）
     · 降 AI 效果（40%）= (before - after) / 60，上限 100

  ▼ 最终 JSON 响应
```

---

## 会员与非会员区别

### 请求权限对比

| 能力 | Free 套餐 | Pro 套餐 |
|------|-----------|----------|
| AI 检测 `/api/scan` | ✅ | ✅ |
| 降 AI 改写 `/api/reduce` | ❌ 返回 403 | ✅ |
| 最大文本长度 | 3000 字符（`FREE_MAX_CHARS`） | 12000 字符（`PRO_MAX_CHARS`） |
| 远端 Pro 二审 | ❌ | ✅（按需触发） |

> 字符上限可通过 `.env` 中的 `FREE_MAX_CHARS` / `PRO_MAX_CHARS` 调整。

---

### 返回字段对比

#### `/api/scan` 返回结构

Free 与 Pro 现在都返回完整检测结果结构；差别不在字段裁剪，而在：
- Free：`review.enabled=false`，不会触发远端二审
- Pro：`review.enabled=true`，满足条件时可能触发远端二审

**Free 套餐**

```json
{
  "label": "Mixed",
  "ai_probability": "58.32",
  "confidence": "medium",
  "plan": "free",
  "document_features": {
    "char_count": 1240,
    "sentence_count": 18,
    "avg_sentence_length": 28.5
  },
  "chunks": [
    {
      "chunk_id": 0,
      "text": "...",
      "ai_score": 62.4,
      "label": "Mixed",
      "confidence": 0.631
    }
  ],
  "analysis": {
    "genre": "general",
    "high_risk_chunks": [0, 2],
    "summary_reasons": [
      "句长波动较小，整体节奏偏均匀。",
      "词汇变化度偏低，表达较集中。"
    ]
  },
  "summary": {
    "confidence_label": "Mixed",
    "percentages": { "ai": 38, "mixed": 42, "human": 20 }
  },
  "metrics": {
    "perplexity": null,
    "burstiness": 0.312,
    "binoculars": null
  },
  "sentences": [
    { "text": "...", "ai_probability": 62, "level": 6 }
  ],
  "vocab": { "count": 3, "words": ["therefore", "furthermore", "utilize"] },
  "engine": {
    "checker": "hybrid-local",
    "qwen_model": "qwen3.5:9b",
    "api_source": "ollama"
  },
  "review": {
    "enabled": false,
    "used": false,
    "provider": null,
    "model": null,
    "reason": "plan_disabled"
  }
}
```

---

**Pro 套餐**

```json
{
  "label": "Mixed",
  "ai_probability": "55.14",
  "confidence": "high",
  "plan": "pro",
  "document_features": {
    "char_count": 1240,
    "sentence_count": 18,
    "avg_sentence_length": 28.5,
    "sentence_length_std": 8.3,
    "burstiness": 0.291,
    "lexical_diversity": 0.573,
    "repeated_ngram_ratio": 0.031,
    "punctuation_density": 0.042,
    "connector_density": 0.018,
    "paragraph_length_variance": 1820.4,
    "detail_signal_count": 4,
    "digit_density": 0.012,
    "list_line_ratio": 0.0,
    "suspicious_terms": ["leverage", "ensure", "implement"]
  },
  "chunks": [
    {
      "chunk_id": 0,
      "start": 0,
      "end": 215,
      "text": "...",
      "ai_score": 62.4,
      "label": "Mixed",
      "confidence": 0.631,
      "reasons": ["句长波动较小，整体节奏偏均匀。"],
      "score_breakdown": {
        "feature_score": 55.2,
        "qwen_score": 65.0,
        "style_score": 58.3
      },
      "features": { "...": "完整特征字典" },
      "signals": {
        "over_smooth": 0.72,
        "template_pattern": 0.31,
        "sentence_uniformity": 0.68,
        "human_detail": 0.42
      },
      "judge_label": "mixed"
    }
  ],
  "analysis": {
    "genre": "general",
    "high_risk_chunks": [0, 2],
    "summary_reasons": [
      "句长波动较小，整体节奏偏均匀。",
      "词汇变化度偏低，表达较集中。"
    ]
  },
  "engine": {
    "checker": "hybrid-local",
    "qwen_model": "qwen3.5:9b",
    "api_source": "ollama"
  },
  "review": {
    "enabled": true,
    "used": true,
    "provider": "openrouter",
    "model": "claude-3.5-haiku",
    "reason": "borderline_score",
    "result": {
      "ai_score": 48.0,
      "confidence": 0.81,
      "label": "mixed",
      "reasons": ["含有具体项目约束，降低 AI 倾向。"]
    }
  }
}
```

> 注意：如果 Pro 本次没有触发二审，仍会返回 `review` 字段，但 `used=false`，并带有 `reason=not_needed` 或其他未触发原因。

---

#### `/api/reduce` 返回结构（仅 Pro）

```json
{
  "reduced": "改写后的完整文本...",
  "rewrite": true,
  "ai_probability": "58.32",
  "ai_reduced_probability": "31.07",
  "quality_score": 74.5,
  "model": "moderate",
  "changes": [
    {
      "original": "In conclusion, it is evident that...",
      "revised": "综合来看，这说明...",
      "reason": "替换过于模板化的结论句式"
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `reduced` | 改写后的完整文本 |
| `rewrite` | 是否成功完成改写；若改写失败并回退原文，则为 `false` |
| `ai_probability` | 原文 AI 概率（改写前） |
| `ai_reduced_probability` | 改写后 AI 概率 |
| `quality_score` | 改写质量综合得分（0-100），兼顾内容保留度和降 AI 效果 |
| `model` | 改写强度：`light` / `moderate` / `deep` |
| `changes` | 各片段的改写记录，含原文、改写文和改写原因 |

---

## API 接口文档

### 认证方式

所有接口需在请求头携带 Bearer Token：

```
Authorization: Bearer <your_api_token>
```

---

### `POST /api/scan`

**请求体**

```json
{
  "content": "待检测文本（最少 50 字符）",
  "model": "qwen3.5:9b",
  "api_source": "ollama"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `content` | string | ✅ | 待检测文本 |
| `model` | string | ❌ | 推理模型，不填使用各来源默认值 |
| `api_source` | string | ❌ | `ollama` / `azure` / `openrouter`，默认读环境变量 |

**错误码**

| 状态码 | 含义 |
|--------|------|
| 400 | 文本太短（< 50 字符）或超出套餐字数上限 |
| 401 | Token 无效 |
| 500 | 检测引擎内部异常 |
| 502 | LLM 返回格式无法解析 |

---

### `POST /api/reduce`

请求体结构与 `/api/scan` 相同。

**额外限制**：Free 套餐调用返回 403。

---

## 评分体系说明

### 分块评分权重

| 来源 | 权重 |
|------|------|
| Qwen LLM 风格判断 | 55% |
| 统计特征评分 | 30% |
| 风格信号评分 | 15% |

### 文体惩罚/奖励

| 文体 | 分数调整 |
|------|----------|
| `academic` / `business_doc` | -8 |
| `list_or_table` | -15 |
| `translation_like` | -6 |
| `marketing` | +3 |

### AI 概率标签映射

| 分数范围 | 标签 |
|----------|------|
| ≥ 70 | AI Generated |
| 40 ~ 69 | Mixed |
| < 40 | Human Written |

### 置信度级别

| 数值范围 | 级别 |
|----------|------|
| ≥ 0.72 | high |
| 0.45 ~ 0.71 | medium |
| < 0.45 | low |

---

## 环境配置

复制 `.env.example` 为 `.env` 并填写以下配置：

```env
# API 鉴权
API_TOKEN=your_legacy_token
FREE_API_TOKENS=token1,token2
PRO_API_TOKENS=pro_token1,pro_token2

# 字数限制
FREE_MAX_CHARS=3000
PRO_MAX_CHARS=12000

# 推理后端（ollama / azure / openrouter）
API_SOURCE=ollama

# Ollama 配置
OLLAMA_BASE_URL=http://218.4.33.190:26316
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_TIMEOUT=120

# Azure OpenAI（可选）
AZURE_API_KEY=
AZURE_ENDPOINT=
AZURE_DEPLOYMENT=gpt-4o-mini
AZURE_API_VERSION=2025-01-01-preview

# OpenRouter（可选）
OPENROUTER_API_KEY=
OPENROUTER_MODEL=claude-3.5-haiku
OPENROUTER_SEND_MODE=self   # self 直连 / proxy 走内部代理

# Pro 二审配置
PRO_REVIEW_SOURCE=openrouter
PRO_REVIEW_MODEL=claude-3.5-haiku
```

---

## 启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 开发模式（交互）
./start.sh

# 后台运行
IS_PROD=y ./start.sh

# 内部测试页面
open http://localhost:8027/
```
