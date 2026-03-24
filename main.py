from dotenv import load_dotenv

load_dotenv()  # 必须在其他 import 之前，确保 os.getenv() 能读到 .env

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.detect import router as detect_router
from services.checker import API_SOURCE, AZURE_DEFAULT_MODEL, OLLAMA_DEFAULT_MODEL, OPENROUTER_DEFAULT_MODEL

app = FastAPI(title="AIGC Checker Engine", version="1.0")
app.include_router(detect_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def get_test_page():
    """返回内部测试用的 HTML 页面，支持 Token 认证、API Source 切换、检测/降 AI 改写两种模式。"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>AIGC Checker - 引擎内部测试</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 30px; max-width: 900px; margin: auto; background-color: #f9fafb; color: #333; }
            h2 { color: #111; }
            .container { background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            .token-row { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; padding: 12px 14px; background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; }
            .token-row label { font-size: 13px; font-weight: 600; color: #475569; white-space: nowrap; }
            .token-row input { flex: 1; padding: 7px 10px; border: 1px solid #cbd5e1; border-radius: 5px; font-size: 14px; font-family: monospace; background: #fff; }
            .token-row input:focus { outline: none; border-color: #4f46e5; }
            .token-status { font-size: 12px; white-space: nowrap; }
            .hint-box { margin: -4px 0 16px; padding: 10px 12px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; color: #1d4ed8; font-size: 13px; line-height: 1.5; }
            textarea { width: 100%; height: 200px; padding: 15px; border: 1px solid #ccc; border-radius: 6px; font-size: 15px; resize: vertical; box-sizing: border-box; }
            button { background-color: #4f46e5; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
            button:hover { background-color: #4338ca; }
            button:disabled { background-color: #9ca3af; cursor: not-allowed; }
            #loading { margin-top: 15px; color: #d97706; font-weight: bold; display: none; }
            .result-box { margin-top: 25px; border-top: 2px solid #e5e7eb; padding-top: 20px; display: none; }
            .examples-section { margin-top: 28px; padding-top: 24px; border-top: 2px solid #e5e7eb; }
            .examples-section h3 { margin-bottom: 14px; }
            .example-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
            .example-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }
            .example-card h4 { margin: 0 0 8px; color: #0f172a; }
            .example-card p { margin: 0 0 12px; color: #64748b; font-size: 13px; }
            pre { background: #1f2937; color: #f8f8f2; padding: 15px; border-radius: 6px; overflow-x: auto; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🧪 AIGC 底层检测引擎 API 测试</h2>

            <div class="token-row">
                <label>🔑 API Token</label>
                <input type="password" id="apiToken" placeholder="输入 API_TOKEN …" oninput="updateTokenStatus()" />
                <span class="token-status" id="tokenStatus">⬜ 未填写</span>
            </div>

            <div class="token-row">
                <label>🌐 API Source</label>
                <select id="apiSource" onchange="onSourceChange()">
                    <option value="ollama">ollama</option>
                    <option value="azure">azure</option>
                    <option value="openrouter">openrouter</option>
                </select>
            </div>

            <div class="token-row">
                <label>🤖 Model</label>
                <input type="text" id="modelInput" value="__DEFAULT_MODEL__" />
            </div>

            <div class="token-row">
                <label>⚙️ Mode</label>
                <select id="modeSelect" onchange="onModeChange()">
                    <option value="detect">detect（AI 检测）</option>
                    <option value="reduce">reduce（降 AI 改写）</option>
                </select>
            </div>

            <div class="hint-box" id="routingHint"></div>

            <textarea id="inputText" placeholder="请输入至少 50 个字符的文章进行测试...
例如：
The advent of electric vehicles has been touted as a cornerstone in the transition towards sustainable transportation. With global efforts to reduce carbon emissions and mitigate climate change, EVs have gained substantial attention and investment. This paper aims to dissect the multifaceted nature of EVs, weighing their advantages against the inherent limitations."></textarea>

            <button onclick="runScan()" id="scanBtn">发起 AI 检测请求</button>
            <div id="loading">⏳ 正在连线大模型进行分析，请稍候...</div>

            <div id="resultBox" class="result-box">
                <h3>📦 返回的 JSON 结构：</h3>
                <pre id="jsonOutput"></pre>
            </div>

            <div class="examples-section">
                <h3>📚 API 调用和返回示例</h3>
                <div class="example-grid">
                    <div class="example-card">
                        <h4>scan 调用示例</h4>
                        <p>`POST /api/scan`</p>
                        <pre>curl -X POST http://127.0.0.1:8000/api/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer &lt;YOUR_API_TOKEN&gt;" \
  -d '{
    "content": "这是待检测的长文本内容，长度至少 50 个字符。",
    "model": "__DEFAULT_MODEL__",
    "api_source": "__API_SOURCE__"
  }'</pre>
                    </div>

                    <div class="example-card">
                        <h4>scan 返回示例</h4>
                        <p>文档级检测结果</p>
                        <pre>{
  "label": "Mixed",
  "ai_probability": "58.32",
  "confidence": "medium",
  "chunks": [
    {
      "chunk_id": 0,
      "label": "mixed",
      "ai_score": 61.4,
      "confidence": 0.68,
      "text": "这是其中一个分块的示例文本。"
    }
  ],
  "review": {
    "enabled": false,
    "used": false,
    "provider": null,
    "model": null,
    "reason": "plan_disabled"
  }
}</pre>
                    </div>

                    <div class="example-card">
                        <h4>reduce 调用示例</h4>
                        <p>`POST /api/reduce`</p>
                        <pre>curl -X POST http://127.0.0.1:8000/api/reduce \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer &lt;YOUR_API_TOKEN&gt;" \
  -d '{
    "content": "这是待降 AI 改写的长文本内容，长度至少 50 个字符。",
    "model": "__DEFAULT_MODEL__",
    "api_source": "__API_SOURCE__"
  }'</pre>
                    </div>

                    <div class="example-card">
                        <h4>reduce 返回示例</h4>
                        <p>包含改写结果与改写状态</p>
                        <pre>{
  "reduced": "这是改写后的完整文本示例。",
  "rewrite": true,
  "ai_probability": "58.32",
  "ai_reduced_probability": "31.07",
  "quality_score": 74.5,
  "model": "deep",
  "changes": [
    {
      "original": "首先，企业需要建立统一规范。",
      "revised": "企业先得把规范定清楚。",
      "reason": "降低模板化表达"
    }
  ]
}</pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const DEFAULT_MODELS = {
                ollama: '__OLLAMA_DEFAULT_MODEL__',
                azure: '__AZURE_DEFAULT_MODEL__',
                openrouter: '__OPENROUTER_DEFAULT_MODEL__'
            };

            window.addEventListener('DOMContentLoaded', () => {
                const saved = sessionStorage.getItem('api_token');
                if (saved) { document.getElementById('apiToken').value = saved; updateTokenStatus(); }
                document.getElementById('apiSource').value = '__API_SOURCE__';
                renderRoutingHint();
            });

            function onSourceChange() {
                const source = document.getElementById('apiSource').value;
                document.getElementById('modelInput').value = DEFAULT_MODELS[source] || '';
                renderRoutingHint();
            }

            function onModeChange() {
                const mode = document.getElementById('modeSelect').value;
                document.getElementById('scanBtn').textContent =
                    mode === 'reduce' ? '发起降 AI 改写请求' : '发起 AI 检测请求';
                renderRoutingHint();
            }

            function renderRoutingHint() {
                const mode = document.getElementById('modeSelect').value;
                const source = document.getElementById('apiSource').value;
                const hint = document.getElementById('routingHint');
                if (mode === 'reduce') {
                    hint.textContent =
                        '当前 reduce 固定策略：前后检测都走 Ollama；rewrite 采用组合流水线：Step1 Azure GPT-4o 深度重写（失败时回退 OpenRouter qwen/qwen-plus），Step2 Ollama Qwen 扰动表达，Step3 规则注入细节与数字锚点。当前选择的 API Source=' + source + ' 仅作为请求兼容参数保留。';
                    return;
                }
                hint.textContent =
                    '当前 scan 主检测链路固定使用 Ollama。API Source=' + source + ' 仍可传递给接口，但不会改变 scan 的实际执行引擎。';
            }

            function updateTokenStatus() {
                const val = document.getElementById('apiToken').value.trim();
                sessionStorage.setItem('api_token', val);
                document.getElementById('tokenStatus').textContent = val ? '✅ 已填写' : '⬜ 未填写';
            }

            async function runScan() {
                const token = document.getElementById('apiToken').value.trim();
                const text  = document.getElementById('inputText').value.trim();
                const model = document.getElementById('modelInput').value.trim();
                const btn   = document.getElementById('scanBtn');
                const loading   = document.getElementById('loading');
                const resultBox = document.getElementById('resultBox');
                const jsonOutput = document.getElementById('jsonOutput');

                if (!token) { alert("请先填写 API Token！"); return; }
                if (text.length < 50) { alert("测试文本太短，请至少输入 50 个字符！"); return; }

                btn.disabled = true;
                loading.style.display = 'block';
                resultBox.style.display = 'none';
                jsonOutput.textContent = '';

                const mode = document.getElementById('modeSelect').value;
                const endpoint = mode === 'reduce' ? '/api/reduce' : '/api/scan';

                try {
                    const response = await fetch(endpoint, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + token
                        },
                        body: JSON.stringify({ content: text, model: model || undefined, api_source: document.getElementById('apiSource').value })
                    });

                    const data = await response.json();
                    jsonOutput.textContent = JSON.stringify(data, null, 2);
                    jsonOutput.style.color = response.ok ? '#f8f8f2' : '#ef4444';
                    resultBox.style.display = 'block';
                } catch (err) {
                    jsonOutput.textContent = "网络请求失败: " + err;
                    jsonOutput.style.color = '#ef4444';
                    resultBox.style.display = 'block';
                } finally {
                    btn.disabled = false;
                    loading.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    """
    # 根据当前配置的 API_SOURCE 确定页面默认显示的模型名
    default_model = (
        OLLAMA_DEFAULT_MODEL if API_SOURCE == "ollama"
        else AZURE_DEFAULT_MODEL if API_SOURCE == "azure"
        else OPENROUTER_DEFAULT_MODEL
    )
    # 将 HTML 模板中的占位符替换为实际运行时配置值，再返回给浏览器
    html = (
        html_content
        .replace("__API_SOURCE__", API_SOURCE)
        .replace("__DEFAULT_MODEL__", default_model)
        .replace("__OLLAMA_DEFAULT_MODEL__", OLLAMA_DEFAULT_MODEL)
        .replace("__AZURE_DEFAULT_MODEL__", AZURE_DEFAULT_MODEL)
        .replace("__OPENROUTER_DEFAULT_MODEL__", OPENROUTER_DEFAULT_MODEL)
    )
    return HTMLResponse(content=html)
