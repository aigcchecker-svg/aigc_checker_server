from dotenv import load_dotenv

load_dotenv()  # 必须在其他 import 之前，确保 os.getenv() 能读到 .env

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.detect import router as detect_router

app = FastAPI(title="AIGC Checker Engine", version="1.0")
app.include_router(detect_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def get_test_page():
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
            textarea { width: 100%; height: 200px; padding: 15px; border: 1px solid #ccc; border-radius: 6px; font-size: 15px; resize: vertical; box-sizing: border-box; }
            button { background-color: #4f46e5; color: white; border: none; padding: 12px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; margin-top: 15px; transition: background 0.2s; }
            button:hover { background-color: #4338ca; }
            button:disabled { background-color: #9ca3af; cursor: not-allowed; }
            #loading { margin-top: 15px; color: #d97706; font-weight: bold; display: none; }
            .result-box { margin-top: 25px; border-top: 2px solid #e5e7eb; padding-top: 20px; display: none; }
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

            <textarea id="inputText" placeholder="请输入至少 50 个字符的文章进行测试...
例如：
The advent of electric vehicles has been touted as a cornerstone in the transition towards sustainable transportation. With global efforts to reduce carbon emissions and mitigate climate change, EVs have gained substantial attention and investment. This paper aims to dissect the multifaceted nature of EVs, weighing their advantages against the inherent limitations."></textarea>

            <button onclick="runScan()" id="scanBtn">发起 API 检测请求</button>
            <div id="loading">⏳ 正在连线大模型进行分析，请稍候...</div>

            <div id="resultBox" class="result-box">
                <h3>📦 返回的 JSON 结构：</h3>
                <pre id="jsonOutput"></pre>
            </div>
        </div>

        <script>
            // token 存入 sessionStorage，刷新页面不需重填
            window.addEventListener('DOMContentLoaded', () => {
                const saved = sessionStorage.getItem('api_token');
                if (saved) { document.getElementById('apiToken').value = saved; updateTokenStatus(); }
            });

            function updateTokenStatus() {
                const val = document.getElementById('apiToken').value.trim();
                sessionStorage.setItem('api_token', val);
                document.getElementById('tokenStatus').textContent = val ? '✅ 已填写' : '⬜ 未填写';
            }

            async function runScan() {
                const token = document.getElementById('apiToken').value.trim();
                const text  = document.getElementById('inputText').value.trim();
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

                try {
                    const response = await fetch('/api/scan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': 'Bearer ' + token
                        },
                        body: JSON.stringify({ content: text })
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
    return HTMLResponse(content=html_content)
