import json
import os
from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from services.checker import API_SOURCE, DEFAULT_MODEL, run_check, run_reduce

router = APIRouter()

_security = HTTPBearer()
_API_TOKEN = os.getenv("API_TOKEN")


def _verify_token(credentials: HTTPAuthorizationCredentials = Security(_security)):
    if not _API_TOKEN:
        raise HTTPException(status_code=500, detail="服务端未配置 API_TOKEN")
    if credentials.credentials != _API_TOKEN:
        raise HTTPException(status_code=401, detail="Token 无效，拒绝访问")


class ScanRequest(BaseModel):
    content: str
    model: str = DEFAULT_MODEL
    api_source: str = API_SOURCE


@router.post("/scan", dependencies=[Security(_verify_token)])
async def scan_document(request: ScanRequest):
    text = request.content.strip()

    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="文本太短，无法进行有效分析，请至少输入50个字符。",
        )

    try:
        result = await run_check(content=text, model=request.model, api_source=request.api_source)
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="底层引擎返回了无法解析的格式，请重试。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测引擎服务异常: {str(e)}")


@router.post("/reduce", dependencies=[Security(_verify_token)])
async def reduce_document(request: ScanRequest):
    text = request.content.strip()

    if len(text) < 50:
        raise HTTPException(
            status_code=400,
            detail="文本太短，无法进行有效处理，请至少输入50个字符。",
        )

    try:
        result = await run_reduce(content=text, model=request.model, api_source=request.api_source)
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="底层引擎返回了无法解析的格式，请重试。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"降 AI 引擎服务异常: {str(e)}")
