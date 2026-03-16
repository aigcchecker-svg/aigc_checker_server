import logging
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from services.checker import API_SOURCE, run_check, run_reduce

logger = logging.getLogger(__name__)

router = APIRouter()

_security = HTTPBearer()
_API_TOKEN = os.getenv("API_TOKEN")
_FREE_API_TOKENS = {token.strip() for token in os.getenv("FREE_API_TOKENS", "").split(",") if token.strip()}
_PRO_API_TOKENS = {token.strip() for token in os.getenv("PRO_API_TOKENS", "").split(",") if token.strip()}
_FREE_MAX_CHARS = int(os.getenv("FREE_MAX_CHARS", "3000"))
_PRO_MAX_CHARS = int(os.getenv("PRO_MAX_CHARS", "12000"))


def _token_context(token: str) -> dict:
    if token in _PRO_API_TOKENS:
        return {
            "plan": "pro",
            "can_reduce": True,
            "can_remote_review": True,
            "max_chars": _PRO_MAX_CHARS,
            "client_id": f"pro:{token[-6:]}",
        }
    if token in _FREE_API_TOKENS:
        return {
            "plan": "free",
            "can_reduce": False,
            "can_remote_review": False,
            "max_chars": _FREE_MAX_CHARS,
            "client_id": f"free:{token[-6:]}",
        }
    if _API_TOKEN and token == _API_TOKEN:
        return {
            "plan": "pro",
            "can_reduce": True,
            "can_remote_review": True,
            "max_chars": _PRO_MAX_CHARS,
            "client_id": "legacy:default",
        }
    raise HTTPException(status_code=401, detail="Token 无效，拒绝访问")


def _verify_token(credentials: HTTPAuthorizationCredentials = Security(_security)) -> dict:
    if not (_API_TOKEN or _FREE_API_TOKENS or _PRO_API_TOKENS):
        raise HTTPException(status_code=500, detail="服务端未配置 API Token")
    return _token_context(credentials.credentials)


class ScanRequest(BaseModel):
    content: str
    model: Optional[str] = None
    api_source: str = API_SOURCE


def _log(tag: str, request: ScanRequest, text_len: int) -> None:
    logger.info("[%s] model=%s api_source=%s chars=%d", tag, request.model or "default", request.api_source, text_len)


@router.post("/scan")
async def scan_document(request: ScanRequest, auth_context: dict = Security(_verify_token)):
    text = request.content.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="文本太短，无法进行有效分析，请至少输入50个字符。")
    if len(text) > auth_context["max_chars"]:
        raise HTTPException(
            status_code=400,
            detail=f"当前套餐最多支持 {auth_context['max_chars']} 个字符，当前文本过长。",
        )
    try:
        _log("scan", request, len(text))
        return await run_check(
            content=text,
            model=request.model,
            api_source=request.api_source,
            plan=auth_context["plan"],
            can_remote_review=auth_context["can_remote_review"],
        )
    except ValueError as exc:
        logger.warning("scan parse error: %s", exc)
        raise HTTPException(status_code=502, detail="底层引擎返回了无法解析的格式，请重试。")
    except Exception as exc:
        logger.exception("scan failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"检测引擎服务异常: {str(exc)}")


@router.post("/reduce")
async def reduce_document(request: ScanRequest, auth_context: dict = Security(_verify_token)):
    text = request.content.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="文本太短，无法进行有效处理，请至少输入50个字符。")
    if not auth_context["can_reduce"]:
        raise HTTPException(status_code=403, detail="当前套餐不支持降 AI 改写，请升级会员版。")
    if len(text) > auth_context["max_chars"]:
        raise HTTPException(
            status_code=400,
            detail=f"当前套餐最多支持 {auth_context['max_chars']} 个字符，当前文本过长。",
        )
    try:
        _log("reduce", request, len(text))
        return await run_reduce(
            content=text,
            model=request.model,
            api_source=request.api_source,
            plan=auth_context["plan"],
            can_remote_review=auth_context["can_remote_review"],
        )
    except ValueError as exc:
        logger.warning("reduce parse error: %s", exc)
        raise HTTPException(status_code=502, detail="底层引擎返回了无法解析的格式，请重试。")
    except Exception as exc:
        logger.exception("reduce failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"降 AI 引擎服务异常: {str(exc)}")
