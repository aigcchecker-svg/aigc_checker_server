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
    """根据 token 值返回对应的权限上下文（套餐、字数上限、功能开关等）。

    优先匹配 PRO 套餐 Token，其次是 FREE 套餐 Token，最后兼容旧版单一 API_TOKEN。
    未匹配到任何有效 Token 时抛出 401 异常。
    """
    if token in _PRO_API_TOKENS:
        # Pro 套餐：支持降 AI 改写和远端二审，字数上限更高
        return {
            "plan": "pro",
            "can_reduce": True,
            "can_remote_review": True,
            "max_chars": _PRO_MAX_CHARS,
            "client_id": f"pro:{token[-6:]}",
        }
    if token in _FREE_API_TOKENS:
        # Free 套餐：不支持改写和二审，字数有限制
        return {
            "plan": "free",
            "can_reduce": False,
            "can_remote_review": False,
            "max_chars": _FREE_MAX_CHARS,
            "client_id": f"free:{token[-6:]}",
        }
    if _API_TOKEN and token == _API_TOKEN:
        # 旧版单一 Token，向下兼容，权限等同 Pro
        return {
            "plan": "pro",
            "can_reduce": True,
            "can_remote_review": True,
            "max_chars": _PRO_MAX_CHARS,
            "client_id": "legacy:default",
        }
    raise HTTPException(status_code=401, detail="Token 无效，拒绝访问")


def _verify_token(credentials: HTTPAuthorizationCredentials = Security(_security)) -> dict:
    """FastAPI 安全依赖：从请求头 Bearer Token 中提取并校验权限上下文。

    若服务端未配置任何 Token 则返回 500，Token 无效则返回 401。
    """
    if not (_API_TOKEN or _FREE_API_TOKENS or _PRO_API_TOKENS):
        raise HTTPException(status_code=500, detail="服务端未配置 API Token")
    return _token_context(credentials.credentials)


class ScanRequest(BaseModel):
    content: str
    model: Optional[str] = None
    api_source: str = API_SOURCE


def _log(tag: str, request: ScanRequest, text_len: int) -> None:
    """记录请求日志，包含操作标签、模型名、API 来源和文本长度，便于问题排查。"""
    logger.info("[%s] model=%s api_source=%s chars=%d", tag, request.model or "default", request.api_source, text_len)


@router.post("/scan")
async def scan_document(request: ScanRequest, auth_context: dict = Security(_verify_token)):
    """AI 内容检测接口，对输入文本进行多维度分析并返回 AI 概率及详细报告。

    - 文本长度需在 50 ~ max_chars 之间，否则返回 400
    - ValueError 对应引擎返回格式异常，统一映射为 502
    """
    text = request.content.strip()
    # 最小长度限制：过短的文本无法提取有效特征
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="文本太短，无法进行有效分析，请至少输入50个字符。")
    # 最大长度限制：由套餐权限决定
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
    """降 AI 改写接口，对高 AI 概率文本进行改写并返回改写前后的概率对比。

    - 仅 Pro 套餐可用（can_reduce=True），Free 套餐返回 403
    - 文本长度限制同 /scan 接口
    """
    text = request.content.strip()
    if len(text) < 50:
        raise HTTPException(status_code=400, detail="文本太短，无法进行有效处理，请至少输入50个字符。")
    # 功能权限校验：Free 套餐不允许使用降 AI 改写
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
