"""FastAPI gateway for external bot channels."""

import json
from typing import Dict

from hyperagent.runtime.channels.config import ChannelConfigStore
from hyperagent.runtime.channels.router import ChannelRouter
from hyperagent.runtime.platform_runtime import PlatformStatusReporter


def create_channel_app(router: ChannelRouter, config_store: ChannelConfigStore):
    """Create a FastAPI app lazily so non-channel commands do not require FastAPI."""

    try:
        from fastapi import FastAPI, HTTPException, Request
    except ImportError as exc:  # pragma: no cover - exercised by CLI smoke behavior.
        raise RuntimeError(
            "FastAPI is not installed. Install the HyperAgent environment dependencies "
            "or run `python -m pip install fastapi uvicorn` in the HyperAgent env."
        ) from exc

    app = FastAPI(
        title="HyperAgent Channel Gateway",
        version="0.1.0",
        description="Feishu and QQ official bot webhook gateway for HyperAgent.",
    )

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"status": "ok", "service": "hyperagent-channel-gateway"}

    @app.get("/channels")
    def channels() -> Dict[str, object]:
        configs = config_store.ensure_defaults()
        env_summary = config_store.env_summary()
        env_configured = config_store.env_configured_summary()
        return {
            "channels": [
                {
                    "provider": item.provider,
                    "enabled": item.enabled,
                    "display_name": item.display_name,
                    "default_llm_provider": item.default_llm_provider,
                    "default_model": item.default_model,
                    "default_mode": item.default_mode,
                    "env_vars": env_summary.get(item.provider, []),
                    "env_configured": env_configured.get(item.provider, {}),
                    "chat_query_only": True,
                }
                for item in configs
            ]
        }

    @app.get("/status")
    def status() -> Dict[str, object]:
        return PlatformStatusReporter(
            router.workspace,
            router.conversations,
            router.providers,
            channel_store=config_store,
        ).report()

    @app.post("/webhooks/feishu")
    async def feishu(request: Request):
        return await _handle("feishu", request)

    @app.post("/webhooks/qq")
    async def qq(request: Request):
        return await _handle("qq", request)

    async def _handle(provider: str, request: Request):
        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="request JSON root must be an object")
        result = router.handle_webhook(
            provider,
            payload,
            headers={str(key): str(value) for key, value in request.headers.items()},
            body=body,
        )
        if result.status == "error":
            raise HTTPException(
                status_code=401 if "signature" in result.error or "token" in result.error else 400,
                detail=result.to_dict(),
            )
        if result.status == "verified" and result.response_payload:
            return result.response_payload
        return result.to_dict()

    return app
