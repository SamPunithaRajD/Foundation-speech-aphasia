"""FoundationSpeech-Aphasia research backend (FastAPI)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import analysis, resources
from app.config import API_VERSION, FORMAL_PROJECT_TITLE, PROJECT_NAME, Settings, get_settings
from app.models.hardware import hardware_summary
from app.services.container import Services
from app.utils.errors import ApiError, ErrorCode

log = logging.getLogger("foundationspeech")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = Services.build(settings)
        services.analysis.startup_housekeeping()
        hardware_summary(settings.device)  # warm heavy imports (torch) so the first /api/models call is fast
        app.state.services = services
        log.info("%s backend ready (db=%s)", PROJECT_NAME, settings.sqlite_path.name)
        yield
        services.close()

    app = FastAPI(title=f"{PROJECT_NAME} API", description=FORMAL_PROJECT_TITLE, version=API_VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["*"], expose_headers=["Content-Disposition"])

    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
        return JSONResponse(status_code=422, content=ApiError(
            ErrorCode.INVALID_REQUEST, "Request validation failed.", 422, {"fields": fields}).to_dict())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        code = ErrorCode.ANALYSIS_NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST
        return JSONResponse(status_code=exc.status_code, content=ApiError(code, str(exc.detail), exc.status_code).to_dict())

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content=ApiError(ErrorCode.INTERNAL_ERROR, "Internal server error.", 500).to_dict())

    app.include_router(analysis.router)
    app.include_router(resources.router)
    _mount_frontend(app, settings)
    return app


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the built React app (`npm run build` -> dist/) from the same origin as the API.

    Registered after the API routers so /api/*, /docs and /openapi.json keep priority. Unknown /api paths
    still return the structured JSON 404, never the HTML shell.
    """
    dist = settings.frontend_dist
    index = dist / "index.html"
    if not settings.serve_frontend or not index.is_file():
        log.info("frontend not served (build missing at %s)", dist)
        return

    @app.api_route("/{path:path}", methods=["GET", "HEAD"], include_in_schema=False)
    async def frontend(path: str):
        if path == "api" or path.startswith("api/"):
            raise ApiError(ErrorCode.INVALID_REQUEST, "Unknown API endpoint.", 404)
        target = (dist / path).resolve()
        if path and target.is_file() and dist.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(index)  # SPA shell


app = create_app()

if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port)
