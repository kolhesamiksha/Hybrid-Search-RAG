from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_oauth2_redirect_html

from .api import rag_router


def includer_router(app) -> None:
    app.include_router(rag_router)


def start_application() -> FastAPI:
    app = FastAPI(title="API", version="0.0.1", description="API")
    includer_router(app)
    return app


app = start_application()


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect() -> get_swagger_ui_oauth2_redirect_html:
    return get_swagger_ui_oauth2_redirect_html()
