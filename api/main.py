from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db import Base, engine
from api.routers import health, conversations, analyze, advisor, screenshot

app = FastAPI(title="ATBG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(conversations.router)
app.include_router(analyze.router)
app.include_router(advisor.router)
app.include_router(screenshot.router)


@app.on_event("startup")
def on_startup():
    # Dev convenience only — in a real deployment, migrations own this.
    Base.metadata.create_all(bind=engine)
