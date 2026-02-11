"""
Refactored FastAPI Application - Now config-driven and domain-agnostic.

This module handles app creation, middleware, and state initialization.
All endpoint handlers live in app/routes.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables FIRST before any other imports that might use them
load_dotenv()

# Import from app structure
from app.config_loader import load_config, get_domain_instance
from app.core import DataLoader, GeminiInterpreter, configure
from app.routes import router

# ── Configuration ────────────────────────────────────────────────────────────

config = load_config()
domain = get_domain_instance(config)

# Apply ML parameters from YAML to analysis module
configure(
    random_forest=config.get('random_forest'),
    pacmap_params=config.get('pacmap'),
)

# ── App creation ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=f"{domain.name} Dashboard API",
    description="Backend API for Tensor Data Visualization Dashboard",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('cors_origins', ["http://localhost:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (accessed by routes via request.app.state) ──────────────────

app.state.domain = domain
app.state.data_loader = DataLoader(
    str(Path(__file__).parent.parent / domain.data_dir), domain
)
app.state.ai_interpreter = GeminiInterpreter(domain)
app.state.tulca_model = None  # lazy-initialized on first request
app.state.tulca_params = config.get('tulca', {
    's_prime': 10,
    'v_prime': 3,
    'optimization_method': 'evd',
})
app.state.colors = config.get('colors', {
    'class_colors': ["#F58518", "#54A24B", "#B279A2"],
    'cluster1': '#C0392B',
    'cluster2': '#2874A6',
})

# ── Register routes ──────────────────────────────────────────────────────────

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
