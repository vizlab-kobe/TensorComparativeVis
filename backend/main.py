"""
FastAPIアプリケーション エントリーポイント

設定駆動・ドメイン非依存のアーキテクチャで構成されている。
このモジュールは以下を担当する:
  - YAML設定ファイルの読み込みとドメイン戦略の生成
  - FastAPIアプリケーションの作成とCORSミドルウェアの設定
  - 共有状態（ドメイン、データローダー、AIインタープリター等）の初期化
  - ルーターの登録
全エンドポイントハンドラは app/routes.py に定義されている。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 環境変数を最初に読み込む（他のインポートが参照する可能性があるため）
load_dotenv()

# アプリケーション構造からのインポート
from app.config_loader import load_config, get_domain_instance
from app.core import DataLoader, GeminiInterpreter, configure
from app.routes import router

# ── 設定読み込み ─────────────────────────────────────────────────────────────

# YAML設定ファイルからアプリケーション設定を読み込む
config = load_config()

# 設定に基づいてドメイン戦略インスタンスを生成（HPC / AirData 等）
domain = get_domain_instance(config)

# YAMLの機械学習パラメータを分析モジュールに適用
configure(
    random_forest=config.get('random_forest'),
    pacmap_params=config.get('pacmap'),
)

# ── アプリケーション生成 ──────────────────────────────────────────────────────

app = FastAPI(
    title=f"{domain.name} Dashboard API",
    description="テンソルデータ比較可視化ダッシュボードのバックエンドAPI",
    version="2.0.0",
)

# CORSミドルウェア設定（フロントエンドからのクロスオリジンリクエストを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('cors_origins', ["http://localhost:5173"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 共有状態の初期化（各ルートから request.app.state でアクセス） ─────────────

# ドメイン戦略インスタンス（変数名、ラベル変換、プロンプト生成等を提供）
app.state.domain = domain

# データローダー（テンソルデータの遅延読み込みを担当）
app.state.data_loader = DataLoader(
    str(Path(__file__).parent.parent / domain.data_dir), domain
)

# Gemini APIを使用したAI解釈エンジン
app.state.ai_interpreter = GeminiInterpreter(domain)

# TULCAモデル（初回リクエスト時に遅延初期化される）
app.state.tulca_model = None

# TULCAハイパーパラメータ（YAMLから取得、デフォルト値あり）
app.state.tulca_params = config.get('tulca', {
    's_prime': 10,
    'v_prime': 3,
    'optimization_method': 'evd',
})

# UI表示用の色設定
app.state.colors = config.get('colors', {
    'class_colors': ["#F58518", "#54A24B", "#B279A2"],
    'cluster1': '#C0392B',
    'cluster2': '#2874A6',
})

# ── ルーター登録 ──────────────────────────────────────────────────────────────

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
