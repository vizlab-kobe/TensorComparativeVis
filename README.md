# TensorComparativeVis

TensorComparativeVis は、高階テンソルとして表現された多変量時系列データの探索的比較分析を支援する、Human-in-the-Loop 型ビジュアルアナリティクスフレームワークです。事前定義されたグループラベルに依存せず、非線形パターンの発見と解釈を可能にします。

## 主な機能

- **探索的比較分析**: 2D 散布図上で任意のクラスタをインタラクティブに選択し、局所的な時間的異常や予期しないパターンを発見
- **テンソルベース分析**: TULCA（Tensor Unified Linear Comparative Analysis）による柔軟なコアテンソル抽出と PaCMAP による非線形次元削減を組み合わせ、潜在構造を保持した可視化を実現
- **定量的解釈**: Random Forest による特徴量重要度分析と逆投影を通じて、時間・空間・変数の影響度の高い組み合わせを特定。統計的検証（Welch の t 検定、Mann-Whitney U 検定、Cohen's d 効果量、Benjamini-Hochberg FDR 補正）付き
- **LLM による自然言語解釈**: Google Gemini API を活用し、定量的結果とドメイン知識を統合した自然言語での解釈を自動生成
- **比較分析サポート**: 複数の分析結果を保存・比較し、探索的比較を一貫した調査ストーリーに統合
- **ドメイン非依存アーキテクチャ**: YAML 設定ファイルとドメイン戦略パターンにより、異なるデータドメイン（HPC、大気データ等）に柔軟に対応
- **時系列詳細モーダル**: 特徴量ランキングから直接時系列データの詳細を確認可能なインタラクティブモーダル

## ワークフロー

本フレームワークは以下の2つの反復的フェーズで動作します：

1. **潜在パターン抽出と可視化**: ラベル付きテンソルデータに対し、TULCA によるコアテンソル抽出 → PaCMAP による非線形次元削減 → 2D 散布図への可視化を実行
2. **探索的比較と解釈**: 散布図上でクラスタを対話的に選択 → Random Forest による特徴量重要度の算出 → 元データ空間への逆投影 → 統計的検定による差異の検証 → LLM による自然言語解釈の生成

## 技術スタック

| レイヤー | 技術 |
|---------|------|
| **フロントエンド** | React 19 + TypeScript, Chakra UI, D3.js, Zustand |
| **バックエンド** | FastAPI + Python 3.9+ |
| **AI 解釈** | Google Gemini API |
| **分析** | TULCA, PaCMAP, Random Forest, SciPy (統計検定) |

## クイックスタート

### 前提条件

- Node.js 18+ および npm
- Python 3.9+
- Google Gemini API キー（LLM 機能を使用する場合）

### データの配置

テンソルデータを `data/processed/` に配置します：

```
data/processed/
├── tensor_X.npy      # 標準化テンソル (T, S, V)
├── tensor_y.npy      # クラスラベル
├── time_axis.npy     # タイムスタンプ
└── time_original.npy # 元データ値
```

### バックエンドのセットアップ

```bash
cd backend

# 仮想環境の作成（推奨）
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 依存パッケージのインストール
pip install -r requirements.txt

# Gemini API キーの設定（LLM 機能を使用する場合）
# .env ファイルを作成し、以下を記載: GEMINI_API_KEY=your_key_here

# ドメイン設定の指定（環境変数で切り替え可能、デフォルト: hpc_default）
# set APP_CONFIG=air_data  # Windows
# export APP_CONFIG=air_data  # macOS/Linux

# サーバーの起動
uvicorn main:app --reload --port 8000
```

- API: `http://localhost:8000`
- Swagger UI ドキュメント: `http://localhost:8000/docs`

### フロントエンドのセットアップ

```bash
cd frontend
npm install
npm run dev
```

- アプリケーション: `http://localhost:5173`

## プロジェクト構成

```
TensorComparativeVis/
├── configs/                        # ドメイン別 YAML 設定ファイル
│   ├── hpc_default.yaml            #   HPC スーパーコンピュータデータ用
│   └── air_data.yaml               #   米国大気質データ用
│
├── backend/
│   ├── main.py                     # FastAPI エントリーポイント
│   ├── models.py                   # Pydantic スキーマ定義
│   └── app/
│       ├── config_loader.py        # YAML 設定読み込み・ドメイン生成
│       ├── routes.py               # 全 API エンドポイント
│       ├── core/                   # コアロジック
│       │   ├── tulca.py            #   TULCA アルゴリズム
│       │   ├── analysis.py         #   特徴量重要度・統計分析
│       │   ├── interpreter.py      #   Gemini API 連携 LLM 解釈
│       │   └── data_loader.py      #   テンソルデータ読み込み
│       └── domains/                # ドメイン戦略パターン
│           ├── base_domain.py      #   基底クラス（抽象インターフェース）
│           ├── hpc_domain.py       #   HPC ドメイン実装
│           └── air_data_domain.py  #   大気データドメイン実装
│
├── frontend/src/
│   ├── App.tsx                     # メインダッシュボードレイアウト
│   ├── theme.ts                    # Chakra UI テーマ設定
│   ├── components/
│   │   ├── ScatterPlot.tsx         #   2D 埋め込み＋投げ縄選択
│   │   ├── FeatureRanking.tsx      #   特徴量重要度ランキング
│   │   ├── Heatmap.tsx             #   寄与度ヒートマップ
│   │   ├── TimeSeriesPlot.tsx      #   時系列比較プロット
│   │   ├── TimeSeriesModal.tsx     #   時系列詳細モーダル
│   │   ├── AIInterpretation.tsx    #   LLM 解釈パネル
│   │   ├── Sidebar.tsx             #   パラメータ設定サイドバー
│   │   ├── GeoMapVis.tsx           #   地理的可視化
│   │   ├── SpatialVisualization.tsx#   空間可視化ラッパー
│   │   └── ScreenshotButton.tsx    #   スクリーンショット機能
│   ├── store/                      # Zustand 状態管理
│   ├── api/                        # API クライアント
│   └── types/                      # TypeScript 型定義
│
└── data/                           # テンソルデータ（gitignore 対象）
```

## 使い方

1. **パラメータの設定**: サイドバーで TULCA の重みパラメータ（w_tg, w_bw, w_bg）を調整
2. **分析の実行**: 「Execute Analysis」ボタンをクリックして埋め込みを計算
3. **クラスタの選択**: 投げ縄ツールで散布図上の Red（クラスタ1）と Blue（クラスタ2）領域を選択
4. **結果の確認**: 両クラスタが選択されると自動的に分析が実行され、特徴量ランキング・ヒートマップ・時系列比較・AI 解釈が表示
5. **分析の保存**: 「Save」ボタンで現在の分析結果を保存
6. **比較分析**: 保存した2つの分析を選択し「Compare」ボタンで LLM による比較解釈を生成

## ドメインの追加方法

新しいデータドメインを追加するには：

1. `backend/app/domains/` に `BaseDomain` を継承した新しいドメインクラスを作成
2. `configs/` に対応する YAML 設定ファイルを作成
3. 環境変数 `APP_CONFIG` でドメインを切り替え

```bash
# 例: 大気データドメインで起動
set APP_CONFIG=air_data  # Windows
uvicorn main:app --reload --port 8000
```

## ライセンス

BSD-3-Clause
