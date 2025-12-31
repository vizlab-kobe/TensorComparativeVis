# TensorComparativeVis

TensorComparativeVis is a human-in-the-loop visual analytics framework for exploratory comparative analysis of multivariate time-series data represented as high-order tensors. The system enables analysts to discover and interpret nonlinear patterns that may not correspond to predefined group labels.

## Key Features

- **Exploratory Comparison**: Interactively select arbitrary clusters from 2D scatterplots without being constrained by predefined group labels, enabling discovery of localized temporal anomalies and unexpected patterns.

- **Tensor-Based Analysis**: Combines TULCA (Tensor Unified Linear Comparative Analysis) for flexible core tensor extraction with PaCMAP for nonlinear dimensionality reduction, preserving meaningful latent structures in visualization.

- **Quantitative Interpretation**: Identifies influential combinations of temporal, spatial, and variable elements through back-projected feature importance analysis with statistical validation (Welch's t-test, Cohen's d effect sizes).

- **LLM-Based Explanation**: Generates natural language interpretations using large language models, synthesizing quantitative results with domain knowledge to reduce cognitive load during iterative exploration.

- **Comparative Analysis Support**: Compares multiple analysis results across different cluster selections, helping analysts connect exploratory comparisons into coherent investigative narratives.

## Workflow

The framework operates in two iterative phases:

1. **Latent Pattern Extraction and Visualization**: Labeled tensor data is processed through TULCA core tensor extraction, nonlinear dimensionality reduction via PaCMAP, and 2D scatterplot visualization.

2. **Exploratory Comparison and Interpretation**: Analysts interactively select clusters, compute feature importance via random forest classification, back-project importance scores to the original data space, validate differences through statistical tests, and generate natural language interpretations.

## Tech Stack

- **Frontend**: React 19 + TypeScript, Chakra UI, D3.js, Zustand
- **Backend**: FastAPI + Python 3.9+
- **AI**: Google Gemini API

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+

### Data Setup

Place tensor data files in `data/processed/`:

```
data/processed/
├── tensor_X.npy      # Standardized tensor (T, S, V)
├── tensor_y.npy      # Class labels
├── time_axis.npy     # Timestamps
└── time_original.npy # Original values
```

### Backend Setup

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key (optional, for LLM features)
# Create .env file with: GEMINI_API_KEY=your_key_here

# Run the server
uvicorn main:app --reload --port 8000
```

API: `http://localhost:8000` | Docs: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

App: `http://localhost:5173`

## Project Structure

```
TensorComparativeVis/
├── backend/
│   ├── main.py           # FastAPI endpoints
│   ├── tulca.py          # TULCA algorithm
│   ├── analysis.py       # Feature importance analysis
│   ├── interpreter.py    # LLM-based interpretation
│   ├── data_loader.py    # Data loading utilities
│   └── models.py         # Pydantic schemas
│
├── frontend/src/
│   ├── App.tsx           # Main application
│   ├── components/       # React components
│   │   ├── ScatterPlot   # 2D embedding with lasso selection
│   │   ├── FeatureRanking# Feature importance visualization
│   │   ├── Heatmap       # Contribution heatmap
│   │   ├── TimeSeriesPlot# Time series comparison
│   │   └── AIInterpretation # LLM summary panel
│   ├── store/            # Zustand state management
│   └── api/              # API client
│
└── data/                 # Tensor data (gitignored)
```

## Usage

1. **Configure Weights**: Adjust TULCA weight parameters (w_tg, w_bw, w_bg) in sidebar
2. **Execute Analysis**: Click "Execute Analysis" to compute embedding
3. **Select Clusters**: Use lasso tool to select Red (Cluster 1) and Blue (Cluster 2) regions
4. **View Results**: Analysis runs automatically when both clusters are selected
5. **Save Analysis**: Click "Save" to store current analysis for later comparison
6. **Compare**: Select 2 saved analyses and click "Compare" for LLM-generated comparison

## License

BSD-3-Clause
