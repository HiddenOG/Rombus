# AutoML Platform — AI-Powered Data Science Copilot

An interactive, AI-powered AutoML platform that guides users through the complete machine learning lifecycle using a conversational interface. Powered by **Google Gemini** for intelligent reasoning and **scikit-learn** for model training.

## Features

- **ASEMIC Workflow** — Guided pipeline: Acquire → Scrub → Explore → Model → Interpret → Communicate
- **AI Copilot** — Gemini-powered chat that suggests data science actions contextually
- **20+ ML Tools** — Data cleaning, encoding, scaling, EDA, model training, prediction
- **AutoML Mode** — One-click best-model search with hyperparameter tuning
- **Interactive Charts** — Histograms, boxplots, scatterplots, correlation heatmaps
- **Session Persistence** — SQLite-backed sessions survive server restarts
- **Undo/Rollback** — Revert any data transformation step
- **SHAP Explanations** — Model interpretability for the Interpret stage

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google Gemini API key

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
cp .env.example .env         # Edit with your API key
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker (Production)
```bash
docker-compose up --build
```
App available at `http://localhost:3000`, API at `http://localhost:8000`.

## Architecture

```
automl-platform/
├── backend/
│   ├── app/                  # Application package
│   │   ├── config.py         # Environment settings
│   │   ├── session.py        # Session management + persistence
│   │   ├── auth.py           # API key authentication
│   │   ├── gemini.py         # Gemini AI client
│   │   ├── models.py         # Pydantic schemas
│   │   ├── tools/            # ML tool implementations
│   │   └── routers/          # API route handlers
│   ├── tests/                # pytest test suite
│   ├── main.py               # Entry point
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/            # Custom hooks
│   │   ├── lib/              # API client
│   │   └── App.jsx
│   └── package.json
├── docker-compose.yml
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload CSV/Excel dataset |
| POST | `/api/chat` | Chat with AI copilot |
| POST | `/api/action` | Execute a data science tool |
| GET | `/api/session/{id}` | Recover session state |

## Environment Variables

See [`.env.example`](backend/.env.example) for all configurable options.

## License

MIT
