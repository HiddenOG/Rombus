# Rhombus: a data science agent you can check

A notebook where you do data science by talking to an AI, and where **nothing the AI does is hidden
or unchecked**. The AI suggests steps; you run them. Every step becomes a notebook cell with real,
editable Python behind it. The data is protected from the most common way machine-learning results
lie to you: leakage from the test set.

Built with FastAPI, scikit-learn, pandas, React and Supabase. Works with free AI providers or a small
local model.

---

## The problem

AI agents can now clean data and train models in seconds. That speed hides three ways the result
can be wrong while looking right:

| How results lie | What usually happens | What this project does |
|---|---|---|
| **Test-set leakage** | The data is cleaned *before* it's split, so the "unseen" test rows have already shaped the cleaning (the fill-in averages, the scaling, the category lists). Scores come out too good. | **The test rows are locked away first**, before any statistic is shown. Every cleaning step learns only from the training rows. |
| **Peeking at the test** | You check the test score, tweak, check again... until the test set has quietly become part of training. | **A final exam** that retrains on all training rows and scores once on the locked rows. Every extra look is counted and flagged. Each score is shown next to a "just guess" baseline. |
| **Unverifiable AI steps** | The agent says it "handled missing values". How? On which rows? | **Every step is a cell with its code.** Read it, edit it, rerun it, or export the whole notebook as one Python script that reproduces it exactly. |

## How it works

### 1. Split first, in strict mode
When you choose the column to predict, a share of the rows (5–50%, your choice, default 20%) is
locked away as the test set. Before that, the app shows only a preview and the column names: no
statistics, no charts.

### 2. The recipe: learn on training rows, apply everywhere
Each cleaning step works in two moments, like writing a sticky note:

```
learn(training rows)  -> remembers what it needs   e.g. "fill Age blanks with 28.0"
apply(any rows, note) -> uses only the note        on training rows, test rows, and new customers
```

So the test rows and future predictions get *exactly* the training rows' treatment, never their own.
Steps that only make sense for training (dropping duplicate rows) are marked as such and skipped
for test and new rows. Steps have stable IDs and can be edited, reordered or deleted. The recipe then
replays from scratch, and refuses a change that would break a later step.

### 3. A notebook of conversations
Like Jupyter, but each cell is a question to the AI, its answer, the action, and the result.
Cells can be edited (settings, words, or code), copied, rerun, reordered and deleted. Kernel
commands work too: restart, run all, clear outputs.

### 4. Honest scoring
- Practice scores come from cross-validation on the training rows only.
- The final exam reports the score, the baseline ("always guess the most common answer", or
  "always guess the average"), how many times the test rows have been looked at, and a warning
  when the model is no better than guessing.

### 5. Take it with you
Export the cleaned data (CSV/Excel), the trained model (`.pkl`, bundled with the recipe), or the
**whole pipeline as a runnable Python script**. A test checks that the exported script reproduces
the app's cleaned training and test rows exactly, custom code steps included.

## Case study: a public house-price notebook, rebuilt honestly

A widely shared Bengaluru house-price notebook (13,320 listings) cleans the data, removes price
outliers, adds features, then splits and trains a Random Forest. I rebuilt every step in this app and
scored the same model three ways on the same data:

| How | Test R² |
|---|---|
| The notebook's way: `price_per_sqft` (price ÷ area) used as a feature | **0.98** |
| The notebook's way without that feature: outliers still removed from *all* rows before the split | 0.79 |
| This app: split first, outliers removed from training rows only, honest final exam | **0.55** |

- **0.98 is leakage.** `price_per_sqft` is built from the price itself, so the model is handed the
  answer. It can't exist for a new house, whose price is unknown. The app refuses to build a feature
  from the target.
- **0.79 is still flattered.** Removing price outliers *before* splitting cleans the test set too, so
  the model is only examined on "typical" houses. In the app, outlier rules apply to training rows
  only; the locked test rows stay as real listings come.
- **0.55 is what the model really does** on listings it has never seen, extremes included.

The cleaning itself is the same in all three (ranges like "2100 - 2850" averaged, rare locations
grouped into "other", per-location outlier rules). Only *when* and *on which rows* it happens differs.

## Does the AI give good advice? Measured, not assumed

`backend/evals/copilot_eval.py` asks the copilot to suggest the next step across 9 situations
(messy columns, leakage traps, finished cleaning, and so on). It then checks each suggestion: is it
correct, and does it actually run?

| Model | Correct | Runs | Median reply |
|---|---|---|---|
| Qwen 3.8 27B (Groq, free tier) | 93% | 100% | 2.7 s |
| Qwen 3.5 2B (local, Ollama, laptop CPU) | 94% | 100% | ~90 s |
| Nemotron 3.5 Lightning (OpenRouter, free) | 67% | not measured | 11 s |

A chain of providers (`AI_CHAIN=groq,gemini,openrouter,ollama`) falls back automatically when one
is rate-limited or down, and each reply says which model answered. The AI is never offered tools
that run arbitrary code, and its view of the data respects the test-row lock.

## Architecture

```
frontend/  React 19 + Vite + Tailwind        backend/  FastAPI + pandas + scikit-learn
  notebook of cells, code editor,              app/recipe.py         learn/apply steps, replay
  exports, sign-in (Supabase Auth)             app/session.py        lock, recipe, notebook state
                │                              app/tools/            ~30 tools (views, actions, steps)
                │  REST + login token          app/copilot*.py       prompt, provider chain, failures
                └──────────────────────────▶   app/access.py         owners, rate limits
                                               app/storage.py        Postgres/SQLite + Supabase Storage/disk
                                               app/export.py         cleaned data, pipeline script
Supabase: Auth (accounts), Postgres (notebook records), Storage (private bucket for data files)
```

## Security model

- **Accounts:** Supabase Auth. The backend verifies login tokens against Supabase's published public
  key, so no auth secret lives on the server. Signed-out visitors get a random per-browser ID, and
  their notebooks move into their account when they sign in.
- **Isolation:** every notebook has an owner. Anyone else gets "not found", never "forbidden",
  so the server doesn't reveal that a notebook exists. The database table has Row Level Security
  switched on, so the public browser key can't read it. Data files live in a private bucket.
- **Custom code:** users can write their own steps, views and models, like a Jupyter cell. That code
  runs with the server's permissions, so it is **switched off** (`ALLOW_CUSTOM_CODE=false`) on any
  shared deployment. The check sits where code runs, not in each route.
- **Abuse limits:** per-user and per-network rate limits for AI, uploads and training. A daily AI
  budget. Request size caps. A cap on one-hot encoding (an ID column could otherwise add thousands
  of columns). Only the most recently used notebooks stay in memory.
- **Responsiveness:** training and replays run on worker threads, one request at a time per
  notebook, so one person's AutoML doesn't freeze the server for everyone.
- The frontend is served with a Content Security Policy and other security headers.

## Run it locally

You'll need Python 3.12 and Node 20.19+ (or 22.12+).

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on Mac/Linux)
pip install -r requirements-dev.txt
cp .env.example .env             # add at least one AI key, e.g. GROQ_API_KEY (free)
uvicorn app:app --reload

# Frontend (second terminal)
cd frontend
npm install
cp .env.example .env             # optional: Supabase URL + publishable key for sign-in
npm run dev                      # http://localhost:5173
```

Without Supabase, notebooks are stored on your machine (SQLite and Parquet files in
`backend/data`), and sign-in is hidden.

**Tests:** `cd backend && python -m pytest`. That's 264 tests, and none of them call a real AI or Supabase.
**Everything in containers:** `docker compose up --build`, then open http://localhost:3000.

## Deploy (Railway + Supabase)

The repo deploys as two Railway services, each built from its own Dockerfile.

1. **Supabase:** create a project, then copy these into the Railway variables below:
   - the Project URL and publishable key;
   - the secret key;
   - the Session pooler connection string.
2. **Railway → New project → Deploy from GitHub repo**, then add two services from it:

   | Service | Root directory | Config file | Variables |
   |---|---|---|---|
   | backend | `backend` | `/backend/railway.json` | AI keys + `AI_CHAIN`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `DATABASE_URL`, `ALLOW_CUSTOM_CODE=false`, `SESSION_TTL_HOURS=72`, `CORS_ORIGINS=https://<frontend domain>` |
   | frontend | `frontend` | `/frontend/railway.json` | `VITE_API_URL=https://<backend domain>`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY` |

   Generate a public domain for each service. Then set `CORS_ORIGINS` and `VITE_API_URL` to the
   other service's domain and redeploy. (Railway's config file path doesn't follow the root
   directory, hence the full paths.)
3. **Supabase → Authentication → URL Configuration:** set the Site URL to the frontend domain.

The backend runs **one worker on purpose**: rate limits, the per-notebook queue and the notebook
cache live in the process. To scale up, use a bigger machine rather than more workers.

GitHub Actions runs the tests, lint, both Docker builds, a container start-up check and a
dependency audit on every push.

## Configuration

All settings, with explanations: [`backend/.env.example`](backend/.env.example) and
[`frontend/.env.example`](frontend/.env.example).

## License

MIT
