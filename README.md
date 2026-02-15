# ANES 2024 Ideological Clusters

> **An interactive exploration of the political landscape behind Trump vs Harris (2024)**

This project uses K-means clustering on ~50 policy attitude questions from the [ANES 2024 Time Series Study](https://electionstudies.org/) to identify 15 distinct ideological clusters among likely voters. Explore them in 3D, take a quiz, and chat with AI-powered personas.

---

## Features

- **3D & 2D Visualizations**: Explore clusters in interactive Plotly plots with customizable axes
- **10-Question Quiz**: Discover which ideological cluster you belong to
- **Story Mode**: Guided narrative explaining the findings
- **Persona Chat**: Interact with AI personas representing each cluster (requires API key)
- **LLM Validation Report**: Per-question accuracy of LLM crime policy predictions on 200 individuals

---

## Repository Structure

```
.
├── docs/                           # Static website (deploy this)
│   ├── index.html
│   ├── assets/
│   │   ├── app.js
│   │   └── styles.css
│   ├── data/                       # All JSON data artifacts
│   └── docs/                       # Validation report, technical note
├── api/
│   └── chat.js                     # Vercel serverless function (LLM chat)
├── helpers/
│   ├── preprocessing_v2.py         # Data cleaning and feature definitions
│   ├── persona_generator.py        # LLM-based persona generation
│   ├── llm_validation_individuals.py  # Individual-level validation logic
│   ├── feature_selection.py        # Feature importance analysis
│   └── ml_inference.py             # ML holdout predictions
├── build_site_data.py              # Main pipeline: clustering → JSON artifacts
├── run_individual_llm_validation.py   # LLM validation experiment (200 individuals)
├── generate_individual_validation_report.py  # HTML report generator
├── generate_question_mappings.py   # Variable-to-label mappings
├── server.py                       # Flask dev server (local testing with chat)
├── docs/technical_note.md          # Full methodology documentation
├── vercel.json                     # Vercel deployment config
├── render.yaml                     # Render deployment config
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- ANES 2024 Time Series data CSV (download from [electionstudies.org](https://electionstudies.org/data-center/))

### Setup

```bash
git clone https://github.com/guillelezama/anes-2024-personas.git
cd anes-2024-personas

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

Place the ANES CSV in `anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv`.

### Run the Pipeline

```bash
# Build all site data (clustering, personas, quiz, etc.)
python build_site_data.py --universe likely_voters

# Optional: generate with LLM personas
export OPENAI_API_KEY="your-key-here"
python build_site_data.py --universe likely_voters --use-llm
```

### Preview Locally

**Static site only** (no LLM chat):
```bash
python -m http.server 8080 --directory docs
# Open http://localhost:8080
```

**With LLM persona chat** (requires Flask + API key):
```bash
pip install flask flask-cors
export OPENAI_API_KEY="your-key-here"
python server.py
# Open http://localhost:5000
```

---

## Deployment

### Option A: GitHub Pages (static site, no chat)

1. Push the repo to GitHub
2. Go to Settings → Pages → Source: Deploy from branch `main`, folder `/docs`
3. Site will be live at `https://yourusername.github.io/repo-name/`

### Option B: Vercel (static site + LLM chat)

1. Connect your GitHub repo to [Vercel](https://vercel.com)
2. Set environment variables in Vercel dashboard:
   - `OPENAI_API_KEY` — your OpenAI key
   - `ANTHROPIC_API_KEY` — (optional) Anthropic key
3. Deploy — Vercel uses `vercel.json` to serve `docs/` as static and `api/chat.js` as a serverless function

### Option C: Render (Flask backend + chat)

1. Connect repo to [Render](https://render.com)
2. Set environment variables (`OPENAI_API_KEY`, etc.)
3. Render uses `render.yaml` to deploy `server.py` with Gunicorn

---

## LLM Validation

We tested whether GPT-4o-mini can predict individual respondents' crime policy positions from their other 46 policy positions. See the [full validation report](docs/docs/llm_validation_report.html) and [technical note](docs/technical_note.md) for details.

To re-run:
```bash
export OPENAI_API_KEY="your-key-here"
python run_individual_llm_validation.py
python generate_individual_validation_report.py
```

---

## Data Privacy

- **Do NOT publish raw ANES microdata** — the CSV is gitignored
- Only aggregated cluster-level statistics are included in `docs/data/`
- Personas are **simulated and fictional**, not real individuals

## Technical Documentation

See [docs/technical_note.md](docs/technical_note.md) for the full methodology: data universe, feature selection, variance weighting, K selection, stability analysis, ML inference, and LLM validation.

## License

- Code: MIT License
- ANES data subject to [ANES Terms of Use](https://electionstudies.org/data-center/)

## Citation

```
@misc{lezama2025anes,
  author = {Guillermo Lezama},
  title = {ANES 2024 Ideological Clusters},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/guillelezama/anes-2024-personas}
}
```

Data source:
```
American National Election Studies. 2024. ANES 2024 Time Series Study.
www.electionstudies.org
```

## Contact

Guillermo Lezama — [guillelezama.com](https://guillelezama.com) | [LinkedIn](https://linkedin.com/in/guillelezama) | [GitHub](https://github.com/guillelezama)

Questions or feedback? [Open an issue](https://github.com/guillelezama/anes-2024-personas/issues).
