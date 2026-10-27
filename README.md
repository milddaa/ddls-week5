## Data-Driven Life Sciences course Week 5 project
This repository contains everything related to DDLS course's week 5 project on single-cell data analysis.

## Run the PBMC Explorer app

From the project directory:

1. Create and activate the environment if needed:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install the app's Python dependencies if they are not already installed:
   ```bash
   uv pip install fastapi uvicorn scanpy numpy
   ```
3. Start the app:
   ```bash
   uv run uvicorn app:app --reload
   ```
4. Open <http://127.0.0.1:8000> in your browser.

At startup, the app loads `data/pbmc3k.h5ad`. Stop the server with **Ctrl+C** in the terminal.