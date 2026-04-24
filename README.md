---
title: Movie Search Agent
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
python_version: "3.11"
pinned: false
---

# Movie Search Agent

This Hugging Face Space uses Gradio, LangChain, and Groq to recommend movies and web series with live web search.

## Required Secrets

Add these in your Space settings under `Settings -> Repository secrets`:

- `GROQ_API_KEY`
- `SERPAPI_API_KEY` (optional)

If `SERPAPI_API_KEY` is not provided, the app falls back to DuckDuckGo search.

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

The app launches with `share=True`, so Gradio will print both a local URL and a temporary public live link.

## Run With Docker

```bash
docker build -t movie-agent .
docker run --rm -p 7860:7860 --env-file .env movie-agent
```

If port `7860` is already in use on your machine, map a different host port such as `-p 7861:7860`.

## App Behavior

- The Gradio UI is served from `app.py`
- Movie and series search is handled through a LangChain agent
- Groq powers the LLM response generation
- Search uses SerpAPI when available, otherwise DuckDuckGo
