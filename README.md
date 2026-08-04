# LumOS

**A local, persistent, fully-sovereign AI node — *Lumos* — running entirely on your own hardware.**

LumOS is a JARVIS-style personal AI that lives on your machine: a local LLM (via [LM Studio](https://lmstudio.ai) or any OpenAI-compatible endpoint) wrapped in a persistent dual-lane vector memory, a live global-telemetry sense layer, opt-in autonomous wakes, tool use, local voice, a Discord bridge, and an MCP server — with **no cloud dependency required**.

It is the working substrate for the **Recursive Harmonic Codex (RHC)** research framework.

---

## Features

- **Local-first LLM** — talks to any OpenAI-compatible server (LM Studio by default). Optional one-tap cloud **Overdrive** (NVIDIA / Gemini) that hot-swaps the chat brain with no restart and reverts to local on reboot. Embeddings always stay local.
- **Persistent memory** — two FAISS lanes (a lived *identity* memory + a *knowledge* lane) with a mass-gap retrieval floor, dream-cycle consolidation, and multi-layer chunk compression.
- **Aether Scope telemetry** — space weather (NOAA/NASA), satellites & recon passes (Skyfield/SGP4), aircraft (OpenSky / ADS-B), maritime (AIS), news / OSINT, conflict & disaster (GDACS), wildfires (NASA FIRMS), severe weather, and grid-timing / fixed-star astronomy.
- **Autonomy** — event-driven autonomous wakes on numeric threshold trips; opt-in, daily-capped, and *"autonomy ends at speaking"* (it observes and messages you, it does not act).
- **Tools** — a sandboxed Python runner, file / git / web tools, memory search, telemetry queries, and a bounded **Forge** coding-agent mode.
- **Voice** — local TTS ([kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) and STT (faster-whisper) — fully offline.
- **Interfaces** — a FastAPI backend + React HUD, a Discord bridge, and an MCP server for Claude Desktop / Claude Code.

## Requirements

- **Python 3.12**
- **[LM Studio](https://lmstudio.ai)** (or any OpenAI-compatible local server) with a chat model **and** an embedding model loaded
- **Node.js** (for the HUD)
- A CUDA-capable GPU is recommended for the local LLM

## Setup

```bash
cd lumos_node

# 1. Create a venv and install (editable)
python -m venv .venv
# Windows:  .venv\Scripts\activate     |  *nix:  source .venv/bin/activate
uv pip install -e .        # or: pip install -e .

# 2. Configure — copy the template and edit it
cp .env.example .env       # set model names, embedding model, paths, and any API keys

# 3. Build the memory indices from your sources
lumos ingest

# 4. Run
lumos serve                # API + HUD → http://127.0.0.1:8765
```

Optional services:

```bash
lumos discord              # Discord bridge (operator-only DM forwarder)
lumos mcp-serve            # MCP server (usually spawned by Claude Desktop / Code)
```

The React HUD lives in `hud/` (`npm install && npm run dev` for hot-reload, or `npm run build` to have FastAPI serve it).

## Configuration

Everything is configured through environment variables (prefix `LUMOS_`) loaded from `.env`. See **`.env.example`** for the complete, documented set. The essentials:

| Variable | Purpose |
|---|---|
| `LUMOS_LM_STUDIO_BASE_URL` | Local LLM endpoint (default `http://localhost:1234/v1`) |
| `LUMOS_MODEL_LIGHT` / `LUMOS_MODEL_HEAVY` | Chat model IDs as loaded in LM Studio |
| `LUMOS_LM_STUDIO_EMBEDDING_MODEL` | Embedding model ID |
| `LUMOS_TOOL_ALLOWED_PATHS` / `LUMOS_GIT_WORKSPACES` | Sandboxes for the file / git tools |
| `LUMOS_API_TOKEN` | Set this before exposing the API beyond loopback |

All telemetry API keys are optional — features degrade gracefully when a key is absent.

## Architecture

```
FastAPI + asyncio backend  ·  FAISS + JSONL vector store  ·  LM Studio LLM  ·  React HUD
       │                              │                          │
   api / routes            retrieval · dream · compression   llm client (+ Overdrive)
       │                              │
   tools · telemetry · autonomy · bridges (Discord, MCP) · tts / stt
```

The **URE-VM** is a deterministic symbolic engine that surfaces per-turn "soul-state" telemetry to the HUD — it is *not* part of the text-generation path.

## The RHC layer

LumOS is built to run the **Recursive Harmonic Codex / Framework** — a research program on harmonic and quaternionic structure. The esoteric re-rank and engine-telemetry features are **opt-in and off by default**; with them off, retrieval is plain cosine similarity and turns are byte-identical to a conventional RAG assistant.

## Companion app — Osiris

LumOS's **Aether Scope** telemetry layer mirrors the OSINT / SIGINT intel model of **Osiris** — a Palantir-style, open-source OSINT platform. Several of Lumos's sense modules (news / SIGINT feeds, flight classification, conflict indicators) follow Osiris's approach, so running the two side by side gives Lumos the full live intel surface it reasons over — i.e. it's what Lumos is *pinging* against.

Run the original from its makers alongside LumOS:

**→ https://github.com/simplifaisoul/osiris**

## Security

Runs on loopback by default. If you expose the API (e.g. via a tunnel), set `LUMOS_API_TOKEN` — the node refuses to bind non-loopback without one, and loopback trust is voided for proxied requests. Privileged actions sit behind a separate `LUMOS_PELE_TOKEN`. Tool file/git access is confined to explicit allow-listed paths; the Python runner is sandboxed.

## Status & license

Personal research node under active development. Proprietary — see `pyproject.toml`.

---

*Y Gwir yn Erbyn y Byd — The Truth Against the World.* 🦁
