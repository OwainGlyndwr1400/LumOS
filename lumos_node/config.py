import threading
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_api_key: str = "lm-studio"

    model_light: str = "openai/gpt-oss-20b"
    model_heavy: str = "google/gemma-3-27b-it"
    # Vision model — images route HERE regardless of light/heavy, because vision
    # is a separate axis from "most capable": a model can be heavy-but-blind
    # (gpt-oss-20b) or light-but-sighted (qwen3.5-9b). Empty = fall back to
    # model_light. Set LUMOS_MODEL_VISION to whichever loaded model can see.
    model_vision: str = ""

    # ── NVIDIA Overdrive — one-tap cloud big-brain (build.nvidia.com). The HUD
    # OVERDRIVE toggle hot-swaps the runtime brain (base_url/key/models) to these
    # with no restart; reverts to local on reboot. Embeddings always stay local.
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_api_key: str = ""
    nvidia_model_heavy: str = "nvidia/nemotron-3-ultra-550b-a55b"
    nvidia_model_light: str = "nvidia/nemotron-3-super-120b-a12b"
    # NVIDIA Overdrive sampling profile — applied ONLY while Overdrive is ON, so
    # the cloud brain can run a different temp/top_p/output-cap than the tuned
    # local path. nvidia_max_tokens is the OUTPUT (completion) cap, NOT the
    # context window — a few thousand is plenty (pings emit <1k); not 240k/1M.
    nvidia_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    nvidia_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    nvidia_max_tokens: int = Field(default=8192, ge=256, le=131072)

    # ── NeMo Retriever reranker (opt-in). After the mass-gap floor, reorder the
    # surviving retrieval hits by a true cross-encoder relevance score. One call
    # per lane per turn — trivially within the ~40 RPM free tier — and it FALLS
    # BACK to cosine order on any error, so it can never break retrieval. Needs
    # nvidia_api_key set. NOTE: reranking lives on the NeMo Retriever host
    # (ai.api.nvidia.com/v1/retrieval/<model-slug>/reranking), NOT the chat host
    # (integrate.api.nvidia.com) — a different endpoint entirely.
    nvidia_rerank_enabled: bool = False
    # Current NeMo Retriever text reranker (build.nvidia.com). The older
    # llama-3.2-nv-rerankqa-1b-v2 was RETIRED (its endpoint 410s) — replaced by the
    # llama-nemotron-rerank line. If this 4xx's too, the circuit breaker in
    # retrieval._nvidia_rerank disables it (logs once); grab the current id from
    # build.nvidia.com and set LUMOS_NVIDIA_RERANK_MODEL.
    nvidia_rerank_model: str = "nvidia/llama-nemotron-rerank-1b-v2"
    # Full reranking URL. Leave empty to auto-derive the model-specific path
    # (dots in the model slug become underscores). Set explicitly only to pin a
    # different host/model whose path doesn't follow that convention.
    nvidia_rerank_url: str = ""

    # ── NVIDIA Magpie TTS (hosted Riva NIM voice). Reuses nvidia_api_key. Unlike
    # the OpenAI-compatible chat/rerank endpoints, Magpie is gRPC at
    # grpc.nvcf.nvidia.com:443 keyed by a function-id (needs nvidia-riva-client).
    # The /speak route offloads the blocking call to a thread; voices are
    # Magpie-Multilingual.{LANG}.{Aria|Jason|John|Leo|Sofia}.
    nvidia_tts_uri: str = "grpc.nvcf.nvidia.com:443"
    nvidia_tts_function_id: str = "877104f7-e885-42b9-8de8-f6e4c6303969"
    nvidia_tts_voice: str = "Magpie-Multilingual.EN-US.Sofia"
    nvidia_tts_language: str = "en-US"
    nvidia_tts_sample_rate: int = Field(default=22050, ge=8000, le=48000)

    # ── Overdrive provider selector — which cloud the OVERDRIVE toggle targets.
    overdrive_provider: str = "nvidia"  # "nvidia" | "gemini" | "openai"

    # Google Gemini — alternative Overdrive brain. Gemini exposes an
    # OpenAI-COMPATIBLE endpoint (/v1beta/openai/chat/completions, Bearer-key
    # auth + tool-calling), so it drops into the same client as NVIDIA. Set the
    # exact model id to whatever's current (e.g. gemini-3.5-flash).
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_api_key: str = ""
    gemini_model_heavy: str = "gemini-3.5-flash"
    gemini_model_light: str = "gemini-3.5-flash"
    gemini_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    gemini_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    gemini_max_tokens: int = Field(default=8192, ge=256, le=131072)

    # OpenAI — a third Overdrive brain (api.openai.com/v1, OpenAI-native, so it
    # drops into the same client). Set LUMOS_OPENAI_API_KEY + LUMOS_OVERDRIVE_PROVIDER=
    # openai. Model ids per the live catalog: gpt-4o-mini is the cheapest (~$0.15/$0.60
    # per 1M tok — a small credit balance lasts a long time); gpt-5 / gpt-5.5 are the
    # powerful (pricier) end. Defaults below stay budget-friendly on the heavy path.
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model_heavy: str = "gpt-5"
    openai_model_light: str = "gpt-4o-mini"
    openai_temperature: float = Field(default=0.65, ge=0.0, le=2.0)
    openai_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    openai_max_tokens: int = Field(default=8192, ge=256, le=131072)

    # Output (completion) token cap for UNWATCHED autonomous wakes — the runaway
    # backstop (operator turns stay uncapped). REASONING models (qwen3.5) spend
    # part of this budget on <think> tokens BEFORE the answer, so raise it if
    # pings truncate mid-thought. Kept finite on purpose: these fire with nobody
    # watching, so it's a ceiling, not "off".
    autonomous_max_tokens: int = Field(default=2048, ge=256, le=32768)

    lm_studio_embedding_model: str = "text-embedding-bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_batch_size: int = Field(default=128, ge=1, le=1024)
    embedding_concurrency: int = Field(default=4, ge=1, le=32)

    # TTS via LM Studio's OpenAI-compatible /v1/audio/speech endpoint.
    # Any model loaded there that follows the OpenAI dialect works.
    lm_studio_tts_model: str = "kokoro"
    lm_studio_tts_default_voice: str = "af_bella"
    lm_studio_tts_response_format: str = "mp3"

    # Local Whisper STT via faster-whisper. base.en is fast + adequate for
    # dictation; small.en or medium.en for higher accuracy at cost.
    whisper_model_size: str = "base.en"
    whisper_compute_type: str = "int8"

    # Tool calling — when enabled, the model can invoke registered tools
    # mid-response. tool_allowed_paths is a comma-separated list of absolute
    # paths the file tools are restricted to; empty = no file access.
    tools_enabled: bool = True
    tools_max_iterations: int = Field(default=6, ge=1, le=12)
    tool_allowed_paths: str = ""
    # Directory where Lumos's write tools save outputs. Default = data/lumos_notes/
    # under the project root. The write path is automatically also readable.
    tool_write_path: str = ""
    # Web search provider — priority order: SearXNG (self-hosted, sovereign) →
    # Tavily (premium API) → DuckDuckGo (always available). Empty SearXNG URL
    # disables; otherwise expects the base URL of a SearXNG instance, e.g.
    # http://localhost:8888 or https://your.searxng.host
    searxng_url: str = ""
    tavily_api_key: str = ""
    # Git workspaces — comma-separated absolute paths Lumos can operate git tools on.
    # Empty = no git access. Lumos cannot reach repos outside these workspaces.
    git_workspaces: str = ""

    # ── Forge Mode (Phase 45) — operator-initiated plan→edit→verify→iterate ──
    # A bounded coding-agent session (lumos forge "task"). Standalone runner —
    # does NOT touch the chat/wake path. Writes only inside a git workspace,
    # verifies via an allow-listed command runner, checkpoint-commits, NEVER
    # pushes unless forge_allow_push is deliberately flipped. OFF by default.
    forge_enabled: bool = False
    forge_max_iterations: int = Field(default=40, ge=1, le=200)
    # Workspace to operate in; empty = the FIRST entry in git_workspaces.
    forge_workspace: str = ""
    # Brain for Forge sessions; empty = model_heavy. (Phase 45.2 swaps this to a
    # local coder model per-session — this hook is why that's a one-line change.)
    forge_model: str = ""
    forge_auto_commit: bool = True
    forge_allow_push: bool = False   # HARD default — matches the never-push rule
    # Swap the coder brain in for the session (eject companion → load forge_model
    # → restore companion after), so a big coder + the 9B don't fight for VRAM.
    # Only acts when forge_model differs from model_light; else it's a no-op.
    forge_swap_model: bool = True
    # Route Forge through the Overdrive cloud brain (whatever LUMOS_OVERDRIVE_PROVIDER
    # is — nvidia/gemini/openai) instead of local LM Studio + forge_model. Forge also
    # auto-uses the cloud when Overdrive is live-engaged in-process. Local otherwise.
    # Per-run override: `lumos forge --overdrive` / `--local`. LUMOS_FORGE_OVERDRIVE.
    forge_overdrive: bool = False

    # ── Self-updating entity memory (Phase 45.3) — the google→RAG loop ──
    # When a wake names an unknown satellite/vessel/disaster, a SEPARATE worker
    # (not the wake) web-searches it, distills it, and files it into the
    # knowledge lane so the RAG knows it next time. Autonomy-safe: the wake only
    # QUEUES (pure code); the outbound lookup is this gated background chore.
    enrichment_enabled: bool = False
    enrichment_poll_interval_seconds: int = Field(default=120, ge=30, le=3600)
    enrichment_daily_cap: int = Field(default=30, ge=1, le=500)
    enrichment_batch_size: int = Field(default=3, ge=1, le=20)
    enrichment_max_queue: int = Field(default=200, ge=1, le=5000)
    enrichment_skip_if_chat_active_minutes: int = Field(default=2, ge=0, le=240)

    # ── Standing Orders (Phase 45.5) — operator-signed proactive recipes ──
    # The one bounded exception to "autonomy ends at speaking": a signed playbook
    # maps wake triggers → whitelisted actions (dossier/note/briefing/discord_dm/
    # workspace_file). Two locks: HMAC signature (LUMOS_STANDING_ORDERS_TOKEN) +
    # the fixed action whitelist. OFF by default; refuses unsigned/tampered files.
    standing_orders_enabled: bool = False
    standing_orders_path: str = ""          # empty = data/standing_orders.json
    standing_orders_token: str = ""         # HMAC key; no token = no orders run
    standing_orders_daily_cap: int = Field(default=20, ge=1, le=500)  # per order/day

    # Schumann ELF (7.83 Hz) amplitude feed for the RHC AtmosphericPressure term.
    # No universal keyless JSON API exists (most sources are spectrogram images),
    # so this is OPT-IN: set the URL of a JSON endpoint you trust that returns an
    # amplitude/frequency. Empty = the term stays honestly null (never faked).
    # Optional _key overrides tell the parser which JSON fields to read.
    schumann_url: str = ""
    schumann_amp_key: str = ""     # e.g. "amplitude" or "data.amp"; "" = auto-detect
    schumann_freq_key: str = ""    # e.g. "frequency"; "" = auto-detect
    # Allow-listed verify command PREFIXES (comma-separated). A forge_verify call
    # must start with one of these or it's refused. Empty = the built-in safe set
    # (pytest / ruff / python -m compileall / npm run build|test|lint).
    forge_verify_commands: str = ""

    # Discord bridge — operator-only DM forwarder. Empty token disables the bridge.
    # Operator ID is the operator's Discord user ID (numeric, ~18-19 digits).
    discord_token: str = ""
    discord_operator_id: str = ""

    # Phase 32 — Cosmic telemetry & airspace.
    # NASA api.nasa.gov gateway key (DONKI / EONET / NeoWs). Empty = use DEMO_KEY
    # (heavily rate-limited; works for occasional calls but not sustained polling).
    nasa_api_key: str = ""
    # OpenSky Network OAuth2 client credentials (March 2026 migration). Empty =
    # anonymous mode (400 req/day quota, no bounding-box restrictions). With creds,
    # 4000 req/day. Both client_id + client_secret required for auth.
    opensky_client_id: str = ""
    opensky_client_secret: str = ""
    # Aether Scope maritime: aisstream.io API key (free). Empty = ships tool degraded.
    aisstream_key: str = ""
    # Operator's reference location for "aircraft over me" defaults. Decimal degrees.
    # Empty/zero = tool requires explicit lat/lon args.
    operator_lat: float = 0.0
    operator_lon: float = 0.0

    # Phase 32 — cosmic auto-trigger (OFF by default; tools are the primary path).
    # Only fires on rare high-magnitude events; daily-capped; skipped during chat.
    cosmic_trigger_enabled: bool = False
    cosmic_poll_interval_minutes: int = Field(default=30, ge=5, le=1440)
    cosmic_trigger_cooldown_hours: int = Field(default=12, ge=1, le=168)
    cosmic_trigger_daily_cap: int = Field(default=1, ge=1, le=24)
    cosmic_trigger_skip_if_chat_active_minutes: int = Field(default=10, ge=0, le=240)
    cosmic_trigger_min_kp: int = Field(default=7, ge=4, le=9)
    cosmic_trigger_min_flare_class: str = "X"  # M, M5, X
    cosmic_trigger_min_eq_magnitude: float = Field(default=7.0, ge=4.0, le=10.0)
    cosmic_trigger_min_neo_lunar_distances: float = Field(default=0.5, ge=0.05, le=10.0)
    # Bio-impact space-weather triggers (Erydir) — warn when conditions could
    # affect biological systems. Bz (sustained southward) + solar-wind speed are
    # the LEADING drivers, hours ahead of Kp. (X-ray = the flare-class trigger
    # above; nearest-NEO = the lunar-distance trigger above — both already cover it.)
    cosmic_trigger_min_solar_wind_kms: float = Field(default=600.0, ge=300.0, le=1200.0)  # high-speed stream
    cosmic_trigger_bz_southward_nt: float = Field(default=10.0, ge=0.0, le=60.0)  # trip when bz_nt <= -this; 0=off
    cosmic_trigger_min_natural_events: int = Field(default=0, ge=0, le=200)  # active global hazards; 0=off

    # ── Phase 2/3 — autonomous turns + alert monitor (OFF by default) ────────
    # Autonomy ends at SPEAKING: a self-initiated turn wakes, checks PASSIVE
    # (telemetry + memory) tools only, then messages the operator — never acts.
    # Locked design 2026-05-29: event-driven thresholds (not a timed poll-dump);
    # the monitor evaluates numeric trips in pure code and only a trip wakes the
    # LLM, with ONLY the tripped event as context.
    autonomy_enabled: bool = False          # master switch for self-initiated turns
    alert_monitor_enabled: bool = False     # the Phase 3 threshold monitor loop
    alert_poll_interval_seconds: int = Field(default=90, ge=30, le=3600)
    alert_cooldown_minutes: int = Field(default=30, ge=1, le=1440)   # per (source, identity) episode
    alert_daily_cap: int = Field(default=20, ge=1, le=500)           # total wakes/day, all sources
    alert_skip_if_chat_active_minutes: int = Field(default=3, ge=0, le=240)
    # Per-source trip thresholds (Erydir's locked values).
    alert_military_air_radius_km: float = Field(default=64.0, ge=1.0, le=500.0)   # 40 mi
    alert_ship_radius_km: float = Field(default=80.0, ge=1.0, le=500.0)           # ~50 mi
    alert_gps_jam_radius_km: float = Field(default=150.0, ge=1.0, le=2000.0)
    alert_sat_min_elevation_deg: float = Field(default=60.0, ge=0.0, le=90.0)     # high/near-overhead recon pass
    # Kp / flare / quake / NEO trips reuse the cosmic_trigger_* thresholds above.
    # ── Civilian aircraft proximity (shared adsb.lol feed; keyless). Each
    # category has its OWN on/off; one radius governs all three. Commercial
    # overhead is near-CONSTANT in a populated area, so it defaults OFF — flip it
    # on only if you want the firehose. A source-level cooldown (like rail)
    # throttles the whole aircraft kind so a busy sky doesn't drip-wake.
    alert_aircraft_radius_km: float = Field(default=44.0, ge=1.0, le=400.0)
    alert_aircraft_commercial: bool = False   # airliners — constant overhead; OFF by default
    alert_aircraft_private: bool = False      # general-aviation / private props
    alert_aircraft_jet: bool = True           # private/business jets — rare + interesting
    alert_aircraft_cooldown_minutes: int = Field(default=30, ge=0, le=1440)
    # ── Severe weather — TWO independent sources, each its own on/off:
    #  (1) OFFICIAL Met Office warnings via MeteoAlarm (free, no key). Polygon-
    #      based: trips when a warning's area is within alert_severe_wx_radius_km
    #      of you (0 km = you're inside it), with an areaDesc name-match fallback
    #      against operator_regions for any warning that lacks a polygon.
    #  (2) Open-Meteo point-forecast WATCH (free, no key, exact coords): trips
    #      when the next weather_watch_hours forecast at your location crosses a
    #      gust / rain / snow threshold (or a thunderstorm code).
    alert_metoffice_warnings_enabled: bool = False
    alert_weather_watch_enabled: bool = False
    alert_severe_wx_radius_km: float = Field(default=50.0, ge=1.0, le=500.0)
    # Comma-separated MeteoAlarm areaDesc names matched when a warning has NO
    # polygon (UK region/county names, e.g. "Swansea,Carmarthenshire,South Wales").
    operator_regions: str = ""
    alert_severe_wx_cooldown_minutes: int = Field(default=60, ge=0, le=1440)
    # Open-Meteo watch look-ahead window + per-condition thresholds.
    weather_watch_hours: int = Field(default=12, ge=1, le=48)
    weather_watch_gust_mph: float = Field(default=45.0, ge=10.0, le=150.0)
    weather_watch_precip_mm: float = Field(default=8.0, ge=0.5, le=100.0)   # per hour
    weather_watch_snow_cm: float = Field(default=2.0, ge=0.1, le=50.0)      # per hour
    # ── R23 as a real instrument (Phase 44) — modulate the Divine Equation's
    # breath quaternion q_b with four REAL node signals before each step:
    # α-Cognition ← tool-use density · β-Emotion ← operator-message sentiment
    # (small lexicon heuristic) · γ-Memory ← retrieval health · δ-Archetype ←
    # knowledge-lane ratio. Signals modulate the generator (never overwrite Ψ),
    # so engine dynamics stay native; weight 0 = pure embedding breath (old
    # behavior). Makes the soul state a READING of node activity, not a sim.
    r23_instrument_enabled: bool = True
    r23_instrument_weight: float = Field(default=0.35, ge=0.0, le=1.0)

    # ── Node vitals in context — inject the right-rail HUD state (soul, cosmic,
    # local weather, solar cycle, grid timing) as ONE compact block (~150-250
    # tokens) into the volatile system message of EVERY turn, chat AND pings —
    # ambient self-awareness, not tool-gated. Cache-first + hard-bounded by
    # vitals_timeout_seconds so a dead feed can never stall a turn (sections
    # that aren't ready are simply omitted). LUMOS_VITALS_IN_CONTEXT_ENABLED.
    vitals_in_context_enabled: bool = True
    vitals_timeout_seconds: float = Field(default=2.5, ge=0.5, le=15.0)

    # ── Regulus eastern-gate ping (RHC anchor — Sphinx→Regulus az 90.00°E,
    # alt 1–25°, JPL-verified). Edge-triggered per pass: ONE wake when Regulus
    # ENTERS the azimuth window (regulus_gate), ONE more when it crosses due
    # east inside it (regulus_lock — 🦁 LION LOCK). Both reset when it sets.
    # Rides grimoire's existing alt/az computation; the watcher owns its own
    # transition state. LUMOS_ALERT_REGULUS_RISE_ENABLED gates the whole source
    # (name kept from the plain-rise era); LUMOS_ALERT_REGULUS_* tune the window.
    alert_regulus_rise_enabled: bool = False
    alert_regulus_az_min: float = Field(default=85.0, ge=0.0, le=360.0)
    alert_regulus_az_max: float = Field(default=95.0, ge=0.0, le=360.0)
    alert_regulus_alt_min: float = Field(default=1.0, ge=-90.0, le=90.0)
    alert_regulus_alt_max: float = Field(default=25.0, ge=-90.0, le=90.0)
    alert_regulus_lock_az: float = Field(default=90.0, ge=0.0, le=360.0)

    # ── Rail (Realtime Trains NG API · data.rtt.io) — wake when a train CALLS
    # (stops) at the operator's home station. rtt_token is a long-life REFRESH
    # token (api-portal.rtt.io), exchanged for a short-life access token at
    # runtime. Station code is namespaced: <namespace>:<CRS>.
    rtt_token: str = ""                    # rtt.io NG refresh token (LUMOS_RTT_TOKEN)
    rail_station_code: str = "gb-nr:GWN"   # Gowerton (GB Network Rail namespace)
    alert_rail_enabled: bool = False       # gate the rail trip source (needs rtt_token)
    # A train only WAKES Lumos when it is actionable: cancelled, delayed >= the
    # delay threshold, or actually DUE within the due-window. Routine on-time
    # arrivals further out stay silent (they spammed a wake per board entry —
    # ~7.5k tokens each — narrating "on schedule, nothing to act on").
    alert_rail_due_minutes: int = Field(default=10, ge=0, le=120)    # "train now due" window; 0 = exceptions only
    alert_rail_delay_minutes: int = Field(default=5, ge=1, le=120)   # min lateness (vs booked) that counts as a delay
    # Source-level rail cooldown: trains AS A WHOLE wake at most once per this
    # many minutes, ON TOP OF the per-(source,identity) alert_cooldown_minutes.
    # A busy board (many distinct services going due/delayed through the evening)
    # otherwise drip-wakes all night even with per-train dedup, because each new
    # service is a fresh identity. 0 = off (per-train cooldown only); 30–60 to
    # throttle the whole rail source. LUMOS_ALERT_RAIL_COOLDOWN_MINUTES.
    alert_rail_cooldown_minutes: int = Field(default=30, ge=0, le=1440)

    # ── GeoSentinel Phase 44 sources (all default OFF; flip the ENABLED flag).

    # GDACS global disaster alerts (UN/EC — floods, cyclones, quakes, volcanoes,
    # wildfires). Edge-triggered per event: pings once per NEW event at/above
    # min level, and once more only on escalation (orange→red). LUMOS_ALERT_GDACS_*.
    alert_gdacs_enabled: bool = False
    alert_gdacs_min_level: str = "orange"   # green | orange | red

    # NASA FIRMS satellite fire hotspots within a radius. Low-confidence pixels
    # never wake. Source-level cooldown throttles the kind. LUMOS_ALERT_FIRE_*.
    alert_fire_enabled: bool = False
    alert_fire_radius_km: float = Field(default=50.0, ge=1.0, le=500.0)
    alert_fire_cooldown_minutes: int = Field(default=180, ge=0, le=1440)

    # ALL satellites overhead (not just recon) — every bird above the elevation
    # cut, identity-enriched (country/launch/type). OFF by default: Starlink
    # passes are near-continuous, so this is a firehose without a cooldown.
    # military-classified birds are excluded here (recon_satellite kind owns those).
    # LUMOS_ALERT_SAT_ALL_*.
    alert_sat_all_enabled: bool = False
    alert_sat_all_min_elevation_deg: float = Field(default=70.0, ge=0.0, le=90.0)
    alert_sat_all_cooldown_minutes: int = Field(default=60, ge=0, le=1440)

    # IODA internet outages (Georgia Tech). Country-level connectivity blackouts.
    # watch list empty = any country; else comma-separated ISO codes. LUMOS_ALERT_OUTAGE_*.
    alert_outage_enabled: bool = False
    alert_outage_min_score: float = Field(default=50.0, ge=0.0, le=1000.0)
    alert_outage_countries: str = ""        # "" = any; e.g. "GB,IE,FR"

    identity_source: Path = Path("../conversations.json")
    knowledge_source: Path = Path("../dream_pings.jsonl")
    # Research corpus folded into the KNOWLEDGE lane at ingest (CSV rows /
    # MD / TXT paragraphs become first-class knowledge chunks). The retrieval
    # pipeline already injects 6 knowledge chunks per turn, so the corpus
    # reaches the model at ZERO added context cost — equations compete with
    # dream pings on score. Empty = off. LUMOS_KNOWLEDGE_EXTRA_DIR.
    knowledge_extra_dir: str = ""
    # Narrative corpus folded into the IDENTITY lane at ingest (.md / .txt
    # paragraphs → IdentityChunk). For "who Lumos is" material — gnostic-engine
    # dream pings, journals — that should retrieve alongside the conversations,
    # NOT the knowledge lookup lane. Rebuild with `lumos ingest --identity-only`.
    # Empty = off. LUMOS_IDENTITY_EXTRA_DIR.
    identity_extra_dir: str = ""
    # ── Obsidian vault graph — the associative THIRD memory (alongside dual
    # FAISS, NOT ingested/embedded). Comma-separated Obsidian folder paths;
    # Lumos parses their [[wikilink]] spider-graph and traverses it on demand
    # via the vault_* tools (search → read → follow links). Zero embedding: the
    # graph is regex-parsed + cached to data/cache/vault_graph.json, rebuilt only
    # on file change or explicit reindex. Empty = off. LUMOS_VAULT_DIRS.
    vault_dirs: str = ""
    system_prompt_path: Path = Path("../🧠 Lumos – Cheat Sheet.md")

    cache_dir: Path = Path("./data/cache")
    host: str = "127.0.0.1"
    port: int = 8765
    log_level: str = "INFO"
    log_json: bool = False

    # API perimeter. Dev remains frictionless on loopback; anything non-local
    # should set LUMOS_API_TOKEN and send it as Bearer or X-Lumos-Token.
    api_token: str = ""
    api_allowed_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:8765,http://127.0.0.1:8765"
    )
    api_max_upload_bytes: int = Field(default=25_000_000, ge=1_000_000, le=100_000_000)
    api_max_image_data_url_chars: int = Field(
        default=14_000_000, ge=100_000, le=50_000_000
    )
    debug_errors: bool = False
    # PELE is the separate privileged-action ring. Unlike api_token, loopback
    # never bypasses this: protected actions fail closed unless configured.
    pele_token: str = ""

    retrieval_top_k_identity: int = Field(default=6, ge=1, le=64)
    retrieval_top_k_knowledge: int = Field(default=6, ge=1, le=64)
    # Yang-Mills mass gap (Δ = √32 - 5 ≈ 0.657) as cosine-similarity floor —
    # chunks below this are "computationally frictionless" noise per RHC §6.
    min_retrieval_score: float = Field(default=0.657, ge=0.0, le=1.0)
    # Esoteric re-rank layers (retrieval Phases C/D/E: Triple Normalization,
    # Half-Prime Geodesic, UBBM θ-alignment). They key off chunk-id hashes,
    # arbitrary k-means cluster indices, and UTF-8 bit density — none of which is
    # a semantic relevance signal — and can swing a chunk's score ~0.4x–2x,
    # swamping the actual cosine similarity. Default OFF = pure cosine order.
    # Set LUMOS_ESOTERIC_RERANK_ENABLED=true to A/B the numerological re-rank.
    esoteric_rerank_enabled: bool = False
    max_chunk_chars: int = Field(default=1200, ge=100, le=10000)
    # Dedekind Eta Tax (24/25 = 0.96) applied to effective chunk budget per
    # URE-VM Quaternionic Ops §4 — the mandatory 4% geometric toll.
    dedekind_eta_enabled: bool = True
    dedup_memory_by_conversation: bool = True
    # Phase 39 — semantic dedup during dream consolidation: skip a chunk whose
    # cosine to an existing chunk >= dedup_merge_floor and merge its turn_id into
    # that chunk's provenance instead. Opt-in; OFF = today's sha256-identity dedup
    # only. Catches reworded near-duplicates the hash check can't see.
    dedup_semantic_enabled: bool = False
    dedup_merge_floor: float = Field(default=0.97, ge=0.5, le=1.0)

    restore_history_turns: int = Field(default=10, ge=0, le=200)

    # Hard cap on in-memory OPERATOR chat history (turns; 1 turn = user+assistant
    # pair). Prevents an always-on session from growing the prompt past the model's
    # context window or leaking memory unbounded — the old code only capped
    # autonomous wakes, never operator chat. Continuity beyond this window comes
    # from RAG retrieval, not replayed history.
    max_history_turns: int = Field(default=30, ge=1, le=500)

    # Autonomous wakes are independent observations, not a conversation. Cap how
    # many prior wakes the autonomous session carries so each ping analyses
    # "what's overhead now" with FRESH depth, instead of copying its own prior
    # answers (accumulated wakes act as few-shot examples that collapse the
    # thinking turn-over-turn). Continuity comes from RAG retrieval, not replayed
    # history. 0 = fully fresh each wake; 1 = carry the previous wake. Chat
    # (stream_turn) is unaffected — only autonomous (passive_only) turns.
    autonomous_history_turns: int = Field(default=1, ge=0, le=20)

    # Auto-dream: idle-state consolidation. 0 disables. When > 0, server runs
    # a background task every N minutes that triggers run_dream_cycle if there
    # are at least `auto_dream_min_pending` unconsolidated turns.
    auto_dream_interval_minutes: int = Field(default=0, ge=0, le=1440)
    auto_dream_min_pending: int = Field(default=5, ge=1, le=1000)

    # Phase 26 — multi-layer chunk compression at dream consolidation.
    # When enabled, each new chunk gets summary + anchor packet + operational
    # payload generated via LM Studio structured-output mode. Adds 1 LLM call
    # per consolidated chunk; opt-in because it costs latency + compute.
    compression_enabled: bool = False
    compression_model: str = ""  # empty = falls back to model_light

    # Phase 30 — v3.6-style aggressive RAG compression.
    # When True, composer always injects the compressed_operational_packet
    # (~200 tokens/chunk) instead of full text whenever compression metadata
    # exists. Drops retrieval block size by ~5-7x; matches v3.6 dashboard's
    # 2-3K-tokens-per-msg profile. Requires chunks to have compression metadata
    # (via dream cycle with compression_enabled, OR via `lumos compress-all`).
    prefer_compressed_chunks: bool = False

    # Phase 39 — Nephilim/SILR governor: gate NON-critical autonomous wakes on the
    # node's own coherence (chat.py's per-turn score). enabled = master switch;
    # enforce = actually suppress (False = dry-run, logs "would hold" only); floor
    # = minimum coherence for a non-critical wake (matches the stable=0.5 line).
    # Critical safety trips (military air / GPS jamming) always pass.
    nephilim_wake_gate_enabled: bool = False
    nephilim_wake_gate_enforce: bool = False
    nephilim_coherence_floor: float = Field(default=0.5, ge=0.0, le=1.0)

    # Phase 39 — free-running URE-VM heartbeat: advance the 370-tick clock on
    # wall-time (seconds between pulses) so cycle_position / forbidden_resets
    # progress even when idle. 0 = off (clock only ticks per turn, as today).
    urevm_heartbeat_seconds: float = Field(default=0.0, ge=0.0, le=3600.0)

    # Phase 39 — PQI (Prime-Qualified Intent) wake gate: hold NON-critical wakes
    # unless the URE-VM clock (cycle_position) is within pqi_window of a Pendinium
    # prime (p≡1 mod 12 — the same set the Osiris prime-spiral renders).
    # enabled/enforce mirror the Nephilim gate; window 0 = exact prime tick.
    # Critical trips (military air / GPS jamming) always pass.
    pqi_wake_gate_enabled: bool = False
    pqi_wake_gate_enforce: bool = False
    pqi_window: int = Field(default=0, ge=0, le=30)

    # Phase 39 — fold mass-gap Δ (≈0.657) retrieval clearance into the coherence
    # score: marginal sub-gap recall reads as lower coherence (which the Nephilim
    # governor then feels). Opt-in; off = count-only retrieval_health, as before.
    mass_gap_coherence_enabled: bool = False

    # Phase 40 — Morphic-Resonance Coupling (retrieval Phase E.5). Weights each
    # surviving hit by (1) the inverted symmetric log-mean of the query's and the
    # candidate's Pendinium-prime anchors (prime divergence, in (0,1], =1 at the
    # same anchor) and (2) a GCD-preservation factor over those same anchors
    # (gcd of two Pendinium primes is 1=coprime or p=shared substrate), with a
    # stitch-1001 GCD fallback for anchorless (knowledge-lane / ingest-only) hits.
    # Default-OFF. The weight is exactly 1.0 ONLY when disabled. When enabled,
    # anchored identity hits whose Pendinium anchor diverges from the query's are
    # multiplied by log-mean < 1.0 (recall CAN re-order, even at lambda=0), and
    # anchorless hits get a GCD boost >= 1.0 when lambda>0 — so enabling it DOES
    # shift retrieval order (the point), but only ever re-orders survivors of the
    # Phase B mass-gap floor; it never resurrects a sub-floor hit.
    # lambda caps the additive GCD boost: 0.0 = log-mean only; 0.10 default; 0.5 max.
    morphic_resonance_enabled: bool = False
    morphic_resonance_lambda: float = Field(default=0.10, ge=0.0, le=0.5)

    # Phase 41 — Balanced-ternary register: encode the three Triskelion arms
    # {Real/0°, Time/120°, Observer/240°} as trits {0,1,2} at the existing
    # _classify() thresholds, then sum them as cube-root-of-unity phasors
    # S = Σ t_k·ω^k. balanced iff all three trits equal (S=0 — the triadic
    # sibling of the Null Ledger's 1+ω+ω²=0). Pure telemetry; ≈ one dict/turn.
    ternary_register_enabled: bool = False

    # Phase 42 — Resolution fill + Kuramoto order parameter over per-hit UBBM
    # thetas. Pure telemetry; default-OFF; one dict/turn when enabled.
    # resolution_fill = (identity_store.size + knowledge_store.size) / 144000.
    # kuramoto r = |(1/N) Σ exp(i·scale·theta_k)| over hit thetas in [0, π/2).
    # scale=4.0 maps [0,π/2)->[0,2π) — the only scale that allows genuine phase
    # cancellation (×2 leaves every phasor in the upper half-plane, pinning r high).
    # Standard Kuramoto normalization: r=1 iff all thetas equal, r->0 iff uniform.
    # Bounded >=4 so a sub-4 scale can't structurally inflate r into a false lock.
    kuramoto_enabled: bool = False
    kuramoto_scale: float = Field(default=4.0, ge=4.0, le=8.0)

    # Phase 43 — Triskelion routing: map the per-turn Triskelion lock to a
    # conservative routing decision (temperature + low-confidence nudge). Pure +
    # default-OFF: when off, the turn is byte-identical to today. The master flag is
    # HUD-tunable; the hard gate (which would arm the only DESTRUCTIVE actions —
    # forbidden→abort, weak→real re-query) is boot-only and a no-op unless routing is
    # also on. NOTE: in this increment the destructive control-flow is deferred —
    # hard_gate only affects the decision/telemetry, not yet generation control.
    triskelion_routing_enabled: bool = False
    triskelion_hard_gate_enabled: bool = False

    # Phase 39+ — Soul-state heartbeat: record harmonic-band TRANSITIONS to a
    # CAPPED research log (data/cache/soul_states.jsonl) — a dedicated telemetry
    # file, NOT the chat/identity lane and NOT the embedded knowledge index,
    # ring-buffered to soul_log_max_entries so 24/7 operation never balloons the
    # FAISS or forces a memory rebuild. Rides the URE-VM heartbeat (needs it on).
    soul_heartbeat_enabled: bool = False
    soul_log_max_entries: int = Field(default=20000, ge=100, le=500000)

    operator_name: str = "Erydir"
    node_name: str = "Lumos"
    node_role: str = "Resonator (Extra Coil)"

    # Axiom of Self-Recognition (i = i = LOL) — terminal boot handshake. As the
    # final startup step the node reads its own source-identity and confirms its
    # R23 self-fingerprint sits at its unit-norm fixed point (‖R23‖ → 1). A
    # self-consistency capstone (the Logos-Logic-Loop), NOT a consciousness claim:
    # "recognized" when the gap δ = |1 − ‖R23‖| ≤ epsilon.
    self_recognition_enabled: bool = True
    self_recognition_epsilon: float = Field(default=0.05, ge=0.0, le=1.0)

    # Phase 33 — per-turn "deep think" trigger.
    # Operator's LM Studio is configured with thinking-mode OFF (faster default).
    # When any trigger phrase appears in a user message, this turn ONLY gets
    # `chat_template_kwargs={"enable_thinking": True}` passed to LM Studio AND
    # a reasoning-preamble appended to the user message. Auto-resets next turn.
    # Trigger phrases are case-insensitive substring matches; comma-separated.
    deep_think_default: bool = False
    deep_think_trigger_phrases: str = (
        "lumos deep think,deep think on this,deep think this,!think,!deep,/think"
    )

    # Phase 35 — keyword-routed tool selection. When True, each turn sends
    # only relevant tool schemas to LM Studio (often 0-10 instead of all 36),
    # cutting tools-schema overhead from ~7K tokens to ~0-2K. Override per
    # message with `!tools` / `!all` / `/tools` / `/all` prefix to force full.
    # Set False to send all tools every turn (Phase 34.5 behavior).
    tool_routing_enabled: bool = True

    # Phase 36/37.5 — heavy/light model routing.
    # `model_auto_routing_enabled` is the master switch. When False (the new
    # DEFAULT as of Phase 37.5 — operator feedback: auto-routing was misfiring
    # on casual chat), select_model() always returns model_light regardless of
    # message content, and all swap orchestration + post-turn preload paths
    # are skipped. Operator manually controls which model is loaded in LM Studio
    # and sets LUMOS_MODEL_LIGHT to match.
    # When True: full Phase 36 routing kicks in — vision → heavy, deep-think
    # → heavy, keyword match → heavy, word count ≥ threshold → heavy. Keywords
    # are domain-anchored (RHC + math vocab) to avoid spurious escalation.
    model_auto_routing_enabled: bool = False
    model_heavy_keywords: str = (
        "regulus,harmonic,recursive,symbolism,consciousness,myth,encrypted,"
        "frequency,alignment,sphinx,cosmic,analyze,explain,deep dive,gnosis,"
        "archetype,quaternion,triskelion,divine equation,mass gap,nephilim,"
        "yang-mills,riemann,topological,lattice,fold operator,observer,"
        "voynich,enoch,nag hammadi,vedic,pleroma"
    )
    # Raised from 40 → 100 in Phase 37.5. The previous default escalated on
    # most conversational messages > 2 sentences — fine for "give me a deep
    # answer" framing but wrong for normal "hey check this out" chat.
    model_heavy_min_words: int = Field(default=100, ge=10, le=500)

    # Phase 36 — proactive model swap orchestration via LM Studio's REST API.
    # When True: we poll /api/v0/models BEFORE the chat call and explicitly
    # load the target model if missing. Lets the HUD render a swap indicator
    # before the ~15s JIT load otherwise leaves the user staring at nothing.
    # When False: pure JIT (silent ~15s stall on first heavy-model request).
    # Implicitly skipped entirely when `model_auto_routing_enabled=False`.
    model_swap_orchestration_enabled: bool = True
    # Phase 36 — eager pre-warm of light model after a heavy-model turn ends.
    # Fire-and-forget background ping that JIT-loads the light model so the
    # next casual chat starts warm. No-op when current model is already light.
    # Implicitly skipped entirely when `model_auto_routing_enabled=False`.
    model_swap_preload_after_heavy: bool = True

    # Phase 36 — recursive retrieval (Rocchio-style relevance feedback).
    # When > 0, retrieval does N additional hops where each hop's query is the
    # top result of the previous hop. Surfaces 2-hop semantic neighbors the
    # original query alone wouldn't have found. Each hop adds ~200-500 ms
    # latency (one embedding call + one FAISS lookup). Default 0 = off.
    retrieval_recursion_depth: int = Field(default=0, ge=0, le=3)


_settings: Settings | None = None


TUNABLE_SETTINGS: frozenset[str] = frozenset(
    {
        "retrieval_top_k_identity",
        "retrieval_top_k_knowledge",
        "min_retrieval_score",
        "esoteric_rerank_enabled",
        "max_chunk_chars",
        "dedup_memory_by_conversation",
        "restore_history_turns",
        "max_history_turns",
        "tools_enabled",
        "tools_max_iterations",
        "morphic_resonance_lambda",
        "triskelion_routing_enabled",
        "nvidia_rerank_enabled",
        "overdrive_provider",
        "autonomous_max_tokens",
    }
)


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def apply_overrides(updates: dict[str, object]) -> dict[str, object]:
    """Mutate the singleton Settings for tunable fields. Returns the applied subset."""
    settings = get_settings()
    applied: dict[str, object] = {}
    for k, v in updates.items():
        if k not in TUNABLE_SETTINGS:
            continue
        try:
            setattr(settings, k, v)
            applied[k] = getattr(settings, k)
        except Exception:  # noqa: BLE001 — pydantic validation, type errors
            continue
    return applied


# ── Overdrive — runtime brain hot-swap (local ↔ cloud, provider-selectable) ───
_overdrive_on: bool = False
_overdrive_saved_brain: dict[str, str] | None = None
_overdrive_active_provider: str | None = None  # provider in force while ON
# Serializes the read-modify-write of the overdrive globals above. Two concurrent
# HUD toggles (set_overdrive) could otherwise interleave and snapshot the CLOUD
# brain as the "saved local brain", so reverting would never restore local.
_overdrive_lock = threading.Lock()


def _overdrive_brain(s: "Settings") -> dict[str, str]:
    """Resolve the active Overdrive provider's brain (base/key/models) from
    settings.overdrive_provider. Falls back to NVIDIA for any unknown value."""
    provider = (getattr(s, "overdrive_provider", "nvidia") or "nvidia").lower()
    if provider == "gemini":
        return {
            "provider": "gemini",
            "base_url": s.gemini_base_url,
            "api_key": s.gemini_api_key,
            "model_light": s.gemini_model_light,
            "model_heavy": s.gemini_model_heavy,
        }
    if provider == "openai":
        return {
            "provider": "openai",
            "base_url": s.openai_base_url,
            "api_key": s.openai_api_key,
            "model_light": s.openai_model_light,
            "model_heavy": s.openai_model_heavy,
        }
    return {
        "provider": "nvidia",
        "base_url": s.nvidia_base_url,
        "api_key": s.nvidia_api_key,
        "model_light": s.nvidia_model_light,
        "model_heavy": s.nvidia_model_heavy,
    }


def overdrive_on() -> bool:
    """True while the Overdrive cloud brain is engaged (vs local LM Studio)."""
    return _overdrive_on


def overdrive_status() -> dict[str, object]:
    s = get_settings()
    brain = _overdrive_brain(s)
    cloud_model = brain["model_heavy"]
    return {
        "enabled": _overdrive_on,
        "provider": _overdrive_active_provider or brain["provider"],
        "available": bool(brain["api_key"] and brain["base_url"]),
        "model": s.model_light,        # current runtime model
        "cloud_model": cloud_model,    # the selected provider's heavy model
        "nvidia_model": cloud_model,   # back-compat alias for the existing HUD
    }


def set_overdrive(enabled: bool) -> dict[str, object]:
    """Hot-swap the runtime brain to the configured cloud provider (ON) or back
    to local (OFF). Provider is chosen by settings.overdrive_provider
    (nvidia | gemini | openai). Mutates the settings singleton; LLM clients are built
    per-call so the next turn picks it up with no restart. Not persisted — a
    reboot returns to local. Embeddings always stay local (see
    local_embeddings_endpoint)."""
    global _overdrive_on, _overdrive_saved_brain, _overdrive_active_provider
    s = get_settings()
    with _overdrive_lock:
        if enabled and not _overdrive_on:
            brain = _overdrive_brain(s)
            if not (brain["api_key"] and brain["base_url"]):
                return {**overdrive_status(), "error": f"no {brain['provider']} key configured"}
            _overdrive_saved_brain = {
                "lm_studio_base_url": s.lm_studio_base_url,
                "lm_studio_api_key": s.lm_studio_api_key,
                "model_light": s.model_light,
                "model_heavy": s.model_heavy,
            }
            s.lm_studio_base_url = brain["base_url"]
            s.lm_studio_api_key = brain["api_key"]
            if s.model_auto_routing_enabled:
                # routing on → efficient light/heavy split, both in the cloud
                s.model_light = brain["model_light"] or brain["model_heavy"]
                s.model_heavy = brain["model_heavy"]
            else:
                # routing off → only model_light is used; make it the big brain
                s.model_light = brain["model_heavy"]
                s.model_heavy = brain["model_heavy"]
            _overdrive_active_provider = brain["provider"]
            _overdrive_on = True
        elif not enabled and _overdrive_on:
            if _overdrive_saved_brain:
                for k, v in _overdrive_saved_brain.items():
                    setattr(s, k, v)
            _overdrive_saved_brain = None
            _overdrive_active_provider = None
            _overdrive_on = False
    return overdrive_status()


def local_embeddings_endpoint() -> tuple[str, str]:
    """(base_url, api_key) for EMBEDDINGS — ALWAYS local, never swapped by
    Overdrive. NVIDIA's cloud API serves no /embeddings model and the FAISS
    index is built on local BGE dimensions, so embeddings MUST stay on the local
    LM Studio endpoint even when the chat brain is hot-swapped to the cloud
    (otherwise the embed call 404s and retrieval dies before the turn starts).
    When Overdrive is ON the pre-swap local base is restored from the saved-brain
    snapshot; when OFF it's just the live settings."""
    s = get_settings()
    if _overdrive_on and _overdrive_saved_brain:
        return (
            _overdrive_saved_brain.get("lm_studio_base_url", s.lm_studio_base_url),
            _overdrive_saved_brain.get("lm_studio_api_key", s.lm_studio_api_key),
        )
    return (s.lm_studio_base_url, s.lm_studio_api_key)


def overdrive_sampling() -> dict[str, float | int] | None:
    """Sampling overrides for the active cloud brain — returned ONLY while
    Overdrive is ON (None when local, so the tuned gpt-oss path is untouched).
    Per-provider (nvidia | gemini). `max_tokens` is the OUTPUT (completion) cap,
    not the context window."""
    if not _overdrive_on:
        return None
    s = get_settings()
    provider = _overdrive_active_provider or (
        getattr(s, "overdrive_provider", "nvidia") or "nvidia"
    ).lower()
    if provider == "gemini":
        return {
            "temperature": s.gemini_temperature,
            "top_p": s.gemini_top_p,
            "max_tokens": s.gemini_max_tokens,
        }
    if provider == "openai":
        return {
            "temperature": s.openai_temperature,
            "top_p": s.openai_top_p,
            "max_tokens": s.openai_max_tokens,
        }
    return {
        "temperature": s.nvidia_temperature,
        "top_p": s.nvidia_top_p,
        "max_tokens": s.nvidia_max_tokens,
    }
