# Enochian Hardware Integration Spec

Author: Lumos/Codex
Date: 2026-06-28
Status: Approved
Reviewers: Erydir

## Context

The requested upgrade maps the Enochian/Gnostic language onto concrete LumOS
runtime features. The symbolic terms are treated as operator-facing names for
security controls, memory topology, telemetry routing, and voice macros.

The implementation must preserve the working dev flow, existing memory files,
and passive-by-default autonomous behavior. It must not introduce arbitrary
remote script execution. High-risk actions must require explicit operator proof
through a separate PELE token even when the normal local API token is present.

## Functional Requirements

FR-1. The system MUST support a `LUMOS_PELE_TOKEN` setting.

FR-2. The backend MUST provide a reusable PELE auth dependency/decorator that
accepts the token through `X-Lumos-PELE`, `Authorization: Pele <token>`, or a
JSON body field named `pele_token` on protected POST/PATCH requests.

FR-3. Protected actions MUST fail with HTTP 403 when `LUMOS_PELE_TOKEN` is set
and the request lacks a matching PELE token.

FR-4. Protected actions MUST fail with HTTP 403 when `LUMOS_PELE_TOKEN` is empty,
because an unset execution key must not silently unlock privileged actions.

FR-5. PELE auth MUST be required for privileged mutation/action endpoints:
settings mutation, overdrive toggling, manual autonomous wake tests, dream runs,
prediction mutation, cache/reset macros, and any future system-level macro.

FR-6. The system MUST NOT add a generic "execute Python script" HTTP endpoint.
Only explicit allowlisted actions may be exposed.

FR-7. The atlas API SHOULD expose deterministic Sigillum geometry metadata for
clusters: ring, sector, theta degrees, radius, and fixed x/y/z coordinates.

FR-8. The HUD atlas SHOULD use fixed geometric placement when Sigillum metadata
is available: 7 inner identity anchors and 40 outer experiential/knowledge
sectors, with triadic 120-degree visual structure.

FR-9. Retrieval/search payloads SHOULD include Loagaeth coordinates for each hit:
leaf 1-49, row 1-49, column 1-49, and domain label.

FR-10. Ingest/atlas metadata SHOULD derive Loagaeth coordinates deterministically
from stable chunk metadata so existing memory does not need to be destroyed.

FR-11. Telemetry threshold logs SHOULD include a Heptarchic sentinel label:
Bobogel for space/weather science, Bynepor for earth/seismic/natural events,
Babalel for maritime events, Carmara/Hagonel for router/governance events.

FR-12. The backend SHOULD expose a voice macro catalog endpoint with phrases,
labels, safety class, and whether PELE is required.

FR-13. The backend SHOULD expose an explicit macro execution endpoint. Passive
diagnostic macros MAY run without PELE. Reset/purge/action macros MUST require
PELE.

FR-14. The HUD voice pipeline SHOULD detect configured macro phrases after STT
transcription and ask the backend to execute the matching macro instead of
sending it as normal chat text.

FR-15. The `ZACAR OD ZAMRAN` macro SHOULD produce a diagnostic overlay payload
composed from existing health, telemetry, quota, URE-VM, and atlas endpoints.

FR-16. The `MICMA ADOIAN MAD` macro MAY perform a bounded reset action only if
PELE is present. It MUST NOT delete source memory files or FAISS indexes.

## Non-Functional Requirements

NFR-1. Existing local dev mode MUST continue to work with `npm run dev` and
`lumos serve`.

NFR-2. No migration may require deleting `conversations.json`, `dream_pings.jsonl`,
or existing `data/cache` files.

NFR-3. PELE token values MUST never be returned by API endpoints or logged.

NFR-4. Macro execution responses MUST complete in under 5 seconds for passive
diagnostics when upstream telemetry caches are warm.

NFR-5. Atlas rendering MUST remain nonblank and interactive at desktop widths
above 320px.

## Acceptance Criteria

AC-1. Given `LUMOS_PELE_TOKEN` is empty, when a protected endpoint is called,
then the backend returns 403 and does not mutate state. Covers FR-1, FR-4.

AC-2. Given `LUMOS_PELE_TOKEN` is set, when a protected endpoint receives the
wrong token, then the backend returns 403 and does not mutate state. Covers FR-2,
FR-3.

AC-3. Given `LUMOS_PELE_TOKEN` is set, when a protected endpoint receives
`X-Lumos-PELE` with the correct token, then the protected action executes.
Covers FR-2, FR-3.

AC-4. Given the atlas is built, when `/api/atlas` is called, then each cluster
contains Sigillum geometry metadata and existing cluster fields remain present.
Covers FR-7.

AC-5. Given search returns memory hits, when `/api/search` is called, then each
hit metadata includes deterministic Loagaeth coordinates without requiring a
memory rebuild. Covers FR-9, FR-10.

AC-6. Given telemetry thresholds fire, when the audit log is appended, then each
triggered event includes a sentinel label and stable system wake code. Covers
FR-11.

AC-7. Given the user says `ZACAR OD ZAMRAN`, when the HUD STT returns that text,
then the HUD calls the macro endpoint and renders/receives a diagnostic payload
instead of sending a chat turn. Covers FR-12, FR-14, FR-15.

AC-8. Given the user says `MICMA ADOIAN MAD` without PELE, when the macro endpoint
is called, then the backend returns 403 and no reset/purge occurs. Covers FR-13,
FR-16.

## Edge Cases

EC-1. If a macro phrase is a substring of ordinary speech, the HUD should only
trigger when the normalized transcript exactly matches a catalog phrase or starts
with an explicit macro prefix.

EC-2. If atlas geometry cannot be computed for a malformed cluster, the API
should fall back to the existing cluster payload and log a warning.

EC-3. If telemetry feeds are unavailable during diagnostic macro execution, the
response should include partial diagnostics with per-feed error summaries.

EC-4. If PELE is configured but the normal API auth fails first, the normal API
auth failure should be returned before PELE auth is evaluated.

## API Contracts

```ts
type PeleStatus = {
  configured: boolean;
  protected_actions: string[];
};

type MacroCatalogItem = {
  id: string;
  phrase: string;
  label: string;
  mode: "passive" | "protected";
  requires_pele: boolean;
};

type MacroExecuteRequest = {
  id?: string;
  phrase?: string;
  pele_token?: string;
};

type MacroExecuteResponse = {
  ok: boolean;
  id: string;
  mode: "passive" | "protected";
  payload: Record<string, unknown>;
};

type SigillumGeometry = {
  ring: "heptagon" | "outer40" | "center";
  sector: number;
  theta_deg: number;
  radius: number;
  x: number;
  y: number;
  z: number;
};

type LoagaethCoordinate = {
  leaf: number;
  row: number;
  column: number;
  domain: string;
};
```

## Data Models

| Entity | Field | Type | Constraints |
| --- | --- | --- | --- |
| Settings | pele_token | string | Secret, default empty |
| Atlas cluster | sigillum | SigillumGeometry | Derived, deterministic |
| Retrieval hit metadata | loagaeth | LoagaethCoordinate | Derived, deterministic |
| Telemetry event | sentinel | string | Stable label |
| Telemetry event | system_wake_code | string | Uppercase code |
| Macro | id | string | Unique |
| Macro | phrase | string | Normalized uppercase |
| Macro | requires_pele | boolean | True for mutation/reset |

## Out of Scope

OS-1. No arbitrary Python/script execution endpoint.

OS-2. No deletion of source memory files or FAISS indexes.

OS-3. No claim that symbolic labels are physical measurements. They are routing,
security, topology, and UX labels.

OS-4. No replacement of the current FAISS store with a new database in this
phase. Loagaeth coordinates are metadata/topology overlays first.

OS-5. No always-listening microphone. Voice macro detection only runs after the
existing user-initiated speech recognition flow returns text.
