# SRE Incident Triage & Runbook Agent

Operational DevOps/SRE co-worker agent (`sre_triage_agent`) that ingests logs and Prometheus/Alertmanager alerts (from bundled scenarios or live sources), correlates them into incidents, reasons via configurable open-source LLM providers (Ollama, self-hosted OpenAI-compatible servers such as vLLM / llama.cpp / LocalAI / LM Studio, or an explicit deterministic offline fallback), proposes evidence-grounded runbook remediation steps, guides operators through interactive or scripted approval with zero-side-effect dry-run simulation, and records a persistent append-only JSONL audit trail.

## Architecture & Safety Guarantees

```text
[Bundled Samples / Live Log File(s) + Prometheus or Alertmanager HTTP]
        │
        ▼  ingestion (logs.py, alerts.py, correlator.py)
  IncidentBundle(alerts, logs)
        │
        ├───────────────────────────────┐
        ▼                               ▼
  runbooks/catalog.py            llm/ (Ollama, OpenAI-Compat, DeterministicFallback)
        │                               │
        └──────────────┬────────────────┘
                       ▼
                 triage/engine.py -> TriageResult (evidence-linked RemediationSteps)
                       │
                       ▼
                 approval/flow.py (InteractivePromptHandler / ScriptedDecisionHandler)
                    ├── approve -> executor/dry_run.py (pure simulation, 0 subprocesses)
                    └── reject / skip -> skip execution, record operator decision
                       │
                       ▼
                 audit/store.py -> append-only JSONL audit log + `audit list/show` CLI
```

### Key Safety & Engineering Guarantees
- **Zero real-infrastructure side effects**: `src/sre_triage_agent/executor/dry_run.py` (`DryRunExecutor`) performs pure string simulation and precondition validation—it never imports or calls `subprocess`, `os.system`, `os.popen`, `socket`, or HTTP clients, and every outcome records `executed_against_real_infra: false`.
- **Shell-injection & traversal defense**: `DryRunExecutor` validates target identifiers (`sanitize_identifier`), rejects unrendered `{placeholder}` templates, detects unsafe shell metacharacters in interpolated alert labels, namespaces, instances, and log attributes, and sanitizes previews (`sanitize_command_preview`).
- **Evidence-grounded remediation**: `TriageEngine` verifies every `RemediationStep.evidence_refs` entry cites real ingested `alert_id` or `log_id` IDs.
- **Zero proprietary LLM SDKs**: Uses only `httpx` against local/self-hosted open-source endpoints (`Ollama` `/api/chat` and OpenAI-compatible `/v1/chat/completions`) plus an offline deterministic runbook engine.
- **Append-only UTF-8 JSONL audit log**: `src/sre_triage_agent/audit/store.py` (`AuditLogger`) records `triage_run_started`, `triage_completed`, `remediation_suggested`, `operator_decision`, and `dry_run_executed` events with UTC ISO-8601 timestamps, `run_id`, and `incident_id`.

## Project Layout

- `src/sre_triage_agent/models.py` — Frozen Pydantic v2 domain schemas (`LogEntry`, `AlertEvent`, `IncidentBundle`, `RemediationStep`, `TriageResult`, `OperatorDecision`, `DryRunOutcome`, `AuditEvent`).
- `src/sre_triage_agent/config.py` — `AgentConfig` (`pydantic-settings` + YAML + `SRE_AGENT_*` environment variables + CLI overrides).
- `src/sre_triage_agent/ingestion/` — Bundled and live log (`logs.py`) and Prometheus/Alertmanager (`alerts.py`) ingestion plus incident correlation (`correlator.py`).
- `src/sre_triage_agent/runbooks/catalog.py` — YAML runbook loader and signal matcher.
- `src/sre_triage_agent/llm/` — `OllamaProvider`, `OpenAICompatibleProvider`, `DeterministicFallbackProvider`, `MockLLMProvider`, and `ResilientLLMProvider`.
- `src/sre_triage_agent/triage/engine.py` — `TriageEngine` orchestration and grounding validation.
- `src/sre_triage_agent/approval/flow.py` — `InteractivePromptHandler`, `ScriptedDecisionHandler`, `ApprovalWorkflow`, and `process_remediations`.
- `src/sre_triage_agent/executor/dry_run.py` — Pure simulated `DryRunExecutor`.
- `src/sre_triage_agent/audit/store.py` — Thread-safe append-only JSONL `AuditLogger` and query/inspection helpers.
- `src/sre_triage_agent/cli.py` & `__main__.py` — Click + Rich CLI exposed as both `sre-agent` and `python -m sre_triage_agent`.

## Installation

Requires **Python 3.11+**.

```bash
pip install -e .
# Include dev/test tools (pytest, fastapi, uvicorn):
pip install -e ".[dev]"
```

After installation, invoke the CLI via either entrypoint:

```bash
sre-agent --help
python -m sre_triage_agent --help
```

## Bundled Scenarios & Runbooks

| Scenario (`--scenario`) | Matched Runbook (`samples/runbooks/`) | Incident Summary |
|---|---|---|
| `pod_oom_killed` | `k8s_oom_killed.yaml` | `checkout-api` pod terminated with `OOMKilled` (exit code 137) and `CrashLoopBackOff`. |
| `api_5xx_db_pool` | `api_db_pool_exhaustion.yaml` | `orders-service` HTTP 503 spike due to PostgreSQL connection pool saturation and row locks. |
| `disk_pressure` | `host_disk_pressure.yaml` | `worker-node-03` `/var/lib/containerd` filesystem exhaustion triggering `KubeNodeDiskPressure`. |
| `tls_cert_expiring` | `tls_cert_renewal.yaml` | Ingress TLS certificate for `api.prod.example.com` expiring within 48h (ACME HTTP-01 403). |

List bundled scenarios with alert and log counts:

```bash
sre-agent scenarios list
sre-agent scenarios list --json
```

## CLI Usage & Showcase Examples

### 1. Check Open-Source LLM Provider Readiness

```bash
# Ollama (default http://localhost:11434)
sre-agent providers check --provider ollama --base-url http://localhost:11434 --model llama3.1:8b

# Self-hosted OpenAI-compatible server (vLLM / llama.cpp / LocalAI / LM Studio)
sre-agent providers check --provider openai_compatible --base-url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct --api-key local-dev-token --json

# Offline deterministic provider
sre-agent providers check --provider deterministic --json
```

### 2. Triage Bundled Scenarios (Interactive & Scripted)

Interactive approval displays Rich panels (with `[LOW RISK]` / `[MEDIUM RISK]` / `[HIGH RISK]` badges, command previews, planned diffs, and evidence references) and prompts:

```text
Decision [a]pprove / [r]eject / [s]kip / [q]uit (optional ': reason'):
```

- Enter `a` / `approve`, `r` / `reject`, `s` / `skip`, or `q` / `quit`.
- Inline notes are supported (`a: verified namespace`, `r: wait for maintenance window`). When `ask_reason` is enabled (CLI interactive default), pressing Enter without an inline note also prompts for an optional reason line.

```bash
# Interactive triage on a single scenario (deterministic offline mode)
sre-agent triage --scenario pod_oom_killed --provider deterministic

# Scripted non-interactive triage across a single scenario with JSON output
sre-agent triage \
  --scenario api_5xx_db_pool \
  --provider deterministic \
  --decisions "approve: inspect active locks,approve: expand pool,reject: defer tx kill" \
  --operator oncall-sre \
  --audit-log ./audit_log.jsonl \
  --json

# Triage all 4 bundled scenarios in one run
sre-agent triage --all-scenarios --provider deterministic --decisions "approve,skip,skip" --json
```

### 3. Triage with Ollama or vLLM / `openai_compatible`

```bash
# Ollama with automatic fallback to deterministic runbook matching if unreachable
sre-agent triage --scenario pod_oom_killed --provider ollama --base-url http://localhost:11434 --model llama3.1:8b --allow-fallback

# Self-hosted OpenAI-compatible endpoint (strict mode: fail with exit code 1 instead of falling back)
sre-agent triage \
  --scenario tls_cert_expiring \
  --provider openai_compatible \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --api-key local-dev-token \
  --timeout 45 \
  --no-allow-fallback
```

### 4. Triage Against Live Logs & Prometheus / Alertmanager

`--log-file` is repeatable, supports JSONL and plain-text logs, and accepts `--tail-lines` (must be `>= 1`).

```bash
sre-agent triage \
  --log-file ./logs/app.log \
  --log-file ./logs/ingress.log.jsonl \
  --tail-lines 500 \
  --alert-url http://localhost:9090/api/v1/alerts \
  --alert-api-type auto \
  --include-resolved \
  --provider ollama \
  --decisions "approve,skip" \
  --json
```

For Alertmanager endpoints, point `--alert-url` to `http://localhost:9093/api/v2/alerts` (with `--alert-api-type alertmanager` or `auto`).

### 5. Inspect Persistent Audit Logs

```bash
# List all audit events (table or --json)
sre-agent audit list --audit-log ./audit_log.jsonl

# Filter by incident, run, event type, or operator decision
sre-agent audit list --audit-log ./audit_log.jsonl --incident-id inc-1234abcd --run-id run-5678ef90 --event-type operator_decision --decision approved --json

# Show full chronological lifecycle for a single incident
sre-agent audit show inc-1234abcd --audit-log ./audit_log.jsonl
sre-agent audit show inc-1234abcd --audit-log ./audit_log.jsonl --json
```

## CLI Reference & Exit Codes

### Common `triage` Flags
- Sample mode: `--scenario <name>`, `--all-scenarios`
- Live mode: `--log-file <path>` (repeatable), `--alert-url <url>`, `--alert-api-type [auto|prometheus|alertmanager]`, `--tail-lines <int>`, `--include-resolved / --no-include-resolved`
- LLM options: `--provider [ollama|openai_compatible|deterministic]`, `--base-url <url>`, `--model <name>`, `--api-key <key>`, `--timeout <float>`, `--allow-fallback / --no-allow-fallback`
- Approval & output: `--decisions <comma-separated-approve-reject-skip-quit>`, `--operator <name>`, `--audit-log <path>`, `--config <yaml-path>`, `--json`

### Exit Codes
- `0` — Success (including interactive `quit` abort, which records `aborted: true` and exits cleanly).
- `1` — Operational, configuration, ingestion, provider (`--no-allow-fallback` / unhealthy `providers check`), approval/audit I/O, or missing `audit show` incident error.
- `2` — Invalid CLI usage (conflicting flags, missing source, invalid `--decisions`, `--timeout <= 0`, `--tail-lines < 1`).

## Configuration: Environment Variables & YAML

Configuration follows strict precedence:
**`CLI flags > SRE_AGENT_* environment variables > YAML config (--config) > built-in defaults`**

See `.env.example` and `config.example.yaml` for ready-to-copy templates (`sre-agent triage --config config.example.yaml ...`).

| Environment Variable | YAML Path (`config.example.yaml`) | Default | Description |
|---|---|---|---|
| `SRE_AGENT_LLM_PROVIDER` | `llm.provider` | `ollama` | `ollama`, `openai_compatible`, or `deterministic` |
| `SRE_AGENT_LLM_BASE_URL` | `llm.base_url` | `http://localhost:11434` | Local/self-hosted LLM endpoint base URL |
| `SRE_AGENT_LLM_MODEL` | `llm.model` | `llama3.1:8b` | Open-source model identifier |
| `SRE_AGENT_LLM_API_KEY` | `llm.api_key` | `None` | Optional Bearer token for OpenAI-compatible servers |
| `SRE_AGENT_LLM_TIMEOUT_SECONDS` | `llm.timeout_seconds` | `30.0` | HTTP timeout in seconds (`> 0`) |
| `SRE_AGENT_LLM_ALLOW_FALLBACK` | `llm.allow_fallback` | `true` | Fall back to deterministic runbooks if LLM fails |
| `SRE_AGENT_PROMETHEUS_URL` | `ingestion.prometheus_url` | `None` | Live Prometheus or Alertmanager alerts URL |
| `SRE_AGENT_ALERT_API_TYPE` | `ingestion.alert_api_type` | `auto` | `auto`, `prometheus`, or `alertmanager` |
| `SRE_AGENT_LOG_FILE` | `ingestion.log_file` | `None` | Default live log file path |
| `SRE_AGENT_TAIL_LINES` | `ingestion.tail_lines` | `None` | Optional positive integer tail limit for live logs |
| `SRE_AGENT_INCLUDE_RESOLVED` | `ingestion.include_resolved` | `false` | Include resolved alerts during ingestion |
| `SRE_AGENT_AUDIT_LOG_PATH` | `audit.audit_log_path` | `./audit_log.jsonl` | Persistent append-only JSONL audit log path |
| `SRE_AGENT_OPERATOR` | `operator` | `sre-operator` | Operator identity recorded in audit events |

## Running Tests

```bash
pytest -v tests/unit
pytest -v tests/e2e
```
