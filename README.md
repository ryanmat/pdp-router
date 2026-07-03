<div align="center">

# pdp-router

**An outcome-fed router for heterogeneous AI models.**

[![ci](https://github.com/ryanmat/pdp-router/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanmat/pdp-router/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11_%7C_3.12_%7C_3.13-3776AB?logo=python&logoColor=white)](https://github.com/ryanmat/pdp-router/blob/main/pyproject.toml)
[![license](https://img.shields.io/github/license/ryanmat/pdp-router)](https://github.com/ryanmat/pdp-router/blob/main/LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

pdp-router routes a single request across many AI models and learns, from real
outcomes, which model to trust for a given job. It starts from a confidence cascade, logs every routing
decision, and lets recorded outcomes reshape the trust weights and Thompson
Sampling posteriors behind future decisions. Point an OpenAI-compatible client
at it, and it picks the model, or convenes a panel of models, for you.

It runs in production for its author, routing chat and agent traffic across
eleven models from six training lineages. The premise: models come and go, but
the routing-and-feedback architecture is the durable part.

## Features

- **Outcome-fed.** Every routing decision is logged, and real outcomes flow back
  into per-model, per-domain trust weights and Thompson Sampling posteriors.
- **Confidence cascade.** Start at the cheapest capable model and escalate only
  when confidence is low. Pay for the big model when the question earns it.
- **Multi-model panel.** On hard queries, fan out to several lineage-diverse
  models in parallel and synthesize their answers with a chair model.
- **Two OpenAI-compatible surfaces.** A default one that reports the routed
  model inline, and a byte-faithful one that strict agent clients accept without
  knowing a router is there.
- **Cognitive diversity by design.** Eleven models across six training lineages,
  chosen because different lineages fail differently.
- **Bring your own outcomes.** Point it at a SQLite trust DB your grading
  process populates. With no DB, it routes on the static cascade.
- **Observability built in.** Optional OpenTelemetry traces, metrics, and GenAI
  instrumentation.

## What this is (and is not)

This is a reference implementation you can actually run, published in full. It
is honest about what it is: a personal system, not a company product, so there
is no support SLA, no roadmap promises, and the repo is a one-way mirror from a
private upstream (see CONTRIBUTING).

Where it fits: LiteLLM and OpenRouter are excellent gateways for reaching many
models behind one API. pdp-router is a smaller, more opinionated thing: a
learned router with a real feedback loop wrapped around that idea. If you want a
gateway, use those. If you want a router that measures its own decisions and
improves from them, clone this and bring your outcomes.

## Honest results

Most router READMEs claim the learned router wins. Here is real data saying it
is not that simple.

At a 100% bandit-routing flip on one production domain, the bandit's outcome
rate was **0.800 vs the static confidence cascade's 0.893** (n=80 graded
decisions, Fisher exact p=0.017), monotonically worsening in 20-row windows.
The flip was reverted to a 10% canary the same day, under a tripwire
pre-committed before the flip.

> [!IMPORTANT]
> The diagnosis is the most useful thing in this repo: the reward signal (did
> the alert clear?) was **blind to output quality**. A model emitting garbage
> text was not penalized as long as the alert resolved. A learned router can
> only beat a good heuristic when its reward measures what you actually care
> about. The fix upstream is a rubric-based judge ensemble as a second outcome
> source, soaking before it earns bandit weight.

Takeaways if you build on this:

1. The router is the easy part. **Outcome plumbing is the product.**
2. Pre-commit your revert tripwires before you flip traffic.
3. Audit your reward signal for blindness before trusting any win.

## Architecture

```mermaid
flowchart TD
    REQ["request<br/>(OpenAI-compatible /v1/chat/completions)"] --> CLS["complexity classifier<br/>(cheap, fast model)"]
    CLS -- "panel_score at threshold<br/>(decomposable query)" --> PANEL["panel: N lineage-diverse<br/>arms in parallel"]
    PANEL --> CHAIR["chair synthesis"]
    CLS -- "otherwise" --> CASCADE["cascade: cheapest capable arm,<br/>escalate on low confidence;<br/>Thompson Sampling when<br/>ROUTING_MODE=bandit"]
    CHAIR --> LOG["routing-decision rows<br/>(JSONL inbox)"]
    CASCADE --> LOG
    LOG --> STORE[("your outcome store<br/>+ grading process")]
    STORE --> RECALC["posterior recalc<br/>trust weights"]
    RECALC -. "the loop closes" .-> CASCADE
```

| Module | What it does |
|---|---|
| `_router.py` | Model registry, capability tiers, confidence cascade, trust-adjusted routing |
| `_bandit.py` | Thompson Sampling over (model, domain, context) with time-discounted observations |
| `_panel.py` | Parallel multi-model panel and chair synthesis, lineage diversity selection |
| `_clients.py` | Provider clients (Anthropic, Gemini and Vertex AI, DeepSeek, OpenRouter) behind one `LLMClient` protocol |
| `_proxy.py` | FastAPI OpenAI-compatible endpoints: classify, route, execute, log. Two surfaces (default and OpenAI-faithful) |
| `_effort.py` | Maps the classifier complexity score to a per-provider reasoning-effort level |
| `_cost.py` | Per-provider pricing and cost estimation |
| `_models.py` | Model ID constants and the live roster |
| `_tracing.py` | Optional OpenTelemetry export (traces, metrics, logs, GenAI instrumentation) |

The roster spans six training lineages: Anthropic (Opus, Sonnet, Haiku), Google
(Gemini Pro, Flash, Flash-Lite), Meta Llama 4 (Scout and Maverick via Vertex
AI), DeepSeek, and OpenAI and Qwen via OpenRouter. Eleven models in total. The
deliberate point is **cognitive diversity**: different training lineages fail
differently, and the panel selects across lineages rather than stacking
near-clones.

## Two surfaces, one of them for strict agent clients

There are two OpenAI-compatible endpoints, and the difference is a small lesson
in how picky real agent clients are.

`/v1/chat/completions` is the default. It leads each stream with a small
non-standard event naming the model the router picked, which is handy when you
control the client and want the routing decision inline.

`/openai/v1/chat/completions` is byte-faithful to OpenAI's streaming shape: a
leading assistant-role delta, then content, then a terminal
`finish_reason: "stop"` and `[DONE]`. Nothing proprietary on the wire.

Why two? Point a strict agent client at the default surface and it rejects the
leading event and the missing terminal `finish_reason` as an unexpected end of
stream. [Crush](https://github.com/charmbracelet/crush) does exactly that. So
the faithful surface exists to speak the dialect those clients expect, and they
talk to it without ever knowing a router is on the other end.

The payoff is the panel. When the classifier scores a query as decomposable,
the faithful surface fans out to several lineage-diverse models and streams the
**chair synthesis** of their answers back through that same OpenAI shape. The
client asked what it thinks is one model a hard question, and transparently got
a multi-model panel. Keep-alive comments bridge the quiet while the panel runs,
so the stream never looks dead. (The panel is opt-in; enable it with
`PROXY_AUTOPANEL_ENABLED=1`.)

> [!IMPORTANT]
> Whether the synthesis actually beats the single best member is an open
> question here, not a settled win. It is measured the same honest way the
> bandit was: capture real turns, grade them, let the data decide. The
> architecture makes the panel cheap to route. It does not make the panel
> automatically right.

## Quickstart

```bash
git clone https://github.com/ryanmat/pdp-router && cd pdp-router
cp .env.example .env          # add your own API keys
uv sync --all-extras
uv run uvicorn pdp_router._proxy:app --host 127.0.0.1 --port 7741
```

Then point any OpenAI-compatible client at it:

```bash
curl -s http://127.0.0.1:7741/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "pdp-auto", "messages": [{"role": "user", "content": "hello"}]}'
```

`model: pdp-auto` engages routing; a concrete model ID pins that model.
`GET /v1/models` lists the roster, `GET /health` shows what it looks like.
Strict agent clients should use the faithful surface at
`/openai/v1/chat/completions`.

Works out of the box with just API keys: with no trust DB present, the
confidence cascade routes on defaults. The learned layer activates when you
bring outcomes (next section).

The panel, streaming, effort routing, and web search are opt-in. Turn them on
with environment variables (see `.env.example`); for example
`PROXY_AUTOPANEL_ENABLED=1` convenes the panel on hard queries. Defaults are
conservative: a fresh clone runs the single-model cascade until you enable the
extras.

## Bring your own outcomes

The bandit and trust weights read from a SQLite DB (default
`~/.pdp-router/pdp_tracker.db`, override `PROXY_TRUST_DB`) that **your** grading
process populates:

```sql
CREATE TABLE model_trust (model_id TEXT, weight REAL);          -- 0.0-1.0
CREATE TABLE bandit_state (
  model_id TEXT, mu REAL, sigma REAL, n_obs INTEGER,
  sum_reward REAL, sum_sq_reward REAL,
  effective_n REAL, effective_sum REAL                          -- discounted
);
```

Every request appends a routing-decision row to a JSONL inbox (default
`~/.pdp-router/inbox/proxy-YYYYMMDD.jsonl`, override `PROXY_ROUTING_INBOX_DIR`)
carrying the selected model, confidence, context bucket, and a request UUID that
also rides the `X-PDP-Prediction-Id` response header. Drain those rows into your
store, grade them against whatever outcome you care about, recompute posteriors,
write them back. That loop, not this code, is where the value lives.

## What is deliberately not here

pdp-router was extracted from a larger private system that runs it in
production. That system's outcome store, judge ensemble and rubrics, drift
watchdog, and earned trust weights are not included here: they are specific to
their domain, and they are exactly the substance you would replace with your
own. What is here is the full routing-and-feedback machinery and the schema
contract you plug outcomes into.

## Further reading

- Chapelle & Li, *An Empirical Evaluation of Thompson Sampling* (NeurIPS 2011)
- Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data* (arXiv:2406.18665)
- Verga et al., *Replacing Judges with Juries* (arXiv:2404.18796), on multi-family
  judge panels and intra-family self-preference bias
- Snell et al., *Scaling LLM Test-Time Compute Optimally* (arXiv:2408.03314), on
  why parallel beats sequential at fixed budget

## License

Apache-2.0. See [LICENSE](LICENSE).

---

ARE YOU LIVING IN THE REAL WORLD?
