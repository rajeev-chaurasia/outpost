# outpost

An agent that answers questions from a customer's messy documents and records, cites the
exact source text behind every sentence it writes, and gets deployed for a new customer in
a different industry by adding a config file rather than editing code.

The interesting claim here is not that a language model can answer questions. Any model
does that. The claim is that a third customer, in a vertical the codebase had never seen,
went live in 123 seconds with zero code changes, kept its data provably separate from the
other two, and refused to answer when it could not find evidence. Each of those is a number
backed by a committed artifact that CI recomputes on every push.

## The problem

Getting an AI system working for one customer is a demo. Getting it working for the next
one is the job, and it is where most of these systems fall apart.

The second customer has different documents, different field names, different vocabulary,
and different rules about what the system is allowed to do on their behalf. If any of that
lives in the code, every new customer is a fork. Four failures show up repeatedly:

1. **The domain leaks into the code.** Entity names, field names, and business vocabulary
   get hardcoded. Customer two needs a rewrite, not a config.
2. **Isolation is bolted on afterward.** Search runs over everything, then filters results
   by tenant. When another tenant's documents score highly, they crowd the top-k window and
   the authorized results never make it in, so recall silently collapses.
3. **The system guesses instead of refusing.** Asked something the documents do not cover,
   it produces a fluent, confident, unsupported answer, which is worse than saying no.
4. **Nothing is measured.** "It works well" is not a number, and cannot be regression
   tested.

outpost is built to make each of those a measurement instead of an opinion.

## Results

Every number below comes from a JSON file in `eval/artifacts/`. CI recomputes the
deterministic ones on every push and fails if a committed file disagrees with a fresh run,
so a number in this README cannot drift from the run that produced it.

### Cold onboarding of a third tenant

The headline measurement. The first two tenants were built alongside the system. The third
was chosen and onboarded after everything else was finished, with a timer running.

| Measure | Result |
| :--- | :--- |
| Wall clock, vertical chosen to grounded answer | **123 seconds** |
| Manual interventions in `src/outpost/` | **0** |
| Fields auto-mapped from messy source data | 77.3 percent (51 of 66) |
| Verification answer | Fully grounded, one citation, rung FULL |
| Cross-tenant leaks | 0 |

Artifact: [`eval/artifacts/onboarding_results.json`](eval/artifacts/onboarding_results.json)

The tenant was energy utility field operations, a vertical touched nowhere else in the
project. Onboarding it meant writing `tenants/utility_ops/config.yaml`, dropping in its
data, and running two CLI commands. Nothing under `src/outpost/` was edited.

### Tenant isolation

14 adversarial probes across three tenants, each querying as one tenant using another
tenant's vocabulary or shared tokens such as bare dates and dollar amounts.

| Measure | Result |
| :--- | :--- |
| Adversarial cases | 14 |
| Cross-tenant leaks | **0** |
| Authorized results, filtering during traversal | **42** |
| Authorized results, post-filtering control | **2** |

Artifact: [`eval/artifacts/isolation_results.json`](eval/artifacts/isolation_results.json)

That last pair is the point. Both approaches leak nothing. The difference is that
post-filtering loses 40 of 42 authorized results, because other tenants' chunks fill the
top-k window before the filter runs. Filtering during index traversal keeps them. The gap
widened as the index grew, which is what you would expect: the more a shared index holds,
the more of the window a post-filter throws away.

### Grounding

Reported per tenant, never pooled, so a weak tenant cannot hide behind a strong one.

| Tenant | Citations | Unsupported sentences | Unsupported rate |
| :--- | :---: | :---: | :---: |
| `dealer_ar` | 1 | 0 | 0.0 percent |
| `claims_intake` | 1 | 0 | 0.0 percent |
| `utility_ops` | 1 | 0 | 0.0 percent |

Artifact: [`eval/artifacts/grounding_results.json`](eval/artifacts/grounding_results.json)

Each answer replays a real recorded `gpt-oss-120b` response, not a hand-written fake. Every
sentence is matched against the retrieved spans; anything that clears the overlap threshold
gets a citation, anything that does not is counted as unsupported rather than quietly
accepted.

### The failure ladder

Five behaviors, each forced by a scenario built to trigger exactly that rung.

| Rung | Trigger | Behavior | Fires correctly |
| :--- | :--- | :--- | :---: |
| 1 FULL | Evidence found, action permitted | Answer with citations | yes |
| 2 PARTIAL | Some claims unsupported | Answer, and name the gaps | yes |
| 3 ACTION_DECLINED | Action outside tenant policy | Draft it, do not run it | yes |
| 4 PROVIDER_FALLBACK | Primary model failed | Secondary answers, fallback recorded | yes |
| 5 REFUSED | Not enough evidence | Refuse, and state what was tried | yes |

Correct rung rate: **5 of 5**.
Artifact: [`eval/artifacts/degradation_results.json`](eval/artifacts/degradation_results.json)

### Latency

20 sequential calls per model against the live API, in three arms. 20 samples cannot
estimate a p99, so p50 and p90 are reported with the max stated separately as the tail.

| Arm | p50 | p90 | max | Answered |
| :--- | ---: | ---: | ---: | :---: |
| `gpt-oss-120b` raw | 3918 ms | 9419 ms | 12023 ms | 20 of 20 |
| `gpt-oss-120b` paced | 3420 ms | 8002 ms | 8501 ms | 20 of 20 |
| `gpt-oss-20b` raw | 1197 ms | 2284 ms | 5456 ms | 16 of 20 |
| `gpt-oss-20b` paced | 1328 ms | 30133 ms | 30363 ms | 15 of 20 |
| **Budget enforced, with fallback** | **3142 ms** | **9443 ms** | **15782 ms** | **19 of 20** |

Artifact: [`eval/artifacts/latency_results.json`](eval/artifacts/latency_results.json)

The last row is what the system actually runs: the primary's transport deadline is set to
the tenant's 8000 ms budget, so a slow call is cut off at the budget rather than run to
completion, and the secondary model takes over. It fell back on 4 of 20 calls and answered
19 of 20, against 16 of 20 for the smaller model alone. Worst case is bounded by two
budgets, one for the cut-off primary plus one for the secondary, and the measured max of
15782 ms sits under that 16000 ms ceiling.


## Null results and known gaps

Measurements that came back negative, and the limits of what the system does today.

**Request pacing does not reliably reduce latency.** Spacing calls two seconds apart
improved `gpt-oss-120b` (p90 9419 ms to 8002 ms) and degraded `gpt-oss-20b` sharply (p90
2284 ms to 30133 ms, five calls hitting the 30 second ceiling). The endpoint is variable
rather than contended, so pacing is not the lever. `PacedProvider` exists so the
measurement can be reproduced, and is deliberately not in the serving path.

What bounds latency instead is the transport deadline. The primary's socket timeout is the
tenant's budget, so a slow call is cut off at 8000 ms and the secondary takes over. That
path answers 19 of 20 against 16 of 20 for the smaller model alone.

**A fallback costs more than one budget.** When the primary is cut off and the secondary
answers, user-visible latency is one cut-off attempt plus one real one, so it exceeds a
single budget by construction. Bounding a request to one budget would mean racing both
models concurrently rather than falling back serially.

**Two of the six models tested could not hold structured output.** Probing the available
models on build.nvidia.com for tool calling, structured output, and latency, three runs
each: `nemotron-3-super-120b` returned malformed JSON on two of three `response_format`
attempts, once emitting `{\n{\n` and omitting a required field, and missed a tool call on
one of three. `kimi-k3` failed structured output three times out of three. `minimax-m3`
does not support tool calling at all. This is why the agent structures everything through
tool calls, which was the reliable path on every model that supported it.

**The primary and fallback models share a family.** Both are `gpt-oss` on the same NVIDIA
endpoint, so the provider abstraction is proven across models rather than across vendors.
`RecordedProvider` is a third implementation of the same protocol with no HTTP at all, and
a test verifies that swapping models is a config change. Genuine vendor diversity would be
a stronger claim.

**Exhaustive questions are answered from top-k, not a full scan.** "Which ones" and "how
many" questions retrieve the top matches rather than every matching record, so a tenant
with hundreds of matching rows could receive a confidently incomplete list. Answering those
correctly needs an aggregation path over the mapped records rather than retrieval.

**Grounding is lexical overlap, not entailment.** A sentence that reuses a source's
vocabulary while inverting its meaning would still be cited. Catching that needs an
entailment model.

**Dense retrieval is weak on exact identifiers.** Invoice numbers, claim numbers, and work
order ids are matched by BM25, not by embedding similarity. Hybrid fusion covers this, but
dense retrieval alone would be the wrong choice for this data.

## Architecture

```
tenants/<id>/config.yaml          the entire per-customer surface
        |
        v
  [ ontology ]  entities, relations, aliases, allowed actions, budgets
        |
        v
  [ connectors ]  csv_export | pdf_text | rest_mock
        |
        v
  [ mapping ]  alias resolution -> coercion -> MappingReport
        |                              (mapped | needs_review | unmapped)
        v
  [ retrieval ]  one shared index across all tenants, holding both
        |          documents and mapped records rendered to text
        |          BM25 postings  +  dense vectors
        |          tenant filter applied DURING traversal
        v
  [ agent ]  plan -> tools -> ground -> degrade -> audit
        |      search, fetch_entity (read)
        |      flag_discrepancy, draft_response (write, action-gated)
        v
  [ serve ]  FastAPI  ->  React dashboard
```

### Five decisions worth explaining

**One shared index, filtered during traversal.** Every tenant's chunks live in the same
BM25 index and the same embedding matrix, which is the realistic deployment shape. Isolation
comes from restricting the candidate set before anything is scored: posting lists are
intersected with the tenant's chunk ids, and embedding matrix rows are selected, before
similarity is computed. The alternative, scoring everything and filtering the results, leaks
nothing but loses authorized recall, measured above at 2 results against 42.

**Structured records are indexed as text, not given a separate lookup path.** Each mapped
record is rendered to one line and indexed like a document, so a question answered from a
CSV row is retrieved, grounded, cited, and tenant-isolated by exactly the same code that
handles a question answered from a PDF. Only fields that mapped cleanly are rendered, so
the agent cannot cite a value a human has not confirmed. Rows sharing a key are suffixed
rather than overwritten, because the mapping layer keeps duplicates deliberately.

**The ontology layer exists to keep domain words out of the code.** A test walks every
Python file under `src/outpost/` and fails if it finds any of eleven domain words drawn
from all three tenants. Customer vocabulary lives in YAML. This test caught two real
collisions during the build and is the reason a third tenant needed no code changes.

**Structured output through tool calls, not `response_format`.** Chosen from measurement,
not preference, after the structured-output failures described above.

**Everything replays in CI without an API key.** Model responses are recorded to
`tests/fixtures/llm/`, keyed by a hash of the normalized request. Embeddings are cached to
a committed `.npz`, keyed by content hash. A cache miss raises a typed error rather than
silently falling back to a live call, so a missing fixture fails loudly instead of quietly
depending on the network. The served app wraps that same cache with a live fallback, since
a real user's question is never pre-cached.

### Layout

| Path | Contents |
| :--- | :--- |
| `src/outpost/ontology/` | Tenant config schema, YAML loader with line-accurate errors, tenant discovery |
| `src/outpost/connectors/` | CSV, PDF text, and a REST mock with typed failure modes |
| `src/outpost/mapping/` | Alias resolution, value coercion, per-field mapping report |
| `src/outpost/retrieval/` | BM25, dense store, rank fusion, traversal-time tenant isolation |
| `src/outpost/llm/` | Provider protocol, NVIDIA client, recorded provider, fallback, budgets |
| `src/outpost/agent/` | Planner, tools, citation grounding, failure ladder, audit log |
| `src/outpost/serve/` | FastAPI application and routes |
| `src/outpost/onboard/` | The onboarding CLI |
| `eval/` | Measurement harnesses and committed artifacts |
| `dashboard/` | React dashboard |

## Running it

Requires Python 3.12 and `uv`. An NVIDIA API key from build.nvidia.com is needed to ask
live questions, but not to run the tests.

```bash
uv sync
cp .env.example .env        # then add your LLM_API_KEY
```

Index a tenant and ask it something:

```bash
uv run python -m outpost.onboard.cli index dealer_ar
uv run python -m outpost.onboard.cli ask dealer_ar "Was invoice INV-1001 paid, and how?"
```

Run the API and the dashboard:

```bash
uv run uvicorn outpost.serve.app:app --port 8000
cd dashboard && npm install && npm run dev
```

Onboard a new tenant by creating `tenants/<id>/config.yaml`, dropping its data under
`tenants/<id>/fixtures/`, and running the two CLI commands above. Nothing else.

Run the checks CI runs, none of which need a key:

```bash
uv run ruff check && uv run ruff format --check
uv run mypy --strict src eval
uv run pytest --cov --cov-branch --cov-fail-under=85
uv run python -m eval.runner --verify-artifacts
```

Regenerate the artifacts that need a live key:

```bash
uv run python -m eval.latency.measure
uv run python scripts/generate_llm_fixtures.py
```

## Limitations

Beyond the null results above:

- **The corpus is small.** 39 indexed chunks across three tenants, 7 from documents and 32
  from structured records, sized to exercise every messy case deliberately rather than to
  prove anything about scale. BM25 and the embedding store are both in-memory and rebuilt
  at startup.
- **Indexes rebuild on every start.** There is no persistence layer for the retrieval
  index, so startup time grows with corpus size.
- **The audit log is a single SQLite file.** Append-only by construction, with no update or
  delete path anywhere in the class, but not sharded, replicated, or retained on a policy.
- **No authentication.** The API trusts the `tenant_id` in the URL. A real deployment needs
  the tenant identity to come from an authenticated session, not a path parameter.
- **Latency is measured on a shared free endpoint.** The numbers reflect that endpoint,
  not the models' capability on dedicated capacity.

## Verification

135 tests, 94 percent branch coverage, `mypy --strict` clean across 70 files, and every
published number recomputed from its artifact on each CI run.

```
uv sync --locked
ruff check
ruff format --check
mypy --strict src eval
pytest --cov --cov-branch --cov-fail-under=85
python -m eval.runner --verify-artifacts
```

## License

MIT.
