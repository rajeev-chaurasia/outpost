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

21 questions across three tenants: 15 the corpus can answer, and 6 it cannot. Two rates,
because either alone is easy to game. Refusing everything scores a perfect refusal rate;
answering only the easy questions scores a perfect unsupported rate.

| Tenant | Answerable | Assertions scored | Unsupported | Refusals correct |
| :--- | :---: | :---: | :---: | :---: |
| `dealer_ar` | 5 | 5 | 0 | 2 of 2 |
| `claims_intake` | 5 | 5 | 0 | 2 of 2 |
| `utility_ops` | 5 | 5 | 0 | 2 of 2 |
| **Total** | **15** | **15** | **0 (0.0 percent)** | **6 of 6 (100 percent)** |

Neither direction of error occurred: no answerable question was refused, and no unanswerable
question was answered.

Artifact: [`eval/artifacts/grounding_results.json`](eval/artifacts/grounding_results.json)

Each answer replays a real recorded `gpt-oss-120b` response, not a hand-written fake. Every
sentence is matched against the retrieved spans; anything that clears the overlap threshold
gets a citation, anything that does not is counted as unsupported rather than quietly
accepted. Reported per tenant, never pooled, so a weak tenant cannot hide behind a strong
one.

An unsupported rate says nothing about whether a citation is deserved, only about whether
one was found. The next table tests that separately.

### Does a citation mean the source supports the sentence?

12 cases pairing a real source with four kinds of sentence. The adversarial ones borrow
almost all of the source's vocabulary while changing what it says, which is precisely what
token overlap cannot see.

| Case type | Cases | Cited | Correct |
| :--- | :---: | :---: | :---: |
| Faithful restatement | 3 | 3 | **3** |
| Negation inverted | 3 | 0 | **3** |
| Value substituted | 3 | 0 | **3** |
| Unrelated subject | 3 | 0 | **3** |

False citations on adversarial cases: **0 of 6**.
Artifact: [`eval/artifacts/entailment_results.json`](eval/artifacts/entailment_results.json)

Overlap alone failed this badly: negation-inverted claims were cited 3 of 3, and 5 of the 6
adversarial cases were wrongly cited. Grounding now rejects a span whose negation cues or
numeric values disagree with the sentence. Faithful restatements are still cited 3 of 3, so
the improvement is not bought by refusing everything.

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

20 calls per arm against a declared budget of 8000 ms. 20 samples cannot support a p99, so
p50 and p90 are reported with the max stated separately as the tail.

| Arm | p50 | p90 | max | Answered |
| :--- | ---: | ---: | ---: | :---: |
| `gpt-oss-120b` raw | 4372 ms | 5304 ms | 7678 ms | 20 of 20 |
| `gpt-oss-120b` paced | 2738 ms | 6254 ms | 6422 ms | 20 of 20 |
| `gpt-oss-20b` raw | 9242 ms | 27545 ms | 28038 ms | **10 of 20** |
| `gpt-oss-20b` paced | 3314 ms | 28777 ms | 29299 ms | **14 of 20** |
| **Budget enforced, with fallback** | **3096 ms** | **4806 ms** | **6413 ms** | **19 of 20** |

Artifact: [`eval/artifacts/latency_results.json`](eval/artifacts/latency_results.json)

The last row is what the system runs. The primary's transport deadline is the tenant's
budget, so a slow call is cut off at 8000 ms rather than run to completion and the secondary
answers instead. On this run it fell back once, answered 19 of 20, and its slowest request
finished in 6413 ms, inside a single budget. The design only guarantees two budgets, one
cut-off attempt plus one real one, and an earlier run did reach 15782 ms.

The comparison worth reading is the last two rows: `gpt-oss-20b` on its own answered 10 of
20, and the same model as a fallback behind a deadline-bounded primary answered 19 of 20.

**Every failure in every arm was a client timeout. Not one HTTP 429 was returned.** That
rules out rate limiting, which would reject fast rather than hang.

These numbers move a lot between runs. `gpt-oss-120b` p90 has come in at 5304, 8034, and
9419 ms across three runs, and `gpt-oss-20b` failures at 4, 10, and 10 of 20. Treat any
single run as one sample of a shared endpoint, not a property of the models.

## Null results and known gaps

Measurements that came back negative, and the limits of what the system does today.

**Request pacing does not reliably reduce latency.** Across three runs it has helped and
hurt in no stable pattern: on the latest it improved `gpt-oss-120b` p50 (4372 ms to 2738 ms)
while worsening its p90, and recovered four of `gpt-oss-20b`'s ten failures. On an earlier
run it made `gpt-oss-20b` sharply worse. Pacing is not the lever, and `PacedProvider` exists
so the measurement can be reproduced rather than because the serving path uses it.

The mechanism is not what a hanging endpoint usually suggests. Every failure recorded across
every arm is a client timeout, with zero HTTP 429 responses, so this is not rate limiting.
Calls are accepted and then not answered within the deadline.

What bounds latency is the transport deadline. The primary's socket timeout is the tenant's
budget, so a slow call is cut off at 8000 ms and the secondary takes over. That path answered
19 of 20 against 10 of 20 for the smaller model on its own.

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

**The contradiction guards initially broke list answers.** Requiring the cited span to
contain every number in the sentence refused any answer drawing on several records, which
cost 3 of 15 answerable questions, one per tenant, all list-style. Fabricated values are now
judged against all retrieved evidence while negation stays per-span. This only surfaced
because the question set was large enough to contain list questions; it was invisible at one
question per tenant.

**Exhaustive questions are answered from top-k, not a full scan.** "Which ones" and "how
many" questions retrieve the top matches rather than every matching record, so a tenant
with hundreds of matching rows could receive a confidently incomplete list. Answering those
correctly needs an aggregation path over the mapped records rather than retrieval.

**Grounding is guarded overlap, still not entailment.** Negation and substituted values are
checked explicitly, which covers the two ways a borrowed sentence usually inverts its
source and takes the measured false-citation rate to zero on the current adversarial set.
Neither check reasons about meaning. A sentence that contradicts its source through word
choice alone, with no negation cue and no changed number, would still be cited. Catching
that needs an entailment model, and the adversarial set is 12 cases, small enough that it
should be read as a regression gate rather than a general guarantee.

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

145 tests, 94 percent branch coverage, `mypy --strict` clean across 71 files, and every
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
