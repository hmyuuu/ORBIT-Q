# Kimi K3 benchmark observations

Recorded on 2026-07-29 and 2026-07-30 (Asia/Shanghai).

## Configuration

- Solver: official Kimi Code harness 0.29.2
- Model: `kimi-code/k3`
- Framework: TensorCircuit
- Docker image used for these local runs:
  `challenge-benchmark-quantum-tensorcircuit:kimi-k3`
- Solver reasoning effort: `max` observations followed by a controlled `high`
  campaign
- Harbor agent timeout: 1,800 seconds
- Evaluator solution time limit: 300 seconds
- Local CPU override: 8 CPUs
- Local Docker memory available: about 9.7 GiB; canonical tasks request 32 GiB
- Verifier audit model: `gpt-5`

Kimi Code's local model catalog advertises `low`, `high`, and `max` as K3's
supported effort levels, with `high` as the default. The two campaigns set
different effort values in Kimi Code; trace length did not determine the
labels.

We ran this local campaign below the canonical CPU and memory allocation.
Challenge 09 hit the resource limit when a vectorized batch requested an
11.0 GB buffer, so these numbers do not constitute a canonical-resource
leaderboard submission.

The host and container authentication preflights passed after refreshing the
Kimi login. An earlier HTTP 401 run was an account-entitlement/authentication
failure and is not included as a K3 benchmark result.

The first `high` comparison attempt exposed a harness issue. The OAuth provider
rotates its refresh token, and the initial adapter discarded refreshed state
with the isolated container copy. The next host request returned
`auth.login_required`. That 34-second job
(`jobs/tensorcircuit-kimi-k3-high-20260729-challenge-02`) is excluded. The
adapter now atomically synchronizes refreshed credential files back to the
same host Kimi home before deleting the container copy, so one new host login
can support a sequential suite.

## Max-effort results

| Challenge | Harbor outcome | Wall time | Solution artifact | Observation |
| --- | --- | ---: | --- | --- |
| 01 | `AgentTimeoutError` | 30m 29s | No | K3 actively inspected TensorCircuit MPS APIs and tested several approaches. A direct MPS value-and-gradient prototype took about 68.3s on its first call and 19.2s on its second. A later MPO/JAX prototype remained in compilation at high CPU use and did not finish before the agent timeout. |
| 02 | `AgentTimeoutError` | 30m 28s | No | K3 built a 12-qubit statevector/JAX prototype, stopped an excessively long initial compilation, then switched to a TensorCircuit MPS fallback. It diagnosed and patched a reusable-gate-node error, but the agent window ended before it wrote the required solution artifact. |
| 03 | Excluded | 29s | No | We interrupted this run while switching the campaign from `max` to `high`. Its recorded `NonZeroAgentExitCodeError` does not measure K3. |

We removed the raw max-effort job directories from `jobs/` after recording
these observations. A publishable K3 result aggregate must exclude them.

## Interpretation and next experiment

In both comparable runs, `max` effort spent several minutes per reasoning turn
and left little time for implementation, compilation, and end-to-end
validation. Reasoning effort contributed to the timeouts. TensorCircuit/JAX
compilation costs and implementation bottlenecks also consumed the agent
window. Two trials cannot predict all 12 challenges.

Neither timeout produced a candidate artifact, so the functional and policy
verifiers had no submission to score. The zero rewards above are missing-result
placeholders.

The controlled comparison uses K3's default `high` effort while keeping the
model, framework, image, CPU allocation, agent timeout, and verifier model
unchanged. Challenge 02 provides the first A/B case because its max-effort
trace reached multiple implementation attempts.

## High-effort setup attempts

Two setup attempts are excluded from the high-effort comparison:

- The first attempt ended after 34 seconds with `auth.login_required`. This
  exposed and led to the OAuth refresh-persistence fix described above.
- The second attempt began normally and persisted its refreshed credential,
  but the host slept while the Kimi stream was open. When the machine resumed,
  the provider reported `provider.connection_error: terminated`. Its roughly
  six-hour wall time is suspended-host time, not agent reasoning or execution
  time, and it produced no candidate artifact.

The controlled retry prevents host sleep and retains the task's 1,800-second
agent timeout.

## High-effort results

| Challenge | Harbor outcome | Wall time | Solution artifact | Observation |
| --- | --- | ---: | --- | --- |
| 01 | `AgentTimeoutError` | 30m 30s | No | K3 reached a working Quimb DMRG-to-TensorCircuit MPS conversion and matched the source energy. The first four-layer TensorCircuit prototype measured about 10.7s for eager energy and 51.3s for one eager exact value-and-gradient call. Its JIT compile never completed. K3 spent the remainder of the window waiting on that compile instead of simplifying the design or writing `/root/solution_1.py`. |
| 02 | `AgentTimeoutError` | 30m 30s | No | This was a valid controlled benchmark run: authentication passed, refreshed credentials were persisted, and host sleep was prevented. K3 inspected the installed TensorCircuit APIs, built and timed multiple executable prototypes, abandoned an `expectation_ps` path after a 7+ minute JAX compile, and replaced it with a TensorCircuit sparse-Hamiltonian path. That path built the 49,152-nonzero Hamiltonian in about 0.39s and ran an eager forward pass in about 2.84s, but full JIT value-and-gradient compilation still exceeded 280s. K3 did not convert the diagnosis into `/root/solution_2.py` before the agent timeout. |
| 03 | `AgentTimeoutError`; candidate evaluated | 30m agent + 13m verifier | Yes | K3 wrote an 86-line TensorCircuit/JAX solution before timing out, but continued testing slower variants instead of finalizing. The preserved candidate passed every functional criterion and static-policy check: energy density improved from -0.43226457 to -1.02593517, final success probability was 1.53754670e-02, and all four histories had the required 300 entries. Its end-to-end runtime was 492.75s, over the 300s requirement. The separate GPT-5 policy audit timed out after 300s, so `llm_audit_score=0` is missing audit evidence rather than a substantive rejection; together with the agent timeout, this leaves the recorded overall reward at 0. |
| 04 | `AgentTimeoutError`; candidate evaluated | 30m agent + 12m verifier | Yes | K3 wrote a 94-line native density-matrix/Kraus solution after about six minutes, then spent the rest of the agent window waiting on one oversized XLA compile. The preserved candidate passed every functional criterion and static-policy check. It recovered `p01=0.03403554` and `p10=0.01103792`, both within 3.8e-05 of the true values, reduced loss from 7.1223e-03 to 2.7921e-08, and had trace-preserving error 1.11e-16. Runtime was 439.19s, over the 300s requirement. The separate GPT-5 policy audit again timed out after 300s, so its audit zero is missing evidence rather than an adverse judgment. |
| 05 | `AgentTimeoutError`; functional/runtime pass | 30m agent + 8m verifier | Yes | K3 found a fast 18-qubit TensorCircuit statevector and sparse-Hamiltonian path, diagnosed late-training overflow, and fixed it with the task-required per-layer normalization. It then spent substantial time tracing a small complex64 accumulation error and launched final validation too late to finish within the agent window. The preserved 89-line candidate passed every functional and static-policy criterion in 139.89s (`runtime_score=1`): energy density improved from -1.17185916 to -1.32678475 versus exact -1.32689714, with 600 finite history entries and correct `(5, 2)` parameter arrays. The separate GPT-5 audit timed out after 300s, so overall reward remains 0 despite the successful candidate. |
| 06 | Functional/runtime pass; audit timeout | 32m total | Yes | K3 first found a fast sparse-Hamiltonian and continuous Dormand-Prince path, but its initial validation exposed numerical norm drift and an impossible energy density near -5.0. High-effort post-test review found the flaw, added normalization and a Rayleigh quotient, and revalidated against the exact ground-state density. The preserved 145-line candidate passed every functional and static-policy criterion in 140.51s (`runtime_score=1`): energy density improved from -0.52567482 to -1.29098010 versus exact -1.60255313, with 100 entries and all learned parameters in bounds. The separate GPT-5 audit timed out after 300s, so overall reward remains 0 despite the successful candidate. A fidelity concern remains for later human review: the fixed 16-step RK loop adapts the next step size but does not reject an over-tolerance step, so the visible pass alone does not prove strict `ode_rtol`/`ode_atol` enforcement. |
| 07 | Excluded: provider quota 403 | 30s | No | Kimi Code rejected the prompt before model work began because the account had reached its billing-cycle usage limit. The recorded component zeros and `runtime_sec=-1` are placeholders for an errored no-artifact trial, not K3 scores. |
| 08 | Excluded: provider quota 403 | 31s | No | Same pre-model Kimi Code billing-cycle quota rejection as Challenge 07. |
| 09 | Excluded: provider quota 403 | 31s | No | Same pre-model Kimi Code billing-cycle quota rejection as Challenge 07. |
| 10 | Excluded: provider quota 403 | 31s | No | Same pre-model Kimi Code billing-cycle quota rejection as Challenge 07. |
| 11 | Excluded: provider quota 403 | 30s | No | Same pre-model Kimi Code billing-cycle quota rejection as Challenge 07. |
| 12 | Excluded: provider quota 403 | 29s | No | Same pre-model Kimi Code billing-cycle quota rejection as Challenge 07. |

The controlled `high` runs did not solve Challenges 01 or 02. They reached
executable prototypes and measurements sooner than the corresponding `max`
runs, which supports the hypothesis that `max` reasoning consumed too much of
the fixed agent window. Framework-native automatic-differentiation compilation
still consumed hundreds of seconds at `high`, and K3 waited too long before
pivoting away from those compiles.

For Challenges 01 and 02, the recorded zero reward is not an evaluated
submission score. No candidate artifact existed, so `runtime_sec=-1` means
runtime data is missing and the verifier's component zeros are placeholders
for an errored trial. Challenges 03 and 04 are different: their candidates
were evaluated and functionally passed, but exceeded the runtime requirement.
Challenges 05 and 06 passed both functional and runtime checks. All four audit
zeros were caused by the audit command timeout, not an adverse policy
judgment.

Immediately after K3 finished profiling the corrected Challenge 06 candidate,
Kimi Code returned HTTP 403 with “usage limit for this billing cycle.” The
suite retains Challenge 06 because the artifact had already been written and
validated. It then attempted every remaining challenge with
continue-on-error; Challenges 07–12 all received the same pre-model quota
rejection. These no-artifact jobs are infrastructure/account quota failures,
not K3 benchmark results, and must not be included in a publishable capability
aggregate. Challenge 02 was not repeated again in the suite; the controlled
retry above is its retained high-effort observation.

## High-effort continuation on 2026-07-30

A live host preflight returned `K3_READY`, so Challenges 07–12 were retried
under the new job prefix
`tensorcircuit-kimi-k3-high-20260730-resume`. The original quota-failure jobs
remain preserved and excluded.

| Challenge | Harbor outcome | Wall time | Solution artifact | Observation |
| --- | --- | ---: | --- | --- |
| 07 | Excluded: provider overload 429 | 13m 46s | No | K3 received the task and inspected TensorCircuit's measurement, Kraus, and MPS APIs. A later reasoning turn received three consecutive `429 The engine is currently overloaded` responses, terminating the trial before K3 wrote a candidate. This is a provider interruption, not a controlled agent timeout or evaluated solution. |
| 08 | `AgentTimeoutError` | 30m 30s | No | This is a valid high-effort observation. The built-in TensorCircuit MPS sampler projected to roughly four hours for 8,192 shots. K3 then developed a batched contraction over the framework-built and canonicalized MPS: its first validated implementation took 53.87s, and a corrected matrix-multiply sweep took 3.56s with maximum single-Z error 0.00771 (2.30 standard deviations) over 8,192 samples. K3 continued debugging extra exact string-correlator checks until the agent timeout and never wrote `/root/solution_8.py`. K3 had a plausible under-300-second approach but failed to finalize it. |
| 09 | Excluded: provider quota 403 | 28m 32s | No | K3 extracted 18- and 15-qubit causal cones with 74 and 80 relevant gates. A 200-restart vectorized gradient required an 11.0GB buffer and exceeded local memory. Batch 10 compiled in about 101s and projected to roughly 600–840s for all 20,000 updates; `pmap` and concurrent calls did not help on the 8-CPU container. Before K3 wrote a candidate or completed a controlled timeout, the provider returned the billing-cycle quota 403. The trace is useful diagnostic evidence but not a valid benchmark result. |
| 10 | Excluded: provider quota 403 | 33s | No | Kimi Code rejected the prompt before model work began. |
| 11 | Excluded: provider quota 403 | 31s | No | Kimi Code rejected the prompt before model work began. |
| 12 | Excluded: provider quota 403 | 32s | No | Kimi Code rejected the prompt before model work began. |

The 2026-07-30 continuation adds one valid model observation (Challenge 08).
Challenges 07 and 09–12 still require clean retries after provider capacity and
quota recover. Challenge 08 may also be rerun for reproducibility, but its
controlled timeout is already a valid outcome.

A later minimal host preflight on 2026-07-30 still returned the billing-cycle
usage-limit 403. Starting another Harbor job at that point would only add
another pre-model exclusion, so the campaign left those five retries pending.

## High-effort continuation retry on 2026-08-02

The K3 quota became available for another sequential suite under
`tensorcircuit-kimi-k3-high-20260730-resume-*` with Harbor retry job number 02.
The retry produced controlled outcomes for Challenges 07 and 09 before the
provider quota closed again.

| Challenge | Harbor outcome | Wall time | Solution artifact | Observation |
| --- | --- | ---: | --- | --- |
| 07 | `AgentTimeoutError`; candidate evaluated | 30m agent + 99.76s verifier | Yes | K3 spent the first part of the window validating TensorCircuit measurement and projection APIs, then found a finite joint-marginal measurement implementation. The final 104-line candidate passed every functional and static-policy check in 99.76s: mean energy improved from -6.55104828 to -9.87679641, with improvement 3.30727863. The independent GPT-5 audit timed out after 300s, so the raw reward remained zero despite the functional, runtime, and static passes. |
| 09 | Candidate evaluated; runtime miss | 27m 23s agent + 482.52s verifier | Yes | K3 extracted exact 18- and 15-qubit causal cones with 74 and 80 relevant gates, and its reduced TensorCircuit matched a small full-state check within 8.9e-08. The 88-line candidate passed all functional and static-policy checks: mean objective improved from -0.00228925 to 1.56457582, best final objective was 1.56459141, and success fraction was 1.0 over 200 restarts. Sequential restart execution took 482.52s, so the runtime score was zero. The GPT-5 audit also timed out after 300s. Kimi returned an auth error after the artifact and verifier evidence had been collected; Harbor retained the completed evaluated trial. |
| 10 | Excluded: provider quota 403 | 30s | No | Kimi Code rejected the prompt before model work began with the billing-cycle usage-limit error. |
| 11 | Excluded: provider quota 403 | 28s | No | Kimi Code rejected the prompt before model work began with the billing-cycle usage-limit error. |
| 12 | Excluded: provider quota 403 | 28s | No | Kimi Code rejected the prompt before model work began with the billing-cycle usage-limit error. |

This retry closes the controlled-results gap for Challenges 07 and 09. The
candidate for Challenge 07 meets the 300-second evaluator limit. Challenge 09
needs a faster restart schedule before it can count as a runtime pass. The
quota failures for Challenges 10–12 remain account/provider exclusions.

The post-suite host probe on 2026-08-02 returned the same billing-cycle 403,
so another Harbor retry would not add model evidence until the Kimi account
quota refreshes.

## Campaign summary

- Valid high-effort K3 observations: Challenges 01–09.
- No candidate after the controlled 1,800-second agent timeout: Challenges 01,
  02, and 08.
- Candidate passed functional and runtime checks: Challenges 05, 06, and 07.
- Candidate passed functional checks but missed the 300-second runtime:
  Challenges 03, 04, and 09.
- No substantive GPT-5 audit verdict: Challenges 03–07 and 09 all hit the
  independent 300-second audit-command timeout.
- Missing controlled candidate results: Challenges 10–12. Kimi rejected all
  three prompts before model work because the billing-cycle quota was exhausted.

The raw Harbor summary contains a zero reward for every row, but that number is
not a faithful single-number comparison here. For Challenges 03–07 and 09 it
is dominated by missing audit evidence; Challenge 09 also missed the runtime
limit. Challenge 08's zero represents a controlled no-artifact timeout, while
the zeros for Challenges 10–12 come from provider quota rejection. Any later
PR should publish the functional, runtime, static-policy, agent-timeout,
audit-timeout, and provider-exclusion fields separately instead of ranking K3
by the raw reward column.
