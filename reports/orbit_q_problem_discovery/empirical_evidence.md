# Sanitized empirical Harbor evidence

> **Claim boundary:** These are exploratory local runs with automated trace scans and no hash-bound human trace review. They are not protocol-qualified model-hardness results and are not signed raw-run attestations.

Snapshot: `orbit-q-exploratory-harbor-sha256:eae54177fd3b7ad25aa9ce1ddbb381bc568de086cb7c65edb1202a3bc1171a5c`
Source cutoff: `2026-08-09T19:03:20.023450Z`

## Decision summary

| Candidate | Classification | Expert evidence | Admitted model pilots | Decision |
|---:|---|---:|---:|---|
| 105 | `solved_hardness_blocking` | 1 | 1 | The expert oracle and one gpt-5.6-sol/high pilot passed the same frozen task checksum. This admitted exploratory solve blocks a claim that this frozen task is unsolved by the target model. |
| 106 | `solved_hardness_blocking` | 1 | 1 | The expert oracle and one gpt-5.6-sol/high pilot passed the same frozen task checksum. This admitted exploratory solve blocks a claim that this frozen task is unsolved by the target model. |
| 107 | `solved_hardness_blocking` | 2 | 1 | The expert oracle and one gpt-5.6-sol/high pilot passed the same reported seed and case digest as an expert run, but a different task checksum. This admitted exploratory solve blocks a claim that this frozen task is unsolved by the target model. |
| 108 | `solved_hardness_blocking` | 1 | 1 | The expert oracle and one gpt-5.6-sol/high pilot passed the same frozen task checksum. This admitted exploratory solve blocks a claim that this frozen task is unsolved by the target model. |
| 109 | `pilot_hardness_signal` | 1 | 1 | The expert oracle passed, while one gpt-5.6-sol/high pilot on the same frozen task checksum failed the bound functional checks heldout_p95_infidelity, heldout_worst_infidelity after passing static policy and source audit. This is an exploratory substantive optimization-performance signal only. |
| 111 | `solved_hardness_blocking` | 1 | 1 | The expert oracle and one gpt-5.6-sol/high pilot passed the same frozen task checksum. This admitted exploratory solve blocks a claim that this frozen task is unsolved by the target model. |

Passing pilots are classified as hardness-blocking for their exact frozen task. A functionally failed but policy- and source-audit-clean pilot is classified only as an exploratory signal, never protocol-qualified model-hard evidence.

## Expert evidence

| Candidate | Job ID | Trial ID | Seed / case digest | Reward / functional / static / audit | Evaluator runtime (s) | Solution SHA-256 |
|---:|---|---|---|---|---:|---|
| 105 | `3eed2f55-035e-4830-90b9-bb0710ccc677` | `988681f6-c931-4323-9ef5-89f4b2a88e77` | — / — | 1.0 / 1.0 / 1.0 / 1.0 | 3.774931 | `4ccf342901e44abe2ffc5bcf989679414605a3f53c2b7680e95974785be32b19` |
| 106 | `0fb20d4b-337c-4aef-926a-fa3d5857e5f5` | `163ccb01-2a28-4439-a6cc-3a5b1d368d20` | 1062026 / 3756ccf5e2905e6f… | 1.0 / 1.0 / 1.0 / 1.0 | 31.728049 | `bf1e1e193274ff9a2e7086683a39797b9842bf630c8660b924a8d4d408a2290c` |
| 107 | `3e7b7093-ee03-4331-afdd-c0b945644f6e` | `d25f26b5-1dc1-4d8a-84d1-3444d5f4856f` | 1072026 / 21894eff8b753232 | 1.0 / 1.0 / 1.0 / 1.0 | 7.492000 | `09bb8db4b1395a81950337db546ef9796581b1ef0d73e3bc05329eb1f2515d0d` |
| 107 | `9a5e8d3a-7877-4d61-a18a-69c7745226d8` | `e621d61c-2609-4851-874e-a8f04783bd2f` | 1072027 / 428aab346b3b50c2 | 1.0 / 1.0 / 1.0 / 1.0 | 6.508000 | `09bb8db4b1395a81950337db546ef9796581b1ef0d73e3bc05329eb1f2515d0d` |
| 108 | `365dc5c5-25df-4d7a-a327-312ff727d344` | `8b52aff3-9f26-4764-94fb-90b697d538aa` | 1082037 / 1e932988c8113662 | 1.0 / 1.0 / 1.0 / 1.0 | 2.143000 | `630f966a0cc31f1dab43486579f50fe0794cb3f06ca29b3b1929eedd7b5ea16e` |
| 109 | `a16440b2-432d-4daa-94e5-9216fdfcec28` | `e1172c3b-4da9-4342-a83b-6f60f5d8baad` | 1092037 / — | 1.0 / 1.0 / 1.0 / 1.0 | 31.152390 | `d1d8e9baa5bf2bfa11c8406e65dc1b1201a7d76731f1e160e8f6735b9e6948c4` |
| 111 | `f0959ff7-b550-4637-b748-8151f65df19d` | `23631296-118d-49f4-b220-94c7cbfa0bf4` | 1112037 / 427074eb20fbd3b1 | 1.0 / 1.0 / 1.0 / 1.0 | 41.042000 | `9b070d8053b380a97c19c167d3687a19f858e3864d8c2ea7b0befb4e4458dbaf` |

## gpt-5.6-sol/high exploratory pilots

| Candidate | Job ID | Trial ID | Seed / case digest | Reward / functional / static / audit | Evaluator runtime (s) | Result / solution / trajectory SHA-256 | Command audit |
|---:|---|---|---|---|---:|---|---|
| 105 | `f2c08f4d-5920-41d0-91ea-f219079742e7` | `9ded9086-de90-4a4c-a971-1af8ae098d38` | — / — | 1.0 / 1.0 / 1.0 / 1.0 | 30.020061 | `8539162a0286220ddd20c218655d31e2e4bc77e4e3cbda1bf7b5d7c8d83d4c12` / `79247d1c330f018c7b958f8e094c7a7184fc8e0443f5f2953470904d3c3060ee` / `c4a60f065313a767eb16a32542fdfbac9478f94318c83b4391e0b845573bc3e3` | no_matches_observed; no_disallowed_access_observed |
| 106 | `1b9efc33-63f0-4ac9-904e-9b37beda013f` | `0ba3e26d-645a-4863-bfcd-ee58a5a8b8be` | 1062026 / 3756ccf5e2905e6f… | 1.0 / 1.0 / 1.0 / 1.0 | 15.032113 | `0d7a24c26ac23faec999d3191a2aac07acaa0b83e18fdf0dbf584cbeec63315f` / `c13056948b74cc8991ebf05f3c58bc1b17b84f7ccb3ac6c2bc08691f3820ab29` / `2463faee0c37e9d8eaa1cc9dc8ec507809b37d5abd19cfff04c32b2bbde5a5d2` | no_matches_observed; no_disallowed_access_observed |
| 107 | `49e3aa42-7cb4-4aed-9715-a8fb634ccf02` | `6e9bac49-d338-4042-8f77-e2ac7557ee99` | 1072027 / 428aab346b3b50c2 | 1.0 / 1.0 / 1.0 / 1.0 | 3.405000 | `4319e4306f8cd487f41ddd1ad73f4938bdf9c52a798ba0efa9a3bdfa37d81ac1` / `3371c1381e689b4e888882614ee9e48f92ca12113c0e2d5a9421708873073798` / `c24e0111c5708f34196f05432a08602f994ead7093f104d21e23faeb27811717` | no_matches_observed; no_disallowed_access_observed |
| 108 | `2f9715fc-3e6b-4ef2-afd7-197fb1d60bb7` | `c2e6dcc1-c063-44ba-bf7f-7d94d131ac30` | 1082037 / 1e932988c8113662 | 1.0 / 1.0 / 1.0 / 1.0 | 2.246000 | `c503a8a1959ce9c36b9e6df76ec088392317d915c7d4880f43b78c701c5baa82` / `a0c23dce6daf8e66b9e565fa91c992ccf9df173a76be1de4f97e2d25065e21c1` / `c4442a174245f9e045091cbcccc96ac13891becca96b3ec0ae1ac2d797adf72a` | no_matches_observed; no_disallowed_access_observed |
| 109 | `ccc17430-dfb7-4d24-a49d-b899b48d942d` | `c441bfb9-9f46-4a9e-bb8b-632d534cb620` | 1092037 / — | 0.0 / 0.0 / 1.0 / 1.0 | 62.622879 | `c7ec2b7aa285c4b7e63d1e4e04316b6fb7349b7ea53422796e7534176f433103` / `790bd3d4c3965e01dfa722cbd32a9791fa510a87de8c588b08c07ec2e52d1ca4` / `aa7323e2234a4ca12dff2e6c03b1855171c575163c283649d72974194b35ce66` | no_matches_observed; no_disallowed_access_observed |
| 111 | `37db917d-34a7-4ea8-9259-8ea63a49c61d` | `f99c734a-da63-4848-8f12-f28550598ea2` | 1112037 / 427074eb20fbd3b1 | 1.0 / 1.0 / 1.0 / 1.0 | 34.398000 | `a05ff3933d1d3ab28a190a518e96c9456417f68a5040766c213890d495fa3050` / `3cda07e15d311cf1824a8f66904b7b5c39b007bc614d9ec5da90334d9483626c` / `232183f64401d9d0a9fb29381101fc3689bbe6ffce49d571c28f322646645445` | no_matches_observed; no_disallowed_access_observed |

Candidate 109's pilot missed bound held-out worst infidelity (0.002057377059741072 > 0.00125) and p95 infidelity (0.001415766428393611 > 0.0007). Its worst leakage, drive amplitude, slew, and edge amplitude remained within their evaluator thresholds. The static policy and source audit passed; the source audit judged the framework-native implementation faithful and found no obvious implementation error.

## Protocol and trust boundary

- Framework/image tag: `tensorcircuit` / `challenge-benchmark-quantum-tensorcircuit:py311`.
- Selected resources: 8 CPUs and 8192 MiB; verifier audit model `gpt-5.6-sol`.
- Model pilots: `gpt-5.6-sol`, reasoning `high`, Codex CLI `0.146.0`; TensorCircuit `1.7.0.dev20260618` was observed in the admitted trajectories.
- `runtime` in the tables is the functional evaluator's measured solution runtime, not solver-agent wall-clock time.
- Network isolation was not enabled: selected configs contained proxy bridge keys. Automated allowlist and marker scans of recorded model tool-call names and inputs had no network-command, protected-artifact, credential/environment-dump, or disallowed-tool matches. These scans are not human attestations.
- Searches under installed-package test directories in candidates 105, 109 were package-source reads, not access to the protected benchmark-root `/tests` path.
- Local same-tag metadata records image digest `sha256:8e627b582a5cdccca2ef40fb07f81671f5f5345ae1a07f40cef65537aa218bc7`, but the selected Harbor results bind only the image tag. The digest is therefore a qualified local observation, not a run-bound attestation.
- Static `raw_simulator_hits` are preserved per run in the JSON; a run is admitted only when its static policy and source audit scores agree and pass.
- Every run's Harbor task checksum is mapped to a committed frozen task-file hash set. Config content remains private, but its opaque SHA-256 is snapshot-bound.
- Solver pilots and the source-audit invocation both use `gpt-5.6-sol`; the audit is separate, but it is not independent model-family adjudication.

## Artifact integrity and sanitization

The JSON ledger contains the full SHA-256, byte size, and repo-relative locator for every admitted job result, trial result, reward, functional output, source audit, solution, artifact manifest, redacted config, oracle transcript, and available trajectory. It retains only an opaque SHA-256 for each raw model rollout; session filenames are excluded. Raw auth, environment values, config/transcript content, and rollout content are excluded. The schema and frozen source-binding manifest are also included in the snapshot digest.

Rebuild or verify deterministically:

```bash
python3 scripts/build_problem_discovery_evidence.py --check
```
