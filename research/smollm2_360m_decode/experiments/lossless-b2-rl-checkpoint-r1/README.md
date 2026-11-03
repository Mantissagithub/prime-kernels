# Normal two-sequence RL startup diagnostic

The normal named 360M preset runs two full-parameter alphabet-sort updates with two vLLM sequence slots, default max_inflight=4, stream64 broadcasts, final per-tensor checkpoint and FP32 optimization/reduction. Only early_trainer_start differs between arms. The private request hook retains distinct deterministic per-prompt seeds at temperature 1; it does not replay outputs or change rewards/admission. Both arms use the same accepted lossless kernel, not the rejected B2 trial.

| Chronological run | Startup | Whole RL seconds | Successful requests | Generated tokens | Startup log interval seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| c1 | control | 75.566662 | 222 | 6832 | 31 |
| e1 | early | 56.313208 | 174 | 5187 | 22 |
| c2 | control | 64.073641 | 172 | 5093 | 30 |

All three runs complete two updates, three weight refreshes and a full checkpoint of about 4.53 GB including optimizer state. Checkpoint file sizes and SHA256 hashes are retained; the final c2 checkpoint remains local. Repeated broadcast files and the first two repeat checkpoints were removed after recording evidence to keep disk use bounded. The elapsed interval includes process startup, rollout, training, checkpoint save and shutdown; post-run checkpoint hashing/cleanup is excluded.

The work gate fails. Of 174 common prompt/seed requests between c1/e1, 33 return different output hashes or token counts. Even the unchanged c1/c2 controls differ on 25 of 172 common requests. Finalized work is respectively 212/166/166 rollouts; training gradients differ. This shows run-to-run trajectory variation under the two-sequence workload despite fixed request seeds; the specific numerical or scheduling cause is not isolated. Lower wall time is not a verified speedup. No default-preset change, kernel adoption or new verified RL point.

The startup log interval runs from launcher inference-start to trainer startup-policy broadcast, with one-second log resolution. Its 31/22/30 seconds suggest startup imports still overlap, but three diagnostic processes do not establish a repeated whole-RL gain. Checkpoint I/O and variable rollout admission also contribute to total time. Following the Explain the Number principle, the five-per-arm sweep stops at the failed comparable-work gate rather than reporting a misleading percentage. All three guards pass, zero job swap, peak GPU 7,398 MiB and at least 3.18 GB available host RAM. Evidence: `prime-kernels/research/smollm2_360m_decode/experiments/lossless-b2-rl-checkpoint-r1/`.

Reproduction: use runtime/run_gpu.sh with run_case.py control ID or early ID. Require a healthy SM89 GPU, the pinned home venv, current cached decoder/codec and 8 GiB free host RAM/disk. compare.py checks complete successful requests, native training work, active batch-two route, library identity, checkpoint manifests and resource guards. analyze.py checks FP32 config, request divergence and startup log intervals. Absolute paths identify this laptop; no compiled libraries, model weights or checkpoint payloads are archived.
