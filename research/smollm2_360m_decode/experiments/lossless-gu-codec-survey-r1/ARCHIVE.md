# Experiment 213 research evidence

Measured source directory is `outputs/megakernel-20261025/lossless-gu-codec-survey-r1` in tiny-prl. The corrected CPU survey completed in guarded unit `lossless-gu-survey-20261101-r2` with exit status zero.

All 64 BF16 gate/up matrices reconstruct bit-for-bit. Encoded bytes are 255,569,568 versus 314,572,800 raw, an 18.7566% storage reduction including headers and raw escape records. This is a storage feasibility result. It contains no GPU decoder measurement or latency improvement. No serving change is adopted.

The source hash matches the report. An independent Luna review recomputed the byte accounting and ran the crafted-pattern CPU self-test, including the permuted escape table. The full checkpoint was not rerun by the reviewer. No weights or compiled libraries are included. Raw report hashes are recorded in `raw-output-manifest.json`.

To run the CPU self-test from this directory with the pinned environment, use `UV_CACHE_DIR=/tmp/tiny-prl-uv-runtime CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --offline --no-project --python /home/pradheep/.venv/bin/python python survey.py --self-test`. Full surveys require the separately bounded operator unit documented in the continuation checkpoint.
