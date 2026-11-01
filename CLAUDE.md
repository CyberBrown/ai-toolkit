# ai-toolkit on Spark (branch flux3-gwen)

Training runs in the `ai-toolkit-training` container, always started by `train_spark.sh`
(gateway GPU claim, OOM shield, supervisor that releases the claim). Never `docker compose up`
the `train` service directly.

- **CLI:** `./train_spark.sh config/<name>.yaml <dataset_dir> [claim_gb]` from this dir.
- **Web UI:** host-native user service `ai-toolkit-ui.service` (copy in this repo; install to
  `~/.config/systemd/user/`), `http://prometheus-spark:8675`. Build: `cd ui && npm install &&
  npx prisma generate && npx prisma db push && npm run build`. `AITK_JOB_LAUNCHER=ui_job_spark.sh`
  makes the worker start jobs through `train_spark.sh` instead of a host python. The compose
  `ui` service (node:22 container) is unused: it could not reach docker to launch jobs.
- The train container mounts this repo and `~/ComfyUI/models` at their **host paths** so
  UI-written configs (absolute host paths, sqlite db) work unchanged inside it. UI datasets live
  in `datasets/` (gitignored); a dataset elsewhere on the host is not visible to UI jobs.
- One job at a time: the container name is fixed.

Traps:
- The container runs as root; its command `chown`s `output/` and `aitk_db.db*` back to 1000 on
  exit. A `docker stop`/SIGKILL skips that — fix ownership by hand.
- PyYAML reads `1e-4` as a **string**; ai-toolkit's own loader does not. Converting a YAML config
  to JSON for the UI API with `yaml.safe_load` gives `TypeError: can't multiply sequence by
  non-int of type 'float'` at step 0. Write `1.0e-4`.
- The jobs API takes `job_config` as an object; a JSON string crashes the worker
  (`Cannot read properties of undefined (reading 'process')`).
- The torchaudio "Error" lines at startup are harmless (audio extensions skipped).
- FLUX.3: train the base `flux3-image-klein-base-bf16`; the distilled Klein is rejected.
