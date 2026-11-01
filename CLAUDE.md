# ai-toolkit on Spark (branch flux3-gwen)

Training runs in the `ai-toolkit-training` container, always started by `train_spark.sh`
(gateway GPU claim, OOM shield, supervisor that releases the claim). Never `docker compose up`
the `train` service directly.

- **CLI:** `./train_spark.sh config/<name>.yaml <dataset_dir> [claim_gb]` from this dir.
- **Web UI:** `ai-toolkit-ui [stop|status|logs|rebuild]` (linked into `~/bin`, listed by `spark-help`); host-native user service `ai-toolkit-ui.service` (copy in this repo; install to
  `~/.config/systemd/user/`), `http://prometheus-spark:8675`. Build: `cd ui && npm install &&
  npx prisma generate && npx prisma db push && npm run build`. `AITK_JOB_LAUNCHER=ui_job_spark.sh`
  makes the worker start jobs through `train_spark.sh` instead of a host python. The compose
  `ui` service (node:22 container) is unused: it could not reach docker to launch jobs.
- The train container mounts this repo and `~/ComfyUI/models` at their **host paths** so
  UI-written configs (absolute host paths, sqlite db) work unchanged inside it. UI datasets live
  in `datasets/` (gitignored); a dataset elsewhere on the host is not visible to UI jobs.
- One job at a time: the container name is fixed.
- **Captioning with a gateway model:** captioner type `OpenAICaptioner` ("Gateway API (Gemma 4)" in
  the UI) posts each image to an OpenAI-style `/chat/completions` endpoint instead of loading a
  model. Default `api_base` is the gateway, `http://host.docker.internal:8080/v1`; `model_name_or_path`
  is the served model name (e.g. `gemma4-heretic-nvfp4`). For these jobs `ui_job_spark.sh` places
  **no** GPU claim (a P0 claim would block or evict the very service being called) and skips
  quiescing. ~1.5 s/image.
  - The gateway token reaches the job as `AITK_CAPTION_API_KEY`, and **only when `api_base` is the
    gateway** — the UI has no login, so a job config could otherwise send it anywhere. The
    captioner refuses redirects and ignores env proxies for the same reason. Never put a token in
    a config.
  - The job starts with a 1-token chat request, so a model that cannot cold-start (gateway 503:
    an active claim holds the memory) fails the job up front. Any API error mid-run fails the job
    instead of marking it completed; captions cut off at `max_new_tokens` are not saved.


## Fork hygiene
`origin` = CyberBrown/ai-toolkit (fork), `upstream` = ostris/ai-toolkit (fetch only — its push URL
is set to `DISABLED-do-not-push-to-ostris`). `gh repo set-default CyberBrown/ai-toolkit` is set;
in a fork `gh pr create` otherwise targets the parent. Still pass `--repo CyberBrown/ai-toolkit
--base flux3-gwen` explicitly. Work branches come off `flux3-gwen` and merge back into it.

Traps:
- The container runs as root; its command `chown`s `output/`, `datasets/` and `aitk_db.db*` back
  to 1000 on exit. `docker stop` is turned into SIGINT (90 s grace) so that still runs; a SIGKILL
  skips it. Captions written outside `datasets/` (another `path_to_caption`) stay root-owned.
- PyYAML reads `1e-4` as a **string**; ai-toolkit's own loader does not. Converting a YAML config
  to JSON for the UI API with `yaml.safe_load` gives `TypeError: can't multiply sequence by
  non-int of type 'float'` at step 0. Write `1.0e-4`.
- The jobs API takes `job_config` as an object; a JSON string crashes the worker
  (`Cannot read properties of undefined (reading 'process')`).
- The torchaudio "Error" lines at startup are harmless (audio extensions skipped).
- FLUX.3: train the base `flux3-image-klein-base-bf16`; the distilled Klein is rejected.
