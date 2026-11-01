import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from collections import OrderedDict

import tqdm

from .BaseCaptioner import BaseCaptioner, CaptionConfig

# The Spark gateway as seen from the training container. The launcher only hands the gateway
# token to jobs whose api_base is exactly this, and it is only ever sent here.
GATEWAY_API_BASE = "http://host.docker.internal:8080/v1"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICaptionConfig(CaptionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # model_name_or_path is the served model name the endpoint routes on
        self.api_base = kwargs.get("api_base", GATEWAY_API_BASE).rstrip("/")
        self.temperature = kwargs.get("temperature", 0.2)
        self.timeout = kwargs.get("timeout", 300)
        self.system_prompt = kwargs.get("system_prompt", None)


class APIError(Exception):
    """The endpoint failed in a way that affects every file (auth, outage, eviction)."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # urllib would carry the Authorization header to wherever a redirect points
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise APIError(f"refusing redirect {code} to {newurl}")


class OpenAICaptioner(BaseCaptioner):
    """Captions through an OpenAI-compatible /chat/completions endpoint (e.g. a vLLM model
    behind the Spark gateway) instead of loading a model in this process."""

    caption_config_class = OpenAICaptionConfig

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(OpenAICaptioner, self).__init__(process_id, job, config, **kwargs)
        self.api_key = None
        # no env proxies, no redirects: the bearer token goes to api_base and nowhere else
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def _request(self, path: str, body=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.caption_config.api_base + path, data, headers)
        try:
            with self.opener.open(req, timeout=self.caption_config.timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise APIError(f"{e.code} from {self.caption_config.api_base}{path}: {e.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise APIError(f"{self.caption_config.api_base}{path}: {e}")

    def _chat(self, content, max_tokens: int):
        cc = self.caption_config
        messages = [{"role": "system", "content": cc.system_prompt}] if cc.system_prompt else []
        messages.append({"role": "user", "content": content})
        body = {"model": cc.model_name_or_path, "messages": messages, "max_tokens": max_tokens, "temperature": cc.temperature}
        if cc.thinking:
            # vLLM chat-template switch; only sent when asked for, other servers reject unknown fields
            body["chat_template_kwargs"] = {"enable_thinking": True}
        return self._request("/chat/completions", body)["choices"][0]

    def load_model(self):
        cc = self.caption_config
        if cc.api_base == GATEWAY_API_BASE:
            self.api_key = os.environ.get("AITK_CAPTION_API_KEY") or None
        self.print_and_status_update(f"Checking {cc.model_name_or_path} at {cc.api_base}")
        # a 1-token request, not /models: it also proves the model can actually be (cold-)started
        self._chat("Reply with OK.", 1)

    def load_loras(self):
        if self.caption_config.loras:
            raise ValueError("LoRAs cannot be applied to a model behind an API")

    def run_caption_loop(self):
        # Same as the base loop, except an APIError ends the job: with a remote model a failure
        # is almost never specific to one file, and the base loop would mark the job completed
        # with no captions written.
        skipped = 0
        for file_path in tqdm.tqdm(self.file_paths, desc="Captioning files", unit="file"):
            if self.is_ui_captioner:
                self.maybe_stop()
                if self.is_stopping:
                    break
            try:
                file_caption = self.get_caption_for_file(file_path)
                if file_caption is not None:
                    self.save_caption_for_file(file_path, file_caption)
                else:
                    skipped += 1
            except APIError:
                raise
            except Exception as e:
                skipped += 1
                print(f"Error captioning file {file_path}: {e}")
            finally:
                self.step_num += 1
                self.update_step()
        if skipped:
            print(f"{skipped} file(s) got no caption; rerun with recaption off to retry only those")

    def get_caption_for_file(self, file_path: str):
        cc = self.caption_config
        img = self.load_pil_image(file_path, max_res=cc.max_res)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92)
        choice = self._chat([
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": self.resolve_prompt(cc.caption_prompt, file_path)},
        ], cc.max_new_tokens)
        if choice.get("finish_reason") == "length":
            # a cut-off caption would be kept forever (recaption off skips files that have one)
            print(f"Caption for {file_path} hit max_new_tokens ({cc.max_new_tokens}); not saved")
            return None
        text = _THINK_RE.sub("", choice["message"].get("content") or "").strip()
        return text or None
