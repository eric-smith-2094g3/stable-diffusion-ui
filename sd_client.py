import base64
import json
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

class SDError(Exception):
    pass

class SDClient:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _post(self, endpoint, payload, retries=3):
        url = f"{self.base}{endpoint}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as e:
                body = e.read().decode()
                last_err = SDError(f"HTTP {e.code}: {body[:200]}")
                if e.code in (502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise last_err
            except Exception as e:
                last_err = SDError(f"request failed: {e}")
                time.sleep(2 ** attempt)
        raise last_err

    def _get(self, endpoint, retries=3):
        url = f"{self.base}{endpoint}"
        last_err = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as e:
                body = e.read().decode()
                last_err = SDError(f"HTTP {e.code}: {body[:200]}")
                if e.code in (502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise last_err
            except Exception as e:
                last_err = SDError(f"request failed: {e}")
                time.sleep(2 ** attempt)
        raise last_err

    def progress(self):
        return self._get("/sdapi/v1/progress?skip_current_image=false")

    def txt2img(self, params, out_dir, save_preview=False):
        payload = {
            "prompt": params.get("prompt", ""),
            "negative_prompt": params.get("negative_prompt", ""),
            "steps": params.get("steps", 20),
            "width": params.get("width", 512),
            "height": params.get("height", 512),
            "cfg_scale": params.get("cfg_scale", 7.0),
            "sampler_name": params.get("sampler_name", "Euler a"),
            "seed": params.get("seed", -1),
            "save_images": False,
            "send_images": True,
        }

        # wait for idle
        for _ in range(60):
            try:
                st = self.progress()
                if st.get("active") is False and st.get("progress", 0) == 0:
                    break
            except SDError:
                pass
            time.sleep(1)
        else:
            raise SDError("timeout waiting for idle state")

        result = self._post("/sdapi/v1/txt2img", payload)
        images = result.get("images", [])
        if not images:
            raise SDError("no images returned")

        out_path = None
        for i, img_b64 in enumerate(images):
            img_data = base64.b64decode(img_b64)
            suffix = f"_{i}" if i > 0 else ""
            out_path = out_dir / f"sd_{int(time.time())}{suffix}.png"
            out_path.write_bytes(img_data)

        if save_preview:
            try:
                st = self.progress()
                preview_b64 = st.get("current_image")
                if preview_b64:
                    preview_data = base64.b64decode(preview_b64)
                    preview_path = out_dir / f"sd_{int(time.time())}_preview.png"
                    preview_path.write_bytes(preview_data)
            except SDError:
                pass

        return out_path

    def samplers(self):
        data = self._get("/sdapi/v1/samplers")
        return [s["name"] for s in data]

    def wait_with_progress(self, poll_interval=1.0, timeout=300):
        """Block until generation finishes, printing progress to stderr."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                st = self.progress()
                p = st.get("progress", 0)
                eta = st.get("eta_relative", 0)
                if st.get("active") is False and p == 0:
                    break
                bar = int(p * 20)
                sys.stderr.write(
                    f"\r[{'#' * bar}{'-' * (20 - bar)}] {p*100:.0f}% eta {eta:.1f}s"
                )
                sys.stderr.flush()
                if p >= 1.0:
                    break
            except SDError:
                pass
            time.sleep(poll_interval)
        sys.stderr.write("\n")
        sys.stderr.flush()
