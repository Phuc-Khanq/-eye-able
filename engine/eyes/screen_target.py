"""UI-Venus 1.5: crop the screen around the gaze point, find the named element, return screen pixels.

Route: Khang and Hung (Khang leads)
Publishes: TargetResult, target_debug (console open only)
Subscribes: TargetRequest
APIs: mss (screen crop), transformers>=4.57, torch (CUDA 12.8), inclusionAI/UI-Venus-1.5-2B.
      Runs in its own worker thread (submit() returns a Future). Screenshots stay in memory, never on disk.

Two steps, so where you look matters (the model itself never knows where your gaze is):
  1. zoom: a small box (640x360) around the gaze, enlarged 2x so small icons are clearer.
  2. wide: if the target isn't in the zoom box, the full 1280x720 box.
The model picks any matching element in the image it sees, so a smaller box near the gaze means fewer
look-alikes (one channel avatar instead of twenty). Experimental hints (hint=) also say which one is meant:
  "center"  adds "the one closest to the center of the screenshot" (the crop is centered on the pointer).
            The default: on YouTube it got 5 of 6 right, against 4 of 7 with no hint and 3 of 5 with "mark"
            (target_bench, Oct 24 2:20-3:10 AM; small samples, rerun on more apps)
  "mark"    draws a red ring at the pointer and adds "the one at or nearest the red circle"
  "none"    UI-Venus's own prompt, unchanged

Prompt and output format come from UI-Venus's own grounding evaluation
(models/grounding/eval_screenspot_pro.py on the UI-Venus-1.5 branch):
  the model answers "[x,y]" on a 0-1000 scale of the image it was shown, or "[-1,-1]" when the
  target isn't there. Some answers are a box "[x1,y1,x2,y2]"; we use its center.
"""
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from engine.config import settings
from engine.core.types import TargetRequest, TargetResult

PROMPT = ("Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\n"
          "The output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task "
          "is infeasible (e.g., the task is not related to the image), the output should be [-1,-1].")

_BOX = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_POINT = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def parse_point(text: str) -> tuple[float, float] | None:
    """Model text -> (x, y) as fractions 0..1 of the image, (-1, -1) for "not there", None if unreadable."""
    m = _BOX.search(text)
    if m:
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        if x1 < 0 or y1 < 0:
            return (-1.0, -1.0)
        return ((x1 + x2) / 2000.0, (y1 + y2) / 2000.0)
    m = _POINT.search(text)
    if m:
        x, y = (int(g) for g in m.groups())
        if x < 0 or y < 0:
            return (-1.0, -1.0)
        return (min(x, 1000) / 1000.0, min(y, 1000) / 1000.0)
    return None


def crop_box(gx: float, gy: float, screen_w: int, screen_h: int, crop_w: int, crop_h: int) -> tuple[int, int, int, int]:
    """(left, top, width, height) of a crop_w x crop_h box centered on the gaze, slid to stay on screen."""
    w, h = min(crop_w, screen_w), min(crop_h, screen_h)
    left = int(round(min(max(gx - w / 2, 0), screen_w - w)))
    top = int(round(min(max(gy - h / 2, 0), screen_h - h)))
    return left, top, w, h


def grab(left: int, top: int, width: int, height: int):
    """Screenshot of a screen region as a PIL image, in memory only. Call os_input.make_dpi_aware()
    once at startup so these are physical pixels, the same as the cursor."""
    import mss
    from PIL import Image
    with mss.mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


def screen_bounds() -> tuple[int, int, int, int]:
    """(left, top, width, height) of the main monitor."""
    import mss
    with mss.mss() as sct:
        m = sct.monitors[1]
        return m["left"], m["top"], m["width"], m["height"]


@dataclass
class Answer:
    raw: str                                  # exactly what the model wrote
    point: tuple[float, float] | None         # fractions of the image, (-1, -1) = not there, None = unreadable
    ms: float
    image_tokens: int = 0                     # how many image tokens the model got (0 = the image didn't reach it)


@dataclass
class Stage:
    name: str                                 # "zoom" or "wide"
    box: tuple[int, int, int, int]            # left, top, width, height in screen pixels
    answer: Answer


class ScreenTarget:
    """Loads UI-Venus once, then answers "where is <label>?" on a crop around the gaze."""

    def __init__(self, model_id: str | None = None, device: str | None = None,
                 crop: tuple[int, int] | None = None, zoom: tuple[int, int] | None = (640, 360),
                 max_new_tokens: int = 32, hint: str = "center") -> None:
        self.model_id = model_id or settings.ui_venus_model
        self.device = device or settings.ui_venus_device
        self.crop_w, self.crop_h = crop or (settings.crop_width, settings.crop_height)
        self.zoom = zoom                      # None = one step with the full box only
        self.trace: list[Stage] = []          # what each step saw and said, for the console and the bench
        self.max_new_tokens = max_new_tokens
        self.hint = hint                      # "center" (default), "mark" or "none"; see the module notes
        self.model = None
        self.processor = None
        self.model_class = ""
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ui-venus")

    # -- loading -------------------------------------------------------------------------------
    def load(self) -> float:
        """Loads the model (downloads it on first use). Returns seconds taken. Safe to call twice."""
        if self.model is not None:
            return 0.0
        t0 = time.perf_counter()
        import torch
        import transformers
        from transformers import AutoProcessor

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        errors = []
        # The UI-Venus eval code uses AutoModelForImageTextToText; the model card uses AutoModelForMultimodalLM.
        for name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(self.model_id, dtype=dtype, low_cpu_mem_usage=True).to(self.device).eval()
                self.model_class = name
                break
            except Exception as e:  # keep going, report everything if nothing works
                errors.append(f"{name}: {type(e).__name__}: {e}")
        if self.model is None:
            raise RuntimeError("Couldn't load UI-Venus:\n  " + "\n  ".join(errors or ["no model class found"]))
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return time.perf_counter() - t0

    # -- asking --------------------------------------------------------------------------------
    def ask(self, image, instruction: str) -> Answer:
        """Ask about one PIL image. Blocking; call from the worker thread (submit) in the live engine."""
        import torch
        self.load()
        prompt = PROMPT.format(instruction=instruction.strip().rstrip("."))
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        with self._lock:
            t0 = time.perf_counter()
            inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                        return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            raw = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            ms = (time.perf_counter() - t0) * 1000
        grid = inputs.get("image_grid_thw")
        tokens = int(grid.prod(dim=-1).sum().item()) // 4 if grid is not None else 0  # 2x2 patches per token
        return Answer(raw=raw, point=parse_point(raw), ms=ms, image_tokens=tokens)

    def find(self, request: TargetRequest, screen=None, origin: tuple[int, int] | None = None) -> TargetResult:
        """Zoom box first, then the wide box. [-1,-1] or unreadable output means found=False; we never
        guess. screen: a PIL image of the whole main monitor (grabbed now if None), origin: its top-left."""
        if screen is None:
            sl, st, sw, sh = screen_bounds()
            screen, origin = grab(sl, st, sw, sh), (sl, st)
        ox, oy = origin or (0, 0)
        gx, gy = request.gaze_x - ox, request.gaze_y - oy
        steps = ([("zoom", self.zoom)] if self.zoom else []) + [("wide", (self.crop_w, self.crop_h))]
        self.trace, total_ms = [], 0.0
        for name, (cw, ch) in steps:
            left, top, w, h = crop_box(gx, gy, screen.width, screen.height, cw, ch)
            image = screen.crop((left, top, left + w, top + h))
            if name == "zoom":  # enlarge so the model gets more detail on small icons
                image = image.resize((w * 2, h * 2), resample=3)  # 3 = bicubic
            ans = self.ask(*self._hinted(image, request.label, gx - left, gy - top, 2.0 if name == "zoom" else 1.0))
            total_ms += ans.ms
            self.trace.append(Stage(name=name, box=(left + ox, top + oy, w, h), answer=ans))
            if ans.point is not None and ans.point[0] >= 0:
                return TargetResult(label=request.label, found=True, x=int(round(left + ox + ans.point[0] * w)),
                                    y=int(round(top + oy + ans.point[1] * h)), ms=total_ms)
        return TargetResult(label=request.label, found=False, ms=total_ms)

    def _hinted(self, image, label: str, px: float, py: float, scale: float):
        """Tells the model which of several look-alikes is meant: the one where the user is pointing.
        px, py: the pointer inside the crop (before enlarging); scale: how much the crop was enlarged."""
        if self.hint == "center":
            return image, f"{label}, the one closest to the center of the screenshot"
        if self.hint == "mark":
            from PIL import ImageDraw
            image = image.copy()
            x, y, r = px * scale, py * scale, 22 * scale
            ImageDraw.Draw(image).ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0), width=int(2 * scale))
            return image, f"{label}, the one at or nearest the red circle"
        return image, label

    def preload(self) -> Future:
        """Load the model and warm it up on the worker thread; returns a Future with the seconds taken.
        Requests submitted meanwhile wait behind it, so the model is never loaded twice."""
        def work() -> float:
            from PIL import Image
            t0 = time.perf_counter()
            self.load()
            self.ask(Image.new("RGB", (640, 360), (255, 255, 255)), "the button")  # the first call is slow
            return time.perf_counter() - t0
        return self._pool.submit(work)

    def submit(self, request: TargetRequest) -> Future:
        """Runs find() on the UI-Venus worker thread, so the camera and audio threads never wait."""
        return self._pool.submit(self.find, request)

    def close(self) -> None:
        self._pool.shutdown(wait=False)
