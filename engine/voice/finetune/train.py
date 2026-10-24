"""Fine-tune Whisper large-v3-turbo with LoRA on dysarthric speech (plus some typical speech).

    python -m engine.voice.finetune.train --name r32-lr1e-4                    # GPU (Kaggle T4)
    python -m engine.voice.finetune.train --name r32 --lr 5e-5 --typical-mix 0.15
    python -m engine.voice.finetune.train --resume checkpoints/r32/last        # continue a run
    python -m engine.voice.finetune.train --tiny                               # smoke test, CPU

What it does:
  * LoRA (peft) on q_proj, k_proj, v_proj, out_proj in the encoder and the decoder
    (default r=32, alpha=64, dropout 0.05). Only the small adapter is trained and saved.
  * fp16 on the T4 (it has no bf16), bf16 automatically on GPUs that support it,
    gradient checkpointing, micro-batches that halve themselves if the GPU runs out of memory.
  * Every batch is about 75% train-split dysarthric/personal speech and 25% (typical_mix)
    LibriSpeech, so the model doesn't forget normal voices.
  * Light augmentation: speed 0.9 / 1.0 / 1.1 and a little noise.
  * Dev WER every eval_every steps (step 0 = the stock model). The best adapter is kept,
    training stops early when dev stops improving. Test data is never used here.
  * The adapter is saved every 30 minutes (last/), so a run cut off by Kaggle's time limit
    loses little; --resume continues it. --max-minutes is a hard time budget.

Writes to checkpoints/<name>/: config.json (incl. seed), dev_history.json, results.json,
best/ (best adapter), last/ (latest adapter + optimizer state), train.log.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from engine.voice.finetune import common


@dataclass
class TrainConfig:
    name: str = "r32-lr1e-4-mix25"
    base_model: str = common.BASE_MODEL
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])
    lr: float = 1e-4
    warmup_steps: int = 50
    weight_decay: float = 0.0
    batch_size: int = 16          # clips per optimizer step
    micro_batch: int = 4          # clips per forward pass (halves itself on out-of-memory)
    max_steps: int = 2000
    max_epochs: float = 5.0       # over the dysarthric/personal train clips
    typical_mix: float = 0.25     # share of LibriSpeech in each batch
    eval_every: int = 250
    patience: int = 4             # evals without improvement before stopping
    min_delta: float = 0.2        # WER points that count as an improvement
    typical_guard: float = 1.0    # LibriSpeech dev may get at most this much worse (WER points)
    dev_max_dys: int = 400
    dev_max_typical: int = 150
    save_every_min: float = 30.0
    max_minutes: float = 0.0      # 0 = no limit
    speed_check_steps: int = 50
    seed: int = 1234
    speed_perturb: list = field(default_factory=lambda: [0.9, 1.0, 1.1])
    noise_prob: float = 0.3
    noise_snr_db: list = field(default_factory=lambda: [20.0, 40.0])
    grad_clip: float = 1.0
    grad_checkpointing: bool = True   # saves GPU memory; also on in --tiny so the CPU test covers it
    num_workers: int = 2
    device: str = "auto"
    tiny: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def tiny_config(cfg: TrainConfig) -> TrainConfig:
    cfg.base_model = common.TINY_MODEL
    cfg.max_steps = 5
    cfg.batch_size = 4
    cfg.micro_batch = 2
    cfg.warmup_steps = 1
    cfg.eval_every = 5
    cfg.dev_max_dys = 10
    cfg.dev_max_typical = 10
    cfg.speed_check_steps = 2
    cfg.num_workers = 0
    cfg.device = "cpu"
    cfg.save_every_min = 0.0     # save every step, to test it
    cfg.tiny = True
    return cfg


# ---------------------------------------------------------------------- data
class ClipDataset:
    """Loads audio, augments (train only), returns (log-mel features, label token ids)."""

    def __init__(self, clips, data_dir: Path, processor, augment: bool, cfg: TrainConfig) -> None:
        self.clips = clips
        self.data_dir = Path(data_dir)
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer
        self.augment = augment
        self.cfg = cfg
        self.start_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self._visits: dict[int, int] = {}    # how often each clip was loaded (per worker)

    def __len__(self) -> int:
        return len(self.clips)

    def _augment(self, audio, rng):
        import numpy as np

        speed = rng.choice(self.cfg.speed_perturb) if self.cfg.speed_perturb else 1.0
        if speed != 1.0:
            from scipy.signal import resample_poly

            # play faster/slower: resample, then treat as 16 kHz again (pitch moves too)
            audio = resample_poly(audio, 100, int(round(100 * speed))).astype(np.float32)
        if rng.random() < self.cfg.noise_prob and len(audio):
            snr = rng.uniform(*self.cfg.noise_snr_db)
            power = float(np.mean(audio ** 2)) + 1e-10
            noise = np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(audio))
            audio = audio + (noise * math.sqrt(power / 10 ** (snr / 10))).astype(np.float32)
        return audio[: common.SAMPLE_RATE * 30]

    def __getitem__(self, i: int):
        for attempt in range(5):          # a broken audio file must not stop a 10-hour run
            c = self.clips[i]
            try:
                audio = common.load_audio(c.audio_path(self.data_dir))
                break
            except Exception as exc:
                print(f"Skipping unreadable clip {c.path}: {exc}", flush=True)
                i = (i + 1) % len(self.clips)
        else:
            raise RuntimeError("5 unreadable clips in a row: check the data folder")
        if self.augment:
            visit = self._visits.get(i, 0)
            self._visits[i] = visit + 1
            # repeatable: the same seed gives the same speed/noise for this clip's n-th use
            audio = self._augment(audio, random.Random(f"{self.cfg.seed}-{i}-{visit}"))
        feats = self.fe(audio, sampling_rate=common.SAMPLE_RATE).input_features[0]
        labels = self.tok(c.text).input_ids
        if labels and labels[0] == self.start_id:    # the model adds it back when shifting
            labels = labels[1:]
        return feats, labels


def collate(batch):
    import numpy as np
    import torch

    feats = torch.from_numpy(np.stack([b[0] for b in batch]).astype(np.float32))
    width = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (_, ids) in enumerate(batch):
        labels[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return feats, labels


class MixedBatches:
    """Endless batches of indices: (1 - mix) from `main`, mix from `typical`, both reshuffled
    every pass. Indices point into main + typical concatenated."""

    def __init__(self, n_main: int, n_typical: int, batch_size: int, mix: float, seed: int) -> None:
        self.n_main, self.n_typ = n_main, n_typical
        self.b = batch_size
        self.k_typ = min(batch_size - 1, round(batch_size * mix)) if n_typical else 0
        self.rng = random.Random(seed)
        self.main_passes = 0.0

    def _cycle(self, n: int, offset: int):
        while True:
            order = list(range(n))
            self.rng.shuffle(order)
            for i in order:
                yield offset + i

    def __iter__(self):
        main = self._cycle(self.n_main, 0)
        typ = self._cycle(self.n_typ, self.n_main) if self.n_typ else None
        k_main = self.b - self.k_typ
        while True:
            batch = [next(main) for _ in range(k_main)]
            if typ is not None:
                batch += [next(typ) for _ in range(self.k_typ)]
            self.rng.shuffle(batch)
            self.main_passes += k_main / max(1, self.n_main)
            yield batch


# ------------------------------------------------------------------- training
class Trainer:
    def __init__(self, cfg: TrainConfig, data_dir: Path, checkpoints: Path,
                 resume: Path | None = None, log=None) -> None:
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.run_dir = Path(checkpoints) / cfg.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or common.Log(self.run_dir / "train.log")
        self.resume = resume
        self.history: list[dict] = []
        self.best: dict | None = None
        self.baseline_dev: dict | None = None
        self.step = 0
        self.stop_reason = ""

    # ............................................................... setup
    def _clips(self):
        clips = common.read_manifest(common.manifest_path(self.data_dir, self.cfg.tiny))
        train = [c for c in clips if c.split == "train"]
        main = [c for c in train if c.source in ("torgo", "personal")]
        typical = [c for c in train if c.source == "librispeech"]
        if not main:                       # tiny smoke test: LibriSpeech only
            main, typical = typical, typical
        dev = [c for c in clips if c.split == "dev"]
        dev_dys = [c for c in dev if c.group in common.DYS_GROUPS][: self.cfg.dev_max_dys]
        dev_typ = [c for c in dev if c.source == "librispeech"][: self.cfg.dev_max_typical]
        return main, typical, dev_dys, dev_typ

    def setup(self) -> None:
        import numpy as np
        import torch

        from engine.voice.finetune import hf

        cfg = self.cfg
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = hf.pick_device(cfg.device)
        self.amp = hf.amp_dtype(self.device)
        self.log(f"Run {cfg.name}: {cfg.base_model} on {common.gpu_name() if self.device == 'cuda' else 'cpu'}, "
                 f"precision {str(self.amp).replace('torch.', '') if self.amp else 'float32'}")

        self.main, self.typical, self.dev_dys, self.dev_typ = self._clips()
        self.log(f"Train: {len(self.main)} main clips, {len(self.typical)} typical clips; "
                 f"dev: {len(self.dev_dys)} dysarthric, {len(self.dev_typ)} typical")

        self.processor = hf.load_processor(cfg.base_model)
        model = hf.load_model(cfg.base_model, device="cpu")
        model.config.use_cache = False
        model.generation_config.forced_decoder_ids = None
        from peft import LoraConfig, PeftModel, get_peft_model

        if self.resume and (Path(self.resume) / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(self.resume), is_trainable=True)
            self.log(f"Resumed adapter from {self.resume}")
        else:
            model = get_peft_model(model, LoraConfig(
                r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                target_modules=list(cfg.target_modules), bias="none"))
        if cfg.grad_checkpointing:
            model.base_model.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(self.device)
        self.model = model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.log(f"Trainable LoRA parameters: {trainable / 1e6:.1f} M of {total / 1e6:.0f} M")

        params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, self._lr_factor)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == torch.float16)
        epoch_steps = cfg.max_epochs * len(self.main) / max(1, cfg.batch_size - round(cfg.batch_size * cfg.typical_mix))
        self.max_steps = max(1, min(cfg.max_steps, math.ceil(epoch_steps)))
        self.micro = max(1, min(cfg.micro_batch, cfg.batch_size))

        if self.resume:
            self._load_state(Path(self.resume))

        ds = ClipDataset(self.main + self.typical, self.data_dir, self.processor, True, cfg)
        self.sampler = MixedBatches(len(self.main), len(self.typical) if self.typical is not self.main else 0,
                                    cfg.batch_size, cfg.typical_mix, cfg.seed + self.step)
        from torch.utils.data import DataLoader

        self.loader = DataLoader(ds, batch_sampler=self.sampler, collate_fn=collate,
                                 num_workers=cfg.num_workers, pin_memory=self.device == "cuda",
                                 persistent_workers=cfg.num_workers > 0,
                                 prefetch_factor=4 if cfg.num_workers > 0 else None)
        common.save_json(self.run_dir / "config.json", asdict(cfg))

    def _lr_factor(self, step: int) -> float:
        warm = max(1, self.cfg.warmup_steps)
        if step < warm:
            return (step + 1) / warm
        span = max(1, self.max_steps - warm)
        return max(0.0, 1.0 - (step - warm) / span)

    # .......................................................... checkpoints
    def _save(self, folder: Path, with_state: bool) -> None:
        import torch

        folder.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(folder))
        common.save_json(folder / "train_config.json", asdict(self.cfg))
        if with_state:
            torch.save({"opt": self.opt.state_dict(), "sched": self.sched.state_dict(),
                        "scaler": self.scaler.state_dict()}, folder / "trainer_state.pt")
            common.save_json(folder / "state.json", {
                "step": self.step, "max_steps": self.max_steps, "micro": self.micro,
                "best": self.best, "baseline_dev": self.baseline_dev, "history": self.history})

    def _load_state(self, folder: Path) -> None:
        import torch

        state_file = folder / "state.json"
        if not state_file.exists():
            return
        state = json.loads(state_file.read_text(encoding="utf-8"))
        self.step = state["step"]          # max_steps is recomputed: a new session has a new budget
        self.micro = state.get("micro", self.micro)
        self.best = state.get("best")
        self.baseline_dev = state.get("baseline_dev")
        self.history = state.get("history", [])
        blob = folder / "trainer_state.pt"
        if blob.exists():
            saved = torch.load(blob, map_location="cpu", weights_only=False)
            self.opt.load_state_dict(saved["opt"])
            self.sched.load_state_dict(saved["sched"])
            self.scaler.load_state_dict(saved["scaler"])
        self.log(f"Continuing at step {self.step} of {self.max_steps}")

    # ............................................................... dev
    def evaluate_dev(self) -> dict:
        from engine.voice.finetune import hf

        self.model.eval()
        self.model.base_model.model.config.use_cache = True
        out = {"step": self.step}
        t0 = time.perf_counter()
        for key, clips in (("dys", self.dev_dys), ("typical", self.dev_typ)):
            if not clips:
                continue
            audios = [common.load_audio(c.audio_path(self.data_dir)) for c in clips]
            hyps = hf.transcribe(self.model, self.processor, audios, self.device,
                                 batch_size=16 if self.device == "cuda" else 4)
            out[key] = common.wer_table([(c.text, h) for c, h in zip(clips, hyps)])
            if key == "dys":
                unseen = [(c.text, h) for c, h in zip(clips, hyps) if c.group == "dys_unseen"]
                out["dys_unseen"] = common.wer_table(unseen)
        self.model.base_model.model.config.use_cache = False
        self.model.train()
        out["seconds"] = round(time.perf_counter() - t0, 1)
        out["score"] = self._score(out)
        return out

    def _score(self, r: dict) -> float:
        """Lower is better. Dysarthric dev WER (averaged with unseen-text WER when there are
        enough unseen sentences), plus a penalty if typical speech got worse than allowed."""
        if "dys" in r and r["dys"]["wer"] is not None:
            score = r["dys"]["wer"]
            u = r.get("dys_unseen") or {}
            if u.get("n", 0) >= 30 and u.get("wer") is not None:
                score = (score + u["wer"]) / 2
        else:
            score = r.get("typical", {}).get("wer") or 0.0
        base_typ = ((self.baseline_dev or r).get("typical") or {}).get("wer")
        typ = (r.get("typical") or {}).get("wer")
        if base_typ is not None and typ is not None:
            score += 3.0 * max(0.0, typ - (base_typ + self.cfg.typical_guard))
        return round(score, 3)

    def _record_eval(self) -> bool:
        """Run dev, keep the best adapter. Returns True when it's time to stop early."""
        r = self.evaluate_dev()
        if self.baseline_dev is None:
            self.baseline_dev = r
            r["score"] = self._score(r)
        self.history.append(r)
        common.save_json(self.run_dir / "dev_history.json", self.history)
        improved = self.best is None or r["score"] < self.best["score"] - self.cfg.min_delta
        dys = (r.get("dys") or {}).get("wer")
        typ = (r.get("typical") or {}).get("wer")
        self.log(f"Dev at step {self.step}: score {r['score']} (dysarthric {dys}, typical {typ}) "
                 f"{'* best' if improved else ''} [{r['seconds']:.0f} s]")
        if improved:
            self.best = r
            self._save(self.run_dir / "best", with_state=False)
            return False
        bad = sum(1 for h in self.history if h["step"] > self.best["step"])
        return bad >= self.cfg.patience

    # ............................................................... loop
    def _micro_steps(self, feats, labels) -> float:
        import torch

        total = 0.0
        n = feats.shape[0]
        while True:
            try:
                self.opt.zero_grad(set_to_none=True)
                total = 0.0
                for i in range(0, n, self.micro):
                    x = feats[i:i + self.micro].to(self.device, non_blocking=True)
                    y = labels[i:i + self.micro].to(self.device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=self.amp,
                                        enabled=self.amp is not None):
                        loss = self.model(input_features=x, labels=y).loss
                    loss = loss * (x.shape[0] / n)
                    self.scaler.scale(loss).backward()
                    total += float(loss.detach())
                return total
            except torch.cuda.OutOfMemoryError:
                if self.micro == 1:
                    raise
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self.micro = max(1, self.micro // 2)
                self.log(f"Out of GPU memory: micro-batch is now {self.micro}")

    def train(self) -> dict:
        import torch

        cfg = self.cfg
        self.setup()
        t_start = time.monotonic()
        deadline = t_start + cfg.max_minutes * 60 if cfg.max_minutes else None
        last_save = time.monotonic()
        if not self.history:
            self._record_eval()                 # step 0 = the stock model on dev
        eval_s = self.history[-1]["seconds"] if self.history else 0.0
        self.model.train()
        loader = iter(self.loader)
        t_speed, step_speed = None, None
        losses = []
        while self.step < self.max_steps:
            feats, labels = next(loader)
            loss = self._micro_steps(feats, labels)
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                           cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            self.step += 1
            losses.append(loss)

            if self.step % 10 == 0 or cfg.tiny:
                self.log(f"step {self.step}/{self.max_steps} loss {sum(losses) / len(losses):.4f} "
                         f"lr {self.sched.get_last_lr()[0]:.2e}")
                losses = []

            # speed check: shorten the run if it won't fit the time budget
            if t_speed is None and self.step >= min(5, cfg.speed_check_steps):
                t_speed, step_speed = time.monotonic(), self.step
            elif step_speed is not None and self.step == step_speed + cfg.speed_check_steps:
                sps = (time.monotonic() - t_speed) / cfg.speed_check_steps
                self.log(f"Speed: {sps:.2f} s per step ({cfg.batch_size} clips)")
                if deadline:
                    left = deadline - time.monotonic()
                    evals = (self.max_steps - self.step) / cfg.eval_every + 1
                    need = (self.max_steps - self.step) * sps + evals * eval_s
                    if need > left:
                        fit = int((left - evals * eval_s) / sps)
                        new_max = max(self.step + 1, self.step + fit)
                        self.log(f"Won't fit in {cfg.max_minutes:.0f} min: max_steps "
                                 f"{self.max_steps} -> {new_max}")
                        self.max_steps = new_max

            if self.step % cfg.eval_every == 0 or self.step == self.max_steps:
                if self._record_eval():
                    self.stop_reason = "early stop (dev stopped improving)"
                    break
            if time.monotonic() - last_save >= cfg.save_every_min * 60:
                self._save(self.run_dir / "last", with_state=True)
                last_save = time.monotonic()
            if deadline and time.monotonic() > deadline - eval_s:
                self.stop_reason = "time budget"
                if self.history[-1]["step"] != self.step:
                    self._record_eval()
                break
        else:
            self.stop_reason = "finished max_steps"
        self._save(self.run_dir / "last", with_state=True)
        result = {"name": cfg.name, "config": asdict(cfg), "steps": self.step,
                  "max_steps": self.max_steps, "stop_reason": self.stop_reason,
                  "minutes": round((time.monotonic() - t_start) / 60, 1),
                  "baseline_dev": self.baseline_dev, "best_dev": self.best,
                  "best_adapter": str(self.run_dir / "best"),
                  "epochs": round(self.sampler.main_passes, 2)}
        common.save_json(self.run_dir / "results.json", result)
        self.log(f"Run {cfg.name} done after {self.step} steps ({self.stop_reason}); "
                 f"best dev score {self.best['score'] if self.best else None} at step "
                 f"{self.best['step'] if self.best else None}")
        return result


def train(cfg: TrainConfig, data_dir: Path, checkpoints: Path, resume: Path | None = None,
          log=None) -> dict:
    return Trainer(cfg, data_dir, checkpoints, resume, log).train()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.data_dir_arg(parser)
    parser.add_argument("--checkpoints", type=Path, default=common.DEFAULT_CHECKPOINTS)
    parser.add_argument("--resume", type=Path, default=None,
                        help="adapter folder to continue from (e.g. checkpoints/<name>/last)")
    parser.add_argument("--tiny", action="store_true", help="smoke test: whisper tiny, 5 steps, CPU")
    for f in fields(TrainConfig):
        if f.name in ("tiny",):
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("list", list) or f.name in ("target_modules", "speed_perturb", "noise_snr_db"):
            parser.add_argument(flag, nargs="+", default=None)
        else:
            caster = {"int": int, "float": float, "str": str,
                      "bool": lambda v: v.lower() in ("1", "true", "yes")}.get(str(f.type), str)
            parser.add_argument(flag, type=caster, default=None)
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    if args.resume and (args.resume / "train_config.json").exists():
        cfg = TrainConfig.from_dict(json.loads((args.resume / "train_config.json").read_text(encoding="utf-8")))
    if args.tiny:
        cfg = tiny_config(cfg)
        cfg.name = "tiny"
    for f in fields(TrainConfig):
        value = getattr(args, f.name, None)
        if value is not None:
            if f.name == "speed_perturb" or f.name == "noise_snr_db":
                value = [float(v) for v in value]
            setattr(cfg, f.name, value)
    train(cfg, args.data_dir, args.checkpoints, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
