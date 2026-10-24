"""LoRA fine-tune Whisper large-v3-turbo on TORGO, then export it for faster-whisper.

Run on the RTX 5080 laptop after prepare_torgo.py:
    .\\.venv\\Scripts\\python.exe -m training.finetune_whisper
    .\\.venv\\Scripts\\python.exe -m training.finetune_whisper --max-steps 300   # quick try

Output:
    models/whisper-turbo-torgo-lora/   small LoRA adapter + checkpoints
    models/whisper-turbo-torgo-ct2/    merged model in faster-whisper format (what the engine loads)

LoRA trains about 1-2 % of the weights in small side layers, so it fits in 16 GB and
can't drift far from what Whisper already knows. Typical speakers stay in the training mix
so the model doesn't forget normal speech.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from training.torgo_utils import training_target

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "openai/whisper-large-v3-turbo"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def decode_wav(data: bytes):
    from training.prepare_torgo import wav_bytes_to_16k

    return wav_bytes_to_16k(data)


@dataclass
class Collator:
    """Turns rows {wav, text} into Whisper inputs. Log-mel features are made here, per batch,
    instead of saved to disk (they'd take about 18 GB)."""

    processor: object
    decoder_start_token_id: int

    def __call__(self, rows: list[dict]) -> dict:
        import torch

        audio = [decode_wav(r["wav"]) for r in rows]
        feats = self.processor.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        texts = [training_target(r["text"]) for r in rows]
        tok = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
        # the model adds <|startoftranscript|> itself; drop it from the labels if present
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats["input_features"], "labels": labels.long()}


def add_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    model.config.use_cache = False
    model.config.apply_spec_augment = True        # small dataset: mask bits of audio while training
    model.config.mask_time_prob = 0.05
    model.config.mask_feature_prob = 0.05
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=LORA_TARGETS, bias="none")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def train(model, processor, train_ds, dev_ds, out_dir: Path, *, max_steps: int, batch: int,
          grad_accum: int, lr: float, eval_every: int, bf16: bool, workers: int,
          gradient_checkpointing: bool = True) -> object:
    from transformers import Trainer, TrainingArguments

    collator = Collator(processor, model.config.decoder_start_token_id)
    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=min(100, max_steps // 10),
        max_steps=max_steps,
        lr_scheduler_type="linear",
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if dev_ds is not None else "no",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=2,
        logging_steps=max(1, min(25, max_steps // 4)),
        remove_unused_columns=False,      # our collator needs the raw columns
        dataloader_num_workers=workers,
        report_to="none",
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                      data_collator=collator)
    trainer.train()
    return trainer.model


def merge_and_save(peft_model, processor, merged_dir: Path) -> Path:
    merged = peft_model.merge_and_unload()
    merged.config.use_cache = True
    merged.config.apply_spec_augment = False
    merged.save_pretrained(str(merged_dir))
    processor.save_pretrained(str(merged_dir))
    return merged_dir


def export_ct2(merged_dir: Path, ct2_dir: Path, quantization: str = "float16") -> Path:
    """Convert a Hugging Face Whisper folder into the format faster-whisper loads."""
    import ctranslate2

    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if (merged_dir / f).exists()]
    ctranslate2.converters.TransformersConverter(str(merged_dir), copy_files=copy).convert(
        str(ct2_dir), quantization=quantization, force=True)
    return ct2_dir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--data", default="data/torgo")
    p.add_argument("--name", default="whisper-turbo-torgo")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    args = p.parse_args()

    import torch
    from datasets import load_from_disk
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if not torch.cuda.is_available():
        print("No CUDA GPU found. Train on the RTX 5080 laptop.")
        return 1
    data = load_from_disk(str(ROOT / args.data))
    processor = WhisperProcessor.from_pretrained(args.base)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model = add_lora(model, r=args.lora_r, alpha=2 * args.lora_r, dropout=0.05)

    out = ROOT / "models" / f"{args.name}-lora"
    model = train(model, processor, data["train"], data["dev"], out,
                  max_steps=args.max_steps, batch=args.batch, grad_accum=args.grad_accum,
                  lr=args.lr, eval_every=args.eval_every,
                  bf16=torch.cuda.is_bf16_supported(), workers=args.workers)
    model.save_pretrained(str(out / "adapter"))

    merged = merge_and_save(model, processor, ROOT / "models" / f"{args.name}-merged")
    ct2 = export_ct2(merged, ROOT / "models" / f"{args.name}-ct2")
    print(f"\nDone. faster-whisper model: {ct2}")
    print("Next: python -m training.evaluate   (decides if it's good enough to use)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
