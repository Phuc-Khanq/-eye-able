"""Hugging Face Whisper helpers: load a model (optionally with a LoRA adapter) and transcribe.

Decoding matches dictation.py: English, transcribe, greedy (one beam), no previous text.
"""

from __future__ import annotations

from pathlib import Path

MAX_NEW_TOKENS = 200


def pick_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def amp_dtype(device: str):
    """bf16 where the GPU supports it, fp16 otherwise (T4), None on CPU."""
    import torch

    if device != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


def load_processor(base: str):
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(base)
    processor.tokenizer.set_prefix_tokens(language="english", task="transcribe",
                                          predict_timestamps=False)
    return processor


def load_model(base: str, adapter: Path | None = None, device: str = "cpu", dtype=None):
    """Whisper (+ adapter) ready for generate(). dtype=None keeps float32."""
    import torch
    from transformers import WhisperForConditionalGeneration
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()      # hides a harmless max_length note on every generate()
    model = WhisperForConditionalGeneration.from_pretrained(base, dtype=dtype or torch.float32)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    model.to(device)
    model.eval()
    return model


def transcribe(model, processor, audios, device: str, batch_size: int = 16) -> list[str]:
    """Greedy English transcription of 16 kHz float32 arrays."""
    import torch

    autocast = amp_dtype(device)
    texts: list[str] = []
    for i in range(0, len(audios), batch_size):
        chunk = audios[i:i + batch_size]
        feats = processor.feature_extractor(chunk, sampling_rate=16000, return_tensors="pt")
        x = feats.input_features.to(device)
        model_dtype = next(model.parameters()).dtype
        if autocast is None:
            x = x.to(model_dtype)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast,
                                             enabled=autocast is not None):
            ids = model.generate(input_features=x, language="en", task="transcribe",
                                 num_beams=1, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
        texts += [t.strip() for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]
    return texts
