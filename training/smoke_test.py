"""Run every step of the training pipeline on a tiny random Whisper, offline, on CPU.

    python -m training.smoke_test

Checks the code works with the installed library versions (transformers, peft,
ctranslate2, faster-whisper) before anyone spends GPU hours. The tiny model learns
nothing useful; the point is that every step runs and connects to the next:
prepare -> LoRA train -> merge -> export -> load in the engine's Dictation -> evaluate.
No downloads, nothing written outside a temp folder.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path


def _bytes_to_unicode() -> list[str]:
    """GPT-2 / Whisper byte-level alphabet: one printable character per byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for _, c in sorted(zip(bs, cs))]


def build_tokenizer(folder: Path):
    """A byte-level tokenizer with Whisper's special tokens, in Whisper's order."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import WhisperTokenizer
    from transformers.models.whisper.tokenization_whisper import LANGUAGES

    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode())}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = (["<|endoftext|>", "<|startoftranscript|>"]
                + [f"<|{code}|>" for code in LANGUAGES]
                + ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
                   "<|nospeech|>", "<|notimestamps|>"]
                + [f"<|{i * 0.02:.2f}|>" for i in range(1501)])
    tok.add_special_tokens(specials)
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "WhisperTokenizer",
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
        "additional_special_tokens": specials[1:-1501],
    }), encoding="utf-8")
    return WhisperTokenizer.from_pretrained(str(folder)), list(LANGUAGES)


def build_model(tokenizer, langs):
    from transformers import GenerationConfig, WhisperConfig, WhisperForConditionalGeneration

    ids = tokenizer.convert_tokens_to_ids
    eot, sot = ids("<|endoftext|>"), ids("<|startoftranscript|>")
    cfg = WhisperConfig(
        vocab_size=len(tokenizer), num_mel_bins=80, d_model=64,
        encoder_layers=2, decoder_layers=2, encoder_attention_heads=4, decoder_attention_heads=4,
        encoder_ffn_dim=128, decoder_ffn_dim=128, max_source_positions=1500,
        max_target_positions=448, decoder_start_token_id=sot, pad_token_id=eot,
        bos_token_id=eot, eos_token_id=eot, suppress_tokens=[], begin_suppress_tokens=[eot],
    )
    model = WhisperForConditionalGeneration(cfg)
    gc = GenerationConfig.from_model_config(cfg)
    gc.lang_to_id = {f"<|{c}|>": ids(f"<|{c}|>") for c in langs}
    gc.task_to_id = {"transcribe": ids("<|transcribe|>"), "translate": ids("<|translate|>")}
    gc.no_timestamps_token_id = ids("<|notimestamps|>")
    gc.is_multilingual = True
    gc.alignment_heads = [[1, 0], [1, 1]]
    gc.max_length = 448
    model.generation_config = gc
    return model


def fake_examples(n_per_speaker: int = 3):
    """Rows shaped like the Hugging Face TORGO rows (audio cast with decode=False)."""
    import numpy as np
    import soundfile as sf

    texts = ["yes", "call grandpa", "the quick brown fox", "[say Ah-P-Eee repeatedly]"]
    out = []
    for spk in ["F01", "M01", "FC01", "MC01", "F04", "MC04"]:
        for i in range(n_per_speaker + 1):
            t = np.arange(16_000) / 16_000
            wav = (0.1 * np.sin(2 * np.pi * (200 + 40 * i) * t)).astype(np.float32)
            buf = io.BytesIO()
            sf.write(buf, wav, 16_000, format="WAV")
            out.append({"audio": {"bytes": buf.getvalue(), "path": f"{spk}_1_arrayMic_{i:04d}.wav"},
                        "transcription": texts[i % len(texts)],
                        "speech_status": "healthy" if "C" in spk else "dysarthria"})
    return out


def main() -> int:
    from datasets import load_from_disk
    from transformers import WhisperFeatureExtractor, WhisperProcessor

    from training import evaluate, finetune_whisper, prepare_torgo
    from training.torgo_utils import Gate

    # ignore_cleanup_errors: on Windows faster-whisper can still hold model.bin open at exit
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        print("1/6 prepare: split fake TORGO rows by speaker")
        train, dev, test, dropped = prepare_torgo.split_rows(fake_examples(), {"F04", "MC04"})
        assert test and all(r["speaker"] in {"F04", "MC04"} for r in test)
        assert not {r["speaker"] for r in train + dev} & {"F04", "MC04"}
        assert dropped["prompt, not speech"] > 0
        prepare_torgo.save_splits(train, dev, test, tmp / "data")
        data = load_from_disk(str(tmp / "data"))

        print("2/6 build a tiny random Whisper")
        tokenizer, langs = build_tokenizer(tmp / "tok")
        processor = WhisperProcessor(feature_extractor=WhisperFeatureExtractor(feature_size=80),
                                     tokenizer=tokenizer)
        processor.tokenizer.set_prefix_tokens(language="english", task="transcribe")
        model = build_model(tokenizer, langs)

        print("3/6 LoRA train 4 steps")
        model = finetune_whisper.add_lora(model, r=4, alpha=8, dropout=0.0)
        model = finetune_whisper.train(model, processor, data["train"], data["dev"], tmp / "lora",
                                       max_steps=4, batch=2, grad_accum=1, lr=1e-3,
                                       eval_every=2, bf16=False, workers=0)
        model.save_pretrained(str(tmp / "lora" / "adapter"))

        print("4/6 merge LoRA and export to faster-whisper (float16, like the real run)")
        merged = finetune_whisper.merge_and_save(model, processor, tmp / "merged")
        ct2 = finetune_whisper.export_ct2(merged, tmp / "ct2", quantization="float16")
        assert (ct2 / "model.bin").exists() and (ct2 / "tokenizer.json").exists()

        print("5/6 load it through engine/voice/dictation.py and transcribe")
        hyps, ms = evaluate.transcribe_all(str(ct2), list(data["test"]), "cpu", "float32")
        assert len(hyps) == len(data["test"])

        print("6/6 score and gate")
        groups = evaluate.score(list(data["test"]), hyps)
        assert "dysarthric" in groups and "typical" in groups
        ok, why = Gate().decide(0.5, groups["dysarthric"]["wer"], 0.1, groups["typical"]["wer"])
        print(f"   tiny model verdict (expected: don't ship): {why}")

    print("\nSmoke test passed: every step runs with these library versions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
