"""Run with: python -m unittest tests.test_rewriter  (no model download: fake tokenizer and model)

The fakes use real torch tensors, so the token slicing and generate() call are the real thing;
only the model's output is scripted.
"""
import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

from engine.voice import rewrite
from engine.voice.rewrite import EXAMPLES, SYSTEM_PROMPT, Rewriter, tidy

HAS_TORCH = importlib.util.find_spec("torch") is not None
SETTINGS = SimpleNamespace(rewrite_model="fake/model", rewrite_device="cpu", rewrite_max_s=3.0)
EOS = 99


class FakeTokenizer:
    """Words <-> ids 1..; the chat template is plain text so the test can read it."""

    eos_token_id = EOS
    pad_token_id = None

    def __init__(self):
        self.vocab, self.words = {}, {}
        self.templates = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        self.templates.append((messages, add_generation_prompt, tokenize))
        return " ".join(f"<{m['role']}> {m['content']}" for m in messages) + (" <assistant>" if add_generation_prompt else "")

    def _ids(self, text):
        out = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab) + 1
                self.words[self.vocab[w]] = w
            out.append(self.vocab[w])
        return out

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        assert add_special_tokens is False         # the template already has them
        ids = self._ids(text)
        if return_tensors == "pt":
            import torch

            return Batch(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.words[i] for i in ids if not (skip_special_tokens and i == EOS))


class Batch(dict):
    """Like transformers' BatchEncoding: a dict with .to(device)."""

    def to(self, device):
        return self


class FakeModel:
    """generate() appends the scripted reply's tokens (and EOS unless cut=True) to the prompt."""

    def __init__(self, tokenizer, reply, cut=False):
        self.tok, self.reply, self.cut = tokenizer, reply, cut
        self.generation_config = SimpleNamespace(eos_token_id=[EOS, 98])
        self.calls = []

    def generate(self, **kw):
        import torch

        self.calls.append(kw)
        new = self.tok._ids(self.reply) + ([] if self.cut else [EOS])
        return torch.cat([kw["input_ids"], torch.tensor([new])], dim=1)


def make(reply, cut=False):
    tok = FakeTokenizer()
    model = FakeModel(tok, reply, cut)
    return Rewriter(SETTINGS, model=model, tokenizer=tok, log=lambda s: None), model, tok


class TestTidy(unittest.TestCase):
    def test_labels_and_quotes(self):
        self.assertEqual(tidy("Rewritten text: Hi mom."), "Hi mom.")
        self.assertEqual(tidy('"Hi mom."'), "Hi mom.")
        self.assertEqual(tidy("“Hi mom.”"), "Hi mom.")
        self.assertEqual(tidy("  Hi mom.  "), "Hi mom.")

    def test_quotes_the_user_said_stay(self):
        self.assertEqual(tidy('"Yes" is all she said.', '"yes" is all she said'), '"Yes" is all she said.')


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestRewriter(unittest.TestCase):
    def test_uses_the_clean_text_when_the_guard_accepts(self):
        r, model, tok = make("I think we should go.")
        clean, used, ms = r.rewrite("um so I I think we should uh go")
        self.assertEqual((clean, used), ("I think we should go.", True))
        self.assertGreaterEqual(ms, 0)

    def test_prompt_is_the_chat_template_with_the_exact_system_message(self):
        r, model, tok = make("Hi mom.")
        r.rewrite("hi mom")
        messages, add_prompt, tokenize = tok.templates[0]
        examples = []
        for said, cleaned in EXAMPLES:     # worked examples between the system message and the real text
            examples += [{"role": "user", "content": said}, {"role": "assistant", "content": cleaned}]
        self.assertEqual(messages, [{"role": "system", "content": SYSTEM_PROMPT}, *examples,
                                    {"role": "user", "content": "hi mom"}])
        self.assertTrue(add_prompt)
        self.assertFalse(tokenize)

    def test_greedy_with_token_and_time_caps(self):
        r, model, _ = make("Hi mom.")
        r.rewrite("hi mom")                                      # 2 tokens -> at least 16
        kw = model.calls[0]
        self.assertFalse(kw["do_sample"])
        self.assertNotIn("temperature", kw)
        self.assertNotIn("top_p", kw)
        self.assertEqual(kw["max_new_tokens"], 16)
        self.assertEqual(kw["max_time"], 3.0)
        self.assertEqual(kw["pad_token_id"], EOS)
        long = " ".join(f"w{i}" for i in range(20))
        r.rewrite(long)
        self.assertEqual(model.calls[1]["max_new_tokens"], 40)    # 2x the input

    def test_only_the_new_tokens_are_decoded(self):
        r, _, _ = make("Hi mom.")
        out = r.rewrite_full("hi mom")
        self.assertEqual(out.output, "Hi mom.")                   # not the prompt echoed back

    def test_a_cut_off_rewrite_is_not_used(self):
        r, _, _ = make("I think we", cut=True)                     # stopped on the cap, no EOS
        out = r.rewrite_full("um I think we should go")
        self.assertEqual((out.clean, out.used), ("I think we should go", True))   # the rule-based clean-up
        self.assertTrue(out.reason.startswith("light clean: model cut off"))
        r, _, _ = make("I think we", cut=True)
        out = r.rewrite_full("I think we should go")              # nothing for the rules to delete
        self.assertEqual((out.clean, out.used), ("I think we should go", False))
        self.assertTrue(out.reason.startswith("cut off"))

    def test_a_reworded_rewrite_falls_back_to_the_rule_based_clean_up(self):
        r, _, _ = make("I believe we should leave at noon.")      # the 5080 run: think -> believe
        out = r.rewrite_full("Um, so, I think we should, uh, leave at noon.")
        self.assertEqual((out.clean, out.used), ("So, I think we should leave at noon.", True))
        self.assertEqual(out.reason, "light clean: model word swapped: think -> believe")
        self.assertEqual(out.output, "I believe we should leave at noon.")

    def test_a_pronoun_swap_falls_back_too(self):
        r, _, _ = make("Please bring the charger for your laptop.")
        clean, used, _ = r.rewrite("Please please bring the charger, the charger for my laptop.")
        self.assertEqual((clean, used), ("Please bring the charger for my laptop.", True))

    def test_a_rejected_rewrite_gives_the_original(self):
        r, _, _ = make("It will be sunny tomorrow.")
        clean, used, _ = r.rewrite("what's the weather tomorrow")
        self.assertEqual((clean, used), ("what's the weather tomorrow", False))

    def test_labels_and_quotes_are_removed_before_the_guard(self):
        r, _, _ = make('Rewritten text: "Hi mom, home at seven."')
        self.assertEqual(r.rewrite("hi mom home at seven")[0], "Hi mom, home at seven.")

    def test_empty_input_never_calls_the_model(self):
        r, model, _ = make("x")
        self.assertEqual(r.rewrite("   "), ("   ", False, 0.0))
        self.assertEqual(r.rewrite(""), ("", False, 0.0))
        self.assertEqual(model.calls, [])


class TestFailOpen(unittest.TestCase):
    def test_any_error_gives_the_original_and_logs_once_a_minute(self):
        logs, clock = [], SimpleNamespace(t=0.0)
        r = Rewriter(SETTINGS, log=logs.append, clock=lambda: clock.t)
        with mock.patch.object(Rewriter, "_generate", side_effect=RuntimeError("CUDA out of memory")):
            self.assertEqual(r.rewrite("hi mom")[:2], ("hi mom", False))
            clock.t = 30
            r.rewrite("hi mom")
            self.assertEqual(len(logs), 1)
            clock.t = 61
            r.rewrite("hi mom")
            self.assertEqual(len(logs), 2)
        self.assertIn("CUDA out of memory", logs[0])

    def test_a_model_error_still_gets_the_rule_based_clean_up(self):
        r = Rewriter(SETTINGS, log=lambda s: None)
        with mock.patch.object(Rewriter, "_generate", side_effect=RuntimeError("CUDA out of memory")):
            out = r.rewrite_full("um so the the bus comes at 7:15")
        self.assertEqual((out.clean, out.used), ("so the bus comes at 7:15", True))
        self.assertTrue(out.reason.startswith("light clean: model error"))

    def test_load_failure_is_an_error_too(self):
        r = Rewriter(SimpleNamespace(rewrite_model="does/not-exist", rewrite_device="cpu", rewrite_max_s=1),
                     log=lambda s: None)
        with mock.patch.object(Rewriter, "load", side_effect=OSError("no such model")):
            out = r.rewrite_full("hi mom")
        self.assertEqual((out.clean, out.used), ("hi mom", False))
        self.assertTrue(out.reason.startswith("error"))


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("transformers"), "transformers not installed")
class TestLoad(unittest.TestCase):
    def test_tries_the_local_cache_first_then_downloads(self):
        import torch

        calls = []
        model = mock.Mock()

        def tok_from(name, local_files_only):
            calls.append(("tok", local_files_only))
            if local_files_only:
                raise OSError("not cached")
            return FakeTokenizer()

        def model_from(name, dtype, local_files_only):
            calls.append(("model", local_files_only, dtype))
            return model

        with mock.patch("transformers.AutoTokenizer.from_pretrained", side_effect=tok_from), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=model_from):
            Rewriter(SETTINGS).load()
        self.assertEqual(calls, [("tok", True), ("tok", False), ("model", False, torch.bfloat16)])
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once()

    def test_float16_on_cuda(self):
        import torch

        model = mock.Mock()
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=FakeTokenizer()), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model) as load:
            Rewriter(SimpleNamespace(rewrite_model="m", rewrite_device="cuda", rewrite_max_s=3)).load()
        self.assertEqual(load.call_args.kwargs["dtype"], torch.float16)
        model.to.assert_called_once_with("cuda")


if __name__ == "__main__":
    unittest.main()
