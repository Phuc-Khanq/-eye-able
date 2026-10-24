"""Run with: python -m unittest tests.test_rewrite_guard  (pure Python: no model)

check_rewrite decides whether a model's rewrite of dictated text is safe to show instead of the
user's own words. Each case says why it must pass or fail.
"""
import unittest

from engine.voice.rewrite import accept_rewrite, check_rewrite, light_clean

KHANG_RAW = ("Valorant Champions Tour. I'm not sure what it is. I'm saying that I'm going to check the "
             "video for me. Dictate Valorant Champions Tour. I'm going to check the video.")
KHANG_CLEAN = "Valorant Champions Tour. I'm not sure what it is, so I'm going to check the video."


class Accepted(unittest.TestCase):
    def ok(self, original, clean):
        self.assertIsNone(check_rewrite(original, clean), (original, clean))
        self.assertTrue(accept_rewrite(original, clean))

    def test_khangs_example(self):
        # 30 -> 16 words (0.53x), 7 of 9 content words kept (dictate, saying dropped), nothing new,
        # the names Valorant Champions Tour survive, "not" survives
        self.ok(KHANG_RAW, KHANG_CLEAN)

    def test_fillers_removed(self):
        self.ok("um so I I think we should uh go", "I think we should go.")

    def test_only_punctuation_and_capitals(self):
        self.ok("hello how are you", "Hello, how are you?")
        self.ok("yes", "Yes.")

    def test_repeated_phrase_collapsed(self):
        self.ok("send it send it to Nhi", "Send it to Nhi.")

    def test_digits_and_number_words_are_the_same_number(self):
        self.ok("meet at seven", "Meet at 7.")
        self.ok("I'm coming at 6", "I'm coming at six.")
        self.ok("my number ends in 4 7 2 9", "My number ends in 4729.")
        self.ok("see you at 3:30", "See you at three thirty.")

    def test_contracted_negation_counts_the_same(self):
        self.ok("I do not want it", "I don't want it.")

    def test_reordering_is_not_a_swap(self):
        self.ok("tomorrow I'll call Khoa", "I'll call Khoa tomorrow.")

    def test_a_stutter_on_a_pronoun_is_fine(self):
        self.ok("I I I want to go to the the park", "I want to go to the park.")

    def test_i_am_is_not_a_name(self):
        # "I'm" mid-sentence is capitalized but isn't a name the rewrite has to keep
        self.ok("so then I'm like going home", "Then I'm going home.")


class Rejected(unittest.TestCase):
    def bad(self, original, clean, reason_starts):
        reason = check_rewrite(original, clean)
        self.assertIsNotNone(reason, (original, clean))
        self.assertTrue(reason.startswith(reason_starts), reason)
        self.assertFalse(accept_rewrite(original, clean))

    def test_an_answer_instead_of_a_rewrite(self):
        self.bad("what's the weather tomorrow", "It will be sunny tomorrow.", ("word swapped", "content changed"))

    def test_a_summary_that_drops_facts(self):
        self.bad("Pick up milk, bread, eggs and butter on the way home from work", "Pick up groceries.",
                 "too short")

    def test_a_dropped_number(self):
        self.bad("meet at 5 at the cafe", "Meet at the cafe.", "number missing")

    def test_a_changed_number(self):
        self.bad("my address is 21 Oak Street", "My address is 12 Oak Street.", "number missing")

    def test_a_dropped_name(self):
        self.bad("call Khoa about the bus", "Call about the bus.", "name missing")

    def test_added_content(self):
        self.bad("send the photos", "Send the photos and the videos from last week.", "too long")
        # 2 of 5 content words new (40% > 25%)
        self.bad("send the photos to Nhi today please", "Send photos, videos and invitations to Nhi.",
                 "content added")

    def test_a_swapped_word(self):
        # not in Khang's list: one noun swapped keeps 80% of the content, so the thresholds alone pass it
        self.bad("please send the photos to Nhi today", "Please send the videos to Nhi today.", "word swapped")
        self.bad("pick up the blue bag", "Pick up the red bag.", "word swapped")

    def test_empty(self):
        self.bad("see you tomorrow", "   ", "empty")
        self.bad("see you tomorrow", "", "empty")
        self.bad("see you tomorrow", "...", "empty")

    def test_much_longer(self):
        self.bad("see you at the station tomorrow morning",
                 "See you at the train station tomorrow morning, okay, and bring the tickets.", "too long")

    def test_a_translation(self):
        self.bad("see you tomorrow at the station", "A demain a la gare.",
                 ("word swapped", "content changed", "point of view changed"))

    def test_a_flipped_negation(self):
        # not in Khang's list, but "keep the meaning" depends on it
        self.bad("I'm not coming tonight", "I'm coming tonight.", "negation changed")
        self.bad("I'm coming to the party tonight", "I'm not coming to the party tonight.", "negation changed")

    def test_a_self_correction_is_rejected_on_purpose(self):
        # The safe side: the 5 goes missing, so the user sees their own words, correction and all.
        self.bad("meet at 5, no wait, at 6", "Meet at 6.", "number missing")

    def test_nothing_to_clean(self):
        self.bad("", "Hello.", "nothing to clean")


class PointOfView(unittest.TestCase):
    """Live run on the 5080 (Oct 24): the model turned "me" into "you" and "my" into "your", and the old
    guard let it through because every pronoun is a stopword."""

    def test_me_to_you_is_rejected(self):
        self.assertTrue(check_rewrite("Can Khang pick me up at 8?", "Can Khang pick you up at 8?")
                        .startswith("point of view changed"))

    def test_my_to_your_is_rejected(self):
        self.assertTrue(check_rewrite("Please please bring the charger, the charger for my laptop.",
                                      "Please bring the charger for your laptop.").startswith("point of view changed"))

    def test_dropping_the_only_i_is_rejected(self):
        self.assertIsNotNone(check_rewrite("I think we should leave at noon.", "We should leave at noon."))


class LightClean(unittest.TestCase):
    """The rule-based clean-up used when the model rewords: fillers and stutters only."""

    def same(self, text):
        self.assertEqual(light_clean(text), text)

    def test_fillers_and_their_commas(self):
        self.assertEqual(light_clean("Um, so, I think we should, uh, leave at noon."), "So, I think we should leave at noon.")
        self.assertEqual(light_clean("Hmm, okay, so the, uh, the bus comes at 7:15."), "Okay, so the bus comes at 7:15.")
        self.assertEqual(light_clean("Uh, can you, like, send me the file, um, tonight?"),
                         "Can you, like, send me the file tonight?")    # "like" can be meant: it stays

    def test_a_filler_at_the_end_keeps_the_full_stop(self):
        self.assertEqual(light_clean("I'll be there at noon, uh."), "I'll be there at noon.")

    def test_stutters_and_repeated_phrases(self):
        self.assertEqual(light_clean("I I I want to go to the the park with Nhi."), "I want to go to the park with Nhi.")
        self.assertEqual(light_clean("Call call Khang when you get home, when you get home."),
                         "Call Khang when you get home.")
        self.assertEqual(light_clean("Tell, uh, tell Khoa that the code is, the code is ready."),
                         "Tell Khoa that the code is ready.")
        self.assertEqual(light_clean("Please please bring the charger, the charger for my laptop."),
                         "Please bring the charger for my laptop.")

    def test_meant_repeats_stay(self):
        self.same("No no, don't open it.")
        self.same("The room is 204 204.")
        self.same("She had had enough.")
        self.same("Bye bye, see you soon.")

    def test_nothing_to_delete_is_unchanged(self):
        self.same("hi mom")
        self.same("I want, I need you to pick up the medicine.")   # a false start needs the model
        self.same("")

    def test_every_case_from_the_5080_run_passes_the_guard(self):
        import csv
        from pathlib import Path

        with (Path(__file__).parent / "rewrite_cases.csv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                lc = light_clean(row["raw"])
                if lc != row["raw"]:
                    self.assertIsNone(check_rewrite(row["raw"], lc), (row["id"], lc))


if __name__ == "__main__":
    unittest.main()
