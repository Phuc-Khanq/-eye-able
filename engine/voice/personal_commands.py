"""Matches a user's own sound against their 3 to 5 recordings.

Route: Khoi and Khoa (Khoa leads, Next tier)
Publishes: commands
Subscribes: utterances from audio_in
APIs: librosa feature.mfcc, sequence.dtw
"""

def match(audio):
    raise NotImplementedError
