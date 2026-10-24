"""Forgiving command matching, plus "click / open / select <name>" for UI-Venus.

Route: Khoi and Khoa (Khoi leads)
Publishes: commands, TargetRequest labels
Subscribes: recognized text
APIs: rapidfuzz process.extractOne, Python re
"""

COMMANDS = [
    "send it", "new line", "delete that", "scroll down", "scroll up",
    "start dictation", "stop listening", "pause", "menu", "help",
]


def match(text: str):
    raise NotImplementedError
