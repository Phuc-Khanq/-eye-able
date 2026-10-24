"""One lock for loading Hugging Face models, so two models never load at the same moment.

Route: Khang (found in the live run on Oct 24)
Used by: engine/eyes/screen_target.py (UI-Venus) and engine/voice/rewrite.py (the clean-up model)

UI-Venus and the dictation clean-up model both start loading when the engine starts, each on its own thread.
`import transformers` is lazy: while one thread is half way through resolving a class such as
AutoModelForImageTextToText, another thread can see it missing. UI-Venus then failed with "Couldn't load
UI-Venus: no model class found", with no other error, while the clean-up model loaded fine. Holding this lock
for the whole load makes the loads run one after the other (it also keeps both from reading the disk and filling
the GPU at once). A re-entrant lock, so a loader that calls another loader on the same thread can't block itself.
"""
import threading

LOAD_LOCK = threading.RLock()
