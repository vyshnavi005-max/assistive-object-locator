import threading
import queue
import json
import os

# Map EVERY spoken variation → YOLO class name
# Covers common mispronunciations and regional phrases
WORD_TO_CLASS = {
    # Bottle
    "bottle": "bottle",
    "water bottle": "bottle",
    "water": "bottle",
    "bottel": "bottle",

    # Cup / Mug
    "cup": "cup",
    "mug": "cup",
    "glass": "cup",
    "coffee": "cup",
    "tea": "cup",

    # Phone
    "phone": "cell phone",
    "phones": "cell phone",
    "mobile": "cell phone",
    "cell": "cell phone",
    "smartphone": "cell phone",
    "iphone": "cell phone",
    "android": "cell phone",

    # Laptop
    "laptop": "laptop",
    "computer": "laptop",
    "mac": "laptop",
    "macbook": "laptop",

    # Remote
    "remote": "remote",
    "remote control": "remote",
    "controller": "remote",

    # Keyboard
    "keyboard": "keyboard",

    # Mouse
    "mouse": "mouse",

    # Backpack / Bag
    "backpack": "backpack",
    "bag": "backpack",
    "rucksack": "backpack",

    # Book
    "book": "book",
    "notebook": "book",
    "textbook": "book",

    # Bowl
    "bowl": "bowl",

    # Chair
    "chair": "chair",
    "seat": "chair",
    "stool": "chair",

    # TV
    "tv": "tv",
    "television": "tv",
    "telly": "tv",
    "screen": "tv",

    # Clock
    "clock": "clock",
    "watch": "clock",

    # Scissors
    "scissors": "scissors",

    # Umbrella
    "umbrella": "umbrella",

    # Vase
    "vase": "vase",

    # Toothbrush
    "toothbrush": "toothbrush",
    "brush": "toothbrush",

    # Food
    "banana": "banana",
    "apple": "apple",
    "orange": "orange",
    "pizza": "pizza",
    "sandwich": "sandwich",
    "donut": "donut",
    "cake": "cake",

    # Handbag / Purse
    "handbag": "handbag",
    "purse": "handbag",

    # Couch / Sofa
    "couch": "couch",
    "sofa": "couch",

    # Teddy bear
    "teddy": "teddy bear",
    "bear": "teddy bear",
    "teddy bear": "teddy bear",

    # Spoon / Fork / Knife
    "spoon": "spoon",
    "fork": "fork",
    "knife": "knife",
    "cutlery": "knife",

    # Sports ball
    "ball": "sports ball",
    "football": "sports ball",
    "cricket ball": "sports ball",
}

# Trigger words — any of these can start a command
TRIGGER_WORDS = [
    "find", "locate", "search", "get", "where", "show",
    "look for", "pick", "grab", "fetch",
    # Also allow JUST the object name without trigger (fallback)
]

# Filler words to strip before matching
FILLERS = ["the", "my", "a", "an", "me", "please", "for", "is"]


def _clean_and_match(text):
    """
    STRICT sequence matching.
    You MUST say a trigger word (like 'find' or 'search') before the object.
    Example: "find my cellphone" or "search for the bottle"
    """
    text = text.lower().strip()
    
    if not text:
        return None

    # Noise rejection: Reject long rambling sentences
    if len(text.split()) > 6:
        return None

    # --- Strategy 1: Trigger-word pattern ONLY ---
    for trigger in TRIGGER_WORDS:
        if trigger in text:
            # Look only at what was said AFTER the trigger word
            trigger_idx = text.find(trigger)
            after = text[trigger_idx + len(trigger):].strip()
            
            # Strip filler words
            for filler in FILLERS:
                if after.startswith(filler + " "):
                    after = after[len(filler) + 1:].strip()
                    
            # Match longest key first
            for word in sorted(WORD_TO_CLASS.keys(), key=len, reverse=True):
                # The object word must exist in the text AT ALL
                # And it must appear closely AFTER the trigger word
                if word in after:
                    return WORD_TO_CLASS[word]

    # No Strategy 2 fallback. If no trigger is heard, return None.
    return None


class VoiceListener:
    """
    Runs offline speech recognition (vosk) in a background thread.
    Uses both final AND partial results for fast response.
    Parses commands like 'find bottle', 'where is my phone', or just 'bottle'.
    """

    def __init__(self, model_path, device_index=None):
        try:
            from vosk import Model, KaldiRecognizer
            import pyaudio
        except ImportError:
            raise ImportError("vosk and pyaudio are required. Run: pip install vosk pyaudio")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Vosk model not found at '{model_path}'.\n"
                "Download: https://alphacephei.com/vosk/models\n"
                "Use: vosk-model-small-en-us-0.15"
            )

        self._model = Model(model_path)
        self._device_index = device_index
        self._command_queue = queue.Queue()
        self._running = False
        self._thread = None
        self._last_triggered = ""  # Prevent duplicate triggers

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[Voice] Ready! Say: 'find bottle' | 'find phone' | 'find laptop' | or just 'bottle'")

    def stop(self):
        self._running = False

    def get_new_target(self):
        """Non-blocking. Returns a new target if spoken, else None."""
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def _try_dispatch(self, text, source=""):
        """Try to parse text and push target to queue if found and not duplicate."""
        if not text:
            return
        target = _clean_and_match(text)
        if target and target != self._last_triggered:
            self._last_triggered = target
            print(f"[Voice] Heard: '{text}' → TARGET: '{target}'")
            self._command_queue.put(target)

    def _listen_loop(self):
        from vosk import KaldiRecognizer
        import pyaudio

        pa = pyaudio.PyAudio()
        recognizer = KaldiRecognizer(self._model, 16000)
        recognizer.SetWords(True)  # Enable word-level confidence

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=8000,
            input_device_index=self._device_index
        )
        stream.start_stream()
        print("[Voice] Microphone active.")

        while self._running:
            data = stream.read(4000, exception_on_overflow=False)

            if recognizer.AcceptWaveform(data):
                # Final result (end of sentence)
                result = json.loads(recognizer.Result())
                self._try_dispatch(result.get("text", ""), source="final")
            else:
                # Partial result (mid-sentence) — enables faster response
                partial = json.loads(recognizer.PartialResult())
                self._try_dispatch(partial.get("partial", ""), source="partial")

        stream.stop_stream()
        stream.close()
        pa.terminate()
