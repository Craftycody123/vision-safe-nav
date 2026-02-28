import pyttsx3
import threading
import time

last_message = None
last_spoken_time = 0
COOLDOWN_SECONDS = 3
is_speaking = False

def speak(message):
    global last_message, last_spoken_time, is_speaking

    now = time.time()
    time_since_last = now - last_spoken_time

    # Speak if: new message OR same message but cooldown has passed
    if message != last_message or time_since_last >= COOLDOWN_SECONDS:
        if not is_speaking:
            last_message = message
            last_spoken_time = now
            thread = threading.Thread(target=_speak_thread, args=(message,), daemon=True)
            thread.start()

def _speak_thread(message):
    global is_speaking
    is_speaking = True
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[Voice Error] {e}")
    finally:
        is_speaking = False