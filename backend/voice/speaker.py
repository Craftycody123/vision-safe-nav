import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 170)

last_message = None

def speak(message):
    global last_message

    if message != last_message:
        engine.stop()   
        engine.say(message)
        engine.runAndWait()
        last_message = message