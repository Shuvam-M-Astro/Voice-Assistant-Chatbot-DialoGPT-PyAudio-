import speech_recognition as sr
import pyttsx3
from chatbot import setup_chatbot, get_response  # ← Import from chatbot.py

# TTS engine setup
tts_engine = pyttsx3.init()
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# STT engine setup
recognizer = sr.Recognizer()
def listen():
    with sr.Microphone(device_index=5) as source:
        print("🎙️ Listening from Blue Snowball mic...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
        except sr.RequestError:
            speak("Speech recognition service failed.")
        return None

def main():
    # List available microphones
    print("Available microphone devices:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"[{index}] {name}")
    
    model, tokenizer = setup_chatbot()
    chat_history_ids = None
    speak("Voice assistant ready. Say something!")

    while True:
        user_input = listen()
        if user_input is None:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            speak("Goodbye!")
            break

        try:
            response, chat_history_ids = get_response(user_input, model, tokenizer, chat_history_ids)
            print(f"Bot: {response}")
            speak(response)
        except Exception as e:
            speak("Error. Restarting conversation.")
            chat_history_ids = None

if __name__ == "__main__":
    main()
