import speech_recognition as sr
import pyttsx3
from chatbot import setup_chatbot, get_response  # ← Import from chatbot.py
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init()

# TTS engine setup
tts_engine = pyttsx3.init()
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# STT engine setup
recognizer = sr.Recognizer()

def select_microphone():
    """Let user select microphone from available devices"""
    print(f"\n{Fore.CYAN}Available microphone devices:{Style.RESET_ALL}")
    mic_names = sr.Microphone.list_microphone_names()
    for index, name in enumerate(mic_names):
        print(f"{Fore.GREEN}[{index}]{Style.RESET_ALL} {name}")
    
    while True:
        try:
            selection = input(f"\n{Fore.YELLOW}Enter the number of your preferred microphone (or press Enter for default): {Style.RESET_ALL}")
            if not selection:
                return None  # Use default microphone
            
            device_index = int(selection)
            if 0 <= device_index < len(mic_names):
                return device_index
            else:
                print(f"{Fore.RED}Invalid selection. Please choose a number between 0 and {len(mic_names)-1}{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Please enter a valid number{Style.RESET_ALL}")

def listen(device_index=None):
    """Listen for voice input using selected microphone"""
    try:
        with sr.Microphone(device_index=device_index) as source:
            print(f"\n{Fore.CYAN}🎙️ Listening...{Style.RESET_ALL}")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            try:
                text = recognizer.recognize_google(audio)
                print(f"{Fore.GREEN}You said: {text}{Style.RESET_ALL}")
                return text
            except sr.UnknownValueError:
                print(f"{Fore.YELLOW}Sorry, I didn't catch that.{Style.RESET_ALL}")
            except sr.RequestError:
                print(f"{Fore.RED}Speech recognition service failed.{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error with microphone: {str(e)}{Style.RESET_ALL}")
    return None

def main():
    # Let user select microphone
    device_index = select_microphone()
    
    # Setup chatbot
    print(f"\n{Fore.CYAN}Setting up the chatbot...{Style.RESET_ALL}")
    model, tokenizer, device = setup_chatbot()
    chat_history_ids = None
    
    print(f"\n{Fore.GREEN}Voice assistant ready! {Style.RESET_ALL}")
    speak("Voice assistant ready. Say something!")

    while True:
        user_input = listen(device_index)
        if user_input is None:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye']:
            speak("Goodbye!")
            print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
            break

        try:
            response, chat_history_ids = get_response(
                user_input, 
                model, 
                tokenizer, 
                chat_history_ids,
                device
            )
            print(f"{Fore.CYAN}Bot: {response}{Style.RESET_ALL}")
            speak(response)
        except Exception as e:
            error_msg = "I encountered an error. Let's start a new conversation."
            print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")
            speak(error_msg)
            chat_history_ids = None

if __name__ == "__main__":
    main()
