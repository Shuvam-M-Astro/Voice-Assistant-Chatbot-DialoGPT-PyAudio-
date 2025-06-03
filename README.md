# 🎙️ Voice Assistant Chatbot (DialoGPT + PyAudio)

An interactive **voice-controlled chatbot** powered by Microsoft's DialoGPT and Hugging Face Transformers. Speak into your microphone, get smart replies, and hear the bot respond back in real time!

---

## 🚀 Features

- 🎤 Converts your voice to text using Google Speech Recognition
- 💬 Generates human-like replies using `DialoGPT-medium`
- 🔊 Reads replies aloud with `pyttsx3` (offline TTS)
- 🔁 Maintains conversation history for multi-turn dialogue
- ✅ Clean modular design (chat logic separated from voice I/O)

---

## 🧠 Model Info

This bot uses [`microsoft/DialoGPT-medium`](https://huggingface.co/microsoft/DialoGPT-medium), a fine-tuned version of GPT-2 optimized for conversational AI.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/voice-assistant-chatbot.git
cd voice-assistant-chatbot
```

# (Optional) Create a virtual environment
```bash
conda create -n chatbot python=3.10
conda activate chatbot
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# If PyAudio fails on Windows:
Reinstall pyaudio

```bash
pip install pyaudio
```
## ⚙️ Configuration
If you have multiple microphones, choose your preferred device:

```bash
with sr.Microphone(device_index=5) as source:
```
You can list available mics like this:

```bash
print(sr.Microphone.list_microphone_names())
```

# # 🙏 Acknowledgments
🤗 Hugging Face Transformers
🎤 PyAudio & SpeechRecognition
🔊 pyttsx3 (offline text-to-speech)








