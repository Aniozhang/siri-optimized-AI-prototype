# Siri-Optimized AI: Vision-Language Processing for Smart Assistants

## Overview
This project builds a Siri-like assistant that processes **both voice commands and visual input** to generate intelligent responses. It integrates **speech recognition (Whisper), vision-language processing (CLIP), and contrastive learning** for improved contextual understanding.

## Features
- **Speech-to-Text:** Converts voice input into text using OpenAI Whisper.
- **Vision Recognition:** Uses CLIP to identify objects in an image.
- **Multimodal AI:** Combines speech and vision to provide intelligent responses.
- **On-Device ML:** Optimized for Core ML deployment.

## Setup
```sh
pip install -r requirements.txt
```

## Running the Assistant
```sh
python assistant.py
```

---

# requirements.txt

torch
openai-whisper
clip-by-openai
numpy
pillow
coremltools

---
