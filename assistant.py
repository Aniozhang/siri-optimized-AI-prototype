from speech_recognition import transcribe_audio
from vision_processing import recognize_object

def assistant_response(speech_text, detected_object):
    if "find the apple" in speech_text and "apple" in detected_object:
        return "I see an apple on the table."
    elif "find the laptop" in speech_text and "laptop" in detected_object:
        return "The laptop is in front of you."
    else:
        return "I couldn't find what you're looking for."

if __name__ == "__main__":
    speech_text = transcribe_audio("./data/input_audio.wav")
    detected_object = recognize_object("./data/scene.jpg", ["a red apple", "a laptop", "a book"])
    response = assistant_response(speech_text, detected_object)
    print("Assistant:", response)