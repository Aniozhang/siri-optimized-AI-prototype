import coremltools as ct
import torch
import clip

def convert_model_to_coreml():
    model, _ = clip.load("ViT-B/32")
    mlmodel = ct.convert(model, inputs=[ct.ImageType()])
    mlmodel.save("Siri_Assistant.mlmodel")
    print("Model saved for Core ML.")

if __name__ == "__main__":
    convert_model_to_coreml()
