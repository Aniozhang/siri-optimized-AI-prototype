import torch
import clip
from PIL import Image

def recognize_object(image_path, text_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_inputs = clip.tokenize(text_labels).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)
    
    similarities = (image_features @ text_features.T).softmax(dim=-1)
    return text_labels[similarities.argmax()]

if __name__ == "__main__":
    detected = recognize_object("./data/scene.jpg", ["a red apple", "a laptop", "a book"])
    print("Detected object:", detected)
