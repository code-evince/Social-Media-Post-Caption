import torch
from PIL import Image
from torchvision import transforms
import json

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your image transformations (same as used during training)
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def generate_caption_from_model(image_path, model, word_to_idx, idx_to_word, max_length=50):
    # Print vocabulary information
    print("Sample word_to_idx entries:", list(word_to_idx.items())[:10])
    print("Sample idx_to_word entries:", list(idx_to_word.items())[:10])
    print("Vocabulary size:", len(word_to_idx))
    print("Model vocabulary size:", model.decoder.fc.out_features)

    model.eval()
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image_transforms(image)
    image = image.unsqueeze(0).to(device)  # Add a batch dimension and send to the device

    # Encode the image
    with torch.no_grad():
        features = model.encoder(image)

    # Initialize the caption generation with <start> token
    start_idx = word_to_idx.get('<start>', 0)
    end_idx = word_to_idx.get('<end>', -1)
    caption = [start_idx]
    states = None  # For storing LSTM states

    for _ in range(max_length - 2):  # Exclude <start> and <end> tokens from the length
        with torch.no_grad():
            inputs = torch.tensor([caption[-1]], dtype=torch.long, device=device).unsqueeze(0)  # Last word only
            embeddings = model.decoder.embed(inputs)  # Get embeddings

            # Concatenate the features and embeddings along feature dimension for every step
            lstm_input = torch.cat((features.repeat(1, 1, 1), embeddings), dim=2)

            if states is None:
                hiddens, states = model.decoder.lstm(lstm_input)  # First step, no states yet
            else:
                hiddens, states = model.decoder.lstm(lstm_input, states)  # Use the states for subsequent steps

            outputs = model.decoder.fc(hiddens.squeeze(1))
            
            # Debug print statements
            print("Raw model output:", outputs)
            
            next_word_idx = outputs.argmax(dim=1).item()
            print("Next word index:", next_word_idx)
            
            caption.append(next_word_idx)

            if next_word_idx == end_idx:
                break
    
    # Debugging print statements
    print("Caption indices from model:", caption)
    
    # Check model output range
    max_idx = max(caption)
    print(f"Max index in model output: {max_idx}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # Handle missing keys gracefully
    caption_text = []
    for idx in caption[1:-1]:  # Skip <start> and <end>
        if str(idx) in idx_to_word:
            word = idx_to_word[str(idx)]
            caption_text.append(word)
        else:
            print(f"Warning: Index {idx} not found in idx_to_word.")
            caption_text.append('<unk>')  # Placeholder for unknown words
    
    return ' '.join(caption_text)