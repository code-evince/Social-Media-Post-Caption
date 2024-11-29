import json
from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from werkzeug.utils import secure_filename
from model import generate_caption_from_model  # Import the function from model.py
from torchvision import models
from rnn_decoder import RNNDecoder
from encoder_decoder import EncoderDecoder
from groq import Groq  # Import Groq for LLaMA 3 integration

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load word_to_idx and idx_to_word mappings
with open('deep_learning_model/word_to_idx.json', 'r') as f:
    word_to_idx = json.load(f)

with open('deep_learning_model/idx_to_word.json', 'r') as f:
    idx_to_word = json.load(f)

# Model architecture definition
embed_size = 512
hidden_size = 512
num_layers = 2
vocab_size = len(word_to_idx)
encoded_image_size = 512  # Make sure this matches your training

# Define the encoder (ResNet) and decoder (RNN)
cnn_encoder = models.resnet18(pretrained=False)
cnn_encoder.fc = torch.nn.Identity()  # Remove the classification layer
cnn_encoder = cnn_encoder.to(device)

decoder = RNNDecoder(embed_size, hidden_size, vocab_size, num_layers, encoded_image_size).to(device)

# Instantiate the full encoder-decoder model
model = EncoderDecoder(cnn_encoder, decoder).to(device)

# Load the pre-trained weights
model.load_state_dict(torch.load('deep_learning_model/model.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Initialize Groq client for LLaMA 3
groq_client = Groq()

def refine_caption_with_llama(caption):
    # prompt = f"Refine the following caption to make it more engaging and interesting, note just give me the caption in inverted commas and nothing else, also it should 1 liner: {caption}"
    prompt = f"Using this caption generate 10 witty instagram caption in 10 words and one hashtag.: {caption}"
    
    completion = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )
    
    refined_caption = ""
    for chunk in completion:
        refined_caption += chunk.choices[0].delta.content or ""
    
    return refined_caption.strip()

# Routes for uploading image and generating captions
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return render_template('index.html', filename=filename)
    return render_template('index.html', filename=None)

@app.route('/generate-caption', methods=['POST'])
def generate_caption_route():
    filename = request.form.get('filename')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    caption = generate_caption_from_model(image_path, model, word_to_idx, idx_to_word)
    print("original caption : ",caption)
    refined_caption = refine_caption_with_llama(caption)
    print("refined caption : ",refined_caption)
    return render_template('results.html', caption=refined_caption, filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/results')
def results():
    return render_template('results.html', caption="Your caption will appear here.")

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)
