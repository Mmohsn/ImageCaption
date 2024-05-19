from flask import Flask, render_template, request, jsonify

from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from PIL import Image
from base64 import b64encode
from io import BytesIO
import time

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
# Initialize the MarianMT model and tokenizer for translation
translation_model_name = 'Helsinki-NLP/opus-mt-en-ar'
translation_model = MarianMTModel.from_pretrained(translation_model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)


def generate_caption(image):
    with Image.open(image) as img:
        raw_image = img.convert("RGB")

        inputs = processor(raw_image, return_tensors="pt", max_new_tokens=100)

        start_time = time.time()
        out = model.generate(**inputs)
        generation_time = time.time() - start_time

        caption = processor.decode(out[0], skip_special_tokens=True)

        # Translate the caption into Arabic
        translated_inputs = translation_tokenizer(caption, return_tensors="pt")
        translated_out = translation_model.generate(**translated_inputs)
        translated_caption = translation_tokenizer.decode(translated_out[0], skip_special_tokens=True)

        return translated_caption

def convert_image_to_base64(image):
    pil_image = Image.open(image).convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_data = b64encode(buffered.getvalue()).decode('utf-8')

    return img_data

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']

        try:
            image_base64 = convert_image_to_base64(image)
        except Exception as e:
            return render_template('index.html', generation_message="Error processing image")
        
        try:
            caption, generation_time = generate_caption(image)
        except Exception as e:
            return render_template('index.html', generation_message="Error generating Captcha")

        generation_message = f"generated in {generation_time:.2f} seconds" if generation_time is not None else "generated in -.-- seconds"

        return render_template('index.html', image=image_base64, caption=caption, generation_message=generation_message)
    
    return render_template('index.html')

@app.route('/api/generate_caption', methods=['POST'])
def generate_caption_api():
    if 'image' in request.files:
        image = request.files['image']
        caption = generate_caption(image)
        image_name = image.filename

        response = {
            'image_name': image_name,
            'description': caption
        }

        return jsonify(response)
    else:
        return jsonify({'error': 'No image uploaded'})
