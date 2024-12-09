from flask import Flask, request, send_file, jsonify
import os
import io
import torch
import torchvision.transforms as transforms
from torchvision.ops import nms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn
import cv2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import pytesseract
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image as PILImage

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

NUM_CLASSES = 2

# Load the pre-trained model
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Get the input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

# Move model to GPU if available

model.to(device)

model.load_state_dict(torch.load(r'C:\Users\klmno\OneDrive\Desktop\Proj\Code\sdd\Working\trained_models\FasterRCNN_ResNet50_FPN_model.pth', weights_only=True))

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed").to(device)
model_ocr.eval()

original_image = None
bounding_boxes = None

def predict(model, image, device, score_threshold=0.5, iou_threshold=0.2):
    model.eval()
    original_height, original_width = image.shape[:2]
    
    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),  # Resize to 800x800 or any appropriate size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform and resize the image for the model's input
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Extract the boxes, labels, and scores from the model's output
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    
    # Filter by score threshold
    keep_indices = scores > score_threshold
    boxes = boxes[keep_indices]
    labels = labels[keep_indices]
    scores = scores[keep_indices]

    # Apply Non-Maximum Suppression
    nms_indices = nms(boxes, scores, iou_threshold)
    
    # Select boxes, labels, and scores after NMS
    boxes = boxes[nms_indices].cpu().numpy()
    labels = labels[nms_indices].cpu().numpy()
    scores = scores[nms_indices].cpu().numpy()
    
    # Rescale boxes to match the original image size
    resized_height, resized_width = 800, 800  # Assuming the model input is resized to (800, 800)
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    rescaled_boxes = []
    for box in boxes:
        x_min = int(box[0] * scale_x)
        y_min = int(box[1] * scale_y)
        x_max = int(box[2] * scale_x)
        y_max = int(box[3] * scale_y)
        rescaled_boxes.append([x_min, y_min, x_max, y_max])
    
    return rescaled_boxes, labels, scores

def draw_boxes_on_image(image, boxes, labels, scores, threshold=0.2):

        for i in range(len(boxes)):
            if scores[i] > threshold:  # Only draw boxes above the threshold
                x1, y1, x2, y2 = boxes[i]  # Extract box coordinates
                color = (0, 255, 0)  # Green color for bounding box
                thickness = 2
                # Draw the rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                # Optionally, add label and score

        return image
    
def combine_text_on_same_line(boxes, texts):
    # Combine text outputs that are on the same line
    line_groups = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box
        added_to_existing_line = False

        for group in line_groups:
            # Check if the current box is on the same line as any existing group
            if abs(group['y'] - y_min) < 15:  # Adjust threshold as needed
                group['text'] += ' ' + texts[i]
                group['y'] = min(group['y'], y_min)  # Use the topmost `y` as the line reference
                added_to_existing_line = True
                break

        if not added_to_existing_line:
            # Create a new line group if no match was found
            line_groups.append({'y': y_min, 'text': texts[i]})

    # Sort the lines by their `y` coordinate to ensure top-to-bottom order
    sorted_lines = sorted(line_groups, key=lambda x: x['y'])

    # Create a single string by joining all lines with newlines
    combined_text = '\n'.join([line['text'] for line in sorted_lines])

    return combined_text

# Function to format the recipe with tokens
def format_recipe_with_tokens(ingredients, cooking_time):
    # Define the separator for the ingredients
    ingredient_separator = " <sep> "

    # Convert all ingredients to lowercase and join them with <sep> token
    formatted_ingredients = ingredient_separator.join([ingredient.lower() for ingredient in ingredients])
    
    # Format the recipe with start and end tokens
    return (
        f"<start-time> {cooking_time} minutes <end-time> "
        f"<start-ingredients> {formatted_ingredients} <end-ingredients> "
    )
    
import re

def format_steps(text):
    # Extract the first occurrence of steps between <start-steps> and <end-steps>
    steps_match = re.search(r'<start-steps>(.*?)<end-steps>', text, re.DOTALL)
    if not steps_match:
        return "No steps generated"

    # Clean and split the steps into a list
    steps_raw = steps_match.group(1).strip()
    steps_list = steps_raw.split("', '")

    # Clean up leading and trailing brackets from the first and last steps
    steps_list[0] = steps_list[0].lstrip("['")  # Remove leading brackets from the first step
    steps_list[-1] = steps_list[-1].rstrip("']")  # Remove trailing brackets from the last step

    # Remove any ',' within the steps themselves
    steps_list = [step.replace("'", "").strip() for step in steps_list]

    # Capitalize the first letter of each step and format them with numbering
    formatted_steps = []
    for i, step in enumerate(steps_list, start=1):
        step_cleaned = step.strip().capitalize()  # Ensure proper capitalization
        formatted_steps.append(f"Step {i}: {step_cleaned}")

    # Join the formatted steps with new lines
    return "\n".join(formatted_steps)

####################################################################################################################################
@app.route('/upload', methods=['POST'])
def upload_file():
    
    global original_image, bounding_boxes
    
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Save the image temporarily
    in_memory_file = file.stream.read()
    np_array = np.frombuffer(in_memory_file, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    print(img.shape)
    
    original_image = img
    
    # Make predictions using the model
    with torch.no_grad():
        boxes, labels, scores = predict(model, img, device)
        image_with_boxes = draw_boxes_on_image(img, boxes, labels, scores, threshold=0.6)

    processed_image_path = 'uploads/processed_image_with_boxes.jpg'
    cv2.imwrite(processed_image_path, image_with_boxes)
    
    original_image_path = 'uploads/original_image.jpg'
    cv2.imwrite(original_image_path, original_image)
    
    bounding_boxes = boxes

    # Optionally: you can return a processed image (e.g., with a bounding box) back to the phone
    # Save the processed image to a byte stream
    _, img_encoded = cv2.imencode('.jpg', image_with_boxes)
    img_byte_arr = io.BytesIO(img_encoded.tobytes())

    # Return the processed image as a response
    return send_file(img_byte_arr, mimetype='image/jpeg')

@app.route('/process_ocr', methods=['GET'])
def process_ocr():
    
    global original_image, bounding_boxes
    
    if original_image is None or bounding_boxes is None:
        return "No processed image or boxes available", 404

    texts = []

    # Iterate over the bounding boxes and extract text using TrOCR
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        cropped_img = original_image[y_min:y_max, x_min:x_max]

        # Convert cropped image to RGB format for PIL
        img_rgb_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        image_pil = PILImage.fromarray(img_rgb_cropped)

        # Prepare image for the TrOCR model
        pixel_values = processor(image_pil, return_tensors='pt').pixel_values.to(device)
        generated_ids = model_ocr.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        texts.append(generated_text)

    combined_text = combine_text_on_same_line(bounding_boxes, texts)
    
    print(combined_text)
    
    return jsonify({
        "ocr_text": combined_text
    })

@app.route('/receive_data', methods=['POST'])
def receive_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    ingredients = data.get("ingredients")
    cooking_time = data.get("cooking_time")

    # Log the received data
    print("Received Ingredients:", ingredients)
    print("Received Cooking Time:", cooking_time)

    formatted_recipe = format_recipe_with_tokens(ingredients, cooking_time)
    
    current_folder = os.path.dirname(os.path.abspath(__file__))

    # Combine it with the relative path to the model
    model_path = os.path.join(current_folder, "trained_gpt2_model")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    
    # Define special tokens, including separate pad and eos tokens
    special_tokens = {
        'eos_token': '<eos>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
        'additional_special_tokens': [
            '<start-time>', '<end-time>', 
            '<start-ingredients>', '<end-ingredients>', 
            '<start-steps>', '<end-steps>', '<sep>'
        ]
    }

    tokenizer.eos_token = "<eos>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"

    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Add the special tokens to the tokenizer vocabulary
    tokenizer.add_special_tokens(special_tokens)
    
    model.resize_token_embeddings(len(tokenizer))

    input_ids = tokenizer(formatted_recipe, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    # Generate steps
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=256, top_k=100, top_p=.87, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    formatted_steps_output = format_steps(generated_text)

    return jsonify({"recipe_directions": formatted_steps_output})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5555)
