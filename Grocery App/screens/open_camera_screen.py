from kivymd.uix.screen import MDScreen
import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.button import Button
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
from PIL import Image as PILImage
import os
import requests
from kivymd.uix.boxlayout import BoxLayout
import pytesseract
from screens.helper.process_ocr import OCRProcessor
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
import json

class BoundingBoxMobileNet(nn.Module):
    def __init__(self, num_classes=2):
        super(BoundingBoxMobileNet, self).__init__()
        self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.fc1 = nn.Linear(self.backbone.features[-1][0].out_channels, 512)
        self.fc2 = nn.Linear(512, num_classes * 4)

    def forward(self, x):
        x = self.backbone.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 2, 4)

class OpenCameraScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "open_camera_screen"
        
        print("Initializing OpenCameraScreen")

        # Create the main layout to hold the image and button
        self.main_layout = BoxLayout(orientation='vertical')

        self.capture = cv2.VideoCapture(0)
        
        # Image widget for displaying the camera feed
        self.image_widget = Image(size_hint=(1, 1))
        self.main_layout.add_widget(self.image_widget)

        # Create an AnchorLayout to position the button at the bottom
        self.button_layout = AnchorLayout(anchor_x='center', anchor_y='bottom', size_hint_y = 0.2, height=60)
        self.btn = Button(
            text="Capture Receipt",
            size_hint=(1, 1),
            on_release=self.capture_receipt# Adjust size_hint to change button size
        )
        self.button_layout.add_widget(self.btn)

        # Add the button layout to the main layout
        self.main_layout.add_widget(self.button_layout)

        # Add the main layout to the screen
        self.add_widget(self.main_layout)

        # Schedule the update function for live camera feed
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        # Load the model
        self.model = BoundingBoxMobileNet(num_classes=2)
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        
        model_directory = os.path.dirname(os.path.abspath(__file__))  # Current directory
        model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]

        if not model_files:
            raise FileNotFoundError("No .pth model file found in the current directory.")
        
        model_path = os.path.join(model_directory, model_files[0])
        
        self.quantized_model.load_state_dict(torch.load(model_path, weights_only=True)['model_state_dict'])
        self.quantized_model.eval()

        # Transformation for input images
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            receipt_box = self.detect_receipt(frame)
            if receipt_box is not None:
                cv2.polylines(frame, [np.int32(receipt_box)], isClosed=True, color=(0, 255, 0), thickness=2)

            # Convert the frame to a format Kivy can use
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture
    
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect
    
    def calculate_overlap(self, box1, box2):
        
        box2 = self.convert_to_full_corners(box2)
        
        """Calculate the area of overlap between two bounding boxes."""
        # Convert boxes to a format suitable for overlap calculation
        box1 = box1.astype(int)
        box2 = box2.astype(int)

        # Get bounding rectangles for both boxes
        x1 = max(box1[:, 0].min(), box2[:, 0].min())
        y1 = max(box1[:, 1].min(), box2[:, 1].min())
        x2 = min(box1[:, 0].max(), box2[:, 0].max())
        y2 = min(box1[:, 1].max(), box2[:, 1].max())

        # Calculate the area of overlap
        if x2 <= x1 or y2 <= y1:  # No overlap
            return 0
        return (x2 - x1) * (y2 - y1)
    
    def convert_to_full_corners(self, box):
        # Extract the top left and bottom right coordinates
        x1, y1, x2, y2 = box
        
        # Calculate the top right and bottom left corners
        top_left = [x1, y1]        # (x1, y1)
        top_right = [x2, y1]       # (x2, y1)
        bottom_right = [x2, y2]    # (x2, y2)
        bottom_left = [x1, y2]     # (x1, y2)

        # Create a numpy array with the four corners
        corners = np.array([top_left, top_right, bottom_right, bottom_left])

        return corners
    
    def detect_receipt(self, frame):
        original_height, original_width = frame.shape[:2]
        input_tensor = self.preprocess_frame(frame)

        with torch.no_grad():
            predictions = self.quantized_model(input_tensor)

        receipt_box_model, _ = self.extract_bounding_boxes(predictions)
        receipt_box_model = self.rescale_bounding_box(receipt_box_model, original_width, original_height)

        self.model_receipt_box = receipt_box_model
        
        # Optional: Perform contour detection for extra processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 65, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=5)

        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges = cv2.cvtColor(cv2.addWeighted(edges_colored, 0.8, line_image, 1, 0), cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes = []

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                receipt_box = np.array([point[0] for point in approx], dtype="float32")
                receipt_box = self.order_points(receipt_box)
                detected_boxes.append(receipt_box)

        return self.get_best_receipt_box(detected_boxes)

    def preprocess_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_frame)
        input_tensor = self.transform(pil_image).unsqueeze(0)
        return input_tensor

    def rescale_bounding_box(self, box, original_width, original_height):
        input_size = 512
        scale_x = original_width / input_size
        scale_y = original_height / input_size
        box[0] = int(box[0] * scale_x)
        box[1] = int(box[1] * scale_y)
        box[2] = int(box[2] * scale_x)
        box[3] = int(box[3] * scale_y)
        return box

    def extract_bounding_boxes(self, predictions):
        predictions = predictions.squeeze().numpy()
        receipt_box = predictions[0].astype(int)
        text_box = predictions[1].astype(int)
        return receipt_box, text_box

    def get_best_receipt_box(self, detected_boxes):
        if not detected_boxes:
            return None
        max_overlap, best_box = 0, None
        for box in detected_boxes:
            overlap = self.calculate_overlap(box, self.model_receipt_box)
            if overlap > max_overlap:
                max_overlap, best_box = overlap, box
        return best_box

    def capture_receipt(self, instance):
        # Capture a frame from the camera
        # Stop the camera stream
        
        # Hide the Capture Receipt button
        self.main_layout.remove_widget(self.button_layout)
        
        ret, frame = self.capture.read()
        
        self.capture.release()
        
        if ret:
            # Encode the image frame as JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            if img_encoded is not None:
                # Convert the encoded image to bytes
                img_bytes = img_encoded.tobytes()

                # Send the image to the server
                url = 'http://192.168.2.117:5555/upload'
                files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}

                # Send the request
                response = requests.post(url, files=files)

                # Handle the response (it could be an image or text)
                if response.status_code == 200:
                    # Define the directory where you want to save the image
                    # This will save the image in the 'processed_images' folder in the same directory as the script
                    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
                    save_dir = os.path.join(script_dir, 'processed_images')  # Create a 'processed_images' folder path
                    
                    # Create the directory if it doesn't exist
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    # Define the path to save the image
                    save_path = os.path.join(save_dir, 'processed_image.jpg')

                    # Save the response content as an image
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Processed image saved as '{save_path}'")

                    # Display the processed image on the screen
                    self.image_widget.source = save_path
                    self.image_widget.reload()

                    # Add Yes/No buttons at the bottom
                    self.show_buttons()
                else:
                    print(f"Error: {response.text}")
            else:
                print("Error encoding image")
        else:
            print("Failed to capture frame from camera")
        
    def show_buttons(self):
        # Create a new layout with Yes/No buttons
        button_layout = BoxLayout(size_hint_y=None, height=50, spacing=20)

        # Yes button (for proceeding with the image)
        yes_button = Button(text="Yes", on_release=self.on_yes)
        
        # No button (for re-capturing the image)
        no_button = Button(text="No", on_release=self.on_no)

        # Add the buttons to the layout
        button_layout.add_widget(yes_button)
        button_layout.add_widget(no_button)

        # Add the button layout to the screen
        self.main_layout.add_widget(button_layout)

    def on_yes(self, instance):
        # Implement your "Yes" logic here (e.g., proceed with saving or further processing the image)
        # Send a GET request to the server to process OCR
        url = 'http://192.168.2.117:5555/process_ocr'
        response = requests.get(url)

        if response.status_code == 200:
            ocr_result = response.json().get('ocr_text', 'No OCR result returned')
            print("OCR Result:", ocr_result)
            # Display or further handle the OCR result as needed
            
            json_output = json.loads(OCRProcessor.process_ocr_text(ocr_result))

            # Ensure the `ingredients` directory exists
            current_dir = os.path.dirname(__file__)

            # Create the full path for the 'ingredients' folder
            ingredients_path = os.path.join(current_dir, 'ingredients')

            # Create the 'ingredients' folder if it doesn't exist
            if not os.path.exists(ingredients_path):
                os.makedirs(ingredients_path)

            self.manager.current = 'edit_item_screen'  # Switch to EditItemScreen
            edit_item_screen = self.manager.get_screen('edit_item_screen')
            edit_item_screen.load_data(json_output)
        else:
            print(f"Error performing OCR: {response.text}")

        print("User is happy with the image.")
        pass

    def on_no(self, instance):
        # User is not happy with the image, restart the camera feed
        print("User is not happy with the image. Restarting the camera.")
        
        # Start the camera feed again
        self.capture = cv2.VideoCapture(0)

        # Remove the buttons and display the live feed again
        self.main_layout.remove_widget(instance.parent)
        self.main_layout.add_widget(self.button_layout)

        # Schedule the camera update function
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        # Clear the image widget
        self.image_widget.source = ''
        self.image_widget.reload()
