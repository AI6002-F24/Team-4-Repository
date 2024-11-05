import cv2
import numpy as np
import pytesseract
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage

class CameraScreen(MDBoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.image_widget = Image()
        self.add_widget(self.image_widget)
        
        self.edge_image_widget = Image()
        self.add_widget(self.edge_image_widget)
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        self.btn = MDRaisedButton(text="Capture Receipt", on_release=self.capture_receipt)
        self.add_widget(self.btn)

        class BoundingBoxMobileNet(nn.Module):
            def __init__(self, num_classes=2):
                super(BoundingBoxMobileNet, self).__init__()
                self.backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
                # Remove the classifier of MobileNetV3 as we'll define our own
                self.backbone.classifier = nn.Identity()

                # Add additional fully connected layers
                self.fc1 = nn.Linear(self.backbone.features[-1][0].out_channels, 512)
                self.fc2 = nn.Linear(512, num_classes * 4)

            def forward(self, x):
                x = self.backbone.features(x)  # Pass input through the MobileNet backbone
                x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling to reduce to (batch_size, channels, 1, 1)
                x = torch.flatten(x, 1)  # Flatten the output to (batch_size, channels)
                x = F.relu(self.fc1(x))  # Pass through first FC layer
                x = self.fc2(x)  # Output layer (bounding box coordinates)
                return x.view(-1, 2, 4)  # Reshape to (batch_size, 2, 4) for two bounding boxes (with 4 coordinates each)

        # Load the quantized MobileNetV3 model
        model = BoundingBoxMobileNet(num_classes=2)  # Assuming MobileNetV3-Large
        self.quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.quantized_model.load_state_dict(torch.load(r'C:\Users\klmno\OneDrive\Desktop\Proj\Bounding Box\quantized_model.pth')['model_state_dict'])
        self.quantized_model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),  # Resize to 512x512
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            receipt_box = self.detect_receipt(frame)
            if receipt_box is not None:
                # Draw bounding box on frame
                cv2.polylines(frame, [np.int32(receipt_box)], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                print("No receipt detected.")

            # Convert image to Kivy Texture and display it
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image_widget.texture = texture

        # Process and display the Canny edge detection result
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 65, 150)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=5)

        # Create an empty image to draw the lines
        line_image = np.zeros_like(frame)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine edges and line image
        edges = cv2.cvtColor(cv2.addWeighted(edges_colored, 0.8, line_image, 1, 0), cv2.COLOR_BGR2GRAY)
        
        print(frame.shape, line_image.shape, edges.shape)
        
        # Display edges in the edge image widget
        edge_buf = cv2.flip(edges, 0).tobytes()
        edge_texture = Texture.create(size=(edges.shape[1], edges.shape[0]), colorfmt='luminance')
        edge_texture.blit_buffer(edge_buf, colorfmt='luminance', bufferfmt='ubyte')
        self.edge_image_widget.texture = edge_texture
        
    def detect_receipt(self, frame):
        
        original_height, original_width = frame.shape[:2]
        
        # Preprocess the frame for the quantized model
        input_tensor = self.preprocess_frame(frame)

        # Predict bounding boxes using the quantized model
        with torch.no_grad():
            predictions = self.quantized_model(input_tensor)
        
        # Parse predictions (assuming the model predicts 8 coordinates, 4 for the receipt and 4 for the text area)
        # Parse predictions (2 bounding boxes, each with 4 coordinates)
        receipt_box_model, text_box = self.extract_bounding_boxes(predictions)

        # Rescale bounding boxes to match the original camera frame size
        receipt_box_model = self.rescale_bounding_box(receipt_box_model, original_width, original_height)

        self.model_receipt_box = receipt_box_model
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blur, 65, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=5)

        # Create an empty image to draw the lines
        line_image = np.zeros_like(frame)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine edges and line image
        edges = cv2.cvtColor(cv2.addWeighted(edges_colored, 0.8, line_image, 1, 0), cv2.COLOR_BGR2GRAY)
        
        # Find contours and filter by rectangularity and area
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_boxes = []

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # Filter for 4-point contours
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                # Order points for perspective correction
                receipt_box = np.array([point[0] for point in approx], dtype="float32")
                receipt_box = self.order_points(receipt_box)
                
                # Check aspect ratio
                (tl, tr, br, bl) = receipt_box
                width_a = np.linalg.norm(br - bl)
                width_b = np.linalg.norm(tr - tl)
                height_a = np.linalg.norm(tr - br)
                height_b = np.linalg.norm(tl - bl)

                # aspect_ratio = max(width_a, width_b) / max(height_a, height_b)
                # if not (1.5 < aspect_ratio < 4):  # Typical receipt aspect ratio range
                #     continue

                # Check edge density within bounding box for receipt texture
                mask = np.zeros(gray.shape, dtype="uint8")
                cv2.drawContours(mask, [np.int0(receipt_box)], -1, 255, -1)
                masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
                edge_density = cv2.countNonZero(masked_edges) / cv2.contourArea(approx)
                
                if edge_density < 0.05:  # Edge density threshold for text regions
                    continue

                detected_boxes.append(receipt_box)

        # Find the best box by comparing with the model's predicted receipt box
        best_box = self.get_best_receipt_box(detected_boxes)

        return best_box

    def preprocess_frame(self, frame):
        """Preprocesses the frame for the quantized model."""
        # Convert the image from BGR to RGB (since OpenCV loads images in BGR format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the image to a PIL image (required for torchvision transforms)
        pil_image = PILImage.fromarray(rgb_frame)

        # Apply the transformations (resize, normalize, convert to tensor)
        input_tensor = self.transform(pil_image)

        # Add batch dimension (1, C, H, W)
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor
    
    def rescale_bounding_box(self, box, original_width, original_height):
        """Rescales the bounding box from the 512x512 model input to the original frame size."""
        input_size = 512  # The input size for the model is 512x512
        scale_x = original_width / input_size
        scale_y = original_height / input_size

        # Rescale the coordinates
        box[0] = int(box[0] * scale_x)  # x1
        box[1] = int(box[1] * scale_y)  # y1
        box[2] = int(box[2] * scale_x)  # x2
        box[3] = int(box[3] * scale_y)  # y2

        return box

    def extract_bounding_boxes(self, predictions):
        """Extracts the bounding boxes from the model's predictions."""
        # Assuming predictions output 2 bounding boxes, each with [x1, y1, x2, y2]
        predictions = predictions.squeeze().numpy()  # Shape: (2, 4)
        
        receipt_box = predictions[0].astype(int)  # First bounding box for receipt
        text_box = predictions[1].astype(int)     # Second bounding box for text area

        return receipt_box, text_box
    
    def get_best_receipt_box(self, detected_boxes):
        if not detected_boxes:
            return None

        max_overlap = 0
        best_box = None

        # Compare each detected box with the model's predicted box
        for box in detected_boxes:
            overlap = self.calculate_overlap(box, self.model_receipt_box)
            if overlap > max_overlap:
                max_overlap = overlap
                best_box = box

        return best_box

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

    def calculate_overlap(self, box1, box2):
        
        box2 = self.convert_to_full_corners(box2)
        
        """Calculate the area of overlap between two bounding boxes."""
        # Convert boxes to a format suitable for overlap calculation
        box1 = box1.astype(int)
        box2 = box2.astype(int)

        print(box1, box2)
        
        # Get bounding rectangles for both boxes
        x1 = max(box1[:, 0].min(), box2[:, 0].min())
        y1 = max(box1[:, 1].min(), box2[:, 1].min())
        x2 = min(box1[:, 0].max(), box2[:, 0].max())
        y2 = min(box1[:, 1].max(), box2[:, 1].max())

        # Calculate the area of overlap
        if x2 <= x1 or y2 <= y1:  # No overlap
            return 0
        return (x2 - x1) * (y2 - y1)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def capture_receipt(self, instance):
        # Capture the receipt region and apply OCR if a receipt is detected
        ret, frame = self.capture.read()
        if ret:
            receipt_box = self.detect_receipt(frame)
            if receipt_box is not None:
                # Unwarp and crop the detected receipt
                transformed = self.unwarp_receipt(frame, receipt_box)
                text = pytesseract.image_to_string(transformed)
                print("Detected Text:", text)

    def unwarp_receipt(self, frame, receipt_box):
        (tl, tr, br, bl) = receipt_box
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")
        
        matrix = cv2.getPerspectiveTransform(receipt_box, dst)
        warped = cv2.warpPerspective(frame, matrix, (width, height))
        return warped

class ReceiptApp(MDApp):  # Inherit from MDApp instead of App
    def build(self):
        self.title = 'Receipt Detection App'
        return CameraScreen()

ReceiptApp().run()
