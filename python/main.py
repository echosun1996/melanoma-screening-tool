# Melanoma Analysis Backend
from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
import cv2
import time
import uuid
import logging
import argparse
import os
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('MelanomaAnalysis')

# Create Flask application
app = Flask(__name__)

def calculate_asymmetry(image):
    """Calculate the asymmetry score of a lesion."""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to get a binary mask of the lesion
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Get moments and center of mass
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Create a vertical line through the center
        height, width = image.shape[:2]

        # Create masks for left and right halves
        left_half = np.zeros((height, width), dtype=np.uint8)
        right_half = np.zeros((height, width), dtype=np.uint8)

        # Draw the contour on both halves
        cv2.drawContours(left_half, [contour], 0, 255, -1)
        cv2.drawContours(right_half, [contour], 0, 255, -1)

        # Mask the halves
        left_half_masked = left_half[:, :cx]
        right_half_masked = right_half[:, cx:]

        # Flip the right half for comparison
        right_half_flipped = cv2.flip(right_half_masked, 1)

        # Resize if needed
        min_width = min(left_half_masked.shape[1], right_half_flipped.shape[1])
        left_half_masked = left_half_masked[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]

        # Calculate the difference
        diff = cv2.bitwise_xor(left_half_masked, right_half_flipped)

        # Calculate asymmetry score
        left_area = cv2.countNonZero(left_half_masked)
        right_area = cv2.countNonZero(right_half_flipped)
        diff_area = cv2.countNonZero(diff)

        if max(left_area, right_area) == 0:
            return 0.0

        asymmetry_score = diff_area / max(left_area, right_area)

        return min(asymmetry_score, 1.0)  # Normalize to [0, 1]
    except Exception as e:
        logger.error(f"Error calculating asymmetry: {e}")
        return 0.5  # Return a default value in case of error

def calculate_border_irregularity(image):
    """Calculate the border irregularity score."""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to get binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Calculate the area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area == 0:
            return 0.0

        # Calculate circularity: 4*pi*area/perimeter^2
        # A perfect circle has circularity of 1, lower values indicate irregularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Convert to irregularity score (inverse of circularity)
        irregularity = 1.0 - circularity

        return min(irregularity, 1.0)  # Normalize to [0, 1]
    except Exception as e:
        logger.error(f"Error calculating border irregularity: {e}")
        return 0.5  # Return a default value in case of error

def calculate_color_variation(image):
    """Calculate the color variation score."""
    try:
        # Convert to grayscale for masking
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to get binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply mask to the original image
        masked = cv2.bitwise_and(image, image, mask=thresh)

        # Split into RGB channels
        r, g, b = cv2.split(masked)

        # Calculate standard deviation for each channel
        std_r = np.std(r[thresh > 0])
        std_g = np.std(g[thresh > 0])
        std_b = np.std(b[thresh > 0])

        # Calculate color variation score
        color_variation = (std_r + std_g + std_b) / 255

        return min(color_variation, 1.0)  # Normalize to [0, 1]
    except Exception as e:
        logger.error(f"Error calculating color variation: {e}")
        return 0.5  # Return a default value in case of error

def calculate_dimensions(image):
    """Calculate the dimensions score based on lesion size."""
    try:
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to get binary mask
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.0

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        _, _, width, height = cv2.boundingRect(contour)

        # Calculate maximum diameter
        diameter = max(width, height)

        # Normalize based on the typical size of concerning lesions (>6mm)
        # Assuming the image is normalized and a value of 100+ pixels represents a large mole
        dimensions_score = min(diameter / 100.0, 1.0)

        return dimensions_score
    except Exception as e:
        logger.error(f"Error calculating dimensions: {e}")
        return 0.5  # Return a default value in case of error

def calculate_risk_probability(asymmetry, border, color, dimensions):
    """Calculate overall risk probability using ABCD features."""
    # Weights for each feature
    asymmetry_weight = 0.3
    border_weight = 0.3
    color_weight = 0.25
    dimensions_weight = 0.15

    # Calculate weighted probability
    probability = (
        asymmetry * asymmetry_weight +
        border * border_weight +
        color * color_weight +
        dimensions * dimensions_weight
    )

    return min(probability, 1.0)  # Ensure between [0, 1]

def analyze_lesion_image(image_data, lesion_id):
    """Analyze a lesion image and return results."""
    try:
        # Extract the base64 image data (removing data:image/jpeg;base64, prefix if present)
        if isinstance(image_data, str) and image_data.startswith('data:'):
            image_data = image_data.split(',')[1]

        # Decode base64 to image
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = np.array(Image.open(io.BytesIO(image_bytes)))
        else:
            # Assume it's already a numpy array
            image = image_data

        # Ensure image is in RGB
        if image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Calculate ABCD features
        asymmetry = calculate_asymmetry(image)
        border = calculate_border_irregularity(image)
        color = calculate_color_variation(image)
        dimensions = calculate_dimensions(image)

        # Calculate overall risk probability
        probability = calculate_risk_probability(asymmetry, border, color, dimensions)

        result = {
            "id": lesion_id,
            "asymmetry": asymmetry,
            "border": border,
            "color": color,
            "dimensions": dimensions,
            "probability": probability
        }

        return result
    except Exception as e:
        logger.error(f"Error analyzing lesion {lesion_id}: {e}")
        # Return default values in case of error
        return {
            "id": lesion_id,
            "asymmetry": 0.5,
            "border": 0.5,
            "color": 0.5,
            "dimensions": 0.5,
            "probability": 0.5
        }

@app.route('/analyze', methods=['GET', 'POST'])
def analyze_lesions():
    """Handle the analysis request from the Electron app."""
    try:
        if request.method == 'POST':
            data = request.json
        else:  # GET method with params
            data = request.args.to_dict()
            if 'lesions' in data and isinstance(data['lesions'], str):
                # Parse the JSON string if needed
                import json
                data['lesions'] = json.loads(data['lesions'])

        lesions_data = data.get('lesions', [])

        if not lesions_data:
            return jsonify({"error": "No lesion data provided"}), 400

        logger.info(f"Received {len(lesions_data)} lesions for analysis")

        # Process each lesion
        results = []
        for lesion in lesions_data:
            lesion_id = lesion.get('id')
            image_data = lesion.get('image')

            if not lesion_id or not image_data:
                continue

            # Analyze the lesion
            result = analyze_lesion_image(image_data, lesion_id)
            results.append(result)

            # Simulate some processing time
            time.sleep(0.1)

        response = {
            "lesions": results,
            "timestamp": datetime.now().isoformat()
        }

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}")
        return jsonify({"error": str(e)}), 500

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Analysis Backend')
    parser.add_argument('--port', type=int, default=7074, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Starting Melanoma Analysis Backend on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)