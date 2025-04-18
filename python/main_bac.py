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
from logging.handlers import RotatingFileHandler


log_dir = '../logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"Created log directory: {log_dir}")

# Setup log file path
log_file = os.path.join(log_dir, 'melanoma_analysis.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),  # 10MB per file, max 5 files
        logging.StreamHandler(sys.stdout)  # Keep console output
    ]
)
logger = logging.getLogger('MelanomaAnalysis')

# Create Flask application
app = Flask(__name__)

def analyze_lesion(lesion_data, lesion_id):
    """
    Analyze lesion data and return results.
    This is a simplified version that just logs the received parameters.
    """
    try:
        # Extract and log all input parameters
        logger.info(f"Analyzing lesion with ID: {lesion_id}")

        # Log all received parameters
        for key, value in lesion_data.items():
            if key == 'image':
                logger.info(f"Received image data (not showing content)")
            else:
                logger.info(f"Parameter {key}: {value}")

        # Generate results based on actual parameters
        # In a real implementation, these would be calculated by a machine learning model

        area_score = min(1.0, lesion_data.get('areaMM2', 0) / 100)
        color_score = min(1.0, lesion_data.get('color_std_mean', 0) / 20)
        asymmetry_score = 1 - (lesion_data.get('symm_2axis', 0.5) or 0.5)
        border_score = lesion_data.get('norm_border', 0.5) or 0.5



        # Calculate mock scores
        ud_scores_image = round(0.4 + (asymmetry_score * 0.3) + (border_score * 0.3), 2)
        ud_scores_tabular = round(0.3 + (area_score * 0.4) + (color_score * 0.3), 2)
        ud_scores_imageTabular = round((ud_scores_image + ud_scores_tabular) / 2 + 0.1, 2)
        risk_score = round((ud_scores_image * 0.3) + (ud_scores_tabular * 0.3) + (ud_scores_imageTabular * 0.4), 2)

        result = {
            "id": lesion_id,
            "ud_scores_image": ud_scores_image,
            "ud_scores_tabular": ud_scores_tabular,
            "ud_scores_imageTabular": ud_scores_imageTabular,
            "risk_score": risk_score
        }

        logger.info(f"Analysis complete for lesion {lesion_id}. Results: {result}")

        return result
    except Exception as e:
        logger.error(f"Error analyzing lesion {lesion_id}: {e}", exc_info=True)
        # Return default values in case of error
        return {
            "id": lesion_id,
            "ud_scores_image": 0.5,
            "ud_scores_tabular": 0.5,
            "ud_scores_imageTabular": 0.5,
            "risk_score": 0.5
        }

@app.route('/analyze', methods=['POST'])
def analyze_lesions():
    """Handle the analysis request from the Electron app."""
    try:
        logger.info(f"Received POST request to /analyze endpoint")

        # Get JSON data
        data = request.json
        logger.info("Received POST request with JSON data")

        # Get lesions array
        lesions_data = data.get('lesions', [])

        # Get request metadata if available
        request_metadata = data.get('requestMetadata', {})
        current_index = request_metadata.get('currentIndex', 0)
        total_count = request_metadata.get('totalCount', len(lesions_data))

        if not lesions_data:
            logger.warning("No lesion data provided in request")
            return jsonify({"error": "No lesion data provided"}), 400

        logger.info(f"Received {len(lesions_data)} lesions for analysis (Lesion {current_index + 1} of {total_count})")

        # Process each lesion
        results = []
        for lesion in lesions_data:
            lesion_id = lesion.get('id')

            if not lesion_id:
                logger.warning("Skipping lesion with no ID")
                continue

            # Analyze the lesion
            result = analyze_lesion(lesion, lesion_id)
            results.append(result)

        response = {
            "lesions": results,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "currentIndex": current_index,
            "totalCount": total_count
        }

        logger.info(f"Analysis complete. Returning results for {len(results)} lesions.")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing analysis request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/hello', methods=['GET'])
def hello():
    """Simple endpoint to test if the server is running."""
    logger.info("Hello endpoint accessed")
    return jsonify({
        "status": "success",
        "message": "Melanoma Analysis Backend is running",
        "timestamp": datetime.now().toISOString()
    })

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Analysis Backend')
    parser.add_argument('--port', type=int, default=7074, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info("Starting server...")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Log file location: {log_file}")

    try:
        app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)