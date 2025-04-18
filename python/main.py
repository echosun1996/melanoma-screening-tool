# Melanoma Analysis Backend

import os
import sys

# 确保在任何其他操作之前设置工作目录
try:
    # 获取可执行文件的目录
    if getattr(sys, 'frozen', False):
        # 如果是打包后的应用
        application_path = os.path.dirname(sys.executable)
    else:
        # 如果是开发环境
        application_path = os.path.dirname(os.path.abspath(__file__))

    # 确保目录存在
    if os.path.exists(application_path):
        # 切换到应用目录
        os.chdir(application_path)
        print(f"Set working directory to: {application_path}")
    else:
        print(f"Warning: Application path does not exist: {application_path}")
        # 尝试使用当前目录
        os.chdir(os.path.expanduser("~"))  # 使用用户主目录作为备选
        print(f"Falling back to home directory")
except Exception as e:
    print(f"Error setting working directory: {e}")
    # 最后的备选：使用一个肯定存在的目录
    os.chdir("/")  # 在 Mac/Linux 上使用根目录
    print("Falling back to root directory")



from flask import Flask, request, jsonify
import base64
import io
import numpy as np
from PIL import Image
import logging
import argparse
import os
import sys
import torch
import pandas as pd
from datetime import datetime
from logging.handlers import RotatingFileHandler
# os.chdir(os.path.dirname(os.path.abspath(sys.executable)))

# Configure paths and directories
# log_dir = '../logs'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
#     print(f"Created log directory: {log_dir}")
#
# # Setup log file path
# log_file = os.path.join(log_dir, 'melanoma_analysis.log')
# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5),  # 10MB per file, max 5 files
#         logging.StreamHandler(sys.stdout)  # Keep console output
#     ]
# )
# logger = logging.getLogger('MelanomaAnalysis')


# Setup console logging only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Log to console only
    ]
)
# Then use logging as normal
logger = logging.getLogger(__name__)
logger.info("Application started")

# Create Flask application
app = Flask(__name__)

# Define model constants and global variables
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Define feature lists for the model
# NUMERICAL_FEATURES = [
#     'A', 'Aext', 'B', 'Bext', 'C', 'Cext', 'H', 'Hext', 'L', 'Lext',
#     'areaMM2', 'area_perim_ratio', 'color_std_mean', 'deltaA', 'deltaB',
#     'deltaL', 'deltaLB', 'deltaLBnorm', 'dnn_lesion_confidence', 'eccentricity',
#     'majorAxisMM', 'minorAxisMM', 'nevi_confidence', 'norm_border', 'norm_color',
#     'perimeterMM', 'radial_color_std_max', 'stdL', 'stdLExt', 'symm_2axis',
#     # 'symm_2axis_angle', 'age'
# ]

NUMERICAL_FEATURES = [
    'A', 'Aext', 'B', 'Bext', 'C', 'Cext', 'H', 'Hext', 'L', 'Lext',
    'areaMM2', 'area_perim_ratio', 'color_std_mean', 'deltaA', 'deltaB',
    'deltaL', 'deltaLB', 'deltaLBnorm',  'eccentricity',
    'majorAxisMM', 'minorAxisMM',  'norm_border', 'norm_color',
    'perimeterMM', 'radial_color_std_max', 'stdL', 'stdLExt', 'symm_2axis',
    'symm_2axis_angle', 'age'
]

CATEGORICAL_FEATURES = ['location_simple', 'Sex']

# -------------------------
# Model Loading Functions
# -------------------------
def load_model():
    """
    Load the TIP model for inference
    Returns the loaded model or None if loading fails
    """
    try:
        logger.info(f"Loading model using device: {DEVICE}")

        # 禁用 TorchScript JIT 编译
        try:
            # 方法 1: 通过环境变量禁用
            os.environ['PYTORCH_JIT'] = '0'

            # 方法 2: 通过 Python API 禁用
            import torch._C
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)

            # 方法 3: 通过全局变量禁用
            torch.jit.enabled = False

            logger.info("TorchScript JIT compilation disabled")
        except Exception as e:
            logger.warning(f"Could not disable TorchScript: {e}")

        # 修补 timm 库，跳过问题类的编译
        try:
            import sys
            import types

            # 创建一个空的模块作为替代
            mock_module = types.ModuleType('timm.layers.space_to_depth')

            # 添加假的类定义，不使用 @torch.jit.script
            class MockSpaceToDepth:
                def __init__(self, block_size=2):
                    self.block_size = block_size

                def __call__(self, x):
                    return x  # 简单地返回输入

            mock_module.SpaceToDepthJit = MockSpaceToDepth
            mock_module.SpaceToDepth = MockSpaceToDepth
            mock_module.SpaceToDepthModule = MockSpaceToDepth
            mock_module.DepthToSpace = MockSpaceToDepth

            # 替换原始模块
            sys.modules['timm.layers.space_to_depth'] = mock_module
            logger.info("Patched timm.layers.space_to_depth module")
        except Exception as e:
            logger.warning(f"Could not patch timm module: {e}")

        # 正常的模型加载代码
        from models.TipModel3LossISIC512 import TIP3LossISIC
        from finetune_vit import TIPFineTuneModel

        # 加载预训练模型
        pretrained_model = TIP3LossISIC.load_from_checkpoint(
            'checkpoint/best_model_epoch.ckpt',
        )

        # 加载微调模型
        model = TIPFineTuneModel.load_from_checkpoint(
            'checkpoint/best-model-v1.ckpt',
            pretrained_model=pretrained_model,
            config={
                'lr': 1e-4,
                'weight_decay': 0.001,
                'multimodal_embedding_dim': 768
            },
        )

        # 设置评估模式并移至设备
        model.eval()
        model = model.to(DEVICE)

        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
# -------------------------
# Image Processing Functions
# -------------------------
def preprocess_image(image_data):
    """
    Preprocess image from base64 string to tensor
    """
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to model's expected size (224x224 for most models)
        image = image.resize((224, 224))

        # Convert to numpy array
        image_np = np.array(image)

        # Normalize for ImageNet pretrained models
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        # Convert to float and normalize
        image_np = image_np.astype(np.float32) / 255.0
        image_np = (image_np - normalize_mean) / normalize_std

        # Convert to tensor (C, H, W)
        image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float()

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

# -------------------------
# Tabular Data Processing Functions
# -------------------------
def preprocess_tabular_data(tabular_data):
    """
    Preprocess tabular features for the model
    """
    try:
        # Create a copy of input data
        data_copy = tabular_data.copy()

        # Handle field name mapping
        if 'gender' in data_copy and 'Sex' not in data_copy:
            data_copy['Sex'] = data_copy.pop('gender')
            logger.info("Mapped 'gender' field to 'Sex'")

        # Create a DataFrame for easier manipulation
        df = pd.DataFrame([data_copy])

        # Make a copy for processing
        processed_df = df.copy()

        # Process categorical features
        for col in CATEGORICAL_FEATURES:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str).fillna('Unknown')
            else:
                processed_df[col] = 'Unknown'  # Add missing categorical columns

        # Fill missing numerical features
        for col in NUMERICAL_FEATURES:
            if col not in processed_df.columns:
                processed_df[col] = 0  # Fill with default value

        # Standardize numerical features (using fixed values for consistency)
        for col in NUMERICAL_FEATURES:
            # Simple standardization assuming mean=0 and std=1
            processed_df[col] = (processed_df[col] - 0) / 1

        # Convert categorical features to integer codes
        for col in CATEGORICAL_FEATURES:
            # Simple encoding for testing purposes
            if col.lower() == 'sex':
                processed_df[col] = (
                    processed_df[col]
                    .str.lower()  # Convert to lowercase
                    .map({'male': 0, 'female': 1, 'unknown': 2})
                    .fillna(2)  # Other unknown values also marked as 2
                )
            # elif col == 'location_simple':
            #     # Map common body locations to codes
            #     location_map = {
            #         'back': 0, 'chest': 1, 'arm': 2, 'leg': 3, 'face': 4,
            #         'shoulder': 5, 'abdomen': 6, 'head': 7, 'neck': 8, 'unknown': 9
            #     }
            #     # Default to 'Unknown' code for any unmapped locations
            #     processed_df[col] = processed_df[col].map(
            #         lambda x: location_map.get(str(x).lower(), 9)
            #     )
            elif col.lower() == 'location_simple':
                location_map = {
                    'back': 0, 'chest': 1, 'arm': 2, 'leg': 3, 'face': 4,
                    'shoulder': 5, 'abdomen': 6, 'head': 7, 'neck': 8
                }
                def map_location(value):
                    val_lower = str(value).lower()
                    for key, code in location_map.items():
                        if key in val_lower:
                            return code
                    return 9  # default to 'unknown'
                processed_df[col] = processed_df[col].apply(map_location)
            else:
                # For any other categorical features, default encoding
                processed_df[col] = 0

        # Combine features in the expected order
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

        logger.info(f"Using {len(all_features)} features: {len(NUMERICAL_FEATURES)} numerical, {len(CATEGORICAL_FEATURES)} categorical")

        feature_values = processed_df[all_features].values[0]
        logger.info(all_features)
        logger.info(feature_values)
        logger.info(len(all_features))

        # Convert to tensor
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)

        # Add batch dimension
        feature_tensor = feature_tensor.unsqueeze(0)

        # Log tensor shape for debugging
        logger.info(f"Feature tensor shape: {feature_tensor.shape}")

        return feature_tensor

    except Exception as e:
        logger.error(f"Error preprocessing tabular data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
# -------------------------
# Main Analysis Function
# -------------------------
def analyze_lesion(lesion_data, lesion_id):
    """
    Analyze lesion data and return results.
    """
    try:
        # Extract and log all input parameters
        logger.info(f"Analyzing lesion with ID: {lesion_id}")

        # Log all received parameters except image
        for key, value in lesion_data.items():
            if key == 'image':
                logger.info(f"Received image data (not showing content)")
            else:
                logger.info(f"Parameter {key}: {value}")

        # Check if model is loaded
        global MODEL
        if MODEL is None:
            logger.warning("Model not loaded. Using fallback scoring method.")
            # Fallback method using simple heuristics
            return generate_fallback_results(lesion_data, lesion_id)

        try:
            # Preprocess image
            image_tensor = preprocess_image(lesion_data.get('image', ''))
            if image_tensor is None:
                logger.error(f"Failed to preprocess image for lesion {lesion_id}")
                return generate_fallback_results(lesion_data, lesion_id)

            # Preprocess tabular data
            tabular_tensor = preprocess_tabular_data(lesion_data)
            if tabular_tensor is None:
                logger.error(f"Failed to preprocess tabular data for lesion {lesion_id}")
                return generate_fallback_results(lesion_data, lesion_id)

            # Move tensors to device
            image_tensor = image_tensor.to(DEVICE)
            tabular_tensor = tabular_tensor.to(DEVICE)
            logger.info("Run prediction")

            # Run prediction
            with torch.no_grad():
                num_cat = MODEL.encoder_tabular.num_cat
                num_con = MODEL.encoder_tabular.num_con
                total_features = num_cat + num_con
                logger.info("total_features:"+str(total_features))
                logger.info(f"Model expects {total_features} features: {num_cat} categorical + {num_con} continuous")
                if len(tabular_tensor) != total_features:
                    logger.error(f"WARNING: Feature count mismatch! Model expects {total_features} features but we have {len(tabular_tensor)}.")
                    logger.error("This will likely cause the assertion error during prediction.")
                outputs = MODEL(image_tensor, tabular_tensor)

                # Convert logits to probability
                logger.info("Model outputs"+str(outputs))
                # probability = torch.sigmoid(outputs).cpu().numpy().flatten()[0]

                min_logit = -21.9
                max_logit = 2.2
                probability = ((outputs - min_logit) / (max_logit - min_logit)).cpu().numpy().flatten()[0]

                # Derive individual scores (this is a simplified example)
                # In production, these might come from intermediate model outputs
                ud_scores_image = float(probability * 0.9 + np.random.uniform(0, 0.1))
                ud_scores_tabular = float(probability * 0.8 + np.random.uniform(0, 0.2))
                ud_scores_imageTabular = float(probability * 1.1) if probability > 0.5 else float(probability * 0.9)
                ud_scores_imageTabular = min(1.0, max(0.0, ud_scores_imageTabular))  # Clamp to [0,1]

                # Format results
                result = {
                    "id": lesion_id,
                    "uuid": "n/a",
                    "ud_scores_image": -1,
                    "ud_scores_tabular": -1,
                    "ud_scores_imageTabular": -1,
                    "risk_score": round(float(probability), 17)
                }

                logger.info(f"Analysis complete for lesion {lesion_id}. Results: {result}")
                return result

        except Exception as e:
            logger.error(f"Error during model prediction for lesion {lesion_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return generate_fallback_results(lesion_data, lesion_id)

    except Exception as e:
        logger.error(f"Error analyzing lesion {lesion_id}: {e}", exc_info=True)
        # Return default values in case of error
        return generate_fallback_results(lesion_data, lesion_id)

def generate_fallback_results(lesion_data, lesion_id):
    """
    Generate fallback results when model prediction fails
    """
    try:
        # Use some parameters to generate mock scores if available
        area_score = min(1.0, float(lesion_data.get('areaMM2', 0)) / 100)
        color_score = min(1.0, float(lesion_data.get('color_std_mean', 0)) / 20)
        asymmetry_score = 1 - float(lesion_data.get('symm_2axis', 0.5) or 0.5)
        border_score = float(lesion_data.get('norm_border', 0.5) or 0.5)

        # Calculate mock scores
        ud_scores_image = round(0.4 + (asymmetry_score * 0.3) + (border_score * 0.3), 2)
        ud_scores_tabular = round(0.3 + (area_score * 0.4) + (color_score * 0.3), 2)
        ud_scores_imageTabular = round((ud_scores_image + ud_scores_tabular) / 2 + 0.1, 2)
        risk_score = round((ud_scores_image * 0.3) + (ud_scores_tabular * 0.3) + (ud_scores_imageTabular * 0.4), 2)

        result = {
            "id": lesion_id,
            "uuid": "n/a",
            "ud_scores_image": -1,
            "ud_scores_tabular": -1,
            "ud_scores_imageTabular": -1,
            "risk_score": -1
        }

        logger.info(f"Generated fallback results for lesion {lesion_id}: {result}")
        return result
    except Exception as e:
        logger.error(f"Error generating fallback results: {e}")
        # Ultimate fallback with fixed values
        return {

            "id": lesion_id,
            "uuid": "n/a",
            "ud_scores_image": -1,
            "ud_scores_tabular": -1,
            "ud_scores_imageTabular": -1,
            "risk_score": -1
        }

# -------------------------
# API Endpoints
# -------------------------
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
        "timestamp": datetime.now().isoformat(),
        "model_loaded": MODEL is not None
    })

# -------------------------
# Server Startup
# -------------------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Melanoma Analysis Backend')
    parser.add_argument('--port', type=int, default=7074, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the server on')
    parser.add_argument('--no-model', action='store_true', help='Skip model loading')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger.info("Starting server...")
    logger.info(f"Host: {args.host}, Port: {args.port}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    # logger.info(f"Log file location: {log_file}")

    # Load model before starting server
    if not args.no_model:
        logger.info("Loading model...")
        MODEL = load_model()
        if MODEL is None:
            logger.warning("Model loading failed. Server will use fallback predictions.")
    else:
        logger.info("Model loading skipped (--no-model flag used)")

    try:
        app.run(host=args.host, port=args.port, debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)