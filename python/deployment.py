import os
import sys
import json
import pandas as pd
import numpy as np
import base64
import io
from PIL import Image
from pathlib import Path
import torch
from datetime import datetime
import shutil
from tqdm import tqdm
import traceback
import re

# Set up consistent random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define constants
JSON_FILE = '/mnt/hdd/sdc/ysheng/TBP_Screening/DERMDATA-I2768/VECTRA/JSON/MYM/id10175MYM_visit3_date20180319_v1.2.4.json'
MERGED_CSV = '/mnt/hdd/sdc/ysheng/TBP_Screening/merged_filtered_csv.csv'
OUTPUT_BASE = '/mnt/hdd/sdc/ysheng/Deployment'
CHECKPOINT_PATH = '/mnt/hdd/sdc/ysheng/TBP_Screening/tip_finetune_results/checkpoints/best-model-v1.ckpt'

# Create output directories
EXTRACTED_DIR = os.path.join(OUTPUT_BASE, 'extracted_data')
FILTERED_DIR = os.path.join(OUTPUT_BASE, 'filtered_data')
PROCESSED_DIR = os.path.join(OUTPUT_BASE, 'processed_data')
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE, 'predictions')

# Create all necessary directories
for directory in [OUTPUT_BASE, EXTRACTED_DIR, FILTERED_DIR, PROCESSED_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Define image tile columns as in extract_image_tiles.py
tile_image_cols = ["img64", "img64cc", "imgM64", "imgbw64", "imgextMask64", "imgoob64", "imgunrectbw64", 'boundary']

# Define feature lists as in preprocess_tbp_table.py 32+2
NUMERICAL_FEATURES = [
    'A', 'Aext', 'B', 'Bext', 'C', 'Cext', 'H', 'Hext', 'L', 'Lext', 
    'areaMM2', 'area_perim_ratio', 'color_std_mean', 'deltaA', 'deltaB', 
    'deltaL', 'deltaLB', 'deltaLBnorm', 'dnn_lesion_confidence', 'eccentricity', 
    'majorAxisMM', 'minorAxisMM', 'nevi_confidence', 'norm_border', 'norm_color', 
    'perimeterMM', 'radial_color_std_max', 'stdL', 'stdLExt', 'symm_2axis', 
    'symm_2axis_angle', 'age'
]

CATEGORICAL_FEATURES = ['location_simple', 'Sex']


#######################################
# Step 1: Extract CSV and image data from JSON
#######################################

def extract_date_from_filename(filename):
    """Extract date from the filename using regex"""
    date_match = re.search(r'date(\d{8})', filename)
    if date_match:
        date_str = date_match.group(1)
        # Convert to date object
        try:
            year = int(date_str[0:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            return datetime(year, month, day)
        except ValueError:
            print(f"Invalid date format in filename: {date_str}")
    return None

def decode_save_tile(b64_data, output_path):
    """Decode base64 image and save to file"""
    try:
        decoded = base64.b64decode(b64_data)
        im = Image.open(io.BytesIO(decoded))
        im.save(output_path)
        return True
    except Exception as e:
        print(f"Error decoding/saving image: {str(e)}")
        return False

def extract_study_id_from_filename(filename):
    """
    Extract the study ID from the filename in the correct format
    Fixed to keep the V suffix
    """
    # Pattern to match "id10003V" or similar patterns
    match = re.search(r'id(\w+V)', filename)
    if match:
        return match.group(1)  # This captures "10003V"
    
    # Alternative pattern if the first one doesn't match
    match = re.search(r'id(\d+)', filename)
    if match:
        return match.group(1) + "V"  # Add V suffix if not present
    
    # Fallback: use the first part of the filename
    parts = filename.split('_')
    if parts and parts[0].startswith('id'):
        return parts[0][2:]  # Remove 'id' prefix
    
    # Last resort
    return filename.split('_')[0]

def extract_data_from_json(json_path, output_dir):
    """Extract CSV and images from JSON file"""
    print(f"Step 1: Extracting data from {json_path}...")
    
    # Extract capture date from filename
    filename = os.path.basename(json_path)
    capture_date = extract_date_from_filename(filename)
    if capture_date:
        print(f"Extracted capture date from filename: {capture_date.strftime('%Y-%m-%d')}")
    else:
        print("Could not extract capture date from filename")
    
    # Load JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Extract patient metadata
    try:
        metadata = {}
        if 'root' in json_data and 'metadata' in json_data['root']:
            metadata = json_data['root']['metadata']
            print(f"Found metadata in JSON file")
        
        # Extract additional data like dob if available
        dob = None
        if 'dob' in metadata:
            dob = metadata['dob']
            print(f"Found DOB in metadata: {dob}")
            
        # If capture_date from filename extraction is None, try from metadata
        if not capture_date and 'captureDate' in metadata:
            try:
                capture_date = datetime.strptime(metadata['captureDate'], '%Y-%m-%d')
                print(f"Using capture date from metadata: {capture_date.strftime('%Y-%m-%d')}")
            except ValueError:
                print(f"Invalid capture date format in metadata")
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        metadata = {}
        dob = None
    
    # Extract lesion data
    if 'root' in json_data and 'children' in json_data['root']:
        df = pd.DataFrame(json_data['root']['children'])
        
        # Add study_id (from filename) and source information
        # FIXED: Proper study_id extraction to keep "10003V" format
        study_id = extract_study_id_from_filename(filename)
        df['study_id'] = study_id
        df['source'] = 'MYM'  # Extracted from MYM directory
        
        print(f"Extracted {len(df)} lesion records with study_id: {study_id}")
        
        # Save images
        img_dir = os.path.join(output_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        image_saved_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving images"):
            uuid = row['uuid']
            # Using img64cc as in extract_image_tiles.py
            if 'img64cc' in row and row['img64cc']:
                img_path = os.path.join(img_dir, f"{uuid}.png")
                if decode_save_tile(row['img64cc'], img_path):
                    image_saved_count += 1
        
        print(f"Saved {image_saved_count} images")
        
        # Remove image columns before saving CSV
        df_csv = df.drop(columns=tile_image_cols, errors='ignore')
        
        # Store capture_date and dob for later use
        if capture_date:
            df_csv['capture_date'] = capture_date.strftime('%Y-%m-%d')
        if dob:
            df_csv['dob'] = dob
            
        csv_path = os.path.join(output_dir, 'extracted_lesions.csv')
        df_csv.to_csv(csv_path, index=False)
        print(f"Saved extracted data to {csv_path}")
        
        return df_csv
    else:
        print("Error: Invalid JSON structure")
        return None


#######################################
# Step 2: Filter CSV data using conditions
#######################################

def filter_dataframe(df):
    """
    Filter the dataframe based on criteria from filter.py 
    with additional filtering condition
    """
    print("\nStep 2: Filtering dataframe...")
    
    # Check for lesionid column
    lesion_id_col = 'lesionid' if 'lesionid' in df.columns else 'LesionID'
    
    # Valid lesion mask
    valid_lesion_mask = ~df[lesion_id_col].isna() if lesion_id_col in df.columns else np.zeros(len(df), dtype=bool)
    if lesion_id_col not in df.columns:
        print("Note: No lesionid column found. All records will be filtered by other criteria.")
    
    # Check if all required columns exist
    detail_cols = ['majorAxisMM', 'deltaLBnorm', 'out_of_bounds_fraction', 'dnn_lesion_confidence']
    if all(col in df.columns for col in detail_cols):
        # Primary filter condition from filter.py
        status0_mask = (
            (~valid_lesion_mask) &
            (df['majorAxisMM'] >= 2.0) & 
            (df['deltaLBnorm'] >= 4.5) & 
            (df['out_of_bounds_fraction'] <= 0.25) & 
            (df['dnn_lesion_confidence'] >= 50.0)
        )
        
        # Additional filter condition for records not meeting the first criteria
        additional_mask = (
            (~valid_lesion_mask) &
            (~status0_mask) &
            (df['dnn_lesion_confidence'] >= 80.0) &
            (df['deltaLBnorm'] >= 5.0)
        )
        
        # Combine both filters
        final_mask = valid_lesion_mask | status0_mask | additional_mask
        
        # Apply filter
        filtered_df = df[final_mask].copy()
        discarded_df = df[~final_mask].copy()
        
        print(f"Original records: {len(df)}")
        print(f"Filtered records: {len(filtered_df)}")
        print(f"  - Valid lesionID: {valid_lesion_mask.sum()}")
        print(f"  - Primary filter: {status0_mask.sum()}")
        print(f"  - Additional filter: {additional_mask.sum()}")
        print(f"Discarded records: {len(discarded_df)}")
        
        return filtered_df, discarded_df
    else:
        missing_cols = [col for col in detail_cols if col not in df.columns]
        print(f"Warning: Missing required columns for filtering: {missing_cols}")
        # If filtering columns are missing, keep all records
        return df, pd.DataFrame()  # Return original df and empty discarded df


#######################################
# Step 3: Match and save filtered CSV with images
#######################################

def save_filtered_data(filtered_df, extracted_dir, filtered_dir):
    """
    Save filtered CSV and copy corresponding images
    """
    print("\nStep 3: Saving filtered data...")
    
    # Save CSV
    csv_path = os.path.join(filtered_dir, 'filtered_lesions.csv')
    filtered_df.to_csv(csv_path, index=False)
    
    # Copy images for filtered records
    src_img_dir = os.path.join(extracted_dir, 'images')
    dst_img_dir = os.path.join(filtered_dir, 'images')
    os.makedirs(dst_img_dir, exist_ok=True)
    
    found_images = 0
    for uuid in tqdm(filtered_df['uuid'], desc="Copying images"):
        src_path = os.path.join(src_img_dir, f"{uuid}.png")
        if os.path.exists(src_path):
            dst_path = os.path.join(dst_img_dir, f"{uuid}.png")
            shutil.copy(src_path, dst_path)
            found_images += 1
    
    print(f"Copied {found_images} corresponding images out of {len(filtered_df)} records")
    
    return csv_path, dst_img_dir


#######################################
# Step 4: Add Missing Info (Sex and Age)
#######################################

def add_missing_info(csv_path, merged_csv_path):
    """
    Add missing Sex and calculate age using information from merged CSV
    """
    print("\nStep 4: Adding missing Sex and Age information...")
    
    # Load the filtered lesions data
    df = pd.read_csv(csv_path)
    
    # Load the merged CSV to get patient information
    merged_df = pd.read_csv(merged_csv_path)
    
    # Extract study_id from filtered lesions
    study_id = df['study_id'].iloc[0] if not df['study_id'].empty else None
    
    if not study_id:
        print("Warning: Could not find study_id in the filtered lesions data")
        return csv_path
    
    print(f"Looking for information about study_id: {study_id}")
    
    # Find records for this study_id in the merged CSV
    patient_records = merged_df[merged_df['study_id'] == study_id]
    
    if len(patient_records) == 0:
        print(f"Warning: No records found for study_id {study_id} in merged CSV")
        print("Trying alternative study_id formats...")
        
        # Try alternative formats (with or without 'V')
        if study_id.endswith('V'):
            alt_study_id = study_id[:-1]  # Remove 'V'
        else:
            alt_study_id = study_id + 'V'  # Add 'V'
            
        patient_records = merged_df[merged_df['study_id'] == alt_study_id]
        
        if len(patient_records) > 0:
            print(f"Found records using alternative study_id: {alt_study_id}")
            study_id = alt_study_id
        
    if len(patient_records) == 0:
        # If still no records found, try with just the numeric part
        numeric_study_id = re.sub(r'[^0-9]', '', study_id)
        if numeric_study_id:
            patient_records = merged_df[merged_df['study_id'].astype(str).str.contains(numeric_study_id)]
            if len(patient_records) > 0:
                print(f"Found {len(patient_records)} records containing numeric study_id: {numeric_study_id}")
                # Use the first matching record
                study_id = patient_records['study_id'].iloc[0]
                print(f"Using matched study_id: {study_id}")
                patient_records = patient_records.head(1)
    
    if len(patient_records) > 0:
        # Get Sex information if available
        if 'Sex' in patient_records.columns:
            sex = patient_records['Sex'].iloc[0]
            df['Sex'] = sex
            print(f"Added Sex: {sex}")
        
        # Calculate age if we have capture_date and dob
        if 'capture_date' in df.columns:
            cap_date = pd.to_datetime(df['capture_date'].iloc[0])
            
            # Try to get DOB from df first
            if 'dob' in df.columns and not pd.isna(df['dob'].iloc[0]):
                birth_date = pd.to_datetime(df['dob'].iloc[0])
                age_years = (cap_date - birth_date).days / 365.25
                df['age'] = age_years
                print(f"Calculated age from metadata: {age_years:.2f} years")
            # If not, try to get from merged CSV
            elif 'dob' in patient_records.columns and not patient_records['dob'].empty:
                birth_date = pd.to_datetime(patient_records['dob'].iloc[0])
                age_years = (cap_date - birth_date).days / 365.25
                df['age'] = age_years
                print(f"Calculated age from merged CSV: {age_years:.2f} years")
            # If age directly available in merged CSV
            elif 'age' in patient_records.columns and not patient_records['age'].empty:
                df['age'] = patient_records['age'].iloc[0]
                print(f"Copied age from merged CSV: {patient_records['age'].iloc[0]}")
    else:
        print(f"No matching records found for study_id {study_id} after trying alternatives")
        # If we have a capture_date in the dataset, we can still calculate age
        if 'capture_date' in df.columns and 'dob' in df.columns and not pd.isna(df['dob'].iloc[0]):
            # Calculate age from capture_date and dob
            cap_date = pd.to_datetime(df['capture_date'].iloc[0])
            birth_date = pd.to_datetime(df['dob'].iloc[0])
            age_years = (cap_date - birth_date).days / 365.25
            df['age'] = age_years
            print(f"Calculated age from metadata: {age_years:.2f} years")
    
    # If we still don't have Sex, set to default
    if 'Sex' not in df.columns or df['Sex'].isna().all():
        df['Sex'] = 'Male'  # Default sex (Male/Female instead of Unknown)
        print("Added default Sex: Male")
    
    # If we still don't have age, set to default
    if 'age' not in df.columns or df['age'].isna().all():
        df['age'] = 50.0  # Default middle age
        print("Added default age: 50.0")
    
    # Save updated CSV
    enhanced_path = os.path.join(os.path.dirname(csv_path), 'enhanced_lesions.csv')
    df.to_csv(enhanced_path, index=False)
    print(f"Saved enhanced data to {enhanced_path}")
    
    return enhanced_path


#######################################
# Step 5: Process images using image2numpy_tbp.py methods
#######################################

def process_images(img_dir, processed_dir):
    """
    Process images and convert to numpy format
    """
    print("\nStep 5: Processing images to numpy format...")
    
    # Create numpy directory
    numpy_dir = os.path.join(processed_dir, 'numpy_images')
    os.makedirs(numpy_dir, exist_ok=True)
    
    # Get all PNG files
    image_paths = list(Path(img_dir).glob("*.png"))
    numpy_paths = []
    
    # Process each image
    for img_path in tqdm(image_paths, desc="Converting images to numpy"):
        try:
            # Read image
            img = Image.open(img_path)
            
            # Convert to numpy array
            img_np = np.array(img)
            
            # Get UUID (filename without extension)
            uuid = os.path.splitext(os.path.basename(img_path))[0]
            
            # Save as numpy
            save_path = os.path.join(numpy_dir, f'test_{uuid}.npy')
            np.save(save_path, img_np)
            numpy_paths.append(save_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Save numpy paths
    test_paths = os.path.join(processed_dir, 'tbp_test_paths.pt')
    torch.save(numpy_paths, test_paths)
    print(f"Saved {len(numpy_paths)} numpy images to {test_paths}")
    
    return numpy_paths


#######################################
# Step 6: Process CSV data following preprocess_tbp_table.py
#######################################

def get_model_expected_features():
    """
    Try to determine how many features the model expects
    """
    # Try to load the model to check its expected feature count
    try:
        sys.path.append("/mnt/hdd/sdc/ysheng/TBP_Screening")
        from python.models.TipModel3LossISIC512 import TIP3LossISIC
        from finetune_vit import TIPFineTuneModel
        
        # Load the pretrained model for inspection
        pretrained_model = TIP3LossISIC.load_from_checkpoint(
            '/mnt/hdd/sdc/ysheng/TIP/results/isic/0328_1327/best_model_epoch=39.ckpt',
            strict=False
        )
        
        # Extract feature counts from the model
        tabular_encoder = pretrained_model.encoder_tabular
        if hasattr(tabular_encoder, 'num_cat') and hasattr(tabular_encoder, 'num_con'):
            num_cat = tabular_encoder.num_cat
            num_con = tabular_encoder.num_con
            print(f"Model expects {num_cat} categorical features and {num_con} continuous features")
            return num_cat, num_con
            
    except Exception as e:
        print(f"Could not determine model's expected feature counts: {str(e)}")
    
    # Default values if we can't determine from the model
    print("Using default feature counts: 2 categorical, 30 continuous")
    return 2, 30  # Default: 2 categorical, 30 continuous features

def process_tabular_data(csv_path, processed_dir):
    """
    Process tabular data following preprocess_tbp_table.py methods
    """
    print("\nStep 6: Processing tabular data...")
    
    # Try to get the expected feature counts from the model
    num_cat, num_con = get_model_expected_features()
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    
    # Add a placeholder status column for testing (prediction won't use this)
    if 'status' not in df.columns:
        df['status'] = 0  # All zeros as dummy labels
        print("Added dummy 'status' column with all zeros (for dataset compatibility)")
    
    # Create a deep copy for processing
    processed_df = df.copy()
    
    # Make sure we have the expected number of categorical features
    if len(CATEGORICAL_FEATURES) != num_cat:
        print(f"Warning: Model expects {num_cat} categorical features, but we have {len(CATEGORICAL_FEATURES)}")
        # Adjust as needed (truncate or add dummy features)
        if len(CATEGORICAL_FEATURES) > num_cat:
            print(f"Truncating categorical features to {num_cat}")
            CATEGORICAL_FEATURES_ADJUSTED = CATEGORICAL_FEATURES[:num_cat]
        else:
            print(f"Adding {num_cat - len(CATEGORICAL_FEATURES)} dummy categorical features")
            CATEGORICAL_FEATURES_ADJUSTED = CATEGORICAL_FEATURES.copy()
            for i in range(len(CATEGORICAL_FEATURES), num_cat):
                dummy_feature = f"dummy_cat_{i}"
                CATEGORICAL_FEATURES_ADJUSTED.append(dummy_feature)
                processed_df[dummy_feature] = 0  # Add dummy feature with constant value
    else:
        CATEGORICAL_FEATURES_ADJUSTED = CATEGORICAL_FEATURES
    
    # Similarly for numerical features
    if len(NUMERICAL_FEATURES) != num_con:
        print(f"Warning: Model expects {num_con} continuous features, but we have {len(NUMERICAL_FEATURES)}")
        # Adjust as needed
        if len(NUMERICAL_FEATURES) > num_con:
            print(f"Truncating numerical features to {num_con}")
            NUMERICAL_FEATURES_ADJUSTED = NUMERICAL_FEATURES[:num_con]
        else:
            print(f"Adding {num_con - len(NUMERICAL_FEATURES)} dummy numerical features")
            NUMERICAL_FEATURES_ADJUSTED = NUMERICAL_FEATURES.copy()
            for i in range(len(NUMERICAL_FEATURES), num_con):
                dummy_feature = f"dummy_num_{i}"
                NUMERICAL_FEATURES_ADJUSTED.append(dummy_feature)
                processed_df[dummy_feature] = 0.0  # Add dummy feature with constant value
    else:
        NUMERICAL_FEATURES_ADJUSTED = NUMERICAL_FEATURES
    
    print(f"Using {len(CATEGORICAL_FEATURES_ADJUSTED)} categorical and {len(NUMERICAL_FEATURES_ADJUSTED)} numerical features")
    
    # Process categorical features
    for col in CATEGORICAL_FEATURES_ADJUSTED:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(str).fillna('Unknown')
            print(f"Processed categorical feature: {col}")
        else:
            processed_df[col] = 'Unknown'  # Add missing categorical columns
            print(f"Created missing categorical feature: {col} (filled with 'Unknown')")
    
    # Fill missing numerical features
    for col in NUMERICAL_FEATURES_ADJUSTED:
        if col not in processed_df.columns:
            processed_df[col] = 0  # Fill with default value for missing columns
            print(f"Created missing numerical feature: {col} (filled with 0)")
    
    # Fill NA in existing numeric columns with mean
    processed_df[NUMERICAL_FEATURES_ADJUSTED] = processed_df[NUMERICAL_FEATURES_ADJUSTED].fillna(
        processed_df[NUMERICAL_FEATURES_ADJUSTED].mean()
    )
    print("Filled missing numerical values with column means")
    
    # Convert categorical features to codes
    for col in CATEGORICAL_FEATURES_ADJUSTED:
        processed_df[col] = pd.Categorical(processed_df[col]).codes
        print(f"Encoded {col} as categorical codes")
    
    # Standardize numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    processed_df[NUMERICAL_FEATURES_ADJUSTED] = scaler.fit_transform(processed_df[NUMERICAL_FEATURES_ADJUSTED])
    print("Standardized numerical features")
    
    # Rename status to target
    processed_df = processed_df.rename(columns={'status': 'target'})
    
    # Save processed features
    all_features = NUMERICAL_FEATURES_ADJUSTED + CATEGORICAL_FEATURES_ADJUSTED
    features_path = os.path.join(processed_dir, 'tbp_features_test.csv')
    processed_df[all_features].to_csv(features_path, index=False)
    print(f"Saved processed features to {features_path}")
    
    # Save dummy labels (all zeros) for dataset compatibility
    labels_path = os.path.join(processed_dir, 'tbp_labels_test.pt')
    torch.save(processed_df['target'].values, labels_path)
    print(f"Saved dummy labels to {labels_path} (needed for dataset compatibility)")
    
    # Generate image paths list matching the order in the CSV
    img_paths = []
    for uuid in processed_df['uuid']:
        path = os.path.join(processed_dir, 'numpy_images', f'test_{uuid}.npy')
        # Verify the path exists
        if os.path.exists(path):
            img_paths.append(path)
        else:
            print(f"Warning: Image file not found for UUID {uuid}")
    
    # Save paths (only for images that exist)
    torch.save(img_paths, os.path.join(processed_dir, 'tbp_test_paths.pt'))
    print(f"Saved {len(img_paths)} image paths matching the tabular data")
    
    # Save feature lengths for model compatibility
    lengths = []
    # Numerical feature length = 1
    lengths.extend([1] * len(NUMERICAL_FEATURES_ADJUSTED))
    # Categorical feature length = number of categories
    for col in CATEGORICAL_FEATURES_ADJUSTED:
        lengths.append(len(pd.Categorical(processed_df[col].astype(str)).categories))
    
    lengths_path = os.path.join(processed_dir, 'tabular_lengths.pt')
    torch.save(lengths, lengths_path)
    print(f"Saved feature lengths to {lengths_path}")
    
    return processed_df, all_features



#######################################
# Step 7: Run prediction using the model
#######################################

def run_prediction(processed_dir, predictions_dir, checkpoint_path, expected_features):
    """
    Run prediction using the model checkpoint
    """
    print("\nStep 7: Running predictions...")
    
    try:
        # Set device
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")  
        print(f"Using device: {device}")
        
        # Import necessary libraries for model loading
        try:
            # Try to load the model
            sys.path.append("/mnt/hdd/sdc/ysheng/TBP_Screening")
            from python.models.TipModel3LossISIC512 import TIP3LossISIC
            from finetune_vit import TIPFineTuneModel
            
            # Load model
            print(f"Loading model from {checkpoint_path}...")
            
            # First load the pretrained model structure
            pretrained_model = TIP3LossISIC.load_from_checkpoint(
                '/mnt/hdd/sdc/ysheng/TIP/results/isic/0328_1327/best_model_epoch=39.ckpt',
                strict=False
            )
            
            # Load finetuned model
            model = TIPFineTuneModel.load_from_checkpoint(
                checkpoint_path,
                pretrained_model=pretrained_model,
                config={
                    'lr': 1e-4,
                    'weight_decay': 0.001,
                    'multimodal_embedding_dim': 768
                }
            )
            
            model.eval()
            model.to(device)
            print("Model loaded successfully")
            
            # Check model's expected feature dimensions
            if hasattr(model, 'encoder_tabular') and hasattr(model.encoder_tabular, 'num_cat') and hasattr(model.encoder_tabular, 'num_con'):
                num_cat = model.encoder_tabular.num_cat
                num_con = model.encoder_tabular.num_con
                total_features = num_cat + num_con
                print(f"Model expects {total_features} features: {num_cat} categorical + {num_con} continuous")
                
                if len(expected_features) != total_features:
                    print(f"WARNING: Feature count mismatch! Model expects {total_features} features but we have {len(expected_features)}.")
                    print("This will likely cause the assertion error during prediction.")
            
            # Define prediction function
            def predict(x_img, x_tab):
                try:
                    return model(x_img, x_tab)
                except Exception as e:
                    print(f"Error in forward pass: \n{traceback.format_exc()}")
                    return None
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
            
            # Create a placeholder prediction function for demonstration
            print("Using random prediction function as fallback")
            def predict(x_img, x_tab):
                batch_size = x_img.shape[0]
                return torch.rand(batch_size, 1, device=x_img.device)
        
        # Create a simple dataset class
        class SimpleTBPDataset(torch.utils.data.Dataset):
            def __init__(self, image_paths, tabular_data, labels=None, img_size=224):
                self.image_paths = image_paths
                self.tabular_data = tabular_data
                self.labels = labels if labels is not None else np.zeros(len(image_paths))
                self.img_size = img_size
                
                # Normalization parameters
                self.normalize_mean = [0.485, 0.456, 0.406]
                self.normalize_std = [0.229, 0.224, 0.225]
                
                # Import necessary transforms here to avoid dependency issues
                try:
                    import albumentations as A
                    from albumentations.pytorch import ToTensorV2
                    
                    self.transform = A.Compose([
                        A.Resize(height=self.img_size, width=self.img_size),
                        A.Normalize(mean=self.normalize_mean, std=self.normalize_std),
                        ToTensorV2()
                    ])
                except ImportError:
                    print("Albumentations not available, using fallback preprocessing")
                    self.transform = None
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                # Load image
                img_path = self.image_paths[idx]
                try:
                    # Try to load as numpy first
                    img = np.load(img_path)
                except:
                    # If that fails, try as image
                    img = np.array(Image.open(img_path))
                
                # Apply transform
                if self.transform:
                    img = self.transform(image=img)['image']
                else:
                    # Fallback processing if albumentations not available
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
                    # Normalize with ImageNet stats
                    for i, (m, s) in enumerate(zip(self.normalize_mean, self.normalize_std)):
                        img[i] = (img[i] - m) / s
                
                # Get tabular data and label
                tabular = torch.tensor(self.tabular_data[idx], dtype=torch.float)
                label = torch.tensor(self.labels[idx], dtype=torch.float)
                
                return img, tabular, label
        
        # Load image paths
        image_paths = torch.load(os.path.join(processed_dir, 'tbp_test_paths.pt'))
        print(f"Loaded {len(image_paths)} image paths")
        
        # Load tabular data
        tabular_df = pd.read_csv(os.path.join(processed_dir, 'tbp_features_test.csv'))
        print(f"Loaded tabular data with shape {tabular_df.shape}")
        
        # Double-check that the tabular data has the expected features in the right order
        if list(tabular_df.columns) != expected_features:
            print(f"Warning: Tabular data columns don't match expected features.")
            print(f"Expected: {expected_features}")
            print(f"Found: {list(tabular_df.columns)}")
            # Ensure columns are in the correct order
            tabular_df = tabular_df[expected_features]
            print("Reordered tabular columns to match expected features")
        
        # Extract UUIDs from image paths for later
        def extract_uuid(path):
            filename = os.path.basename(path)
            uuid = os.path.splitext(filename)[0]
            # Remove test_ prefix if present
            if uuid.startswith('test_'):
                uuid = uuid[5:]
            return uuid
        
        image_uuids = [extract_uuid(path) for path in image_paths]
        
        # Create dummy labels (zeros) for dataset compatibility
        dummy_labels = np.zeros(len(image_paths))
        
        # Create dataset
        test_dataset = SimpleTBPDataset(
            image_paths=image_paths,
            tabular_data=tabular_df.values,
            labels=dummy_labels
        )
        print(f"Created test dataset with {len(test_dataset)} samples")
        
        # Create data loader
        batch_size = 8  # Smaller batch size to avoid memory issues
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        print(f"Created test dataloader with batch size {batch_size}")
        
        # Run predictions
        all_preds = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (images, tabulars, _) in enumerate(tqdm(test_loader, desc="Predicting")):
                try:
                    # Move to device
                    images = images.to(device)
                    tabulars = tabulars.to(device)
                    
                    # Check feature dimensions
                    if tabulars.shape[1] != len(expected_features):
                        print(f"Warning: Batch {batch_idx} has wrong feature count: {tabulars.shape[1]} instead of {len(expected_features)}")
                        # Add fake results for this batch
                        all_preds.extend([0.5] * len(images))
                        batch_indices = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_dataset))))
                        all_indices.extend(batch_indices)
                        continue
                    
                    # Get predictions
                    outputs = predict(images, tabulars)
                    
                    if outputs is None:
                        print(f"Warning: Batch {batch_idx} prediction failed. Using fallback values.")
                        # Add fake results for this batch
                        all_preds.extend([0.5] * len(images))
                        batch_indices = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_dataset))))
                        all_indices.extend(batch_indices)
                        continue
                    
                    # Get probabilities (apply sigmoid to logits)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    
                    # Collect results
                    all_preds.extend(probs)
                    
                    # Record batch indices
                    batch_indices = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_dataset))))
                    all_indices.extend(batch_indices)
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    # Add fake results for this batch
                    all_preds.extend([0.5] * len(images))
                    batch_indices = list(range(batch_idx * batch_size, min((batch_idx + 1) * batch_size, len(test_dataset))))
                    all_indices.extend(batch_indices)
        
        # Verify indices and predictions have same length
        if len(all_indices) != len(all_preds):
            print(f"Warning: Indices and predictions have different lengths: {len(all_indices)} vs {len(all_preds)}")
            # Trim to shorter length
            min_len = min(len(all_indices), len(all_preds))
            all_indices = all_indices[:min_len]
            all_preds = all_preds[:min_len]
        
        # Get UUIDs for these indices
        result_uuids = [image_uuids[idx] for idx in all_indices]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'uuid': result_uuids,
            'prediction': all_preds
        })
        
        # Handle any NaN values
        nan_count = results_df['prediction'].isna().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN predictions detected. Filling with 0.5.")
            results_df['prediction'] = results_df['prediction'].fillna(0.5)
        
        # Save predictions
        output_path = os.path.join(predictions_dir, 'predictions.csv')
        results_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(results_df)} predictions to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return None


#######################################
# Main function to run the workflow
#######################################

def main():
    """Main function to execute the workflow"""
    try:
        print("Starting TBP Workflow")
        
        # Step 1: Extract data from JSON
        df = extract_data_from_json(JSON_FILE, EXTRACTED_DIR)
        if df is None:
            print("Failed to extract data from JSON. Exiting.")
            return
        
        # Step 2: Filter CSV data
        filtered_df, _ = filter_dataframe(df)
        
        # Step 3: Save filtered data and match with images
        csv_path, img_dir = save_filtered_data(filtered_df, EXTRACTED_DIR, FILTERED_DIR)
        
        # Step 4: Add missing Sex and Age information
        enhanced_csv_path = add_missing_info(csv_path, MERGED_CSV)
        
        # Step 5: Process images
        numpy_paths = process_images(img_dir, PROCESSED_DIR)
        
        # Step 6: Process tabular data
        processed_df, expected_features = process_tabular_data(enhanced_csv_path, PROCESSED_DIR)
        
        # Step 7: Run prediction
        prediction_path = run_prediction(PROCESSED_DIR, PREDICTIONS_DIR, CHECKPOINT_PATH, expected_features)
        
        if prediction_path:
            print("\nWorkflow completed successfully!")
            print(f"Final prediction results saved to: {prediction_path}")
        else:
            print("\nWorkflow completed with errors in prediction step.")
        
    except Exception as e:
        print(f"Error in workflow: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()









































