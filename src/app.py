from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import joblib
import os
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = None
pca = None
scaler = None

try:
    # Try several likely locations relative to this file and cwd
    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_paths = [
        os.path.join(current_dir, 'face_recognition_pipeline.pkl'),
        os.path.join(current_dir, '..', 'face_recognition_pipeline.pkl'),
        os.path.join(current_dir, '..', '..', 'face_recognition_pipeline.pkl'),
        'face_recognition_pipeline.pkl'
    ]
    found = False
    for mp in candidate_paths:
        if os.path.exists(mp):
            print(f"Found model file at: {mp}")
            pipeline = joblib.load(mp)
            print(f"Loaded pipeline object of type: {type(pipeline)} from {mp}")
            if isinstance(pipeline, dict):
                print(f"Pipeline keys: {list(pipeline.keys())}")
                model = pipeline.get('model')
                pca = pipeline.get('pca')
                scaler = pipeline.get('scaler')
                print(f"✓ Extracted components: model={type(model)}, pca={type(pca)}, scaler={type(scaler)}")
            else:
                model = pipeline
                print(f"✓ Model loaded successfully as object: {type(model)}")
            found = True
            break
    if not found:
        print(f"⚠ Model file not found in any candidate paths: {candidate_paths}")
except Exception as e:
    print(f"✗ Error loading model: {str(e)}")


def get_confidence(model_obj, X):
    """Return confidence percentage for prediction.
    Tries predict_proba, otherwise decision_function (sigmoid), otherwise 0.
    """
    try:
        if hasattr(model_obj, 'predict_proba'):
            probs = model_obj.predict_proba(X)
            # For binary classification, take max prob
            return float(np.max(probs) * 100)
        elif hasattr(model_obj, 'decision_function'):
            scores = model_obj.decision_function(X)
            # decision_function can return array; handle binary/multi
            if np.ndim(scores) == 1:
                # sigmoid to map to (0,1)
                probs = 1 / (1 + np.exp(-scores))
                return float(np.max(probs) * 100)
            else:
                # multiclass scores -> apply softmax
                exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp / np.sum(exp, axis=1, keepdims=True)
                return float(np.max(probs) * 100)
        else:
            return 0.0
    except Exception as e:
        print(f"Error computing confidence: {e}")
        return 0.0

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    img = cv2.resize(img, (100, 100))  # same size as training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.flatten()
    img = img.reshape(1, -1).astype(np.float32)
    
    # Apply PCA transformation if available
    if pca is not None:
        img = pca.transform(img)
    
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')

    if not file:
        return jsonify({"error": "No image file provided"}), 400

    if model is None:
        return jsonify({"error": "Model not loaded. Check if face_recognition_pipeline.pkl exists."}), 500

    try:
        # sanitize filename and save
        filename = file.filename.replace(' ', '_').replace('\\', '_')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Image saved to: {filepath}")

        # Preprocess image
        processed_img = preprocess_image(filepath)
        print(f"Processed image shape: {processed_img.shape}")

        # Make prediction
        prediction = model.predict(processed_img)[0]
        confidence = float(np.max(model.predict_proba(processed_img)) * 100)

        pred_label = "Arnold Schwarzenegger" if int(prediction) == 1 else "Other Person"
        print(f"Prediction: {pred_label}, Confidence: {confidence}%")

        return jsonify({
            "success": True,
            "prediction": pred_label,
            "prediction_value": int(prediction),
            "confidence": round(confidence, 2),
            "image_path": filepath
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Prediction error: {str(e)}"
        print(f"✗ {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)
