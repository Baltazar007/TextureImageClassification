from flask import Flask, jsonify, request
import joblib
from extraction import extract_features
import numpy as np
import tensorflow as tf
import cv2
import os
from sklearn import metrics
from collections import defaultdict

app = Flask(__name__)

models = {
    'decision_tree_original': joblib.load('dtc_original.pkl'),
    'svm_original': joblib.load('svm_original.pkl'),
    'catboost_original': joblib.load('catboost_original.pkl'),
    'knn_original': joblib.load('knn_original.pkl'),
    'decision_tree_std': joblib.load('dtc_std.pkl'),
    'svm_std': joblib.load('svm_std.pkl'),
    'catboost_std': joblib.load('catboost_std.pkl'),
    'knn_std': joblib.load('knn_std.pkl'),
    'decision_tree_mm': joblib.load('dtc_mm.pkl'),
    'svm_mm': joblib.load('svm_mm.pkl'),
    'catboost_mm': joblib.load('catboost_mm.pkl'),
    'knn_mm': joblib.load('knn_mm.pkl')
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the image
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(
            image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Verify the image size
        if image.shape[0] < 32 or image.shape[1] < 32:
            return jsonify({'error': 'Image size must be at least 32x32 pixels'})

        # Extract features
        features = extract_features(image)

        # Load the models
        results = defaultdict(dict)
        for model_name, model in models.items():
            # Make predictions
            predictions = model.predict(features.reshape(1, -1))
            predicted_class_indices = np.argmax(predictions, axis=1)

            # Get the class labels
            label_names = os.listdir(os.path.join(os.getcwd(), 'data'))
            label_names.sort()

            # Compute metrics
            accuracy = metrics.accuracy_score(
                [label_names.index(os.path.basename(
                    os.path.dirname(image_file.filename)))],
                predicted_class_indices)
            precision = metrics.precision_score(
                [label_names.index(os.path.basename(
                    os.path.dirname(image_file.filename)))],
                predicted_class_indices, average='micro')
            recall = metrics.recall_score(
                [label_names.index(os.path.basename(
                    os.path.dirname(image_file.filename)))],
                predicted_class_indices, average='micro')

            # Store the results
            results[model_name]['class'] = label_names[predicted_class_indices[0]]
            results[model_name]['accuracy'] = accuracy
            results[model_name]['precision'] = precision
            results[model_name]['recall'] = recall

        # Return the results
        return jsonify(results)
    except:
        return jsonify({'error': 'Unexpected error.'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
