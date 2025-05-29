from flask import Flask, request, jsonify
from app.models.lstm_model import StandardModel
from app.models.custom_model import CustomModel
from app.utils.data_processing import DataProcessor
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# تهيئة النماذج
standard_model = StandardModel()
custom_model = CustomModel()
data_processor = DataProcessor()

@app.route('/predict/standard', methods=['POST'])
def predict_standard():
    data = request.json['data']
    prediction = standard_model.predict(data)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/predict/custom', methods=['POST'])
def predict_custom():
    if 'file' not in request.files:
        return jsonify({'error': 'No model file uploaded'})
    
    model_file = request.files['file']
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', model_file.filename)
    model_file.save(model_path)
    
    custom_model.load(model_path)
    prediction = custom_model.predict(request.json['data'])
    return jsonify({'prediction': prediction.tolist()})

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json['data']
    epochs = request.json.get('epochs', 50)
    
    trained_model = custom_model.train(data, epochs)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'models', f'custom_{int(time.time())}.h5')
    trained_model.save(model_path)
    
    return jsonify({'message': 'Model trained successfully', 'path': model_path})

if __name__ == '__main__':
    app.run(debug=True)
