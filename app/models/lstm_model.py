from tensorflow.keras.models import load_model
import os

class StandardModel:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), '../../standard_model.h5')
        self.model = load_model(model_path)
        self.sequence_length = self.model.input_shape[1]

    def predict(self, data):
        # منطق التنبؤ باستخدام النموذج المدرب
        return self.model.predict(data)
