from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class CustomModel:
    def __init__(self):
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=(60, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=50, verbose=0)
        return self.model
