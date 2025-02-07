import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        # Initialiser le filtre de Kalman
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        # Initialiser la position
        self.is_initialized = False

    def initialize(self, coordX, coordY):
        self.kf.statePre = np.array([[coordX], [coordY], [0], [0]], np.float32)
        self.is_initialized = True

    def correct(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]], np.float32)
        self.kf.correct(measured)

    def predict(self):
        predicted = self.kf.predict()
        return predicted[0][0], predicted[1][0]

