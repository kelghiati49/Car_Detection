
from KalmanFilter import KalmanFilter
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
import base64
import cv2
import re
import os



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def car_detect(frame):
    #Initialize our Models
    model = YOLO("./best.pt")
    ocr = PaddleOCR(use_angle_cls = True, use_gpu = False)
    kalman = KalmanFilter()

    #Initialize the car_searched regstration plate number
    car_searched = "85040"

    
    def comparaison(label_ocr, label_cherche):
    # Comparaison des trois premiers éléments detectes de la plaque
        return label_ocr[:3] == label_cherche[:3]

    def paddle_ocr(frame, x1, y1, x2, y2):
        frame = frame[y1:y2, x1: x2]
        result = ocr.ocr(frame, det=False, rec = True, cls = False)
        text = ""
        for r in result:
            #print("OCR", r)
            scores = r[0][1]
            if np.isnan(scores):
                scores = 0
            else:
                scores = int(scores * 100)
            if scores > 50:
                text = r[0][0]
        pattern = re.compile('[\W]')
        text = pattern.sub('', text)
        text = text.replace("???", "")
        text = text.replace("O", "0")
        text = text.upper()
        return str(text)


    results = model.predict(frame, conf = 0.45)
    for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = paddle_ocr(frame, x1, y1, x2, y2)
                    xc = (x1 + x2)/2
                    yc = (y1 + y2)/2
                    label_ocr = label

                    # Utilisation de la fonction comparaison
                    if comparaison(label_ocr, car_searched):
                        if not kalman.is_initialized:
                                kalman.initialize(xc, yc)
                        else:
                                kalman.correct(xc, yc)

                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(frame, (x1, y1), c2, (255, 0, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

    if kalman.is_initialized:
            predicted_x, predicted_y = kalman.predict()
            cv2.circle(frame,(int (predicted_x),int(predicted_y)),5, (255, 200, 0), 2)

        # #convertir l'image en JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    frame = base64.b64encode(buffer).decode('utf-8')
    
    return frame