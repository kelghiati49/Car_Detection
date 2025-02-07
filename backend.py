from starlette.websockets import WebSocketDisconnect
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from Car_detection import car_detect
import numpy as np
import base64
import cv2




'''commande a utilise pour execute le backend :
      uvicorn backend:app --reload --port 8000
'''
app = FastAPI()

@app.get("/")
def greet():
    return {"message": "bonjour"}

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

@app.websocket("/video_stream")
async def video_stream(websocket: WebSocket):
    """ Gère le streaming de la vidéo via WebSocket """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            # Lire la vidéo envoyée par le frontend
            video_np = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(video_np, cv2.IMREAD_COLOR)

            # Appliquer la détection
            processed_frame = car_detect(frame)

            if processed_frame is None:
                print("Erreur: Frame non traitée")
                continue

            # Convertir la chaîne Base64 en image NumPy
            buffer = base64.b64decode(processed_frame)
            nparr = np.frombuffer(buffer, np.uint8)
            processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame_bytes = buffer.tobytes()

            # Envoyer l'image traitée encodée en base64
            await websocket.send_bytes(processed_frame_bytes)
            
            # await asyncio.sleep(0.05)  # Simuler un délai pour éviter la surcharge CPU

    except WebSocketDisconnect:
        print("Client déconnecté")
    except Exception as e:
        print("Erreur WebSocket :", e)
    finally:
        await websocket.close()



