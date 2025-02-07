import streamlit as st
import numpy as np
import websockets
import asyncio
import base64
import cv2


# ajout de l'image au background
def get_base64(file_path):
    with open(file_path, "rb") as file:
        data = base64.b64encode(file.read()).decode()
    return data
img_base64 = get_base64("car_1.jfif")  # Image de fond
page_bg_img = f"""
<style>
    .stApp {{
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: cover;
        background-position: center;
      }}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("# Détection de véhicules et reconnaissance de plaques")


# st.title("Détection de plaques en temps réel")

uploaded_file = st.file_uploader("Téléchargez une vidéo", type=["mp4"])
if uploaded_file is not None:
    st.video(uploaded_file)  # Affiche la vidéo originale
    
    # Convertir la vidéo en flux de frames
    cap = cv2.VideoCapture(uploaded_file.name)
    st.markdown("## Vidéo traitée")
    # Connexion WebSocket au backend
    async def stream_video():
        async with websockets.connect("ws://localhost:8000/video_stream") as websocket:
            frame_placeholder = st.empty()  # Placeholder pour afficher les images
            while cap.isOpened():
                ret, im = cap.read()
                if not ret:
                    break
                
                # Convertir la frame en bytes et envoyer au backend
                _, buffer = cv2.imencode(".jpg", im)
                await websocket.send(buffer.tobytes())

                # Recevoir l'image traitée et la décoder
                try:
                    response = await websocket.recv()
                    print("Réponse reçue :", response)
                except Exception as e:
                    print("Erreur WebSocket :", e)
                    break
                processed_frame = np.frombuffer(response, np.uint8)
                processed_frame = cv2.imdecode(processed_frame, cv2.IMREAD_COLOR)

                # Afficher la frame traitée
                frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

            cap.release()

    asyncio.run(stream_video())
    
