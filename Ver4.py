import cv2 as cv
import numpy as np
import os
import mediapipe as mp
import time
import math
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Inisialisasi model YOLOv8n dari Ultralytics
yolo_model = YOLO('best100epoch.pt')

# Inisialisasi model Mediapipe untuk mendeteksi pose dan landmark
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kelas yang digunakan untuk LSTM: Kanan, Maju, Stop, Mundur, Kiri
actions = np.array(['Kanan', 'Maju', 'Stop', 'Mundur', 'Kiri'])

sequence = []
predictions = []

# Parameter focal length dan tinggi objek nyata
focal_length_pixel = 481
tinggi_objek_nyata = 181

# Fungsi untuk menghitung jarak menggunakan tinggi bounding box
def hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata):
    if tinggi_bounding_box == 0:
        return float('inf')
    jarak = (tinggi_objek_nyata * focal_length_pixel) / tinggi_bounding_box
    return jarak / 100  # Konversi ke meter

# Fungsi untuk deteksi pose dan landmark menggunakan MediaPipe
def media_pipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results

# Fungsi menggambar landmark pada gambar
def draw_land_marks(image, results):
    custom_pose_connections = list(mp_pose.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, connections=custom_pose_connections)
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Fungsi untuk ekstraksi dan normalisasi keypoints
def extract_keypoints_normalize(results):
    midpoint_shoulder_x, midpoint_shoulder_y = 0, 0
    shoulder_length = 1

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]

        midpoint_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        midpoint_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        shoulder_length = math.sqrt(
            (left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)

        selected_pose_landmarks = results.pose_landmarks.landmark[11:23]
        pose = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                          (res.y - midpoint_shoulder_y) / shoulder_length] for res in selected_pose_landmarks]).flatten()
    else:
        pose = np.zeros(12 * 2)

    if results.left_hand_landmarks:
        left_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                               (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 2)

    if results.right_hand_landmarks:
        right_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length,
                                (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 2)

    return np.concatenate([pose, left_hand, right_hand])

# Fungsi untuk memuat model LSTM yang sudah dilatih
def load_lstm_model():
    model = load_model('agung.keras')  # Ganti dengan path model LSTM kamu
    return model

# Muat model
lstm_model = load_lstm_model()

# Inisialisasi dua kamera
cap_lstm = cv.VideoCapture(0)  # Kamera 1 untuk LSTM
cap_yolo = cv.VideoCapture(1)  # Kamera 2 untuk YOLO

# Set resolusi untuk kamera YOLO
cap_yolo.set(cv.CAP_PROP_FRAME_WIDTH, 1366)
cap_yolo.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

sequence = []
threshold = 0.4

stop_signal = False  # Variabel untuk mendeteksi apakah ada obstacle
jarak_yolo = 0  # Variabel untuk menyimpan jarak YOLO

while cap_lstm.isOpened() and cap_yolo.isOpened():
    ret_lstm, frame_lstm = cap_lstm.read()
    ret_yolo, frame_yolo = cap_yolo.read()

    # Deteksi obstacle menggunakan YOLO (kamera YOLO)
    hasil_yolo = yolo_model(frame_yolo)
    stop_signal = False  # Reset stop signal untuk setiap frame
    jarak_yolo = 0  # Reset jarak YOLO setiap frame

    for hasil in hasil_yolo:
        for box in hasil.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Mendapatkan koordinat bounding box
            tinggi_bounding_box = y2 - y1

            # Hitung jarak berdasarkan tinggi bounding box
            jarak_yolo = hitung_jarak(tinggi_bounding_box, focal_length_pixel, tinggi_objek_nyata)

            # Gambar bounding box di sekitar objek yang terdeteksi
            cv.rectangle(frame_yolo, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Tampilkan jarak dan STOP jika jarak kurang dari 2 meter
            if jarak_yolo < 2:
                cv.putText(frame_yolo, 'STOP', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                stop_signal = True  # Set stop signal menjadi True
                print("STOP")
            else:
                cv.putText(frame_yolo, f'Jarak: {jarak_yolo:.2f}m', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Tampilkan jarak YOLO di pojok kiri atas setiap saat
    cv.putText(frame_yolo, f'Jarak YOLO: {jarak_yolo:.2f}m', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    # Deteksi pose dan landmark menggunakan MediaPipe (kamera LSTM)
    image_lstm, results_lstm = media_pipe_detection(frame_lstm, holistic_model)
    draw_land_marks(image_lstm, results_lstm)

    # Ekstraksi keypoints untuk LSTM
    keypoints = extract_keypoints_normalize(results_lstm)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    # Prediksi jika sequence sudah lengkap (30 frame) dan tidak ada obstacle
    if len(sequence) == 30 and not stop_signal:
        res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
        
        # Jika confidence lebih dari threshold, tampilkan prediksi
        if res[np.argmax(res)] > threshold:
            predicted_class = actions[np.argmax(res)]
            
            # Tampilkan teks berdasarkan prediksi
            if predicted_class == 'Kanan':
                cv.putText(image_lstm, 'Kanan', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
                print("Aksi: Kanan")
            elif predicted_class == 'Maju':
                cv.putText(image_lstm, 'Maju', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                print("Aksi: Maju")
            elif predicted_class == 'Stop':
                cv.putText(image_lstm, 'Stop', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
                print("Aksi: Stop")
            elif predicted_class == 'Mundur':
                cv.putText(image_lstm, 'Mundur', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
                print("Aksi: Mundur")
            elif predicted_class == 'Kiri':
                cv.putText(image_lstm, 'Kiri', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
                print("Aksi: Kiri")
    
    # Tampilkan frame LSTM dan YOLO
    cv.imshow('Kontrol Kursi Roda - LSTM', image_lstm)
    cv.imshow('Deteksi Obstacle - YOLO', frame_yolo)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap_lstm.release()
cap_yolo.release()
cv.destroyAllWindows()
