import pickle
import string
import cv2
import mediapipe as mp
import numpy as np
from sklearn.decomposition import PCA

# Memuat model dari file pickle
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Menginisialisasi kamera
cap = cv2.VideoCapture(0)

# Menginisialisasi objek Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Membuat dictionary dari A hingga Z
labels_dict = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

# Memuat PCA dari model
pca = model_dict.get('pca', None)

# Menambahkan variabel untuk menyimpan jumlah frame yang diproses dan yang diklasifikasikan dengan benar
total_frames = 0
correctly_classified_frames = 0

while True:
    # Inisialisasi variabel untuk menyimpan data tangan
    data_aux = []
    x_ = []
    y_ = []

    # Membaca frame dari kamera
    ret, frame = cap.read()

    # Mendapatkan dimensi frame
    H, W, _ = frame.shape

    # Mengubah format frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Memproses frame menggunakan objek Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Mendeteksi landmark tangan dan menggambar pada frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Inisialisasi variabel koordinat batas
            x1, y1 = W, H
            x2, y2 = 0, 0

            for hand_landmarks in results.multi_hand_landmarks:
                # Mendapatkan koordinat tangan
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Normalisasi koordinat tangan
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = min(x1, int(min(x_) * W) - 10)
                y1 = min(y1, int(min(y_) * H) - 10)
                x2 = max(x2, int(max(x_) * W) - 10)
                y2 = max(y2, int(max(y_) * H) - 10)

            # Memastikan data_aux memiliki dimensi yang benar
            if len(data_aux) != 42:
                continue

            # Melakukan transformasi PCA jika tersedia
            if pca:
                data_reduced = pca.transform([data_aux])
            else:
                data_reduced = [data_aux]

            # Melakukan prediksi menggunakan model
            prediction = model.predict(data_reduced)

            # Mendapatkan karakter yang diprediksi berdasarkan label
            predicted_character = labels_dict[int(prediction[0])]

            # Menampilkan karakter yang diprediksi pada frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

         
            # Meningkatkan jumlah frame yang diproses
            total_frames += 1

            # Memeriksa apakah prediksi benar dan meningkatkan jumlah frame yang diklasifikasikan dengan benar
            expected_character = labels_dict 
            if predicted_character == expected_character:
                correctly_classified_frames += 1

            # Menghitung dan menampilkan akurasi pada setiap iterasi
            accuracy = (correctly_classified_frames / total_frames) * 100
            print(f'Accuracy: {accuracy:.2f}% for {total_frames} frames (Expected: {expected_character}, Predicted: {predicted_character})')
                


    # Menampilkan frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela tampilan
cap.release()
cv2.destroyAllWindows()
