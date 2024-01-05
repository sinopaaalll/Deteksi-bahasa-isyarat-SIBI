import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Mengimpor modul yang diperlukan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Menginisialisasi objek Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Menentukan direktori tempat data disimpan
DATA_DIR = './data1'

# Membuat list kosong untuk menampung data dan label
data = []
labels = []

# Iterasi melalui setiap direktori dalam DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
    # for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:
        data_aux = []
        x_ = []
        y_ = []

        # Membaca gambar dari setiap direktori
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#         plt.figure()
#         plt.imshow(img_rgb)
# plt.show()


        # Memproses gambar dengan model Hands
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            # Mengambil landmark tangan dan menghitung posisi relatif terhadap titik referensi
            for hand_landmarks in results.multi_hand_landmarks:
                # Melakukan deteksi landmark tangan
                # ...
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Menambahkan data dan label ke dalam list
            data.append(data_aux)
            labels.append(dir_)

# Membuka file pickle dan menyimpan data serta label ke dalamnya
f = open('data1.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
