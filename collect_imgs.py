import os
import cv2

# Inisialisasi direktori untuk menyimpan data gambar
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Jumlah kelas yang akan dikumpulkan datanya
number_of_classes = 26
# Jumlah data gambar yang akan dikumpulkan untuk setiap kelas
dataset_size = 100

# Mengambil video dari perangkat menggunakan kamera
cap = cv2.VideoCapture(0)

# Iterasi untuk setiap kelas yang akan dikumpulkan datanya
for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    # Menampilkan instruksi dan menunggu pengguna menekan tombol 'Q' untuk memulai pengambilan gambar
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    # Mengambil data gambar sebanyak dataset_size dan menyimpannya ke dalam direktori yang sesuai dengan kelasnya
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

# Mematikan kamera dan menutup semua jendela tampilan
cap.release()
cv2.destroyAllWindows()
