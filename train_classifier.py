import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Memuat data dari file pickle
data_dict = pickle.load(open('./data1.pickle', 'rb'))

# Membagi data menjadi data fitur (data) dan label (labels)
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Membagi data menjadi data latih dan data uji
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Menginisialisasi model Random Forest Classifier
model = RandomForestClassifier()

# Melatih model menggunakan data latih
model.fit(x_train, y_train)

# Memprediksi label menggunakan data uji
y_predict = model.predict(x_test)

# Menghitung skor akurasi model
score = accuracy_score(y_predict, y_test)

# Menampilkan hasil akurasi
print('{}% of samples were classified correctly !'.format(score * 100))

# Menyimpan model ke dalam file pickle
f = open('model2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
