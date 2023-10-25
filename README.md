# Laporan Proyek Machine Learning
### Nama : Moch Reki Hadiyanto 
### Nim : 211351083 
### Kelas : Pagi B

## Domain Proyek

Web App yang saya kembangkan ini sebaiknya digunakan oleh/berdampingan dengan seorang profesional agar variabel-variabel yang diinputkan tidak semena-mena dimasukkan begitu saja, Web App ini dikembangkan untuk memudahkan pengguna dalam menentukan proses pengobatan selanjutnya tergantung dari hasil output Web App ini. Namun jika anda bukanlah seorang profesional sebaiknya mendatangi langsung ahlinya.

## Business Understanding

Memungkinkan seorang profesional/dokter bekerja lebih cepat dan tepat, dengan itu lebih banyak pasien akan mendapatkan penanganan langsung dari seorang dokter.

### Problem Statements
- Semakin banyaknya orang yang didiagnosa mengidap diabetes dikarenakan pola hidup modern yang tidak teratur/buruk, maka semakin banyak pula pasien yang harus ditangani oleh ahli profesional

### Goals
- Memudahkan dokter/ahli profesional dalam menentukan pengobatan selanjutnya bagi pasien yang mengidap/tidak mengidap penyakit diabetes dengan hasil yang dikeluarkan oleh Web App.

## Data Understanding
Diabetes prediction dataset adalah datasets yang saya gunakan, saya dapatkan dari kaggle.com. Data-data yang terdapat di dalam datasets ini didapatkan dari data medikal pasien (tentunya dikumpulkan dengan izin mereka) alias data riil. Datasets ini mengandung 9 Attribut(Kolom) dan 100,000 data(baris) pada saat sebelum pemrosesan data cleasing dan EDA.
<br> 

[Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

### Variabel-variabel pada Diabetes prediction dataset adalah sebagai berikut:
- gender : merupakan jenis kelamin pasien. [Female, Male]
- age    : merupakan umut pasien. [Numbers, min: 0, max: 80]
- hypertension : merupakan kondisi medis yang dialami oleh pasien dimana tekanan darah semakin meningkat. [1: True, 0: False]
- heart_disease: merupakan kondisi medis yang dialami oleh pasien dimana mereka memiliki penyakit jantung. [1: True, 0: False]
- smoking_history : menunjukkan mengenai riwayat merokok pasien. [No info, never, former, current, not current]
- bmi     : merupakan pengukuran lemak tubuh berdasarkan berat dan tinggi badan [Float, min: 10, max: 95.7]
- HbA1c_level  : merupakan pengukuran rata-rata level gula pada darah dari 2-3 bulan yang lalu.[Float, min: 3.5, max: 9]
- blood_glucose_level : merupakan pengukuran rata-rata level gula pada darah dari 1-3 hari yang lalu.[Numbers, min: 80, max: 300]
- diabetes  : menunjukkan apakah pasien mengidap diabetes atau tidak. [1: True, 0: False]

## Data Preparation
Untuk data preparation ini saya melakukan EDA (Exploratory Data Analysis) terlebih dahulu, lalu melakukan proses data cleansing agar model yang dihasilkan memiliki score yang lebih tinggi.

Sebelum memulai data preparation, kita akan mendownload datasets dari kaggle yang akan kita gunakan, langkah pertama adalah memasukkan token kaggle,
``` bash
from google.colab import files
files.upload()
```
Lalu kita harus membuat folder untuk menampung file kaggle yang tadi telah diupload,
``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu, download datasets menggunakan code dibawah ini, 
``` bash
!kaggle datasets download -d iammustafatz/diabetes-prediction-dataset
```
Setelah download telah selesai, langkah selanjutnya adalah mengektrak file zipnya kedalam sebuah folder,
``` bash
!unzip diabetes-prediction-dataset.zip -d diabetes_prediction
!ls diabetes_prediction
```
Datasets telah diekstrak, seharusnya sekarang ada folder yang bernama diabetes_prediction dan di dalamnya terdapat file dengan ektensi .csv, <br>
Langkah selanjutnya adalah mengimport library yang dibutuhkan untuk melaksanakan data Exploration, data visualisation, dan data cleansing,
``` bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
Selanjutnya, mari baca file .csv yang tadi kita ekstrak, lalu melihat 5 data pertama yang ada pada datasets,
``` bash
data = pd.read_csv("diabetes_prediction/diabetes_prediction_dataset.csv")
data.head()
```
Lalu untuk melihat jumah data, mean data, data terkecil dan data terbesar bisa dengan kode ini,
``` bash
data.describe()
```
Untuk melihat typedata yang digunakan oleh masing-masing kolom bisa menggunakan kode ini,
``` bash
data.info()
```
Selanjutnya kita akan melihat korelasi antar kolomnya,
``` bash
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True)
```
![download](https://github.com/RexBaee/prediksi_diabetes/assets/130348460/a273a2b8-7c60-4732-9a96-5d6ae30167de)
<br>
Korelasi antar kolom numerik terlihat aman namun saya merasa terlalu banyak data yang tidak berkaitan erat, selanjutnya melihat apakah di dalam datasetsnya terdapat nilai null,
``` bash
sns.heatmap(data.isnull())
```
![download](https://github.com/RexBaee/prediksi_diabetes/assets/130348460/0669c8e2-47d4-4f79-9992-fb90ee407ad5) <br>
Semuanya merah yang menandakan bahwa datasetsnya tidak memiliki data null di dalamnya, selanjutnya akan melihat apakah ada data duplikasi,
``` bash
data[data.duplicated()]
```
Terdapat keterangan bahwa 3854 baris merupakan data duplikasi, untuk menghapus data duplikasinya bisa menggunakan kode berikut,
``` bash
data.drop_duplicates(inplace=True)
```
selanjutnya kita bisa periksa jumlah data yang tersisa dengan kode berikut,
``` bash
data.info()
```
96146 baris data yang akan digunakan untuk diproses menjadi model nantinya. Selanjutnya memisahkan data-data unique menjadi integer, seperti berikut, <br>
Pertama harus mencari terlebih dahulu data unique yang bertype string(object),
``` bash
pd.unique(data.smoking_history)
pd.unique(data.gender)
```
Selanjutnya membuat fungsi untuk menjadikan masing-masing string menjadi sebuah integer,
``` bash
def change_string_to_int(column):
    variables=pd.unique(data[column])
    for item in range(variables.size):
        data[column]=[item if each==variables[item] else each for each in data[column]]
    return data[column]
```
Lalu mengimplementasikan fungsi yang tadi sudah dibuat dan memasukkan hasilnya pada kolom yang sesuai, 
``` bash
data["gender"]=change_string_to_int("gender")
data["smoking_history"]=change_string_to_int("smoking_history")
```
Langkah selanjutnya adalah melihat korelasi antara semua kolom,
``` bash
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(data.corr(numeric_only=True), annot=True, linewidths=.5, fmt= '.1f',ax=ax,)
plt.show()
```
![download](https://github.com/RexBaee/prediksi_diabetes/assets/130348460/bcdfb4d3-bce8-4aa9-a376-a1b11aed2f8b)
Bisa dilihat bahwa korelasi kolom gender dengan kolom yang lainnya sangatlah rendah, ini bisa mempengaruhi model kita, maka sebaiknya untuk dihilangkan saja,
``` bash
data.drop("gender",axis=1,inplace=True)
```
Dan proses EDA dan data cleaning sudah diselesaikan. Selanjutnya adalah membuat modelnya.

## Modeling
Model machine learning yang akan digunakan disini adalah logistic regression, langkah pertama yang harus kita lakukan adalah memasukkan semua library yang akan digunakan pada saat proses pembuatan model,
``` bash
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score,confusion_matrix
```
Lalu membuat variable yang akan menampung fitur-fitur dan targetnya,
```
x = data.drop("diabetes", axis=1)
y = data.diabetes
```
Langkah selanjutnya adalah membuat train test split, dengan persentase 30% test dan 70% train
``` bash
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
```
Lalu kita akan reshape bentuk arraynya,
``` bash
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
```
Dan selanjutnya adalah mengimplementasikan model Logistic Regression dan melihat tingkat akurasinya,
``` bash
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 200,)
print("test accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_test, y_test)))
print("train accuracy: {} ".format(logreg.fit(x_train, y_train).score(x_train, y_train)))
```
Score yang kita dapatkan adalah 95.85% untuk test dan 95.89% untuk train, lalu akhirnya kita akan uji dengan data inputan kita sendiri,
``` bash
data = np.array([[80, 0, 1, 0, 25.19, 6.6, 140]])
print(logreg.predict(data))
```
Dan hasilnya adalah 0 yang artinya tidak berpotensi mengidap diabetes. Sebelum mengakhiri ini, kita harus ekspor modelnya menggunakan pickle agar nanti bisa digunakan pada media lain.
``` bash
import pickle
filename = "prediksi_diabetes.sav"
pickle.dump(logreg,open(filename,'wb'))
```

## Evaluation
Matrik evaluasi yang saya gunakan disini adalah confusion matrix, karena ianya sangat cocok untuk kasus pengkategorian seperti kasus ini. Dengan membandingkan nilai aktual dengan nilai prediksi, kita bisa melihat jumlah hasil prediksi saat model memprediksi diabetes dan nilai aktual pun diabetes, serta melihat saat model memprediksi diabetes sedangkan data aktualnya tidak diabetes.
``` bash
y_pred = logreg.fit(x_train, y_train).predict(x_test)
cm = confusion_matrix(y_test,y_pred)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['diabetes','not_diabetes']); ax.yaxis.set_ticklabels(['diabetes','not_diabetes']);
```
![download](https://github.com/RexBaee/prediksi_diabetes/assets/130348460/852cf0cc-7bb1-468a-8514-23bb03eccb80) <br>
Disitu terlihat jelas bahwa model kita berhasil memprediksi nilai diabetes yang sama dengan nilai aktualnya sebanyak 26017 data.

## Deployment
[Diabetes Prediction App](https://prediksidiabetes-reki.streamlit.app/)

![image](https://github.com/RexBaee/prediksi_diabetes/assets/130348460/e9be203b-a687-4488-82cf-63846da7f3df)
