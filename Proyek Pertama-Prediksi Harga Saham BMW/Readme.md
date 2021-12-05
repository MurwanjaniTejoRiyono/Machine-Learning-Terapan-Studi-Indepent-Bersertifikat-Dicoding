# Markdown Machine Learning Terapan Menggunakan LSTM

## Murwanjani Teho Riyono-M1

### **1. Domain Proyek**
Latar belakang dibuatnya program machine learning ini selain sebagai syarat kelulusan submission 1 pada kelas machine learning terapan dicoding adalah permasalah yang berkaitan dengan trader maupun investor baru. Para trader maupun investor yang masih baru cenderung ragu dalam melakukan penjualan saham mereka. Kebanyakan dari mereka akan berfikir 'apakah saya harus menjualnya sekarang? apakah harga sekarang merupakan harga tertinggi dari saham saya?'. Para trader maupun investor tersebut akan merasa lebih bimbang ketika setelah mereka menjual saham tersebut, justru harga saham melambung tinggi. Oleh karena itu, penting bagi seorang trader maupun investor untuk mengetahui harga tertinggi dari suatu saham dikemudian hari. Sehingga apabila mereka rugi pada hari ini, mereka akan mengetahui kapan dan berapa harga tertinggi dari saham tersebut akan berulang kembali. Perlu diketahui bahwa saham maupun bitcoin memiliki pola yang berulang dalam kurun waktu tertentu.

### **2. Business Understanding**
Para trader maupun investor yang masih baru cenderung ragu dalam melakukan penjualan saham mereka. Kebanyakan dari mereka akan berfikir 'apakah saya harus menjualnya sekarang? apakah harga sekarang merupakan harga tertinggi dari saham saya? besok harga tertingginya berapa ya biar dapat saya jual?'. Oleh karena itu, penting bagi seorang trader maupun investor untuk mengetahui harga tertinggi dari suatu saham setiap harinya. Peran machine learning dalam memprediksi harga tertinggi pada hari esok sangatlah penting dalam kasus ini. Dengan adanya teknologi machine learning kita dapat membuat model yang dapat memprediksi harga tertinggi dari suatu saham pada esok hari maupun beberapa hari kedepan. Hal itu tentunya akan mempermudah trader maupun investor yang masih baru. Metrik digunakan untuk mengevaluasi model machine learning ini adalah Root Mean Square Error (RMSE). Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan. RMSE tidak memiliki satuan.

#### Problem Statements
- Bagaimana seorang yang baru dalam dunia trading dapat memperkirakan harga tertinggi dari suatu saham pada tiap harinya?
- Bagaimana membangun model yang dapat memprediksi harga tertinggi saham setiap harinya dengan akurat ?

#### Goals
- Membuat model Machine Learning yang dapat memprediksi atau memperkirakan harga tertinggi saham dalam skala hari
- Membangun model yang dapat memprediksi harga tertinggi saham setiap harinya dengan akurat

#### Solution statements
Machine learning ini menggunakan algoritma Long Short Term Memory Network (LTSM) dalam memprediksi harga tertingginya. LSTM merupakan tipe Recurrent Neural Network yang dapat mempelajari data historis atau time series. Ia merupakan algoritma deep learning yang kompleks dan dapat mempelajari informasi jangka panjang dengan sangat baik. LSTM sangat cocok untuk mengklasifikasikan, memproses, dan memprediksi deret waktu yang diberikan jeda waktu dengan durasi yang tidak diketahui. Sehingga algoritma ini sangat cocok untuk memprediksi harga tertinggi saham pada beberapa hari berikutnya.

### **3. Data Understanding**
Pada submission ini menggunakan data saham Top Electric Car Company yang diberikan oleh Yahoo finance yang dapat diunduh pada platfom [kaggle](https://www.kaggle.com/meetnagadia/share-price-of-top-electric-car-company). Dataset ini memiliki data saham dari 10 perusahaan manufaktur ternama. Pada model machine learning memprediksi harga tertinggi saham yang dimiliki oleh BMW. Dataset BMW ini memiliki format .csv dengan 1266 baris dan 7 kolom.Dataset Data set ini terdiri dari 5 type data fload, 1 bertipe integer dan 1 bertipe object. Dataset BMW ini berisikan data saham BMW dari tahun 2016-2021 (5 Tahun). Pada dataset ini secara kesuluruhan tidak terdapat nilai null maupun data yang hilang. 6 parameter yang terdapat pada dataset ini melitupi :
* Date      : Menunjukkan waktu dalam satuan hari
* Open      : Harga pembukaan/awal saham BMW diawal market
* High      : Harga tertinggi saham BMW dihari tersebut
* Low       : Harga terendah saham BMW dihari tersebut
* Close     : Harga penutupan/akhir saham BMW dihari tersebut
* Adj Close : Adjusted close adalah harga penutupan setelah penyesuaian untuk semua pembagian dan pembagian dividen yang berlaku. 
* Volume    : Jumlah sekuritas yang diperdagangkan selama periode waktu tertentu.

### **4. Data Preparation**
* Pada data preparation karena kita hanya memerlukan data harga tertinggi saham BMW tiap harinya, maka kita pisahkan dengan menggunakan fungsi filter dan masukan kedalam dataframe baru yang ditampung variable 'dataDf'. Variable tersebut berisi dataframe yang telah difilter berisikan data harga tertinggi (high).

* Setelah kita filter, kita cek info dari dataframe yang baru untuk memastikan apakah tidak ada yang berubah. Data masih masih seperti semula dengan tidak adanya nilai null dan berjumlah 1266 data dengan pembuktian berupa pengecekan head dan tail dimana head ini akan menampilkan data awal dan tail akan menampilkan data akhir. Head dan tail disetting untuk menampilkan 10 data meliputi 10 data teratas/awal (head) dan 10 data terakhir/terbawah (tail). Setelah itu rubah data ke array numpy dan kita perlu mendapatkan beberapa baris untuk dilakukannya training.

* Proses selanjutnya adalah membuat training dataset. Pada pembuatan training dataset akan dilakukan proses scaled dan membagi dataset menjadi 2 buah unit train. X dan Y train. Keduanya akan dirubah kedalam numpy array dan akan dilakukan penyesuaian dimensi menggunakan reshape sehingga pemrosesan akan menjadi lebih mudah.

* Setelah data frame terubah menjadi array, kita akan melakukan scalling data/normalisasi data dengan MinMaxScaler. MinMaxScaler akan membuat data berada pada rentang 0-1

### **5. Modeling**
Pada tahap modeling, machine learning ini menggunakan algoritma Long Short Term Memory Network (LTSM) dalam memprediksi harga tertingginya. LSTM merupakan tipe Recurrent Neural Network yang dapat mempelajari data historis atau time series. Ia merupakan algoritma deep learning yang kompleks dan dapat mempelajari informasi jangka panjang dengan sangat baik. Pada modeling LSTM digunakan 2 layer LSTM dengan tiap layernya menggunakan 128 dan 64 hidden unit. Hyperparameter yang digunakan selanjutnya adalah digunakanya 2 buah dense layer dengan tiap layernya menggunakan jumlah unit atau node per layernya sebanyak 25 dan 1.

### **6. Evaluation**
- Membuat testing data set dengan nilai yang telah discale sebelumnya. Dilanjutkan dengan proses membuat data sets x_test dan y_test,merubahnya ke numpy array dan melakukan reshape.
- Melakukan prediksi harga
- Mendapatkan root mean squared error (RMSE)
- Melakukan plot data
- Pada project machine learning ini menggunakan matriks Root mean squared error (RMSE). RMSE adalah aturan penilaian kuadrat yang juga mengukur besarnya rata-rata kesalahan. Ini adalah akar kuadrat dari rata-rata perbedaan kuadrat antara prediksi dan observasi aktual. Root Mean Squared Error (RMSE) merupakan salah satu cara untuk mengevaluasi model regresi linear dengan mengukur tingkat akurasi hasil perkiraan suatu model. Nilai RMSE rendah menunjukkan bahwa variasi nilai yang dihasilkan oleh suatu model prakiraan mendekati variasi nilai obeservasinya. RMSE menghitung seberapa berbedanya seperangkat nilai. Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati. Model machine learning menggunakan algoritma LSTM ini menghasilkan nilai RMSE sebesar 2.9126111035298763. Hal tersebut menunjukkan bahwa model machine learning ini layak untuk diterapkan
