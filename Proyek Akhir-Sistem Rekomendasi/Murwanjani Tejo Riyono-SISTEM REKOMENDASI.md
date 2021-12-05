# Laporan Proyek Akhir Machine Learning Sistem Rekomendasi - Murwanjani Tejo Riyono


## Project Overview
Menonton film merupakan suatu kegiatan yang digemari pada masa sekarang ini. Terlebih lagi bagi para pekerja WFH yang menghabiskan sebagian waktunya di rumah, mereka kebanyakan menonton film untuk mengobati rasa bosan mereka. Tentunya akan menghabiskan banyak waktu bagi mereka apabila mereka mencari film yang sesuai dengan genre mereka secara manual. Untuk itu perlu adanya sebuah sistem yang dapat merekomendasikan film atau movie sesuai dengan genre yang mereka inginkan berdasarkan apa yang telah mereka tonton sebelumnya.


## Business Understanding
Perlu adanya sebuah sistem yang dapat merekomendasikan film atau movie sesuai dengan genre yang mereka inginkan berdasarkan apa yang telah mereka tonton sebelumnya menggunakan teknik content-based filtering


### Problem Statements
Bagaimana sebuah sistem dapat merekomendasikan film atau movie sesuai dengan genre yang mereka inginkan berdasarkan apa yang telah mereka tonton sebelumnya?


### Goals
Membuat sebuah sistem yang dapat merekomendasikan film atau movie sesuai dengan genre yang diinginkan berdasarkan apa yang telah ditonton sebelumnya.


### Solution statements
Berdasarkan uraian diatas, dibuatlah sebuah sistem rekomendasi dengan algoritma content-based filtering yang dapat merekomendasikan konten berdasarkan kemiripan konten yang telah disukai pengguna sebelumnya. Model machine learning ini akan dibuat menggunakan teknik TF-IDF Vectorizer dan Cosine Similarity.


## Data Understanding
Pada submission ini menggunakan data IMDb movies extensive dataset yang dapat diunduh pada platfom [kaggle](https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset).

IMDb adalah situs web film paling populer dan menggabungkan deskripsi plot film, peringkat Metastore, peringkat dan ulasan kritikus dan pengguna, tanggal rilis, dan banyak aspek lainnya.

Situs web ini terkenal karena menyimpan hampir setiap film yang pernah dirilis (yang tertua adalah dari tahun 1874 - "Passage de Venus") atau baru saja direncanakan untuk dirilis (film terbaru adalah dari tahun 2027 - "Avatar 5").

IMDb menyimpan informasi yang berkaitan dengan lebih dari 6 juta judul (hampir 500.000 di antaranya adalah film unggulan) dan dimiliki oleh Amazon sejak tahun 1998.
### Deskripsi Variabel
Kumpulan data film mencakup 85.855 film dengan atribut seperti deskripsi film, peringkat rata-rata, jumlah suara, genre, dll.

Dataset peringkat mencakup 85.855 detail peringkat dari perspektif demografis.

Dataset nama mencakup 297.705 anggota pemeran dengan atribut pribadi seperti detail kelahiran, detail kematian, tinggi badan, pasangan, anak-anak, dll.

Dataset prinsipal judul mencakup 835.513 peran pemeran dalam film dengan atribut seperti id judul IMDb, id nama IMDb, urutan kepentingan dalam film, peran, dan karakter yang dimainkan.

Berikut ini penjelasan variable yang digunakan dalam pemrosesan :
* imdb_title_id         : ID Judul Film Pada IMDb
* title                 : Judul Film
* genre                 : Genre Film
* weighted_average_vote : Rating Yang Diberikan


### Eksploratory Data Analysis
Menggunakan 2 buah file csv berupa imdb_movies dan imdb_rating yang dimerge menjadi imdb. Dari data yang telah dimerge, kita dapat mengeksplore data genre dengan memvisualisasikannya sebagai berikut : \
![Genre](https://user-images.githubusercontent.com/52377153/143157594-9d64b2ec-32f3-49f8-9af6-cf2989dd47da.png)\

Dari visualisasi data tersebut, terlihat genre Drama sangat mendominasi dan menjadi favorit untuk ditonton berdasarkan datset yang kita pakai.

Berikut pula struktur data dari dataset yang telah dimerge :

| No | Column | Non-Null Count | Dtype | 
| -- | ------ | -------------- | ----- |
| 0 | imdb_title_id | 16111 non-null | object |
| 1 | title | 16111 non-null | object |
| 2 | genre | 16111 non-null | object |
| 3 | weighted_average_vote | 16111 non-null | float64 |


## Data Preparation

### Drop Feature
* Pada imdb_movies kita melakukan drop kolom yang tidak diperlukan. Pada dataset ini yang kita butuhkan adalah imdb_title_id, title dan genre. Selain ketiga kolom tersebut, kita akan hapus atau kita lakukan drop kolom.

* Pada imdb_rating kita melakukan drop kolom yang tidak diperlukan. Pada dataset ini yang kita butuhkan hanyalah imdb_title_id dan weighted_average_vote.

### Merge Dataset
* Setelah melakukan drop kolom. Kita akan melakukan merge atau penggabungan kedua dataset tersebut dengan imdb_title_id sebagai patokan kesamaan dari kedua dataset tersebut. Dataset baru yang telah di merge ini dimasukan kedalam variable imdb.

* Hasil pengecekan nilai null pada dataset menunjukkan 0 pada keseluruhan data. Sehingga dapat dipastikan pada dataset ini tidak memiliki data yang hilang pada saat proses drop dan merge sebelumnya.

### Sorting Dataset
* Hal yang perlu diperhatikan adalah dataset kita masih belum urut. Maka dari itu kita urutkan dengan menggunakan fungsi sort. Acuan yang digunakan pada saat sorting adalah imdb_title_id secara ascending.

### Konversi Data Series menjadi List
* Proses penting selanjutnya adalah mengkonversi data menjadi list dan membuat dictionary untuk data meliputi movieId, movieTitle dan movieGenre.


## Modelling
Model yang dibuat menggunakan algoritma content-based filtering dan menggunakan teknik TF-IDF Vectorizer dan Cosine Similarity. TF-IDF Vectorizer digunakan untuk merepresentasikan fitur penting dari setiap genre film yang tersedia. Fungsi tfidfvectorizer() digunakan untuk melakukan perhitungan idf dari setiap genre, kemudian melakukan fit dan transformasi vektor TF-IDF ke dalam bentuk matriks menggunakan fungsi todense(). Cosine similarity digunakan untuk menghitung derajat kesamaan antar film dari library sklearn. Setelah itu kita buat model rekomendasi film dengan parameter :
* movieTitle : Judul Film.
* Similarity_data : Dataframe mengenai similarity (cosine_sim_df)
* Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘title’ dan ‘genre’.
* k : Banyak rekomendasi yang ingin diberikan.

Argpartition digunakan untuk mengambil sejumlah nilai 'k' tertinggi dari similarity data. Setelah itu, dilakukan proses pengambilan data dari bobot tertinggi ke terendah. Data dimasukkan ke dalam variabel closest. Pada model machine learning ini akan melakukan pencarian film yang memiliki genre sama atau mirip dengan film "Pengalila", sehingga kita perlu drop movieTitle "Pengalila" agar tidak muncul dalam daftar rekomendasi yang diberikan. 

Hasil dari 10 rekomendasi film yang memiliki genre sama dengan film "Pengalila" :\
![10 recomend](https://user-images.githubusercontent.com/52377153/143158285-14860d5a-1736-440f-9cd2-7ef0cd434cd6.png)\


## Evaluation
Evalution model menggunakan metrik precision, dimana kita menentukan banyaknya jumlah rekomendasi yang relevan dengan genre film. 

Berikut formula precision :\
![Formula Precision](https://dicoding-web-img.sgp1.cdn.digitaloceanspaces.com/original/academy/dos:819311f78d87da1e0fd8660171fa58e620211012160253.png)

Dari hasil rekomendasi yang diberikan, menghasilkan 10 rekomendasi film dengan genre yang sama dengan "Pengalila" yaitu genre Drama. Hal tersebut menunjukkan tingkat relevansi atau precision sebesar 10/10 atau bisa dikatakan 100% karena 10 genre film yang direkomendasikan similar dengan genre film "Pengalila"

Berdasarkan referensi [berikut](https://www.dicoding.com/academies/319/discussions/134402). Hasil evaluasinya adalah :\
![10 recomend](https://user-images.githubusercontent.com/52377153/143225560-58c497c1-f2a1-4d4c-90eb-bb2244422ca6.png)
