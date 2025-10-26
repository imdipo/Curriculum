# Convolutional Neural Network (CNN)
CNN as it names "neural network" work like how human visual cortex systems works. and it used in deeplearning realm.  

jika di sebelumnya kita telah mengenal konsep image gradient vector, dan bagaimana algoritma HOG menyimpulkan informasi dari seluruh gradient vector dari sebuah gambar dll

disini kita membahas cnn classic untuk klasifikasi gambar. yang juga merupakan fondasi untuk nantinya ya 

well, kita akan membahas dulu keterbatasan transformasi afin (affine transformations) dalam jaringan saraf tradisional

Inti dari jaringan saraf adalah transformasi afine: sebuah vektor diterima sebagai input dan dikalikan dengan sebuah matriks untuk menghasilkan output (biasanya ditambahkan vektor bias sebelum hasilnya dilewatkan melalui fungsi nonlinier). Hal ini berlaku untuk segala jenis input, baik itu gambar, klip suara, maupun kumpulan fitur yang tidak terurut: apapun dimensinya, representasinya selalu bisa diratakan menjadi vektor sebelum dilakukan transformasi:

## Mekanisme
<ol>
<li>Input → Vektor Rata: Input apa pun (gambar, klip suara, fitur) harus diratakan (flattened) menjadi satu vektor (satu array 1-dimensi).

<li>Perkalian Matriks: Vektor input ini dikali dengan sebuah matriks (disebut weight matrix).

<li>Penambahan Bias: Hasil perkalian matriks ini ditambah dengan sebuah vektor bias.

<li>Nonlinieritas: Hasil akhir kemudian dilewatkan melalui fungsi nonlinier (seperti ReLU atau Sigmoid).
</ol>

Secara matematis, untuk input vektor x, matriks bobot W, dan vektor bias b, transformasinya adalah:

$$Output=f(Wx+b)$$

(di mana f adalah fungsi nonlinier).

## Masalah utamanya
Masalah-nya dengan Transformasi Afin ini adalah Mengabaikan Struktur Data. karena transformasi afin tidak memanfaatkan struktur intrinsik (bawaan) data terstruktur seperti gambar

| Properti           | Penjelasan                                                                 | Contoh                                                      |
|-------------------|---------------------------------------------------------------------------|------------------------------------------------------------|
| Array Multi-dimensi | Data disimpan dalam bentuk tensor dengan lebih dari satu dimensi.         | Gambar (Tinggi × Lebar × Channel)                         |
| Urutan Penting      | Urutan di sepanjang sumbu tertentu memiliki arti struktural.             | Sumbu lebar dan tinggi pada gambar; sumbu waktu pada klip suara. Mengubah urutan piksel secara acak akan merusak gambar. |
| Sumbu Channel       | Sumbu khusus untuk mengakses "pandangan" berbeda dari data yang sama.     | RGB (Merah, Hijau, Biru) pada gambar berwarna; Kiri/Kanan pada audio stereo. |

Intinya keterbatasannya adalah Ketika data terstruktur (misalnya, gambar 3D) diratakan menjadi vektor 1D. 

**Informasi Topologi Hilang** Hubungan spasial (kedekatan) antara piksel yang berdekatan dalam gambar rusak

**Diperlakukan Sama** dimana semua sumbu (lebar, tinggi, channel) diperlakukan sama setelah diratakan. Jaringan saraf tidak tahu bahwa dua elemen yang berdekatan di vektor mungkin sebenarnya jauh di gambar asli, atau sebaliknya.

ya pada akhirnya **Struktur Tidak Dieksploitasi** Sifat penting seperti urutan dan channel diabaikan.

## Solusi
Sebagai solusinya **discrete convolution** berperan untuk mempertahankan struktur intrinsik data (dimensi spasial dan urutan) dan memanfaatkannya saat memproses data.

## 1.1 Discrete convolutions

Cara penerapan dari Konvolusi diskrit adalah sebagai blok bangunan utama dalam Convolutional Neural Networks (CNNs), yang sangat efektif dalam tugas-tugas yang melibatkan data terstruktur seperti visi komputer dan pengenalan ucapan.

Akhirnya, Dengan menggunakan konvolusi, jaringan dapat belajar pola lokal dan fitur spasial (misalnya, garis tepi atau sudut) dengan menerapkan kernel (filter) kecil yang bergerak melintasi dimensi spasial, sehingga menghormati topologi data.

## sifat utama konvdisk
Konvolusi diskrit adalah jenis transformasi linier yang dirancang untuk melestarikan konsep urutan (atau topologi) data. dengan dua sifat mendasar yang membuatnya sangat efisien dan efektif

### Sparsitas (Konektivitas Lokal)
Berbeda dengan transformasi afine di mana setiap unit input terhubung ke setiap unit output (*fully connected*), konvolusi bersifat *sparse connected*. Artinya, hanya sebagian kecil unit input yang berdekatan yang berkontribusi pada perhitungan satu unit output.

Jika divisualisasikan, setiap piksel output hanya dipengaruhi oleh sekelompok piksel lokal di sekitar posisi yang sama pada gambar input. Hal ini memanfaatkan fakta bahwa informasi yang paling relevan untuk suatu piksel biasanya ada di sekitarnya.

### Parameter Sharing
Konvolusi menggunakan kembali bobot yang sama (disebut *kernel* atau *filter*) di berbagai lokasi input yang berbeda. Bayangkan seperti menggunakan kaca pembesar yang sama untuk mencari pola tertentu (misalnya garis tepi horizontal) di setiap area gambar.

Sifat ini sangat mengurangi jumlah parameter yang harus dipelajari oleh model, sehingga jauh lebih efisien dibandingkan lapisan afine penuh, yang harus mempelajari bobot unik untuk setiap koneksi.

![kernel area](../Asset/Screenshot%202025-10-15%20174032.png)

wujud asli dari kernel, matriks bobot kecil yang disebut filter. Kernel inilah yang digeser (slide) ke seluruh peta fitur input untuk melakukan operasi perkalian dan penjumlahan, menghasilkan satu titik pada peta fitur output.
![discrete conv](../Asset/discreteConv.png)

Kernel (atau filter) bergerak melintasi peta fitur input (input feature map). Pada setiap posisi, dilakukan perhitungan hasil kali antara setiap elemen kernel dengan elemen input yang tumpang tindih, kemudian hasil-hasil tersebut dijumlahkan untuk menghasilkan output pada posisi saat ini.
Prosedur ini dapat diulangi dengan menggunakan berbagai kernel yang berbeda untuk membentuk sebanyak mungkin peta fitur output (output feature maps) yang diinginkan (lihat Gambar 1.3).
Hasil akhir dari prosedur ini disebut output feature maps.

Jika terdapat beberapa peta fitur input, maka kernel harus memiliki dimensi tiga (3D) — atau secara ekuivalen, setiap peta fitur input dikonvolusikan dengan kernel yang berbeda — dan hasil dari tiap konvolusi tersebut dijumlahkan secara elemen demi elemen (elementwise) untuk menghasilkan satu peta fitur output.

Konvolusi yang digambarkan pada Gambar sebelumnya adalah contoh dari konvolusi dua dimensi (2D convolution), namun konsepnya dapat digeneralisasi menjadi konvolusi N-dimensi (N-D convolution).
Sebagai contoh, dalam konvolusi 3D, kernel berbentuk kubus (cuboid) dan bergerak melintasi tinggi (height), lebar (width), serta kedalaman (depth) dari peta fitur input.

Kumpulan kernel yang mendefinisikan konvolusi diskrit memiliki bentuk (shape) yang sesuai dengan suatu permutasi dari $(n, m, k_1, \ldots, k_N)$, di mana:


1. $n$ = jumlah *output feature maps*

Setiap *layer konvolusi* menghasilkan sejumlah *feature map* baru — sesuai dengan berapa banyak *kernel/filter* yang kita gunakan.

Misalnya, kita menggunakan **32 filter**: maka layer tersebut menghasilkan **32 output feature maps**, jadi $n = 32$


2. $m$ = jumlah *input feature maps*  

Bayangkan kita memasukkan gambar RGB ke dalam CNN.  
Gambar RGB memiliki 3 *channel*: **R** (Red), **G** (Green), **B** (Blue)

Artinya jumlah *input feature map* adalah 3 (karena ada 3 *channel*). Jadi $m = 3$

Jika lapisan sebelumnya menghasilkan 16 *feature maps*, maka: $m = 16$


3. $k_j$ = ukuran kernel sepanjang sumbu ke-$j$

$o_j$ yang digunakan untuk menentukan ukuran tiap feature map (tinggi × lebar). Dimana beberapa properti berikut memengaruhi ukuran output $o_j$ dari suatu lapisan konvolusi di sepanjang sumbu ke-$j$:

$$
o_j = \left\lfloor \frac{i_j + 2p_j - k_j}{s_j} \right\rfloor + 1
$$

- $o_j$ : ukuran *output* sepanjang sumbu ke-$j$ 
- $i_j$ : ukuran *input* sepanjang sumbu ke-$j$  
- $k_j$ : ukuran *kernel* sepanjang sumbu ke-$j$  
- $s_j$ : *stride* (jarak antara dua posisi kernel yang berurutan) di sepanjang sumbu ke-$j$  
- $p_j$ : *zero padding* (jumlah nol yang ditambahkan di awal dan akhir suatu sumbu) di sepanjang sumbu ke-$j$

Sebagai contoh, Gambar 1.2 menunjukkan sebuah kernel 3 × 3 yang diterapkan pada input 5 × 5 dengan padding nol 1 × 1 dan stride 2 × 2.

![OutputVal](../Asset/OutputValues.png)

Perlu diperhatikan bahwa stride merupakan salah satu bentuk subsampling.
Sebagai alternatif dari menganggapnya sebagai ukuran seberapa jauh kernel digeser, stride juga dapat dipandang sebagai ukuran seberapa banyak output yang dipertahankan.

gambar ini menunjukkan bagaimana satu set filter (kernel) bekerja pada peta fitur input, menghasilkan peta fitur output
<img src="../Asset/convolutionMapping.png" alt="convolution mapping from two input feature maps to three output" width="400">

Sebuah proses konvolusi memetakan dua *input feature maps* menjadi tiga *output feature maps* menggunakan kumpulan kernel $w$ berukuran $3 \times 2 \times 3 \times 3$. Pada jalur kiri, *input feature map* pertama dikonvolusikan dengan kernel $w_{1,1}$ dan *input feature map* kedua dengan kernel $w_{1,2}$. Hasil dari kedua konvolusi tersebut dijumlahkan secara elemen demi elemen (*elementwise*) untuk membentuk *output feature map* pertama. Prosedur yang sama diulang untuk jalur tengah dan jalur kanan guna membentuk *output feature map* kedua dan ketiga, yang kemudian dikelompokkan bersama (*stacked*) untuk menghasilkan output akhir.

Contohnya gini biar lebih jelas:
![discrete conv](../Asset/Contoh-kernel-bekerja.png)




Sebagai contoh, menggeser kernel dengan langkah dua (stride = 2) setara dengan menggeser kernel dengan langkah satu (stride =w 1), namun hanya mempertahankan elemen output bernomor ganjil (lihat Gambar 1.4).w

### 1.2 Pooling
Selain konvolusi diskrit itu sendiri, operasi pooling merupakan komponen penting lainnya dalam CNN (Convolutional Neural Networks).
Pooling berfungsi untuk mengurangi ukuran peta fitur (feature maps) dengan menggunakan suatu fungsi untuk merangkum subregion (wilayah kecil), misalnya dengan mengambil nilai rata-rata (average pooling) atau nilai maksimum (max pooling).

Pooling bekerja dengan cara menggeser sebuah jendela (window) di atas input, lalu mengumpankan isi jendela tersebut ke fungsi pooling.  
Dalam beberapa hal, mekanisme pooling mirip dengan konvolusi diskrit, namun menggantikan kombinasi linear (dot product) yang dilakukan kernel dengan fungsi lain (seperti maksimum atau rata-rata).

Gambar berikutnya bakalan menunjukkan contoh average pooling, abis itu bakal menunjukkan max pooling.

Ini average pooling
![AvgPool](../Asset/AvgPool.png)

Ini Max pooling
![AvgPool](../Asset/MaxPool.png)

## Aritmetika Konvolusi (Convolution Arithmetic)
Analisis hubungan antara berbagai properti lapisan konvolusi menjadi lebih mudah karena setiap sumbu (axis) bekerja secara independen — artinya, pemilihan ukuran kernel, stride, dan zero padding pada sumbu ke-$j$ hanya memengaruhi ukuran output pada sumbu tersebut, dan tidak berinteraksi dengan sumbu lain.