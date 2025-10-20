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
