## Activation Function
Setelah hasil konvolusi, biasanya diterapkan fungsi aktivasi nonlinier untuk memperkenalkan non-linearitas pada model.
Yang paling umum adalah ReLU (Rectified Linear Unit):
Fungsi aktivasi ReLU (*Rectified Linear Unit*) didefinisikan sebagai:

$$
f(x) = \max(0, x)
$$

Tanpa aktivasi, seluruh jaringan hanyalah kombinasi linier dari input → tidak bisa belajar pola kompleks.

Fungsi lain: Sigmoid, Tanh, LeakyReLU, dsb.
Namun ReLU paling populer karena efisien dan membantu mengatasi vanishing gradient.

## 1.4 Menyusun CNN Bertingkat
Biasanya CNN tidak hanya memiliki satu layer konvolusi.
Lapisan-lapisan tersebut disusun bertingkat agar jaringan bisa belajar fitur dari yang sederhana → kompleks. Contohnya paling sering itu

**Arsitektur CNN:**

Input  
↓  
**Conv → ReLU → Pool**  
↓  
**Conv → ReLU → Pool**  
↓  
**Flatten → Fully Connected → Softmax**

## 1.5 Fully Connected Layer dan Output
Setelah beberapa tahap konvolusi dan pooling, feature map akan diratakan (flatten) dan dimasukkan ke lapisan fully connected (dense).

Lapisan ini berfungsi untu, Menggabungkan semua fitur yang telah diekstrak. Menghasilkan prediksi akhir (misalnya klasifikasi gambar). Dan biasanya diakhiri dengan Softmax untuk menghasilkan probabilitas tiap kelas:

$$
P(y = i \mid x) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

## 1.6 Backpropagation pada CNN

Walaupun operasinya tampak berbeda, CNN tetap dilatih menggunakan backpropagation.
Gradien dihitung terhadap setiap kernel/filter melalui operasi convolution transpose, lalu bobot diperbarui dengan optimizers seperti SGD atau Adam.

## 1.7 Regularisasi dan Modern Enhancements

Untuk meningkatkan performa dan generalisasi:
1. Batch Normalization (BN) → menstabilkan distribusi aktivasi, mempercepat training.

2. Dropout → menonaktifkan neuron acak selama training untuk mencegah overfitting.

3. Data Augmentation → memperbanyak variasi data dengan rotasi, flipping, cropping, dsb.

# Kesimpulannya
Singkatnya, operasi konvolusi menggeser sebuah kernel yang telah ditentukan sebelumnya (juga disebut “filter”) di atas peta fitur input (matriks piksel gambar), kemudian melakukan perkalian dan penjumlahan antara nilai-nilai kernel dengan sebagian nilai fitur input untuk menghasilkan output. Nilai-nilai hasil tersebut membentuk matriks output, karena biasanya ukuran kernel jauh lebih kecil dibandingkan ukuran gambar input.ss