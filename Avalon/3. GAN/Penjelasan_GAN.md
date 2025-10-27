# Generative Adversarial Networks (GAN): Sebuah Tinjauan Profesional

## Abstrak

**Generative Adversarial Network (GAN)** merupakan sebuah kelas *model generatif* dalam *machine learning* yang diperkenalkan oleh **Ian Goodfellow** dan rekan-rekannya pada tahun 2014.  
Arsitektur ini secara inovatif menggunakan dua jaringan saraf yang dilatih secara simultan dalam sebuah skema permainan *zero-sum*.  

- **Generator (G)** bertugas menghasilkan data.  
- **Discriminator (D)** bertugas mengevaluasi keaslian data tersebut.  

Tujuan akhirnya adalah melatih *Generator* hingga mampu menghasilkan data artifisial yang tidak dapat dibedakan dari data asli.

---

## 1. Analogi Intuitif: Pemalsu vs. Ahli Forensik

Untuk memahami konsep GAN secara intuitif, bayangkan sebuah permainan antara dua pihak:

- **Generator (Si Pemalsu)**: Seorang pemalsu yang sangat berbakat yang mencoba membuat karya seni palsu (misalnya, lukisan *Monalisa* palsu). Tujuannya adalah membuat karya yang begitu sempurna sehingga tidak ada yang bisa membedakannya dari yang asli.

Generator adalah jaringan yang akan mengambil input berupa noise acak (z) dan mengubahnya menjadi data palsu/sintetis yang realistis dengan tujuannya yaitu adalah untuk menipu discriminator, enerator belajar dari kesalahan dengan backpropagation, sehingga lama-kelamaan bisa menghasilkan data palsu yang sangat mirip data asli.

![Generator](../Asset/Generator.png)



- **Discriminator (Si Ahli Forensik)**: Seorang ahli seni dan forensik yang bertugas memeriksa setiap karya seni dan menentukan apakah itu asli atau palsu.

Prosesnya berjalan sebagai berikut:

1. Pada awalnya, si pemalsu (`G`) menghasilkan lukisan yang buruk. Ahli forensik (`D`) dengan mudah mengidentifikasinya sebagai palsu.  
2. Ahli forensik (`D`) terus belajar dari karya asli dan karya palsu yang ia lihat, membuatnya semakin pintar dalam mendeteksi ketidaksempurnaan sekecil apa pun.  
3. Si pemalsu (`G`), berdasarkan umpan balik dari kegagalannya menipu si ahli, terus mengasah kemampuannya untuk menghasilkan karya yang lebih baik.  
4. Siklus ini berlanjut hingga si pemalsu menjadi begitu mahir sehingga si ahli forensik tidak lagi yakin dan peluangnya menebak dengan benar hanya 50%. Pada titik ini, si pemalsu telah berhasil menciptakan karya palsu yang sempurna.

---

## 2. Arsitektur dan Komponen

GAN terdiri dari dua model jaringan saraf:

### **Generator (`G`)**

- **Input**: Sebuah vektor acak (*random noise*) $z$ dari *latent space*.  
- **Proses**: `G` merupakan jaringan *deconvolutional* yang memetakan vektor $z$ menjadi data berdimensi sama dengan data asli (misalnya gambar). Hasilnya adalah $G(z)$ — data sintetis.  
- **Tujuan**: Menghasilkan $G(z)$ sedemikian rupa sehingga $D(G(z)) \approx 1$, artinya berhasil menipu *Discriminator*.

### **Discriminator (`D`)**

- **Input**: Data yang bisa berasal dari dataset asli ($x$) atau hasil dari *Generator* ($G(z)$).  
- **Proses**: `D` adalah jaringan *convolutional* (klasifikator biner) yang menghasilkan sebuah nilai probabilitas tunggal, $D(x)$, yang merepresentasikan kemungkinan bahwa $x$ adalah data asli.  
- **Tujuan**: Memberikan $D(x) \to 1$ untuk data asli dan $D(G(z)) \to 0$ untuk data palsu.

---

## 3. Formulasi Matematis: *The Minimax Game*

Inti dari GAN dijelaskan dalam sebuah **fungsi tujuan (objective function)** yang merepresentasikan permainan *minimax* antara `G` dan `D`.

Fungsi nilainya, $V(D, G)$, didefinisikan sebagai:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

### Penjelasan Notasi:

- **$\min_G \max_D$**  
  Notasi *minimax*:  
  - *Discriminator* (`D`) mencoba **memaksimalkan** fungsi ini.  
  - *Generator* (`G`) mencoba **meminimalkannya**.

- **$\mathbb{E}_{x \sim p_{data}(x)}[\log D(x)]$**  
  - $x$ adalah sampel dari distribusi data asli.  
  - $D(x)$ adalah probabilitas yang diberikan oleh `D` bahwa $x$ adalah data asli.  
  - Tujuan `D`: membuat $D(x) \to 1$ untuk data asli (karena $\log(1) = 0$ adalah nilai maksimum logaritma).

- **$\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$**  
  - $z$ adalah sampel *noise* acak.  
  - $G(z)$ adalah data palsu yang dihasilkan oleh `G`.  
  - $D(G(z))$ adalah probabilitas bahwa `D` menganggap data palsu tersebut asli.  
  - Tujuan `D`: membuat $D(G(z)) \to 0$.  
  - Tujuan `G`: membuat $D(G(z)) \to 1$ (menipu `D`), sehingga berusaha **meminimalkan** bagian ini.

---

Secara ringkas, `D` dan `G` memainkan permainan *zero-sum* pada fungsi $V(D, G)$.  
Pelatihan mencapai **Nash Equilibrium** ketika `G` menghasilkan data dengan distribusi yang identik dengan data asli, dan `D` tidak dapat membedakannya — yaitu ketika:

$$
D(x) = 0.5 \quad \text{untuk semua } x
$$
