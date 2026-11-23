# Rencana Project: VecScope

**VecScope** (*vector + scope*) adalah app interaktif untuk melihat cara model bahasa "membaca" teks. Kamu ketik kalimat, lalu app menunjukkan:

- bagaimana kalimat dipotong jadi **token**, dan
- bagaimana makna kata disimpan sebagai **vektor** (deretan angka).

Di versi berikutnya, kamu juga bisa unggah gambar dan membandingkannya dengan teks. Ini bukan chatbot. Ini alat eksplorasi untuk portofolio, dengan semangat yang sama seperti VisionIndetify.

Riset awal: November 2026.

---

## 1. Tujuan

Menjelaskan token dan embedding secara visual. Pengguna bisa:

1. Mengetik kalimat dan membandingkan hasil dua tokenizer.
2. Melihat vektor kata sebagai strip berwarna, seperti heatmap di artikel Jay Alammar.
3. Mengukur seberapa mirip dua kata.
4. Mencoba analogi `king - man + woman ≈ queen`.

Di v2, CLIP menaruh teks dan gambar di ruang vektor yang sama. Jadi skor kemiripan dan peta 2D bisa berisi teks dan gambar sekaligus.

---

## 2. Fitur

### v1: teks dulu

- **Tokenisasi.** Kalimat → daftar token, token ID, dan jumlah token. Bandingkan BPE GPT-2 (`tiktoken`, encoding `gpt2`) dengan WordPiece BERT (`bert-base-uncased`). Jumlah tokennya biasanya beda.
- **Strip vektor berwarna.** Pakai GloVe 50 dimensi. Satu baris per kata. Merah = positif, biru = negatif, putih = dekat nol. Contohnya ada di [Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/).
- **Cosine similarity.** Skor kemiripan antar kata, dalam bentuk tabel dan heatmap kecil.
- **Peta 2D.** Default pakai PCA karena hasilnya stabil. UMAP opsional. Kata yang maknanya dekat seharusnya mengelompok.
- **Analogi.** Hitung `king - man + woman`, lalu tampilkan top-k kata terdekat. `queen` biasanya ada di dekat puncak, tapi tidak selalu nomor satu.

Kalau kata tidak ada di kosakata GloVe (OOV), tampilkan pesan yang jelas. GloVe `glove-wiki-gigaword-*` tidak membedakan huruf besar-kecil, jadi semua input diubah ke huruf kecil.

### v2: gambar dengan CLIP

- Unggah gambar → tampilkan strip embedding-nya dengan gaya yang sama.
- Hitung cosine similarity antara teks dan gambar.
- Cari gambar pakai kueri teks: urutkan gambar sampel/unggahan menurut skor.
- Scatter 2D yang berisi titik teks dan gambar. **Pakai ruang CLIP saja.** Jangan campur dengan vektor GloVe.

---

## 3. Tech stack

| Bagian | Pilihan | Alasan |
| --- | --- | --- |
| Bahasa | Python | Sudah dipakai di VisionIndetify / cifar10-app. |
| Tokenizer BPE | [tiktoken](https://github.com/openai/tiktoken) | Library resmi OpenAI. `tiktoken.get_encoding("gpt2")` untuk GPT-2. Encoding modern (`cl100k_base`, `o200k_base`) bisa ditambah. |
| Tokenizer WordPiece | [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/fast_tokenizers) | `AutoTokenizer.from_pretrained("bert-base-uncased")`. BERT memakai WordPiece. |
| Embedding kata (v1) | [Gensim downloader](https://radimrehurek.com/gensim/downloader.html) | `api.load("glove-wiki-gigaword-50")`. 50 dimensi, cocok untuk gaya visual Alammar, dan jauh lebih ringan dari Word2Vec Google News 300d. Analogi pakai `most_similar(positive=[...], negative=[...])`. |
| Embedding kalimat (v2) | [Sentence Transformers](https://www.sbert.net/) | Untuk membandingkan vektor kata statis dengan embedding kalimat. Pilih model ringan dari daftar resmi. |
| Teks + gambar (v2) | [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) | Teks dan gambar masuk satu ruang vektor. Mulai dari `openai/clip-vit-base-patch32` (ViT-B/32, umum dan cukup ringan) lewat Transformers. Alternatif: `sentence-transformers/clip-ViT-B-32` kalau mau satu API `encode()` untuk keduanya. Butuh Pillow untuk membuka gambar. |
| Reduksi dimensi | scikit-learn PCA; [UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) opsional | PCA sudah cukup untuk v1. |
| Visualisasi | [Plotly](https://plotly.com/python/pca-visualization/) | Heatmap dan scatter interaktif. |
| UI | [Streamlit](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps) | Layout bertab gampang dibuat. Alternatif: [Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast). |

Jangan commit file vektor, bobot model, atau unggahan pengguna. Gensim menyimpan cache di `~/gensim-data`; folder cache masuk `.gitignore`.

---

## 4. Langkah kerja

### v1

1. Siapkan `README.md`, `requirements.txt`, dan `.gitignore` di repo [cendysaputra/VecScope](https://github.com/cendysaputra/VecScope).
2. Buat modul tokenisasi: satu fungsi untuk GPT-2 (tiktoken), satu untuk BERT WordPiece. Output: token, ID, jumlah token.
3. Load GloVe 50d. Buat fungsi ambil vektor, cosine similarity, dan analogi. Tangani kata yang tidak ada di kosakata.
4. Buat heatmap strip berwarna dengan Plotly (skala diverging, misalnya RdBu).
5. Buat scatter 2D: PCA ke 2 komponen, dengan tooltip nama kata. Sediakan pilihan UMAP, tapi PCA tetap default.
6. Gabungkan semua di `app.py` (Streamlit): input teks, pilih tokenizer/model, tampilkan visualisasi.
7. Uji dengan contoh tetap: `king, man, woman, queen`, plus satu kalimat yang tokenisasinya beda di GPT-2 dan BERT.
8. Tulis README: cara menjalankan, ukuran download model, dan **bukti** (lihat bagian 7).

### v2 (setelah v1 jalan)

9. Load CLIP. Ubah gambar unggahan/sampel jadi vektor, lalu tampilkan dengan fungsi strip berwarna yang sama.
10. Hitung cosine teks ↔ gambar. Tambah pencarian gambar dari kueri teks (tampilkan skor dan urutan).
11. Buat scatter campuran di ruang CLIP, dengan penanda berbeda untuk teks dan gambar.

---

## 5. Struktur repo

```text
VecScope/
├── README.md
├── PLAN.md
├── AGENT.md              # aturan kerja untuk agent/kontributor
├── requirements.txt
├── .gitignore            # __pycache__/, .venv/, cache model, unggahan
├── app.py                # UI Streamlit
├── data/samples/         # gambar sampel publik (v2), diunduh lewat script
├── scripts/              # unduh sampel, bukti, screenshot
├── tests/
└── src/
    ├── tokenize.py       # tiktoken vs BERT WordPiece
    ├── embeddings.py     # load GloVe, ambil vektor
    ├── similarity.py     # cosine + analogi
    ├── visualize.py      # heatmap strip + scatter PCA/UMAP
    └── clip.py           # embed gambar & teks CLIP (v2)
```

---

## 6. Milestone

- [x] **v1:** semua fitur teks di bagian 2. Bahasa Inggris, GloVe 50d, UI Streamlit lokal.
- **v2:**
  - [x] Fitur gambar CLIP (strip, kemiripan, pencarian, scatter campuran).
  - [x] Tokenizer modern (`cl100k_base`, `o200k_base`).
  - [x] Pilihan UMAP (perlu `pip install umap-learn`).
  - [ ] Embedding kalimat dengan Sentence Transformers.
  - [ ] Awal dukungan Bahasa Indonesia.
- [ ] **v3:**
  - Embedding Indonesia yang lebih serius, misalnya [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) atau model multilingual dari Sentence Transformers. Contoh analogi: `raja - pria + wanita`.
  - CLIP asli OpenAI lebih kuat di bahasa Inggris. Untuk kueri Indonesia, coba `sentence-transformers/clip-ViT-B-32-multilingual-v1`.
  - Deploy demo ke [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit). Docs resmi menandai SDK Streamlit bawaan sebagai *deprecated*, jadi ikuti panduan terbaru (template Docker untuk Streamlit, atau Gradio).

---

## 7. Bukti di README

Jangan cuma menulis "berhasil". Lampirkan screenshot dan output contoh:

- Kalimat yang sama, token GPT-2 vs BERT (daftar token, ID, jumlah token).
- Heatmap `king` / `man` / `woman` / `queen`. Pola warna `man` dan `woman` seharusnya lebih mirip satu sama lain daripada ke `king`.
- Tabel cosine: `king`–`queen` tinggi, `king`–`water` rendah.
- Scatter 2D: klaster orang (`king`, `queen`, `man`, `woman`) terpisah dari hewan dan kota.
- Analogi: top-k hasil `king - man + woman` beserta skornya.
- **v2:** strip embedding sebuah gambar, skor cosine teks ↔ gambar (kueri yang cocok vs tidak), hasil pencarian gambar, dan scatter campuran teks + gambar di ruang CLIP.

---

## 8. Sumber

1. [The Illustrated Word2vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/): heatmap strip berwarna dan analogi king/queen.
2. [Tokens & Embeddings](https://dranubhaparashar.github.io/projects/lectures/tokens-embeddings/): token, token ID, embedding, cosine, peta kata.
3. [tiktoken (OpenAI)](https://github.com/openai/tiktoken): tokenizer BPE resmi.
4. [WordPiece tokenization (Hugging Face Course)](https://huggingface.co/docs/course/chapter6/6): cara BERT memotong kata.
5. [Gensim downloader](https://radimrehurek.com/gensim/downloader.html): load GloVe / Word2Vec pretrained.
6. [CLIP (Hugging Face Transformers)](https://huggingface.co/docs/transformers/model_doc/clip): docs resmi, checkpoint `openai/clip-vit-base-patch32`.

Semua link dicek masih aktif per November 2026. Versi library bisa berubah, jadi ikuti docs resmi di atas.
