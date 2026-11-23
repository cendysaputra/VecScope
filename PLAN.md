# Rencana Project: VecScope

Riset singkat (November 2026) buat [VecScope](https://github.com/cendysaputra/VecScope), app GitHub interaktif dengan semangat VisionIndetify: ketik teks (nanti juga unggah gambar), lalu *lihat* bagaimana model memotong input jadi token dan merepresentasikan makna sebagai vektor. Namanya dari **vector + scope**: alat buat ngintip isi vektor. Bukan chatbot; ini alat eksplorasi buat portofolio.

---

## 1. Tujuan singkat

Bikin app yang menjelaskan token dan embedding secara visual. Orang bisa mengetik kalimat, membandingkan tokenizer, melihat vektor sebagai strip berwarna (gaya heatmap Jay Alammar), mengukur kemiripan, dan mencoba analogi `king - man + woman ≈ queen`. Di versi berikutnya, CLIP memasukkan teks dan gambar ke ruang vektor yang sama, jadi kemiripan dan peta 2D bisa campur keduanya.

---

## 2. Fitur

**v1 (teks dulu, biar cepat selesai):**

- **Tokenisasi:** input kalimat → daftar token + token ID. Bandingkan BPE GPT-2 (`tiktoken`, encoding `gpt2`) vs WordPiece BERT (`bert-base-uncased`). Tampilkan jumlah token (hasilnya biasanya beda).
- **Strip vektor berwarna:** embedding kata (default: GloVe 50d), satu baris per kata, warna = nilai (merah positif, biru negatif, putih dekat nol) — seperti heatmap king/man/woman di [Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/).
- **Cosine similarity:** skor kemiripan antar kata, plus tabel/heatmap kecil.
- **Peta 2D:** PCA (default; stabil). UMAP opsional. Kata yang maknanya dekat harus mengelompok.
- **Analogi:** `king - man + woman`, tampilkan top-k kata terdekat (biasanya `queen` di dekat hasil, tapi tidak selalu persis sama).

Kalau kata tidak ada di kosakata GloVe (OOV), tampilkan pesan jelas. Model `glove-wiki-gigaword-*` bersifat *uncased*, jadi normalisasi ke huruf kecil.

**v2 (gambar, CLIP):**

- Unggah gambar → strip embedding berwarna, gaya yang sama.
- Cosine similarity teks ↔ gambar.
- Cari gambar dengan kueri teks (urutkan gambar sampel/unggahan menurut skor).
- Scatter 2D campur titik teks dan gambar. **Pakai ruang CLIP saja** — jangan dicampur dengan vektor GloVe.

---

## 3. Tech stack

| Bagian | Rekomendasi | Kenapa |
| --- | --- | --- |
| Bahasa | Python | Sudah kamu pakai di VisionIndetify / cifar10-app. |
| Tokenizer BPE | [tiktoken](https://github.com/openai/tiktoken) | Library resmi OpenAI. `tiktoken.get_encoding("gpt2")` untuk GPT-2; encoding modern seperti `cl100k_base` / `o200k_base` bisa ditambah nanti. |
| Tokenizer WordPiece | [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/fast_tokenizers) | `AutoTokenizer.from_pretrained("bert-base-uncased")` — BERT pakai WordPiece. |
| Embedding kata (v1) | [Gensim downloader](https://radimrehurek.com/gensim/downloader.html) | `api.load("glove-wiki-gigaword-50")` — 50 dimensi, sama gaya visualisasi Alammar, lebih ringan dari Word2Vec Google News 300d. Analogi: `most_similar(positive=[...], negative=[...])`. |
| Embedding kalimat (v2) | [Sentence Transformers](https://www.sbert.net/) | Bandingkan vektor kata statis vs embedding kalimat. Pilih model ringan dari daftar resmi. |
| Teks + gambar (v2) | [CLIP](https://huggingface.co/docs/transformers/model_doc/clip) | Teks dan gambar masuk satu ruang vektor. Mulai dari `openai/clip-vit-base-patch32` (varian ViT-B/32 yang umum dan relatif ringan) lewat Transformers, atau wrapper `sentence-transformers/clip-ViT-B-32` kalau mau API `encode()` yang sama untuk keduanya. Butuh Pillow buat buka gambar. |
| Reduksi dimensi | scikit-learn PCA; [UMAP](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) opsional | PCA sudah cukup buat v1. |
| Visualisasi | [Plotly](https://plotly.com/python/pca-visualization/) | Heatmap interaktif + scatter 2D, cocok untuk app. |
| UI | [Streamlit](https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps) | Layout bertab lebih gampang. Alternatif: [Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast). |

Jangan commit file vektor atau unggahan pengguna. Download Gensim cache-nya di `~/gensim-data`; `.gitignore` folder cache.

---

## 4. Langkah-langkah bikin

**v1**

1. Kerja di repo [cendysaputra/VecScope](https://github.com/cendysaputra/VecScope). Siapkan `README.md`, `requirements.txt`, dan `.gitignore`.
2. Modul tokenisasi: satu fungsi GPT-2/tiktoken, satu fungsi BERT WordPiece; output token, ID, dan jumlah token.
3. Load GloVe 50d; fungsi ambil vektor, cosine similarity, dan analogi. Handle kata yang tidak ada di vocab.
4. Heatmap strip berwarna (Plotly, skala diverging seperti RdBu).
5. Scatter 2D: PCA ke 2 komponen; tooltip nama kata. Siapkan sakelar UMAP, tapi PCA tetap default.
6. Gabungkan di `app.py` (Streamlit): input teks, pilih tokenizer/model, lalu tampilkan visualisasi teks.
7. Uji dengan contoh tetap: `king, man, woman, queen` dan satu kalimat yang tokenisasinya beda di GPT-2 vs BERT.
8. Tulis README dengan cara jalan, ukuran download model, dan **bukti** (lihat bagian 7).

**v2 (setelah v1 jalan)**

9. Load CLIP; embed gambar unggahan/sampel jadi vektor, lalu pakai ulang fungsi strip berwarna.
10. Cosine teks ↔ gambar, plus pencarian gambar dari kueri teks (tampilkan skor dan urutan).
11. Scatter campuran di ruang CLIP, dengan penanda beda untuk titik teks vs gambar.

---

## 5. Struktur repo

```text
VecScope/
├── README.md
├── requirements.txt
├── .gitignore            # __pycache__/, .venv/, cache model, unggahan
├── app.py                # UI Streamlit
├── data/samples/         # beberapa gambar sampel publik (v2)
└── src/
    ├── tokenize.py       # tiktoken vs BERT WordPiece
    ├── embeddings.py     # load GloVe, ambil vektor
    ├── similarity.py     # cosine + analogi
    ├── visualize.py      # heatmap strip + scatter PCA/UMAP
    └── clip.py           # embed gambar & teks CLIP (v2)
```

---

## 6. Milestone

- **v1:** fitur teks di bagian 2, bahasa Inggris, GloVe 50d, UI Streamlit lokal. Belum perlu CLIP.
- **v2:** fitur gambar CLIP, embedding kalimat (Sentence Transformers), opsi tokenizer modern (`cl100k_base` / `o200k_base`), UMAP, dan awal dukungan Bahasa Indonesia.
- **v3:** embedding Indonesia yang lebih serius, misalnya [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) atau model multilingual dari Sentence Transformers; analogi seperti `raja - pria + wanita`. CLIP asli OpenAI lebih diarahkan ke bahasa Inggris; untuk kueri Indonesia bisa coba `sentence-transformers/clip-ViT-B-32-multilingual-v1`. Deploy demo ke [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit). Docs resmi saat ini menandai built-in Streamlit SDK sebagai deprecated — ikuti panduan terbaru (template Docker untuk Streamlit, atau Gradio).

---

## 7. Bukti di README

Jangan cuma klaim “berhasil”. Lampirkan screenshot + output contoh:

- Kalimat yang sama, token GPT-2 vs BERT (daftar token, ID, jumlah token).
- Heatmap `king` / `man` / `woman` / `queen` (pola warna `man`–`woman` lebih mirip daripada ke `king`).
- Tabel cosine, misalnya `king`–`queen` tinggi, `king`–`water` rendah.
- Scatter 2D: klaster orang (`king`, `queen`, `man`, `woman`) terpisah dari hewan atau kota.
- Analogi: top-k hasil `king - man + woman` (tulis skornya; `queen` biasanya di dekat puncak).
- **v2:** strip embedding sebuah gambar; skor cosine teks ↔ gambar (kueri yang cocok vs yang tidak); hasil pencarian gambar; scatter campur teks dan gambar di ruang CLIP.

---

## 8. Sumber

1. [The Illustrated Word2vec (Jay Alammar)](https://jalammar.github.io/illustrated-word2vec/) — heatmap strip berwarna dan analogi king/queen.
2. [Tokens & Embeddings](https://dranubhaparashar.github.io/projects/lectures/tokens-embeddings/) — token, token ID, embedding, cosine, peta kata.
3. [tiktoken (OpenAI)](https://github.com/openai/tiktoken) — tokenizer BPE resmi.
4. [WordPiece tokenization (Hugging Face Course)](https://huggingface.co/docs/course/chapter6/6) — cara BERT memotong kata.
5. [Gensim downloader](https://radimrehurek.com/gensim/downloader.html) — load GloVe / Word2Vec pretrained.
6. [CLIP (Hugging Face Transformers)](https://huggingface.co/docs/transformers/model_doc/clip) — docs resmi, checkpoint `openai/clip-vit-base-patch32`.

Link dicek masih aktif per November 2026. Versi library bisa berubah; ikuti docs resmi di atas.
