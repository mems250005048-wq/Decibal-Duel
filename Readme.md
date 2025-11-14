# 🎧 Conditional GAN for Audio Generation (CGAN) — README

This repository contains a full **Conditional GAN (CGAN)** pipeline that trains a model to generate audio samples from labeled environmental sound categories (e.g., dog bark, siren, drilling). The workflow includes automatic ZIP extraction, dataset discovery, preprocessing, model training, and audio reconstruction.

---

## 📁 Project Overview

This project trains a **Conditional GAN** on log-mel spectrograms derived from `.wav` audio files. The generator learns to synthesize mel‑spectrograms conditioned on a chosen category, and the discriminator distinguishes real vs. fake mel-specs using spectral normalization.

Audio is reconstructed using **InverseMelScale** + **Griffin-Lim**.

---

## 🚀 Features

* Automatic ZIP extraction from Google Drive
* Auto-detection of category folder structure
* Dataset mean/std computation for normalization
* Log‑mel spectrogram generator & discriminator (CGAN)
* Griffin-Lim phase reconstruction
* Audio saving & playback directly in Colab
* Training loop using **LSGAN loss**
* Fixed runtime error using `.reshape(-1)` instead of `.view(-1)`

---

## 📦 Folder Requirements

Your extracted ZIP must contain category folders such as:

```
root_dir/
  ├── dog_bark/
  ├── drilling/
  ├── engine_idling/
  ├── siren/
  └── street_music/
```

Each folder must contain `.wav` files.

---

## 🔧 Installation

```bash
pip install torch torchaudio tqdm matplotlib
```

Google Colab will run this automatically.

---

## 📥 ZIP Extraction

The code extracts your dataset from Google Drive:

```python
ZIP_PATH = "/content/drive/MyDrive/the-frequency-quest.zip"
EXTRACT_ROOT = "/content/data"
```

Make sure your ZIP is correctly located in your Drive.

---

## 🧠 Dataset Class

The dataset:

* Loads audio
* Converts stereo → mono
* Computes log‑mel spectrograms
* Normalizes using dataset mean/std
* Uses `.reshape(-1)` to prevent PyTorch runtime errors

---

## 🏗 Model Architecture

### **Generator**

Takes:

* Random noise `z` (latent vector)
* One‑hot label vector

Outputs a **1×128×512** mel-spectrogram.

### **Discriminator**

Receives:

* Mel-spectrogram
* Label embedding reshaped into a spatial map

Uses **spectral normalization** for training stability.

---

## 🎹 Audio Reconstruction

Mel spectrogram → Reconstructed waveform using:

* `InverseMelScale`
* `GriffinLim (n_iter=64)` for better audio clarity

Generated audio is saved as:

```
gan_generated_audio/<category>_ep<epoch>.wav
```

---

## 🎯 Training

Training uses **Least Squares GAN (LSGAN)** to stabilize convergence.

Key hyperparameters:

```
LATENT_DIM = 100
EPOCHS = 300
BATCH_SIZE = 32
LR = 2e-4
```

After every 10 epochs, sample audio for the first 3 categories.

---

## 🧪 Inference Example

To generate audio for category index 0:

```python
wav = generate_audio_gan(G, 0, DEVICE, train_dataset.mean, train_dataset.std)
save_and_play(wav, 22050, "example.wav")
```

---

## 🛠 Troubleshooting

### ❗ Dataset not found

Check the extracted folder structure and ensure category names match:

```python
train_categories = ['dog_bark', 'drilling', 'engine_idling', 'siren', 'street_music']
```

### ❗ Empty dataset

Ensure your category folders contain `.wav` files.

### ❗ Griffin-Lim slow or noisy

Decrease `n_iter`, but audio quality will drop.

---

## 📌 Notes

* Modify category names to match your dataset.
* Adjust `max_frames` depending on your audio duration.
* Increase epochs for more realistic audio.
* Works best with clean audio datasets.

---

## 📄 License

MIT License. Feel free to modify and build on this! 😊

---

## 🙌 Credits

Developed for training CGANs on environmental sound datasets in Google Colab.

