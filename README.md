# Tibetan OCR Training

A training pipeline for building OCR models for classical Tibetan scripts (Uchen) using deep learning and ONNX. Developed by the [BUDA Project](https://github.com/buda-base) and polished by me.

---

## 📚 Overview

This repository provides:

- 🧠 Training scripts for OCR models
- 🪪 Configurable model and dataset parameters
- 🔤 Unicode and EWTS output support
- 📦 Export to ONNX for deployment

---

## 🛠️ Installation

Clone the repository and install required Python dependencies:

```bash
git clone https://github.com/buda-base/tibetan-ocr-training.git
cd tibetan-ocr-training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Model, dataset, and training parameters are defined in JSON config files:

- `configs/model_config.json`
- `configs/dataset_config.json`
- `configs/training_config.json`

Each config controls architecture, paths, charset, batch size, optimizer settings, etc.

---

## 🚀 Training

To launch training, run:

```bash
python train.py --config configs/training_config.json
```

You can also pass individual flags to override values in the config.

---

## 🔁 ONNX Export

Export trained models to ONNX format:

```bash
python export_onnx.py --checkpoint path/to/checkpoint.pth --output path/to/model.onnx
```

---

## 🖼️ Dataset Format

Training data must be structured as:

```
dataset/
├── images/
│   ├── 000001.jpg
│   ├── 000002.jpg
├── labels/
│   ├── 000001.txt
│   ├── 000002.txt
```

Each label file contains one or more lines of Tibetan text (Unicode).

---

## 🧠 Decoding & Transliteration

The pipeline includes support for:

- CTC decoding using `pyctcdecode`
- EWTS transliteration using `pyewts`
- Vocabulary loading from `charset.txt`

---

## 📊 Metrics

The system computes:

- EMR: Exact Match Rate
- CER: Character Error Rate
- WER: Word Error Rate

Results are logged and can be exported as JSON or CSV for comparison.

---

## 🔍 Dependencies

See [requirements.txt](./requirements.txt) for full list. Core libraries include:

- PyTorch
- ONNX Runtime
- NumPy, SciPy
- OpenCV
- PyEWTS
- PyCTCDecode

---

## 🧾 License

This repository is forked from the part of the BUDA project. License terms TBD.

---

## 🧠 Credits

-Original scripts developed by: [BUDA Project](https://github.com/buda-base)
- OCR Design: Inspired by PaddleOCR, EasyOCR, and U-Net-based line detection

---

## 📬 Contact

For questions or contributions, please open an [issue](https://github.com/buda-base/tibetan-ocr-training/issues) or contact the BUDA team.
