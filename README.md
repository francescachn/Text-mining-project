# 📰 Text Mining Project — News Category Classification

A text mining and NLP pipeline that classifies short news snippets into five categories — **politics, business, entertainment, sports, technology** — using classical topic modeling (NMF, LDA), a from-scratch **BiLSTM**, and fine-tuned **BERT** / **DistilBERT** transformers, wrapped in an interactive **Plotly Dash** dashboard.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/francescachn/Text-mining-project/blob/main/Copia_di_progetto_data_.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. EDA & Data Cleaning](#1-eda--data-cleaning)
  - [2. Avoiding Data Leakage](#2-avoiding-data-leakage)
  - [3. Linguistic Analysis (spaCy)](#3-linguistic-analysis-spacy)
  - [4. Unsupervised Topic Modeling (NMF & LDA)](#4-unsupervised-topic-modeling-nmf--lda)
  - [5. Supervised Classification](#5-supervised-classification)
- [Results](#results)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Tech Stack](#tech-stack)
- [Authors](#authors)
- [License](#license)

---

## Overview

Given a corpus of short, template-generated news headlines, this project builds a full text-mining pipeline end to end:

1. **Explore & clean** the raw text (dedup analysis, length distributions, vocabulary coverage).
2. **Split safely** using a group-aware strategy that eliminates train/validation leakage caused by duplicate sentences.
3. **Profile the language** with spaCy (POS tagging, lemmatization, dependency parsing).
4. **Discover latent topics** with two unsupervised methods (NMF and LDA) and compare them against the ground-truth labels.
5. **Train and compare three classifiers** of increasing complexity — a custom BiLSTM, fine-tuned BERT, and fine-tuned DistilBERT.
6. **Present everything interactively** in a multi-tab Plotly Dash dashboard.

---

## Project Structure

```
Text-mining-project/
├── Copia_di_progetto_data_.ipynb   # main pipeline: EDA, cleaning, spaCy, NMF/LDA, BiLSTM, BERT, DistilBERT
├── dashboard.ipynb                 # generates app.py and launches the dashboard (Colab + localtunnel)
├── app.py                          # standalone Plotly Dash app, produced by `%%writefile` in dashboard.ipynb
├── train.csv                       # training data — 3,200 rows: `content`, `category`
├── test.csv                        # held-out test data — 800 rows: `content` only
└── README.md
```

> **Note:** at the time of writing, `dashboard.ipynb` lives on the `Marco` branch of this repository while the main pipeline notebook is on `main`. Merge (or copy) it into `main` before following the run instructions below, or adjust the paths if your layout differs.

---

## Dataset

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 3,200 | `content`, `category` | Labeled training data |
| `test.csv` | 800 | `content` | Unlabeled held-out test set |

**Categories** (well balanced, ~19.5–20.4% each): `politics` (654), `technology` (652), `sports` (639), `entertainment` (631), `business` (624).

**Important characteristic:** the corpus is *template-generated* — the 3,200 training rows are built from only **152 unique sentence templates**, each repeated ~21 times on average (e.g. *"The senator announced new taxation policy today"* with the policy name swapped). This is expected and intentional (it preserves training volume), but it has two direct consequences that shaped the methodology below:

- Duplicate rows are **not dropped** (doing so would shrink the training set from 3,200 to 152 rows).
- The train/validation split **cannot be a naive random or stratified split** — see [Avoiding Data Leakage](#2-avoiding-data-leakage).

Vocabulary is small and dense: 141 unique word types across 21,893 tokens, with just 88 words covering 80% of the corpus — again a signature of templated text.

---

## Methodology

### 1. EDA & Data Cleaning

- Checked for missing values (none) and duplicates (3,048 / 3,200 duplicate rows in train — expected, see above).
- Verified class balance across the 5 categories via count plots.
- Computed character/word counts per document (avg. ~6–7 words, 40–60 characters).
- `clean_text()`: lowercases, strips HTML tags, removes punctuation/special characters, and normalizes whitespace via regex. Applied to both `train.csv` and `test.csv` (rows that become empty after cleaning are dropped from train only — the test set must keep one prediction per row).
- Plotted word-count and vocabulary-coverage distributions to size later model hyperparameters (`MAX_LEN`, `MAX_VOCAB`).

### 2. Avoiding Data Leakage

Because many rows share **identical text**, a standard `train_test_split` (even stratified by category) can place copies of the same sentence in both the training and validation sets. Verified empirically: with a naive stratified split, **100% of validation rows had their exact sentence already present in training** — a textbook case of data leakage that turns "validation accuracy" into "memorization accuracy."

**Fix:** [`GroupShuffleSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) from scikit-learn, grouping on the cleaned sentence text, so that all copies of a given sentence move together into either train or validation:

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["cleaned_content"]))
```

Result: **2,623 train / 577 validation** rows, with **zero content overlap** — confirmed with an explicit assertion in the notebook.

### 3. Linguistic Analysis (spaCy)

Using `en_core_web_sm`:

- Token-level POS tags, lemmas, syntactic dependencies, and head words.
- A normalized, stacked POS-tag profile **per category**, to check whether different news domains carry distinct grammatical signatures.
- A dependency-tree visualization (`displacy`) for a sample sentence, illustrating subject/verb/object relations.

### 4. Unsupervised Topic Modeling (NMF & LDA)

Two complementary approaches, each configured for 5 topics (matching the 5 known categories):

| Method | Vectorization | Notes |
|---|---|---|
| **NMF** | TF-IDF (`max_features=1000`, English stopwords) | Deterministic, linear-algebra based; sharp, focused keyword clusters |
| **LDA** | Bag-of-words / raw counts | Probabilistic (Dirichlet-based); broader, contextual semantic groupings |

Both algorithms independently converged on the same five macro-themes (business operations, economy/finance, technology, entertainment/arts, public policy/education), which cross-validates the semantic separability of the dataset ahead of supervised modeling.

### 5. Supervised Classification

Three models of increasing complexity, all evaluated on the leakage-free validation split:

- **BiLSTM (PyTorch, from scratch):** a custom vocabulary (`<PAD>`/`<UNK>` reserved indices), post-padded sequences (`MAX_LEN=12`), an embedding layer, a bidirectional LSTM (concatenating the final forward/backward hidden states), and a dropout (p=0.3) + dense classification head.
- **BERT (`bert-base-uncased`, fine-tuned):** Hugging Face `Trainer` API, sequence classification head on top of the pretrained encoder.
- **DistilBERT (`distilbert-base-uncased`, fine-tuned):** same fine-tuning recipe, ~40% fewer parameters than BERT.

---

## Results

| Model | Architecture | Parameters | Epochs | Best Val Loss | Macro F1 |
|---|---|---:|---:|---:|---:|
| **BiLSTM** | RNN (custom, PyTorch) | 76,357 | 5 | 0.0506 | 1.00 |
| **DistilBERT** | Transformer (distilled) | 66,957,317 | 2 | 0.0102 | 1.00 |
| **BERT-Base** | Transformer | 109,486,085 | 2 | 0.0048 | 1.00 |

All three models reach a **perfect macro F1-score (1.00)** on the 577-row validation split. This is a direct consequence of the dataset's templated, low-diversity structure (152 unique sentences) rather than evidence that any architecture "solves" news classification in general — see [Known Limitations](#known-limitations--future-work).

Practical takeaways from the notebook:

- The transformers converge in fewer epochs (2 vs. 5) thanks to self-attention capturing long-range dependencies more directly than a recurrent model.
- **DistilBERT** offers effectively the same accuracy as **BERT** here at ~39% of the parameter count and lower inference latency — a good default choice for resource-constrained deployment.
- **BiLSTM** is by far the lightest model (76K vs. 67M/109M parameters) and is a reasonable choice when compute is heavily constrained, provided the real-world text stays reasonably close in style to the training distribution.

---

## Dashboard

`dashboard.ipynb` writes out a self-contained **Plotly Dash** application (`app.py`, dark theme via `dash-bootstrap-components`'s `CYBORG` theme) and launches it — in its current form, from Google Colab via `localtunnel` for a shareable public URL.

**Tabs:**

| Tab | Contents |
|---|---|
| **Home** | Project overview and the 5 target categories |
| **Dataset** | Paginated, searchable preview of the (sample) data |
| **EDA** | Category distribution bar chart; text-length and word-count histograms |
| **Models & Metrics** | Side-by-side comparison table (architecture, parameters, macro F1/precision/recall, inference latency) and confusion-matrix heatmaps per model |
| **Live Test** | Free-text box + model selector (BiLSTM / DistilBERT / BERT-Base) + "Classify Text" button, returning a predicted category with a per-class probability bar chart and simulated latency for each selected model |

> **Note on the current implementation:** to keep `app.py` fully self-contained and instantly runnable without bundling multi-hundred-MB model checkpoints, it currently (a) generates a small **synthetic placeholder dataset** at startup instead of loading `train.csv`/`test.csv`, and (b) drives the **Live Test** tab with a lightweight **keyword-matching heuristic** rather than the actual trained BiLSTM/BERT/DistilBERT weights. The confusion matrices and metrics table reflect the real results reported in [Results](#results); the live inference is illustrative. See [Future Work](#known-limitations--future-work) for how to wire up the real models.

---

## Installation

**Requirements:** Python 3.10+.

```bash
git clone https://github.com/francescachn/Text-mining-project.git
cd Text-mining-project

pip install -U spacy scikit-learn numpy pandas seaborn matplotlib torch transformers datasets
python -m spacy download en_core_web_sm

# for the dashboard
pip install dash dash-bootstrap-components plotly
```

A GPU is recommended (but not required) for fine-tuning BERT/DistilBERT.

---

## Usage

### Run the analysis notebook

Open `Copia_di_progetto_data_.ipynb` in Jupyter or click the "Open in Colab" badge above. Run cells top-to-bottom — cell 1 installs dependencies, later cells expect `train.csv`/`test.csv` to be present in the working directory.

### Run the dashboard

**Locally:**

```bash
python app.py
# then open http://localhost:8050
```

**From Google Colab** (as originally authored in `dashboard.ipynb`):

```python
!pip install dash dash-bootstrap-components scikit-learn plotly -q
!python app.py & npx -y localtunnel --port 8050 --bypass-tunnel-reminder
```

This prints a public `loca.lt` URL and a tunnel password (your outbound IP) needed to open it.

---

## Key Design Decisions

- **Duplicates were kept, not dropped**, because the corpus is intentionally template-based; dropping them would collapse 3,200 rows into 152.
- **`GroupShuffleSplit` over `train_test_split`**, to eliminate leakage from duplicate sentences between train and validation — verified with an explicit zero-overlap assertion.
- **`MAX_LEN=12`**, sized empirically from the vocabulary-coverage and word-count analysis (documents average 6–7 words, max ~10–12).
- **Dropout (p=0.3) in the BiLSTM**, added specifically to counter the overfitting risk created by a small, repetitive vocabulary.

---

## Known Limitations & Future Work

- **Synthetic, templated corpus.** The perfect F1 scores reflect the low diversity of the dataset (152 unique templates), not necessarily generalization to real, unstructured news text. Before any real-world deployment, models should be evaluated on genuinely novel, noisy text.
- **No early stopping in the BiLSTM/BERT runs** — worth adding if extending training beyond the current epoch counts, especially on less templated data.
- **Dashboard uses a keyword-heuristic + synthetic sample data** for the Live Test tab, rather than the trained checkpoints. To connect real inference:
  1. Save the trained model artifacts from the main notebook (e.g. `torch.save(...)` for the BiLSTM, `model.save_pretrained(...)` for BERT/DistilBERT, plus the tokenizer/vocab and `LabelEncoder`).
  2. Load them once at `app.py` startup.
  3. Replace `run_inference()` with real forward passes through each loaded model.
  4. Replace the synthetic `df_master`/`df_test` with the actual cleaned `train.csv`/`test.csv`.
- **Production serving:** the Dash dev server (and the Colab/localtunnel setup) is explicitly not meant for production; consider `gunicorn`/`waitress` behind a proper reverse proxy for a persistent deployment.

---

## Tech Stack

**NLP / ML:** spaCy, scikit-learn (TF-IDF, NMF, LDA, `GroupShuffleSplit`), PyTorch, Hugging Face `transformers` & `datasets`

**Data / Viz:** pandas, NumPy, matplotlib, seaborn

**Dashboard:** Plotly (graph objects, express), Dash, Dash Bootstrap Components

---

## Authors

Developed by **Francesca** ([@francescachn](https://github.com/francescachn)) as a text mining / NLP course project. Repository branch history suggests additional contributions (e.g. the `Marco` branch) — update this section with full contributor credits.

---

## License

No license file is currently included in this repository. Add a `LICENSE` file (e.g. MIT) if you intend this project to be reused by others; until then, all rights are reserved by default.
