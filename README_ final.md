# TransforMiners — News Text Mining & Classification

A full text-mining pipeline that classifies short news snippets into five categories — politics, business, entertainment, sports, technology — combining classical topic modeling (NMF, LDA) with a from-scratch BiLSTM and fine-tuned BERT / DistilBERT transformers, all served through a fully-wired, real-model Plotly Dash dashboard. Developed for a Text Mining / NLP course project (Università Cattolica del Sacro Cuore).

## Overview

Given a corpus of short, template-generated news headlines, this project builds a full
text-mining pipeline end to end:

- Explore & clean the raw text (dedup analysis, length distributions, vocabulary coverage).
- Split safely using a group-aware strategy that eliminates train/validation leakage
  caused by duplicate sentences.
- Profile the language with spaCy (POS tagging, lemmatization, dependency parsing).
- Discover latent topics with two unsupervised methods (NMF and LDA) and compare them
  against the ground-truth labels.
- Train and compare three classifiers of increasing complexity — a custom BiLSTM,
  fine-tuned BERT, and fine-tuned DistilBERT.
- Present everything interactively in a multi-tab Plotly Dash dashboard, wired to the
  real trained models.

## Project Structure

```
Text-mining-project/
├── Copia_di_progetto_data_.ipynb   # main pipeline: EDA, cleaning, spaCy, NMF/LDA, BiLSTM, BERT, DistilBERT
├── Progetto_con_Dashboard.ipynb    # saves dashboard artifacts, writes app.py, launches it (Colab + localtunnel)
├── app.py                          # standalone Plotly Dash app, produced by `%%writefile` in the dashboard notebook
├── train.csv                       # training data — 3,200 rows: `content`, `category`
├── test.csv                        # held-out test data — 800 rows: `content` only
└── README.md
```

## Dataset

| File | Rows | Columns | Notes |
|---|---|---|---|
| `train.csv` | 3,200 | `content`, `category` | Labeled training data |
| `test.csv` | 800 | `content` | Unlabeled held-out test set |

Categories (well balanced, ~18.9–21.2% each): business, entertainment, politics, sports,
technology.

**Important characteristic**: the corpus is template-generated — the 3,200 training rows
are built from only **152 unique sentence templates**, each repeated ~21 times on average
(e.g. *"The senator announced new taxation policy today"* with the policy name swapped).
This is expected and intentional (it preserves training volume), but it has two direct
consequences that shaped the methodology below:

- Duplicate rows are **not** dropped (doing so would shrink the training set from 3,200 to
  152 rows).
- The train/validation split **cannot** be a naive random or stratified split — see
  *Avoiding Data Leakage*.

Vocabulary is small and dense: 141 unique word types across 21,893 tokens, with just 88
words covering 80% of the corpus — again a signature of templated text.

## Methodology

### 1. EDA & Data Cleaning

- Checked for missing values (none) and duplicates (expected and retained — see above;
  the held-out test set is drawn from the same template pool and is duplicate-heavy too).
- Verified class balance across the 5 categories via count plots.
- Computed character/word counts per document (avg. ~6–7 words, 40–60 characters).
- `clean_text()`: lowercases, strips HTML tags, removes punctuation/special characters,
  and normalizes whitespace via regex. Applied to both `train.csv` and `test.csv` (rows
  that become empty after cleaning are dropped from train only — the test set must keep
  one prediction per row).
- Plotted word-count and vocabulary-coverage distributions to size later model
  hyperparameters (`MAX_LEN`, `MAX_VOCAB`).

### 2. Data Split

Because many rows share identical text, a standard `train_test_split` (even stratified by
category) can place copies of the same sentence in both the training and validation sets.
Verified empirically: with a naive stratified split, a large share of validation rows had
their exact sentence already present in training — a textbook case of data leakage that
turns "validation accuracy" into "memorization accuracy."

Two group-aware strategies were compared:

- **Strategy 1 — `GroupShuffleSplit`**, grouping on the cleaned sentence text:
  ```python
  gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2)
  train_idx, val_idx = next(gss.split(train_df, groups=train_df["cleaned_content"]))
  ```
  Result: 2,623 train / 577 validation rows, zero content overlap — but a noticeably
  unbalanced validation set across categories (e.g. Entertainment ~27%, Business ~13%).

- **Strategy 2 — `StratifiedGroupKFold`** (used for all downstream modeling): keeps
  groups intact **and** balances categories across the split. Final split: **2,562
  train / 638 validation rows**, zero content overlap — confirmed with an explicit
  assertion in the notebook.

### 3. Linguistic Analysis (spaCy)

Using `en_core_web_sm`:

- Token-level POS tags, lemmas, syntactic dependencies, and head words.
- A normalized, stacked POS-tag profile per category, to check whether different news
  domains carry distinct grammatical signatures.
- A dependency-tree visualization (`displacy`) for a sample sentence, illustrating
  subject/verb/object relations.

### 4. Unsupervised Topic Modeling (NMF & LDA)

Two complementary approaches, each configured for 5 topics (matching the 5 known
categories):

| Method | Vectorization | Notes |
|---|---|---|
| NMF | TF-IDF (`max_features=1000`, English stopwords) | Deterministic, linear-algebra based; sharp, focused keyword clusters |
| LDA | Bag-of-words / raw counts | Probabilistic (Dirichlet-based); broader, contextual semantic groupings |

Both algorithms independently recover topics that map cleanly onto the 5 ground-truth
categories, cross-validating the semantic separability of the dataset ahead of supervised
modeling. The LDA topics recovered, for example:

1. *records, release, box, breaks, office* → entertainment (box-office releases)
2. *software, update, brings, improvements, functionality* → technology
3. *new, yesterday, regarding, proposed, legislation* → politics
4. *won, thrilling, match, team, championship* → sports
5. *company, announces, ceo, changes, strategic* → business

### 5. Supervised Classification

Three models of increasing complexity, all evaluated on the leakage-free validation
split (638 rows):

- **BiLSTM** (PyTorch, from scratch): a custom vocabulary (`<PAD>`/`<OOV>` reserved
  indices), padded sequences (`MAX_LEN=12`), an embedding layer, a bidirectional LSTM
  (concatenating the final forward/backward hidden states), and a dropout (`p=0.3`) +
  dense classification head.
- **BERT** (`bert-base-uncased`, fine-tuned): Hugging Face `Trainer` API, sequence
  classification head on top of the pretrained encoder.
- **DistilBERT** (`distilbert-base-uncased`, fine-tuned): same fine-tuning recipe, ~39%
  fewer parameters than BERT.

## Results

| Model | Architecture | Parameters | Epochs | Best Val Loss | Macro F1 |
|---|---|---|---|---|---|
| BiLSTM | RNN (custom, PyTorch) | 76,165 | 5 | 0.0607 | 1.00 |
| DistilBERT | Transformer (distilled) | 66,957,317 | 2 | 0.0334 | 1.00 |
| BERT-Base | Transformer | 109,486,085 | 2 | 0.0200 | 1.00 |

All three models reach a perfect macro F1-score (1.00) on the 638-row validation split.
This is a direct consequence of the dataset's templated, low-diversity structure (152
unique sentences) rather than evidence that any architecture "solves" news classification
in general — see *Known Limitations*.

Practical takeaways from the notebook:

- The transformers converge in fewer epochs (2 vs. 5) thanks to self-attention capturing
  long-range dependencies more directly than a recurrent model.
- DistilBERT offers effectively the same accuracy as BERT here at ~39% fewer parameters
  and lower inference latency — a good default choice for resource-constrained
  deployment.
- BiLSTM is by far the lightest model (76K vs. 67M/109M parameters) and is a reasonable
  choice when compute is heavily constrained, provided the real-world text stays
  reasonably close in style to the training distribution.

## Dashboard

`TransforMiners Project.ipynb` saves the trained model artifacts, computes the real
validation-set metrics and EDA statistics, writes out a self-contained Plotly Dash
application (`app.py`, dark theme via `dash-bootstrap-components`'s CYBORG theme), and
launches it — currently from Google Colab via `localtunnel`, for a shareable public URL.

Unlike an earlier iteration, the dashboard now loads the **actual trained checkpoints**
(BiLSTM weights, fine-tuned BERT/DistilBERT, vocabulary, label encoder) and runs real
forward passes for both the EDA and Live Test tabs — nothing is synthetic or hardcoded.

| Tab | Contents |
|---|---|
| Home | Project overview, dataset characteristics, the data-leakage fix, and the final results table |
| Dataset | Paginated table of the real training data (`category` + `content`) |
| EDA | Seven sub-tabs: Category Distribution, Text Length (char/word count), Vocabulary Coverage Curve, POS-Tag Profile by Category, NMF Topics, LDA Topics, Word Cloud |
| Models & Metrics | Comparison table (architecture, parameters, macro F1/precision/recall) and a confusion-matrix heatmap per model, computed on the real validation set |
| Live Test | Free-text box + model selector (BiLSTM / DistilBERT / BERT-Base, any combination) + "Analyze" button, running real inference and returning a predicted category with a per-class probability bar chart |

### Dashboard artifacts

The dashboard notebook saves the following files (read by `app.py` at startup):

- `bilstm_model_weights.pth`, `bilstm_config.json`, `bilstm_word_index.json` — BiLSTM
  weights and vocabulary.
- `label_encoder.pkl` — shared label encoder across all three models.
- `distilbert_model/`, `bert_model/` — fine-tuned Hugging Face checkpoints
  (`config.json` + weights + tokenizer, via `Trainer.save_model()`).
- `real_train_data.csv` — the cleaned training dataframe, used by the Dataset and EDA tabs.
- `real_metrics.json` — accuracy/macro F1/precision/recall/confusion matrix per model,
  computed on the real validation set.
- `eda_extra.json` — text-length arrays, vocabulary-coverage curve, POS distribution per
  category, and NMF/LDA topic words.

## Installation

Requirements: Python 3.10+.

```bash
git clone https://github.com/francescachn/Text-mining-project.git
cd Text-mining-project

pip install -U spacy scikit-learn numpy pandas seaborn matplotlib torch transformers datasets evaluate accelerate
python -m spacy download en_core_web_sm

# for the dashboard
pip install dash dash-bootstrap-components plotly wordcloud
```

A GPU is recommended (but not required) for fine-tuning BERT/DistilBERT.

## Usage

### Run the analysis notebook

Open `TransforMiners Project.ipynb` in Jupyter or Colab. Run cells top-to-bottom — later
cells expect `train.csv`/`test.csv` to be present in the working directory, and the
BiLSTM/BERT/DistilBERT training cells populate the in-memory objects that the dashboard
notebook depends on.

### Run the dashboard

Locally, once `app.py` and its artifacts exist in the working directory:

```bash
python app.py
# then open http://localhost:8050
```

From Google Colab (as authored in `TransforMiners Project.ipynb`), in the same runtime
right after the main notebook's training cells:

1. Run the cells that save the model artifacts, compute `real_metrics.json` and
   `eda_extra.json`, and write `app.py`.
2. Run the launch cell:
   ```python
   !pip install dash dash-bootstrap-components wordcloud transformers -q
   !pkill -f "app.py"
   !nohup python app.py > server.log 2>&1 &
   ```
3. Check `server.log` to confirm the app started without errors before opening the link.
4. Open the tunnel:
   ```python
   !npx -y localtunnel --port 8050
   ```
   This prints a public `loca.lt` URL and a tunnel password (the Colab VM's outbound IP).

## Key Design Decisions

- Duplicates were kept, not dropped, because the corpus is intentionally template-based;
  dropping them would collapse 3,200 rows into 152.
- `StratifiedGroupKFold` over a naive `train_test_split`, to eliminate leakage from
  duplicate sentences between train and validation while keeping categories balanced —
  verified with an explicit zero-overlap assertion.
- `MAX_LEN=12`, sized empirically from the vocabulary-coverage and word-count analysis
  (documents average 6–7 words, max ~10–12).
- Dropout (`p=0.3`) in the BiLSTM, added specifically to counter the overfitting risk
  created by a small, repetitive vocabulary.

## Known Limitations & Future Work

- **Synthetic, templated corpus.** The perfect F1 scores reflect the low diversity of the
  dataset (152 unique templates), not necessarily generalization to real, unstructured
  news text. Before any real-world deployment, models should be evaluated on genuinely
  novel, noisy text.
- **No early stopping** in the BiLSTM/BERT runs — worth adding if extending training
  beyond the current epoch counts, especially on less templated data.
- **Dependency-parse visualization** (`displacy`) is currently notebook-only; it is not
  embedded in the dashboard's EDA tab.
- **Production serving**: the Dash dev server (and the Colab/localtunnel setup) is
  explicitly not meant for production; consider `gunicorn`/`waitress` behind a proper
  reverse proxy for a persistent deployment.

## Tech Stack

**NLP / ML**: spaCy, scikit-learn (TF-IDF, NMF, LDA, `StratifiedGroupKFold`), PyTorch,
Hugging Face `transformers` & `datasets`

**Data / Viz**: pandas, NumPy, matplotlib, seaborn, `wordcloud`

**Dashboard**: Plotly (graph objects, express), Dash, Dash Bootstrap Components

## Authors

**Group name**: TransforMiners

- Cheng Ting Francesca
- Fonio Gabriella
- Stranges Marco

## Course

Developed for the Data Visualization and Text Mining course at Università Cattolica del
Sacro Cuore, Milan (IT).
