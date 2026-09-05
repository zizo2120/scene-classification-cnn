# Multi-Class Natural Scene Image Classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zizo2120/scene-classification-cnn/blob/main/Computer_Vision_Image_Classification.ipynb)

End-to-end computer vision project classifying natural-scene photographs into six categories
(**buildings, forest, glacier, mountain, sea, street**) using a CNN trained from scratch and a
fine-tuned MobileNetV2 transfer-learning model. Built for the TechTrek Task 4 assignment.

## Contents

| File | Description |
|---|---|
| `Computer_Vision_Image_Classification.ipynb` | Main deliverable — the full notebook (EDA, preprocessing, baseline CNN, transfer learning, evaluation, error analysis, Grad-CAM, predictions). |
| `README.md` | This file. |
| `app.py` | Bonus Streamlit app for interactive image upload + prediction. |
| `requirements.txt` | Python dependencies. |
| `final_scene_classifier.keras` | Exported trained model (produced by running the notebook — not committed until you run it). |

## Dataset

[Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
(Kaggle) — ~17,000 natural-scene RGB images across 6 classes, provided as `seg_train/`,
`seg_test/`, and `seg_pred/` (unlabeled, used for the "new image" prediction demo).

## Setup & Run Instructions

### 1. Environment

Recommended: **Google Colab** with a GPU runtime (`Runtime > Change runtime type > GPU`) — the
notebook is written to run there out of the box.

To run locally instead:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

The notebook's **Section 3 — Dataset Loading** cell checks whether `./dataset/seg_train`
already exists; if not, it downloads and unzips the dataset there via the Kaggle API.

1. Create a Kaggle API token: Kaggle account settings → *Create New API Token* → downloads
   `kaggle.json`.
2. In Colab, upload `kaggle.json` (or place it at `~/.kaggle/kaggle.json`, `chmod 600`, if
   running locally) before running the cell.
3. Run the cell — it invokes
   `kaggle datasets download -d puneet6060/intel-image-classification -p ./dataset --unzip`
   automatically and prints the resolved dataset location.

**Manual download alternative:** download the zip from
[kaggle.com/datasets/puneet6060/intel-image-classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
and unzip it into `./dataset` yourself (so it contains `seg_train/`, `seg_test/`,
`seg_pred/`) — the cell will detect the existing folder and skip the download.

### 3. Run the notebook

Open `Computer_Vision_Image_Classification.ipynb` and run all cells top to bottom. It will:

- Explore and validate the dataset (EDA, corrupted-image check, class balance).
- Build `tf.data` pipelines with a leak-free, exact stratified 70/15/15 train/val/test split.
- Train a baseline CNN from scratch and a fine-tuned MobileNetV2 transfer-learning model.
- Evaluate both on the untouched test set (accuracy, precision, recall, F1, confusion matrix).
- Run error analysis and Grad-CAM explainability.
- Export the best model to `final_scene_classifier.keras`.
- Run predictions on 5+ brand-new images from `seg_pred/`.

### 4. Load the exported model elsewhere

```python
import tensorflow as tf

model = tf.keras.models.load_model("final_scene_classifier.keras")
probs = model.predict(preprocessed_image_batch)  # shape (N, 150, 150, 3), raw [0,255] uint8/float
```

### 5. (Bonus) Run the Streamlit demo

After exporting the model from the notebook:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints, and upload an image to classify.

## Results

Final metrics from a full Colab GPU run (exact 70/15/15 stratified split — 11,923 train /
2,555 val / 2,556 test images), evaluated once on the held-out test set:

| Model | Test Accuracy | Macro F1 | Macro Precision | Macro Recall | Trainable Params | Train Time |
|---|---|---|---|---|---|---|
| Baseline CNN (from scratch) | 81.49% | 0.8150 | 0.8273 | 0.8176 | 456,710 | 1793.9s |
| MobileNetV2 (fine-tuned) | **91.71%** | **0.9185** | n/a* | n/a* | 1,626,246 | 2853.4s |

\*Precision/recall for the fine-tuned model were not captured in this run's saved notebook
output (the evaluation cell's output was cleared before export); accuracy and macro F1 are
confirmed from the results table produced in Section 16.

Transfer learning improved test accuracy by **+10.2 points** and macro F1 by **+0.104**
over the from-scratch baseline. In this run the baseline's `EarlyStopping` triggered after
only 14 of its 30 budgeted epochs, so it also finished training faster in wall-clock time
than the transfer-learning model, which ran its full 25-epoch (15 + 10) schedule.

**Baseline CNN — per-class accuracy:** forest 0.949, sea 0.894, buildings 0.832,
mountain 0.827, street 0.818, glacier 0.586. Glacier was by far the hardest class in this
run — its most common confusion is glacier→mountain (128 images), well ahead of the
next-largest pairs, street→buildings (66) and buildings→street (52) — classes with
overlapping textures/colors (rock vs. mountain terrain, building facades vs. street scenes).

The fine-tuned MobileNetV2 model was selected as the final production model
(`final_scene_classifier.keras`) for its substantially higher accuracy and F1 with a
comparable parameter count, even though it took longer to train in this run.

## Notes

- All random seeds are fixed (`SEED = 42`) for reproducibility.
- Data augmentation is applied to the training split only; validation/test data is never
  augmented, and the test set is only touched once, at final evaluation.
- See the notebook's final "Conclusion" section for a full write-up answering the assignment's
  required reflection questions (architecture choices, overfitting evidence, confusion
  patterns, transfer-learning impact, final metrics, future improvements).
