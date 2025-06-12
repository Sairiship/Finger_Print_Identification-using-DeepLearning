
# 🔍 Fingerprint Identification using Deep Learning

A deep learning-based dual-classification system that identifies:

* 🧑 **Subject Identity** (600 individuals)
* ✋ **Finger Type** (10 fingers: Left and Right hand – little, ring, middle, index, thumb)

The system uses Convolutional Neural Networks (CNNs) trained on the **SOCOFing fingerprint dataset**, using grayscale images resized to `96x96`.

---

## 📁 Dataset

**Source:** [SOCOFing Dataset](https://www.kaggle.com/datasets/latifa99/fingerprint)

**Structure:**

```
data/
├── SOCOFing/
│   ├── Altered/
│   │   ├── Altered-Easy/
│   │   ├── Altered-Medium/
│   │   └── Altered-Hard/
│   └── Real/
```

**Classes:**

* **600** unique subject IDs
* **10** finger types (Left/Right \* Little, Ring, Middle, Index, Thumb)

---

## 🧠 Models Used

Two separate CNN models were trained:

### 1️⃣ Subject ID Model

* 🔹 **Task:** Predicts the person (ID range: `0–599`)
* 🔹 **Final Dense Layer:** 600 units
* 🔹 **Evaluation Set:** `y_SubjectID_test`

### 2️⃣ Finger Type Model

* 🔹 **Task:** Predicts finger type (`0–9`)
* 🔹 **Final Dense Layer:** 10 units
* 🔹 **Evaluation Set:** `y_fingerNum_test`

---

## 🏗️ Architecture Highlights

* **Input Shape:** `96 x 96 x 1` (Grayscale)
* **CNN Stack:**

  * `Conv2D → BatchNormalization → MaxPooling`
  * `Dropout` for regularization
  * `Dense` fully connected layers
* **Loss Function:** `Categorical Crossentropy`
* **Optimizer:** `Adam` with learning rate scheduling
* **Regularization:** `L2` weight decay

---

## 📊 Training Details

* **Epochs:** `20`
* **Batch Size:** `64`
* **Callbacks Used:**

  * `EarlyStopping`
  * `ReduceLROnPlateau`
  * `ModelCheckpoint`
  * `TensorBoard`

---

## 🔬 Evaluation Metrics

* ✅ Training and Validation **Accuracy/Loss** plots
* ✅ **Test Accuracy** for:

  * Subject ID model
  * Finger classification model
* ✅ **Confusion Matrix** (for Finger prediction)
* ✅ **Random Fingerprint Prediction Demo**:

  * Subject ID match check
  * Finger type prediction verification

---

## 🔍 Visualizations

* 📈 Training & Validation **Accuracy/Loss** plots
* 📉 Confusion Matrix (Finger classification)
* 🔎 Random test image prediction:

  * Fingerprint sample displayed
  * Model-predicted vs real ID and finger type

---

## 🔧 Technologies Used

* 🐍 Python 3
* 🧠 Keras & TensorFlow
* 📷 OpenCV (for image processing)
* 📊 Matplotlib & Seaborn (for visualizations)
* 📘 Scikit-learn (metrics, confusion matrix)

---

## 📁 Project Structure

```
Fingerprint_Recognition/
├── main.ipynb                # Core implementation notebook
├── README.md                 # Project overview
└── data/
    └── SOCOFing/
        ├── Altered/
        │   ├── Altered-Easy/
        │   ├── Altered-Medium/
        │   └── Altered-Hard/
        └── Real/
```




