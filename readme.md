# 🧠 Domain Adaptation with DANN (MNIST → MNIST-M)

---

## 🚀 Project Overview

This project implements **Domain-Adversarial Neural Networks (DANN)** for unsupervised domain adaptation using PyTorch.

### 🎯 Goal

Train a model on:

* **Source domain:** MNIST (clean handwritten digits)
* **Target domain:** MNIST-M (digits blended with colored backgrounds)

👉 The challenge is to **generalize across domains** without using target labels during training.

---

## 🏗️ Method

We use **DANN (Domain-Adversarial Neural Network)** which consists of:

* 🧩 **Feature Extractor**
* 🔢 **Label Classifier**
* 🌐 **Domain Classifier**
* 🔁 **Gradient Reversal Layer (GRL)**

### 💡 Key Idea

Learn features that are:

* ✔ discriminative for classification
* ✔ invariant across domains

---

## 🧪 Dataset

### 📦 MNIST (Source)

* Grayscale handwritten digits
* Automatically downloaded via torchvision

### 🎨 MNIST-M (Target)

* MNIST digits blended with colored backgrounds
* Downloaded from GitHub and extracted in Colab

---

## ⚙️ Data Preprocessing

* Resize to **32×32**
* Convert MNIST to **3 channels**
* Normalize both domains:

  ```python
  transforms.Normalize((0.5,), (0.5,))
  ```

---

## 🧠 Model Architecture

### 🔹 Baseline DANN (CNN)

```text
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten
      → Label Classifier (10 classes)
      → GRL → Domain Classifier (2 classes)
```

---

### 🔥 Upgraded Model: ResNet18 DANN

* Pretrained **ResNet18 backbone**
* Replace final FC layer
* Two heads:

  * Classification head
  * Domain head (with GRL)

---

## 🏋️ Training Strategy

### ✅ Key Techniques Used

| Technique                        | Purpose                        |
| -------------------------------- | ------------------------------ |
| Alpha schedule (iteration-based) | Stabilize adversarial training |
| Domain loss weighting (0.1)      | Prevent over-regularization    |
| Warmup (first 3 epochs)          | Avoid early collapse           |
| LR Scheduler (StepLR)            | Reduce plateau                 |
| ResNet18 backbone                | Better feature extraction      |
| Training epochs = 40             | Improve convergence            |

---

## 🔁 Loss Function

```python
loss = classification_loss + λ * domain_loss
```

Where:

* λ increases gradually during training

---

## 📊 Results

### 📌 Baseline CNN DANN

| Metric          | Value |
| --------------- | ----- |
| Source Accuracy | ~99%  |
| Target Accuracy | ~65%  |

---

### 🔥 Upgraded ResNet18 DANN

| Epoch | Target Accuracy |
| ----- | --------------- |
| 1     | 0.49            |
| 10    | 0.66            |
| 20    | 0.69            |
| 30    | 0.70            |
| 40    | **~0.71–0.72**  |

---

## 📈 Analysis

### ✅ Observations

* Source domain quickly reaches near-perfect accuracy
* Target domain improves steadily
* Plateau observed after ~epoch 20
* ResNet significantly boosts performance

---

### ⚠️ Challenges

* Adversarial training instability
* Domain classifier overpowering feature extractor
* Plateau in later epochs

---

## 🖼️ Visualization

### 🔍 Random Prediction (with Confidence)

The project includes visualization functions:

* Random samples from dataset
* Ground truth vs prediction
* Confidence score
* Color-coded correctness:

  * 🟢 correct
  * 🔴 incorrect

---

## 🧪 Example Output

```text
GT: 5 | Pred: 5 | Conf: 0.97  ✅
GT: 3 | Pred: 8 | Conf: 0.62  ❌
```

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib tqdm
```

---

### 2. Clone dataset

```bash
git clone https://github.com/mashaan14/MNIST-M.git
```

---

### 3. Run training

* Open in **Google Colab (GPU T4 recommended)**
* Run all cells sequentially

---

### 4. Evaluate

```python
test_random_images_with_conf(model, mnist_m_test, device)
```

## 🔬 Key Insights

* Domain adaptation is a **minimax optimization problem**
* Proper scheduling (alpha, LR) is critical
* Balance between classification and domain loss is essential
* Strong backbone (ResNet) improves generalization significantly

---

## 🚀 Future Improvements

* 🔥 CDAN (Conditional Domain Adversarial Network)
* 🔥 MMD-based domain alignment
* 🔥 t-SNE visualization of feature space
* 🔥 Pseudo-labeling on target domain
* 🔥 Entropy minimization

---

## 🎯 Conclusion

This project demonstrates a complete pipeline for domain adaptation:

* ✅ Stable DANN training
* ✅ Strong baseline (~65%)
* 🔥 Improved model (~70%+ with ResNet)
* ✅ Visualization and evaluation tools

---

## 👨‍💻 Author

* Name: *Nguyen Hai Tien Phat*
* Field: Computer Vision / Deep Learning
* Focus: Domain Adaptation, Robust AI

---

## ⭐ If you find this useful

Consider giving it a ⭐ on GitHub!

---
