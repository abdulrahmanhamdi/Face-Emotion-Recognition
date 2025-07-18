# Face Emotion Recognition with PyTorch

This project showcases a deep learning pipeline for **facial emotion recognition** using the **FER-2013** dataset. It detects and classifies human emotions from grayscale facial images into 7 categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

---

## 🧪 Technologies Used

- **PyTorch** – Deep learning framework
- **Python 3.11** – Programming language
- **Torchvision** – Image transformations and dataset management
- **Matplotlib & Seaborn** – Data and result visualization
- **Google Colab** – GPU-based training environment
- **KaggleHub** – Automated dataset downloading from Kaggle

---

## 📁 Dataset Overview

- **Name**: [FER-2013 Facial Expression Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
- **Author**: Manas Sambare
- **Image Size**: 48×48 grayscale
- **Classes**:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

### 📊 Class Distribution

**Training Set**\


**Test Set**\


> ⚠️ Class imbalance is significant, with "Happy" dominating the dataset and "Disgust" severely underrepresented.

---

## 🧠 Model Architecture

- **Input Size**: 48×48 grayscale images
- **Model Type**: Custom CNN (Convolutional Neural Network)
- **Layers**:
  - Conv + ReLU + MaxPooling
  - Dropout regularization
  - Fully Connected layers
- **Output Layer**: 7-class Softmax
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 30
- **Train/Validation Split**: 80/20

---

## 📈 Training Results

### 👀 Sample Images by Class



### 📉 Loss Curve



### 📈 Accuracy Curve



> The model improves steadily, but overfitting starts to emerge after epoch 20 due to limited training.

---

## 🧪 Evaluation

### Confusion Matrix (Train)



- ✅ Good accuracy for "Happy"
- ❌ Struggles with minority class "Disgust"
- ❓ Misclassifications occur mainly between **Fear/Sad** and **Neutral/Angry**

### Prediction Samples



> Green = Correct prediction\
> Red = Incorrect prediction

---

## 🚀 Example Usage

```python
from torchvision import transforms
from PIL import Image
from model import EmotionCNN

model = EmotionCNN()
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

img = Image.open("face.jpg").convert("L")
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
input_tensor = transform(img).unsqueeze(0)
prediction = model(input_tensor)
emotion = prediction.argmax(dim=1).item()
```

---

## 📦 Project Structure

```
├── data/
│   └── fer2013/
├── models/
│   └── emotion_model.pth
├── notebooks/
│   └── Face_Emotion_Recognition_Using_Pytorch.ipynb
├── outputs/
│   ├── train_distribution.png
│   ├── test_distribution.png
│   ├── sample_faces.png
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   ├── confusion_matrix.png
│   └── prediction_samples.png
```

---

## ⚠️ Limitations

- Only 30 epochs were used due to time and resource constraints
- "Disgust" class is severely underrepresented
- Accuracy can improve significantly with:
  - Extended training (≥ 200 epochs)
  - Balanced dataset or weighted loss
  - More advanced CNN architectures

---

## 🔮 Future Work

-

---

## 📚 Citation

> Manas Sambare, *FER-2013 Dataset*, [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 🔗 Related Resources

- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [FER-2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- [Facial Expression Recognition Challenge (ICML 2013)](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge)

```
```
