# EmotionDetectionClassifier

A convolutional neural network (CNN) image classifier that detects facial emotions from photos, built with Keras/TensorFlow. The model is trained to recognise four emotional states and can run inference on new, unseen face images.

---

## Emotion Classes

The classifier predicts one of four labels:

| ID | Label |
|---|---|
| 0 | Emotionless |
| 1 | Happy |
| 2 | Amazed |
| 3 | Angry |

---

## Repository Structure

```
EmotionDetectionClassifier/
├── Amazed/               # Training images — Amazed faces
├── Angry/                # Training images — Angry faces
├── Emotionless/          # Training images — Emotionless faces
├── Happy/                # Training images — Happy faces
├── NewUnseenFaces/       # New images for inference / testing
├── EmotionDetection.ipynb  # Main notebook: training + inference pipeline
├── keras_model.h5          # Saved trained Keras model
└── labels.txt              # Class index → label mapping
```

---

## How It Works

**1. Observe** — Face images are organised into labelled folders, one per emotion class. Each folder acts as the ground truth label for its images.

**2. Hypothesise** — A CNN can learn to distinguish facial emotion patterns (muscle configuration, expression geometry) from raw pixel data if given a sufficient number of labelled examples per class.

**3. Experiment** — Images are loaded, resized, and normalised. A Keras CNN model is constructed, compiled, and trained on the labelled image folders using `ImageDataGenerator` or equivalent. The trained weights are saved to `keras_model.h5`.

**4. Analyse** — The saved model is loaded and used to classify images from `NewUnseenFaces/`. Each image is preprocessed to match the training input shape, passed through the model, and the predicted class index is mapped back to a human-readable label via `labels.txt`.

**5. Conclude** — The model outputs a predicted emotion label and confidence score for each new face image.

---

## Quickstart

### Prerequisites

```bash
pip install tensorflow keras numpy pillow jupyter
```

### Clone the repo

```bash
git clone https://github.com/alketcecaj12/EmotionDetectionClassifier.git
cd EmotionDetectionClassifier
```

### Run the notebook

```bash
jupyter notebook EmotionDetection.ipynb
```

Execute all cells in order to:
- Load and preprocess the training images
- Build and train the CNN model
- Save the trained model to `keras_model.h5`
- Run predictions on images in `NewUnseenFaces/`

---

## Using the Saved Model

If you want to skip training and use the pre-trained model directly:

```python
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model and labels
model = load_model("keras_model.h5")
labels = {0: "Emotionless", 1: "Happy", 2: "Amazed", 3: "Angry"}

# Preprocess a new image
img = Image.open("NewUnseenFaces/your_face.jpg").resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_label = labels[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"Predicted emotion: {predicted_label} ({confidence:.2%})")
```

> Adjust the target image size (`224, 224`) to match the input shape used during training if different.

---

## Adding New Training Data

To extend the model with additional emotion classes or more examples:

1. Create a new folder with the class name (e.g. `Sad/`, `Surprised/`)
2. Add labelled face images to that folder
3. Update `labels.txt` with the new class index and name
4. Re-run the training cells in `EmotionDetection.ipynb`
5. Save the updated model

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3 |
| Deep learning | Keras / TensorFlow |
| Image processing | Pillow, NumPy |
| Notebook | Jupyter |
| Model format | HDF5 (`.h5`) |

---

## Author

Alket Cecaj — [github.com/alketcecaj12](https://github.com/alketcecaj12)
