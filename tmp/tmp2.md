# 📘 Loading Frames in TensorFlow vs TensorFlow Lite

## 🔹 1. TensorFlow (Training + Preprocessing)

TensorFlow is designed for **training and preprocessing**, so it provides multiple convenient APIs for loading frames/images.

### a) From a **directory with images and labels**
Typical dataset folder structure:
```
dataset/
  cap/
    img001.jpg
    img002.jpg
  mug/
    img001.jpg
    img002.jpg
  ceiling/
    img001.jpg
```

Load with `image_dataset_from_directory`:
```python
import tensorflow as tf

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    image_size=(224, 224),
    batch_size=32
)
```

### b) From a **video (extracting frames)**
```python
import cv2
import tensorflow as tf

cap = cv2.VideoCapture("cap_video.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))
    tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
    frames.append(tensor)

cap.release()
video_tensor = tf.stack(frames)  # shape: (num_frames, 224, 224, 3)
```

### c) From **NumPy arrays or Kaggle datasets**
```python
import numpy as np
import tensorflow as tf

X = np.load("images.npz")["arr_0"]
y = np.load("labels.npz")["arr_0"]

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32).shuffle(1000)
```

### d) Using **tf.data pipeline**
```python
import tensorflow as tf
import pathlib
import os

data_dir = pathlib.Path("dataset")
list_ds = tf.data.Dataset.list_files(str(data_dir/"*/*"))

def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return img, label

dataset = list_ds.map(process_path).batch(32)
```

---

## 🔹 2. TensorFlow Lite (Inference Only)

### a) Single image
```python
import tensorflow as tf
import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread("cap.jpg")
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img.astype(np.float32)/255.0, axis=0)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print(output)
```

### b) Multiple images
```python
images = ["cap.jpg", "mug.jpg", "ceiling.jpg"]

for path in images:
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = np.expand_dims(img.astype(np.float32)/255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(path, output)
```

### c) Video frames
```python
cap = cv2.VideoCapture("cap_video.mp4")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (224,224))
    frame = np.expand_dims(frame.astype(np.float32)/255.0, axis=0)
    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(output)
cap.release()
```

---

# 📊 Summary Table

| Framework         | Input Methods | Typical Use Case |
|-------------------|--------------|------------------|
| **TensorFlow**    | `tf.keras.utils.image_dataset_from_directory`, `tf.data.Dataset`, NumPy arrays, Kaggle datasets, OpenCV video frames | **Training** (flexible data pipelines) |
| **TensorFlow Lite** | `Interpreter.set_tensor()` with NumPy arrays (images or frames) | **Inference only** (must preprocess manually) |

✅ **Key takeaway**:  
- Use **TensorFlow** for dataset loading, preprocessing, and training (directories, videos, Kaggle datasets).  
- Use **TensorFlow Lite** only for running inference — you must preprocess frames to match the trained model’s input.
