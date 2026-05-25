# Plant Disease Classifier

A deep learning web app that identifies plant diseases from leaf photos.
Built for the Machine Learning elective at Erhvervsakademi København, Spring 2026.

## Live Demo
**http://165.227.128.172/**

Upload a leaf photo → the model identifies the disease and confidence level.

## Run Locally
```bash
git clone https://github.com/juliusbay/plant-disease-ml-exam2026.git
cd plant-disease-ml-exam2026
pip install flask tensorflow pillow numpy
python app.py
```
Then open **http://localhost:5000**

## Model
- Model: MobileNetV2 (transfer learning)
- Dataset: New Plant Diseases Dataset (87,000 images, 38 classes)
- Framework: TensorFlow / Keras
