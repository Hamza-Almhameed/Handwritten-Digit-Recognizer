# Handwritten Digit Recognizer (School Project)

Simple tkinter app to draw a digit (0–9) and predict it using a saved scikit-learn model (`digit_model.pkl`).

## Features
- Draw digits on a 200×200 canvas.
- Image is cropped, resized to 8×8 and scaled to match scikit-learn's digits format.
- Prediction displayed in the window.

## Prerequisites
- Python 3.7+
- Packages: pillow, numpy, joblib, scikit-learn
- tkinter (usually included with Python installers on Windows)
