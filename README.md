# Resistor Detection (OpenCV + CNN)

> Coursework from the **Machine Vision module of my BEng Robotics degree at the University of the West of England (UWE Bristol)**. Not a production system.

Detect whether an image contains a resistor, then segment it with classical OpenCV so its colour bands can be read.

## What's in it
- **Classifier:** small Keras CNN (2 x Conv/MaxPool/Dropout, Dense 256) on 200x200 RGB images, resistor vs not-resistor.
- **Segmentation:** Canny edges, contour finding, median filtering (salt-and-pepper noise), thresholding, bounding boxes and masking to isolate the resistor body.
- `CNN_RESISTOR_report.pdf`: the write-up.

## Data
`data/resistor` (53 images) and `data/not_resistor` (149 images), collected by hand.

## Results and honest limitations
- 87.1% accuracy on a 31-image held-out split.
- This is a small student project. The test set is tiny, and the notebook trains on the full dataset, so that number overstates real performance. Validation accuracy during training was unstable (40% at epoch 10).
- The output layer is `Dense(10)` for a binary task. It works but is wasteful.
- The notebook was written for Google Colab (Drive paths). To run it, change `dir_path` to the local `data/` folder and remove the `google.colab` imports.

## Possible next steps
Train on the training split only, add k-fold cross-validation, use transfer learning (e.g. MobileNet), and finish colour-band decoding.

## Stack
Python, TensorFlow/Keras, OpenCV, scikit-image, scikit-learn.
