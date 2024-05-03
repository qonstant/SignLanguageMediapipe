
# Real Time Sign Language Translator with Mediapipe

This is a Kazakh Sign Language Translator that recognizes keypoints using Google's Mediapipe framework and converts gestures into words.

### Run

1. Clone the repository:
   ```sh
   git clone https://github.com/qonstant/SignLanguageMediapipe.git
2. Install dependencies which was used in this project with Python 3.11.8:
   ```sh
   pip3 install opencv-python==4.9.0
   pip3 install numpy==1.26.4
   pip3 install mediapipe==0.9.2.1

### Introduction:

In Kazakhstan, the absence of a Kazakh Sign Language translator app leaves the deaf and hard of hearing community without a vital tool for communication. This project fills a crucial gap by providing a means to bridge linguistic barriers, ensuring inclusivity and equal access to communication for all. Its importance lies not only in addressing an urgent need but also in pioneering innovation where none exists.

### What is Mediapipe?

The MediaPipe Gesture Recognizer facilitates the creation of machine learning models that track only the position of your hand, rather than the entire picture. This focused approach saves time during training, reduces memory consumption, and accelerates model development. By leveraging data from an Excel table containing coordinates of the fingers, it enables faster model training, allowing for real-time recognition of hand gestures and efficient integration of corresponding application features.

<img width="1073" alt="Mediapipe" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/5114ce60-7060-4f9d-94f1-98d2c86c7ab8">

### Data:

I have used my own dataset.

<img width="1033" alt="Dataset1" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/09072577-3762-4ba5-92db-2c4b87af5a45">

When saving a position of key points, you will need to press a specific key along with its corresponding ID. Consequently, the ID will be stored in the first column, while the coordinates of the keypoints will be stored in subsequent columns.

This is how it gets coordinates of the landmarks(hand key points):

```python
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point
```

### Datasets structure

There are only 14 static words and 4 movements available yet. 3456 rows for static words and 5296 for actions.

Static words save

<img width="974" alt="Dataset2" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/377d50ef-e6d8-46de-abac-7c6c102ac173">

### Model building

Here we have 2 models, one for action with using LSTM ( Long Short-Term Memory) and another for static words without LSTM.

First one:

```python
use_lstm = False
model = None

if use_lstm:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION, )),
        tf.keras.layers.Reshape((TIME_STEPS, DIMENSION), input_shape=(TIME_STEPS * DIMENSION, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16, input_shape=[TIME_STEPS, DIMENSION]),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
else:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
```
* When use_lstm is True, the model includes LSTM layers, suitable for sequence data processing.
* When use_lstm is False, the model is a simpler architecture without LSTM layers.

Second one:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```

* Input Layer: Accepts input data with a shape of 42 (21 * 2), which presumably represents 21 keypoints with 2 coordinates each.
* Dropout Layer (0.2): Applies dropout regularization to randomly deactivate 20% of the input units during training to prevent overfitting.
* Dense Layer (20 units): A fully connected layer with 20 units, applying the Rectified Linear Unit (ReLU) activation function for non-linearity.
* Dropout Layer (0.4): Another dropout layer, this time with a rate of 40%.
* Dense Layer (10 units): Another fully connected layer with 10 units and ReLU activation.
* Output Layer: Produces the final classification output with NUM_CLASSES units and softmax activation, suitable for multi-class classification tasks.

### Model Training

<img width="1149" alt="Training" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/23cf1921-3ef4-4c83-aa44-17486cb2c49f">

### Training Results

Model for recognition of actions:

<img width="735" alt="action_loss" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/839e07c5-1352-4bbc-a27b-848d2650c85a">
<img width="562" alt="cm_action" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/e301083f-0eba-44f5-a4dc-e9a9841c4595">


Model for recognition of static words:

<img width="752" alt="static_loss" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/3fde8efe-5f78-465d-a259-ea797515f89b">
<img width="561" alt="cm_static" src="https://github.com/qonstant/SignLanguageMediapipe/assets/98641240/f277765f-aa18-4ba8-9380-b037527c1d22">

### Results



