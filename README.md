# ASL Detection

This project aims to detect and recognize American Sign Language (ASL) gestures from live video streams or pre-recorded videos. Using computer vision and deep learning techniques, the system is capable of identifying ASL signs and converting them into corresponding English letters or words in real-time. This project can be used to bridge the communication gap between the deaf and hard-of-hearing community and those unfamiliar with sign language.


# Features

- **Real-Time Detection** : The system processes live video input and predicts ASL gestures in real time.
- **Gesture Recognition** : It recognizes individual hand signs representing letters of the ASL alphabet.
- **Deep Learning Model** : A convolutional neural network (CNN) trained on a dataset of ASL gestures.
- **Webcam Integration** : Supports webcam input for real-time predictions.
- **FastAPI Backend** : The backend is built using FastAPI to handle video stream input and return predictions.
- **WebSocket Support** : Enables live streaming of video for real-time sign language detection.

# Technologies Used
- Python
- Google medipipe :[link](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- OpenCV (for image and video processing)
- TensorFlow/Keras (for the deep learning model)
- FastAPI (for the backend API)
- WebSockets (for real-time communication)
 
## Table of Contents
1. Data set creation
    - Hand Landmark dataset
    - Image with Hand Landmark
        
2. Data manipulation and visualization
    - Remove NA and duplicate
    - Check Outliyes
    - Create a scalar dataset
    - Data is summarized()

3. Training the model
    - Clustering(ML)
    - CNN
        - Normal cnn architecture
        - VIT architecture(Vision transformer)

3. Inference
    - Image
    - Vedio
 

4. API
5. How to Setup
6. Others



## 1. Data set creation [Notebook]()

### Data set creation 1: Hand Landmark dataset and Image with Hand Landmark image dataset

- **Dataset (Drive or GitHub URL)**: [Final Datset](https://github.com/username/repo/blob/main/data/house_prices)

    This dataset is designed to aid in the recognition and interpretation of American Sign Language (ASL) gestures using machine learning models. It contains images/csv of hand gestures representing different ASL signs, including alphabets common phrases. The dataset can be used for classification tasks, enabling models to learn and identify specific ASL signs from the provided visual data.

    This dataset includes two types of data images and documents. Reference the following dataset to create these images and document. Hand Landmarks were extracted using Google Mediapipe Hand Landmark recognitionmodel.
    - **Refference (Drive or Cloud URL)**: [Google Meadiapipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)

    - ![Hand Landmark ](https://example.com/swagger1.png)

    The MediaPipe Hand Landmarker task lets you detect the landmarks of the hands in an image. You can use this task to locate key points of hands and render visual effects on them. This task operates on image data with a machine learning (ML) model as static data or a continuous stream and outputs hand landmarks in image coordinates, hand landmarks in world coordinates and handedness(left/right hand) of multiple detected hands.


    - asl-alphabet :[Keggle dataset link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
    - aslamerican-sign-language-aplhabet-dataset : [Keggle dataset link](https://www.kaggle.com/datasets/debashishsau/  aslamerican-sign-language-aplhabet-dataset)
    - synthetic-asl-alphabet :[Keggle dataset link]( https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)

- **Steps of create this datasets :**

    - Detect the hand use (mediapipe/hand_landmarker/detection)
        - ![Detect hand](https://example.com/swagger1.png)
    - Crop the hand area
        - cutoff=30
        - desired_size=(300,300,3)
        - ![Crop Hand](https://example.com/swagger1.png)

    - Extracted the landmark use mediapipe
    ```python
        Landmark 0: (x: 303, y: 473, z: 1.1764698228944326e-06)
        Landmark 1: (x: 355, y: 438, z: -0.06731968373060226)
        Landmark 2: (x: 379, y: 380, z: -0.10316909849643707)
        Landmark 3: (x: 357, y: 342, z: -0.13917222619056702)
        Landmark 4: (x: 327, y: 305, z: -0.1697036474943161)
        Landmark 5: (x: 346, y: 246, z: -0.05628802999854088)
        Landmark 6: (x: 356, y: 155, z: -0.11031270772218704)
        Landmark 7: (x: 358, y: 97, z: -0.14328952133655548)
        Landmark 8: (x: 356, y: 48, z: -0.1625644564628601)
        Landmark 9: (x: 292, y: 257, z: -0.0652509406208992)
        Landmark 10: (x: 252, y: 164, z: -0.1389753520488739)
        Landmark 11: (x: 224, y: 102, z: -0.18423861265182495)
        Landmark 12: (x: 199, y: 49, z: -0.20252306759357452)
        Landmark 13: (x: 250, y: 292, z: -0.08093016594648361)
        Landmark 14: (x: 250, y: 253, z: -0.19030404090881348)
        Landmark 15: (x: 291, y: 314, z: -0.2105305939912796)
        Landmark 16: (x: 316, y: 358, z: -0.18995937705039978)
        Landmark 17: (x: 221, y: 340, z: -0.10010740160942078)
        Landmark 18: (x: 241, y: 324, z: -0.19231398403644562)
        Landmark 19: (x: 280, y: 368, z: -0.20064625144004822)
        Landmark 20: (x: 304, y: 401, z: -0.18099738657474518)
    ```
    - Save the landmark in the data frame : [Keggle dataset link](https://www.kaggle.com/datasets/grassknoted/)

    - Save the copied image with landmark
        - ![Saved Image sample 1](https://example.com/swagger1.png)
        - ![Saved Image sample 2](https://example.com/swagger1.png)

- **Content :**

    - Images : High-quality RGB images of individual ASL gestures with Hand Land mark.
    - Labels: Corresponding labels for each gesture, including alphabets (A-Z) other common ASL gestures or phrases.

    - Document(csv) : ASL gestures with Hand Land mark.
    - Labels: Corresponding labels for each gesture, including alphabets (A-Z) other common ASL gestures or phrases.

## 2. Data manipulation and visualization. [Notebook]()

The dataset used for American Sign Language detection is numerical. Before training the model, the following preprocessing steps were applied:

- Handling Missing Values: Removed rows/columns with missing data to ensure data integrity.
- Removing Duplicates: Identified and dropped duplicate records to avoid redundancy.
- Outlier Detection & Removal:
    - Interquartile Range (IQR): Removed extreme values based on statistical distribution.
    - Isolation Forest (ISO): Applied an anomaly detection technique to filter outliers.

### Processed Datasets 
After preprocessing, two versions of the dataset were saved:

- IQR Dataset: Processed dataset after outlier removal using the IQR method.
- ISO Dataset: Processed dataset after outlier removal using the Isolation Forest method.

## Training model [Notebook]()

After preprocessing, the dataset was used to train four machine learning models for American Sign Language detection. The models were trained and evaluated based on key performance metrics.

### Models Used:
- Random Forest
    - An ensemble learning method that builds multiple decision trees and combines their outputs for better accuracy.
    - Handles high-dimensional data well and reduces overfitting.
- Evaluation Matrix
![image][]

- Gradient Boosting (GBM)
    - A boosting technique that builds models sequentially, correcting errors from previous models.
    - Works well with structured numerical datasets.
- Evaluation Matrix
![image][]

- CatBoost
    - A high-performance gradient boosting algorithm optimized for categorical features.
    - Reduces the need for extensive preprocessing and improves accuracy.
- Evaluation Matrix
![image][]

- XGBoost
    - An optimized gradient boosting framework known for its speed and efficiency.
    - Regularization techniques help prevent overfitting.
- Evaluation Matrix
![image][]

### Training Process:
Each model was trained on the preprocessed dataset (ISO versions).

Hyperparameter tuning was performed using GridSearchCV/RandomizedSearchCV to find the best configurations.

- Models were evaluated using:

    - Accuracy
    - Precision
    - Recall
    - F1-score

## 3. Inference
After training, the models were deployed for real-time inference on both images and videos to detect American Sign Language gestures.

- Inference with Videos
    - The model processes video frames in real time.
    - Each frame is analyzed to detect and classify the sign language gesture.
    - The system provides continuous predictions for dynamic signing.


## 4. API

### Pre-requisites
- python 3.11.11

Follow these steps to set up your development environment for this project:

### Create a New `venv`

1. Navigate to your project directory:

   ```bash
   cd /path/to/your/project
   ```

2. Create a virtual environment:

   ```bash
   python -m venv <venv_name>
   ```

### Activate and Deactivate `venv`

- In `cmd`:

   ```bash
   <venv_name>\Scripts\activate
   ```

- In bash:

   ```bash
   source <venv_name>/Scripts/activate

   # To deactivate the virtual environment:
   deactivate
   ```

### Create, Activate & Deactivate `venv` using conda

- Use Anaconda Navigator to create a venv:

   ```bash
   # Activate the conda environment
   conda activate <venv_name>

   # To deactivate the conda environment
   conda deactivate
   ```

### Install the Dependencies

- You can also use a `requirements.txt` file to manage your project's dependencies. This file lists all the required packages and their versions.
1. Install packages from `requirements.txt`:

   ```
   pip install -r requirements.txt
   ```
    This ensures that your development environment matches the exact package versions specified in `requirements.txt`.

2. Verify installed packages:

   ```bash
   pip list
   ```
    This will display a list of packages currently installed in your virtual environment, including the ones from `requirements.txt`.




### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project-directory
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    python  main.py
    ```

