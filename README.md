# End-to-End Deep Learning Project using SimpleRNN

An **End-to-End Sentiment Analysis** project built using **Simple Recurrent Neural Networks (RNN)** on the **IMDB Movie Reviews dataset**.  
This project demonstrates the full lifecycle of a deep learning application — from data preprocessing and model training to deployment with **Streamlit**.

---

## Overview

The goal of this project is to classify IMDB movie reviews as **Positive** or **Negative** using a **SimpleRNN-based model**.  
It covers all essential deep learning steps including data preparation, model training, evaluation, and deployment as an interactive web app.


---

## ⚙️ Key Steps

### 1 Data Preprocessing
- **Dataset:** IMDB Movie Reviews (Keras built-in dataset)  
- **Steps:**  
  - Load and split into training/testing sets  
  - Tokenize and pad sequences to uniform length (500 words)  
  - Prepare data for RNN input  

### 2 Model Building
The model architecture includes:
- **Embedding Layer:** Converts words into dense vectors  
- **SimpleRNN Layer:** Captures sequential dependencies in reviews  
- **Dense Output Layer:** Sigmoid activation for binary classification  

### 3 Model Compilation
The model was compiled using:

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

### 4 Early Stopping

An **EarlyStopping** callback was used to stop training when the validation loss stopped improving, ensuring better generalization.

---

###  Training Results

 **Final Results:**

- **Accuracy:** 94.42%  
- **Validation Accuracy:** 75.52%  
- **Validation Loss:** 0.6340  

---

###  Model Training Summary

The model was trained using a **Simple RNN (ReLU activation)** for binary sentiment classification on the **IMDB dataset**.  
It utilized **Adam optimizer**, **binary crossentropy loss**, and **accuracy** as the evaluation metric.  
The training process demonstrated strong convergence with stable validation accuracy.

---

###  Deployment (`main.py`)

The `main.py` file serves as the **Streamlit application** for real-time sentiment prediction.

**What it does:**
- Loads the trained model `simple_rnn_imdb.h5`
- Preprocesses and tokenizes user-input reviews
- Predicts sentiment (**Positive / Negative**)
- Displays prediction score and sentiment label interactively

**Live Demo:**  
 [IMDB Sentiment Classifier App](https://simplernn-nz9amgswmmomgbfwrmdm99.streamlit.app/)

---

###  Technologies Used

- **Python 3.x**  
- **TensorFlow / Keras**  
- **NumPy**  
- **Streamlit**  
- **Matplotlib / Seaborn**  
- **tqdm**

---

###  Installation & Usage

**1 Clone the repository**
```bash
git clone https://github.com/ManelHjaoujia/End-to-End-DL-Project-using-SimpleRNN.git
cd End-to-End-DL-Project-using-SimpleRNN
```

**2 Install dependencies**
```bash
pip install -r requirements.txt
```

**3 Run the Streamlit app**
```bash
streamlit run main.py
```

##  Future Improvements

-  Implement **LSTM** and **GRU** models for comparison  
-  Add **word embedding visualization** (t-SNE or PCA)  
-  Integrate **TensorBoard** for real-time model monitoring  
-  Apply **regularization and dropout** to reduce overfitting  

---

##  Author

**Manel Hjaoujia**  
 Master’s student in Information Systems Engineering & Data Science  
 Passionate about Artificial Intelligence, Deep Learning, and NLP  

---

##  Acknowledgments

- [TensorFlow Documentation](https://www.tensorflow.org/)  
- [Keras IMDB Dataset](https://keras.io/api/datasets/imdb/)  
- Inspiration from **Deep Learning** community tutorials and projects

