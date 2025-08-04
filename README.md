# SimpleRNN-MovieReviewAnalysis
Review Sentiment Analyzer (RNN)

This project uses a Recurrent Neural Network (RNN) to perform sentiment analysis on movie reviews. A user-friendly web application, built with Streamlit, allows for real-time classification of a review as either "Positive" or "Negative".

## Live Application

**You can access the live deployed application here:**
[**➡️ Live Sentiment Analyzer App**](https://simplernn-moviereviewanalysis-4ueajt6dyngdyv6ytj2jda.streamlit.app/)

---

## Dataset

The project utilizes the popular **IMDB Movie Review Dataset**, which is built into the Keras library. This large dataset consists of 50,000 movie reviews from the Internet Movie Database. Unnecessary columns are not present as the data is loaded directly as processed text.

---

## Model and Preprocessing

The model is a Sequential Recurrent Neural Network built with TensorFlow and Keras, specifically designed to understand the context and sequence in text data.

* **Preprocessing**:
    * *Tokenization:* Each word in a review is mapped to a unique integer using a pre-built `word_index` dictionary from the IMDB dataset.
    * *Index Offsetting:* The first three indices (0, 1, 2) are reserved for special tokens: `<PAD>` (padding), `<START>` (start of sequence), and `<OOV>` (out-of-vocabulary words). Therefore, every word index from the dictionary is offset by +3.
    * *Padding:* All review sequences are padded with zeros at the end to ensure they have a uniform length of 500. This is required for batch processing in the neural network.

* **Architecture**:
    * *Embedding Layer:* The input layer takes the integer-encoded vocabulary and creates dense vector representations (embeddings) for each word. This helps the model understand relationships between words.
    * *SimpleRNN Layer:* A Simple Recurrent Neural Network layer processes the sequence of word embeddings. It maintains a hidden state that captures information from previous steps in the sequence to influence the current prediction.
    * *Output Layer:* A final `Dense` layer with a single neuron and a `sigmoid` activation function squashes the output to a value between 0 and 1. This value represents the predicted probability of the review being positive.

The model is compiled using the `adam` optimizer and `binary_crossentropy` as the loss function, which are standard choices for binary classification problems.

---

## Web Application

A simple and intuitive web application has been developed using **Streamlit** to interact with the trained model. Users can type or paste a movie review into a text area. Upon clicking the "Classify" button, the application preprocesses the text, feeds it to the model, and instantly displays the result:
* The sentiment classification (**Positive** or **Negative**).
* The model's confidence score for the prediction.

---

## How to Run the Project Locally

Follow these steps in your terminal. The order is important.

**1. Clone the repository:** This downloads the project files to your computer.
```bash
git clone https://github.com/chaitanyananepallicn/SimpleRNN-MovieReviewAnalysis.git
```

**2. Install the required dependencies:** This command reads the `requirements.txt` file and installs all the necessary Python libraries.
```bash
pip install -r requirements.txt
```

**3. Navigate into the project directory:** You must be inside this folder to access the necessary files.
```bash
cd MovieReviewAnalysis(Simple RNN)
```

**4. Run the Streamlit application:**
```bash
streamlit run app.py
```
Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

---

## File Structure

To run this project, your directory should contain the following essential files:
```
└── MovieReviewAnalysis(Simple RNN)/
    ├── app.py             # The Streamlit application script
    ├── model.keras        # The pre-trained Keras model file
    └── requirements.txt   # A file listing the project dependencies
