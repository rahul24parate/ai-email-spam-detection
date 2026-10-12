# AI-Powered Email Spam Detection System

A beginner-friendly machine-learning project that classifies a text message as **spam** or **not spam**. It uses TF-IDF text features and Logistic Regression, with a Streamlit interface for trying sample messages.

> Educational demo only. It is not a security product, and its predictions are not guaranteed to be correct.

## Features

- Downloads the public UCI SMS Spam Collection dataset when first trained.
- Cleans basic label issues and removes duplicate messages.
- Splits the data into training and test sets.
- Converts text into numerical features using TF-IDF.
- Trains a Logistic Regression classifier.
- Reports accuracy, precision, recall, F1-score, and a confusion matrix.
- Saves the trained pipeline and provides a small web app for predictions.

## Tech stack

Python, pandas, scikit-learn, TF-IDF, Logistic Regression, Streamlit, joblib.

## Project structure

```text
ai_email_spam_detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/                 # downloaded automatically; not committed
```

## Run locally

Python 3.10 or newer is recommended.

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Create and activate a virtual environment (recommended).
4. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Train and evaluate the model (internet required for the first dataset download):

   ```bash
   python train_model.py
   ```

6. Start the web app:

   ```bash
   streamlit run app.py
   ```

Streamlit will show a local URL, usually `http://localhost:8501`.

## How it works

1. **Dataset:** labeled SMS messages, marked spam or ham (ordinary message).
2. **TF-IDF:** turns words and phrases into numerical features based on how informative they are in the dataset.
3. **Logistic Regression:** learns patterns that help separate spam from ordinary messages.
4. **Evaluation:** a held-out test set is used to calculate classification metrics.
5. **Prediction:** the Streamlit app passes new text through the same saved pipeline.

## Example messages

- Spam-like: `Congratulations! You won a free prize. Call now to claim your reward.`
- Ordinary: `Hi, are we still meeting at the library at 4 pm?`

Predictions can vary. Do not claim a specific accuracy until you run the training script and record its actual output.

## Limitations and responsible use

- This dataset contains SMS messages, not a representative sample of every modern email.
- Language, slang, and new scam techniques may affect results.
- The displayed probability is a model estimate and is not a guarantee.
- Do not paste private or confidential messages into a public deployment.
- Do not use the model as the sole basis for blocking messages or making high-impact decisions.

## Possible improvements

- Add a larger and more recent labeled dataset.
- Compare Logistic Regression with Multinomial Naive Bayes.
- Add unit tests and error analysis.
- Deploy the app using a hosting service after reviewing privacy and resource limits.

## Data source

UCI Machine Learning Repository — SMS Spam Collection:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Resume description

**AI-Powered SMS Spam Detection | Python, scikit-learn, TF-IDF, Logistic Regression, Streamlit**

Built a text-classification application that learns to distinguish spam from ordinary messages using TF-IDF features and Logistic Regression. Evaluated the model on a held-out test set using classification metrics and created a Streamlit interface for interactive predictions.

Use this description only after you have run the project and can explain the implementation and actual evaluation results.
