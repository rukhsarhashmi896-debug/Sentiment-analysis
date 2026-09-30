# Sentiment Analysis using NLP

A Natural Language Processing (NLP) project for classifying text as **Positive** or **Negative** sentiment using the **Sentiment140** dataset, TF-IDF text features, and Logistic Regression.

## Project Overview

This project demonstrates an end-to-end sentiment analysis workflow:

1. Download the Sentiment140 dataset using KaggleHub.
2. Load and inspect the dataset with Pandas.
3. Select the required `text` and `sentiment` columns.
4. Convert the original sentiment labels:
   - `0` → Negative
   - `4` → Positive
5. Check missing values and remove duplicate text.
6. Clean the text using regular expressions.
7. Split the data into training and testing sets.
8. Convert text into numerical features using TF-IDF.
9. Train a Logistic Regression classifier.
10. Evaluate the model using accuracy and a confusion matrix.
11. Create a prediction function that returns the predicted sentiment and confidence.
12. Test the model on sample reviews and user-provided text.

## Dataset

The project uses the **Sentiment140** dataset downloaded through KaggleHub:

```text
kazanova/sentiment140
```

The original dataset contains **1,600,000 rows and 6 columns**.

Original columns:

- `sentiment`
- `id`
- `date`
- `query`
- `user`
- `text`

For modeling, only the following columns are retained:

- `text`
- `sentiment`

After removing duplicate text, the notebook reports **1,581,466 rows**.

The cleaned sentiment distribution is:

| Sentiment | Count |
|---|---:|
| Positive | 791,281 |
| Negative | 790,185 |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Regular Expressions (`re`)
- Scikit-learn
- KaggleHub
- Seaborn

## NLP / Text Preprocessing

The notebook defines a `clean_text()` function that:

- Converts text to lowercase
- Removes URLs
- Removes user mentions such as `@username`
- Removes hashtags
- Removes non-alphabetic characters
- Normalizes extra whitespace

Empty cleaned texts are removed before model training.

## Train-Test Split

The cleaned dataset is split using:

- **80% training data**
- **20% testing data**
- `random_state=42`
- Stratified split based on the sentiment labels

The notebook reports:

- Training samples: **1,262,150**
- Testing samples: **315,538**

## TF-IDF Feature Extraction

Text is converted into numerical features using `TfidfVectorizer`.

Configuration:

```python
TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2)
)
```

This creates up to **20,000 TF-IDF features** and uses both:

- Unigrams `(1)`
- Bigrams `(2)`

The resulting feature matrices have:

```text
Training shape: (1262150, 20000)
Testing shape:  (315538, 20000)
```

## Machine Learning Model

The project uses **Logistic Regression**:

```python
LogisticRegression(
    max_iter=1000
)
```

The model is trained on the TF-IDF representation of the training text.

## Model Performance

The notebook reports a test accuracy of:

**77.9957%**

or approximately:

**78.00%**

### Confusion Matrix

The reported confusion matrix is:

```text
[[119203  38504]
 [ 30928 126903]]
```

The notebook also visualizes the confusion matrix using a heatmap.

## Prediction Function

A reusable function named `predict_sentiment()` is included.

It:

1. Cleans the input text.
2. Converts it into TF-IDF features.
3. Predicts the sentiment.
4. Calculates the model's maximum class probability as a confidence value.
5. Returns the prediction and confidence.

Example:

```python
sentiment, confidence = predict_sentiment(
    "The phone quality is amazing and the camera is excellent."
)

print(sentiment)
print(confidence)
```

## Sample Predictions

The notebook tests the model with several types of text.

| Example | Prediction | Confidence |
|---|---|---:|
| Phone quality and camera review | Positive | 91.96% |
| Food and service review | Positive | 90.99% |
| Customer support complaint | Negative | 68.38% |
| Late and damaged package | Negative | 85.19% |
| College feedback | Positive | 73.08% |
| Workplace feedback example | Positive | 54.56% |

The notebook also accepts custom text using Python's `input()` function.

Example:

```text
Enter any text: food is good

Predicted Sentiment: Positive
Confidence: 69.9 %
```

## Project Workflow

```text
Sentiment140 Dataset
        ↓
Data Loading
        ↓
Data Cleaning & Selection
        ↓
Sentiment Label Mapping
        ↓
Missing Value & Duplicate Removal
        ↓
Text Cleaning
        ↓
Train / Test Split
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Prediction
        ↓
Accuracy & Confusion Matrix
        ↓
Custom Text Sentiment Prediction
```

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom.
3. KaggleHub downloads the dataset automatically.
4. Train the model.
5. Run the prediction cells or enter your own text.

### Option 2: Local Python Environment

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn kagglehub
```

Then open:

```text
SA_NLP.ipynb
```

using Jupyter Notebook, JupyterLab, or another compatible notebook environment.

## Repository Structure

```text
.
├── SA_NLP.ipynb
└── README.md
```

> **Note:** The dataset is downloaded at runtime through KaggleHub, so the large CSV dataset does not need to be stored directly in this repository.

## Results

The final model achieves approximately **78.00% accuracy** on the test set used in the notebook.

This project provides a complete beginner-friendly example of applying NLP preprocessing, TF-IDF feature extraction, Logistic Regression, and model evaluation to a large-scale sentiment classification dataset.
