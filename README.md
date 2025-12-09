# Self-Projects (Data Science & Machine Learning)

This repository contains a collection of personal data science and machine learning projects. The notebooks cover various domains, including Computer Vision, Natural Language Processing (NLP), Time Series Forecasting, and general predictive modeling.

## 📂 Project Overview

### 🧠 Computer Vision & Deep Learning

#### 1. Handwritten Digits Recognition using Neural Network
* **File:** `Handwritten_digits_recognization_using_Neural_Network.ipynb`
* **Objective:** To classify handwritten digits (0-9) correctly.
* **Algorithm:** Artificial Neural Network (ANN) / Multi-Layer Perceptron (MLP).
* **Dataset:** Likely the **MNIST** dataset (60,000 training images, 10,000 testing images).
* **Key Concepts:**
    * Data flattening (28x28 images to 784 vector).
    * Dense layers with activation functions (ReLU, Softmax).
    * Loss calculation (Sparse Categorical Crossentropy).

#### 2. Image Classification using Convolutional Neural Networks
* **File:** `Image_Classification_using_Convolutional_Neural_Networks.ipynb`
* **Objective:** To improve image classification performance using CNNs, which capture spatial hierarchies in images better than standard NNs.
* **Algorithm:** Convolutional Neural Network (CNN).
* **Dataset:** Likely **CIFAR-10** or **MNIST**.
* **Key Concepts:**
    * Convolutional layers (Conv2D) for feature extraction.
    * Pooling layers (MaxPooling2D) for dimensionality reduction.
    * Flattening and Dense layers for final classification.

#### 3. Handwritten Digits Recognition by Logistic Regression
* **File:** `handwritten_digits_recognize_by_Logistic_Regression.ipynb`
* **Objective:** A baseline approach to the digit recognition problem to compare performance against deep learning models.
* **Algorithm:** Logistic Regression.
* **Dataset:** **MNIST**.
* **Key Concepts:**
    * Linear classification.
    * Understanding the limitations of linear models on image data compared to Neural Networks.

---

### 🗣️ Natural Language Processing (NLP)

#### 4. Sentiment Classification: Russia-Ukraine War
* **File:** `Sentiment_Classification_Russia_Ukrain_War.ipynb`
* **Objective:** To analyze public sentiment regarding the Russia-Ukraine conflict based on textual data.
* **Dataset:** Likely **Twitter data** (Tweets) scraped using keywords related to the war.
* **Key Concepts:**
    * **Text Preprocessing:** Tokenization, Stop-word removal, Lemmatization.
    * **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) or CountVectorizer.
    * **Modeling:** Classification algorithms (Naive Bayes, Logistic Regression, or LSTM).

---

### 📈 Regression & Forecasting

#### 5. Netflix Stock Price Prediction
* **File:** `Netflix_Stock_Price_Prediction.ipynb`
* **Objective:** To predict the future closing price of Netflix (NFLX) stock.
* **Dataset:** Historical stock market data (Open, High, Low, Close, Volume).
* **Key Concepts:**
    * **Time Series Analysis.**
    * Data visualization (Candlestick charts, Trend lines).
    * **Models:** Linear Regression (for simple trends) or LSTM (Long Short-Term Memory) networks for sequential data.

#### 6. Sales Prediction
* **File:** `Sales_prediction.ipynb`
* **Objective:** To forecast sales for a business to aid in inventory and resource planning.
* **Dataset:** Likely the **BigMart Sales** dataset or similar retail data.
* **Key Concepts:**
    * **Feature Engineering:** Handling categorical variables (One-Hot Encoding).
    * **Exploratory Data Analysis (EDA):** Correlation heatmaps, distribution plots.
    * **Models:** Linear Regression, Random Forest Regressor, or XGBoost.

#### 7. Rain Prediction
* **File:** `rain_prediction.ipynb`
* **Objective:** To predict whether it will rain the next day based on weather metrics.
* **Dataset:** Likely the **Rain in Australia** dataset.
* **Key Concepts:**
    * **Binary Classification:** (Yes/No).
    * Handling imbalanced datasets.
    * **Models:** Logistic Regression, Decision Trees, or Random Forest.

---

## 🛠️ Technologies Used
* **Languages:** Python 3.x
* **Libraries:**
    * **Data Manipulation:** Pandas, NumPy
    * **Visualization:** Matplotlib, Seaborn
    * **Machine Learning:** Scikit-Learn
    * **Deep Learning:** TensorFlow / Keras

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Somnath1998g/Self-Projects.git](https://github.com/Somnath1998g/Self-Projects.git)
    ```
2.  Navigate to the directory:
    ```bash
    cd Self-Projects
    ```
3.  Install dependencies (if not already installed):
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```
4.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
