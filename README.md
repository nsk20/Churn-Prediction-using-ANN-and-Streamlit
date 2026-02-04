# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project is a complete end-to-end Machine Learning application that predicts whether a customer is likely to churn (leave a bank) based on their profile. It uses an **Artificial Neural Network (ANN)** built with **TensorFlow/Keras** and is deployed using **Streamlit** for an interactive user experience.

## 🚀 Features

-   **Deep Learning Model**: Uses a multi-layer Artificial Neural Network for high-accuracy predictions.
-   **Interactive Web UI**: Built with Streamlit, allowing users to input customer data and get real-time predictions.
-   **Data Preprocessing**: Includes robust preprocessing steps such as One-Hot Encoding for categorical variables and Standard Scaling for numerical features.
-   **Model Management**: Pre-trained model and preprocessing objects (encoders/scalers) are serialized for fast inference.

## 🛠️ Technology Stack

-   **Frameowrk**: [Streamlit](https://streamlit.io/)
-   **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
-   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
-   **Machine Learning Utilities**: [Scikit-learn](https://scikit-learn.org/)
-   **Model Export**: [Pickle](https://docs.python.org/3/library/pickle.html)

## 📂 Project Structure

```text
├── .venv/                      # Virtual environment
├── Churn_Modelling.csv          # The dataset used for training
├── app.py                      # Main Streamlit application
├── experiments.ipynb           # Notebook for model training & experiments
├── prediction.ipynb            # Notebook for testing predictions
├── model.h5                    # Trained ANN model (Keras format)
├── label_encoder_gender.pkl    # Serialized label encoder for Gender
├── onehot_encoder_geo.pkl      # Serialized one-hot encoder for Geography
├── scaler.pkl                  # Serialized standard scaler
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## 🔧 Installation & Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nsk20/Churn-Prediction-using-ANN-and-Streamlit.git
    cd Churn-Prediction-using-ANN-and-Streamlit
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

To run the Streamlit application:

```bash
streamlit run app.py
```

Once the app is running, you can enter customer details such as Credit Score, Geography, Gender, Age, Tenure, Balance, and more. The model will process the input and display the **Churn Probability** along with a final prediction (Likely to Churn or Not).

## 📊 Dataset

The project uses the `Churn_Modelling.csv` dataset, which contains 10,000 records of bank customers with features like:
-   **Geography**: France, Spain, Germany
-   **Gender**: Male, Female
-   **Age**, **Tenure**, **Balance**
-   **Number of Products**
-   **Credit Score**
-   **Is Active Member**, **Has Credit Card**

## 🧠 Model Architecture

The ANN model was trained using:
-   An input layer corresponding to the processed feature set.
-   Hidden layers with ReLU activation.
-   An output layer with Sigmoid activation (for binary classification).
-   Optimization using the **Adam** optimizer and **Binary Cross-Entropy** loss.

