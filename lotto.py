import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import time
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

# Streamlit UI setup
st.title('Korea Lotto Prediction & Analysis')

# Load data
file_path = 'Lotto_Numbers.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, 0].values.reshape(-1, 1)  # Draw number
y = data.iloc[:, 1:].values  # Drawn numbers

# Display the most recent draw number and winning numbers
most_recent_draw = data.iloc[-1, 0]
most_recent_winning_numbers = data.iloc[-1, 1:].tolist()
st.write(f"Most Recent Round : {most_recent_draw} and the Winning Numbers : {most_recent_winning_numbers}")

# Functions for frequency analysis
def most_frequent_analysis(y, num_elements, num_most_common=10):
    all_elements = []
    for row in y:
        for element in combinations(row, num_elements):
            all_elements.append(tuple(sorted(element)))
    freq_elements = Counter(all_elements).most_common(num_most_common)
    if not freq_elements:
        return [(("None",), 0)]
    return freq_elements

# Analyze and visualize the most frequent single numbers, pairs, triplets, quadruplets, quintuplets, and sextuplets
def visualize_most_frequent(y):
    analysis_config = [
        (1, 'Most Frequent 1-element Sets', 45),
        (2, 'Most Frequent 2-element Sets', 40),
        (3, 'Most Frequent 3-element Sets', 35),
        (4, 'Most Frequent 4-element Sets', 30),
        (5, 'Most Frequent 5-element Sets', 20),  # Added analysis for 5-pair sets
    ]

    for num_elements, title, num_most_common in analysis_config:
        freq_elements = most_frequent_analysis(y, num_elements, num_most_common)
        elements, counts = zip(*freq_elements)
        elements = ['\n'.join(map(str, el)) for el in elements]

        plt.figure(figsize=(14, 6))
        plt.bar(elements, counts, color='skyblue' if num_elements == 1 else 'lightgreen' if num_elements == 2 else 'salmon' if num_elements == 3 else 'gold' if num_elements == 4 else 'purple' if num_elements == 5 else 'orange')
        plt.title(title)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha="right")
        st.pyplot(plt)

# Define models with brief descriptions
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Stacking': StackingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('dt', DecisionTreeClassifier(random_state=42))
        ],
        final_estimator=LogisticRegression()
    ),
    'SVM': SVC(random_state=42, probability=True),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

# Function to train models and predict numbers
def predict_numbers_and_accuracy(models):
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text('Please wait...')
    
    model_predictions = {}
    num_models = len(models)
    for i, (model_name, model) in enumerate(models.items(), start=1):
        accuracies = []
        predictions = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for j in range(y.shape[1]):  # For each position in the drawn numbers
            model.fit(X_train, y_train[:, j])
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test[:, j], y_pred)
            accuracies.append(accuracy)
            next_draw_prediction = model.predict(np.array([[X.max() + 1]]))
            predictions.append(int(next_draw_prediction[0]))
        mean_accuracy = np.mean(accuracies) * 100  # Convert accuracy to percentage
        
        # Sort the predictions for CatBoost model after all numbers have been predicted
        if model_name == 'CatBoost':
            predictions = sorted(predictions)
        
        model_predictions[model_name] = {'Predicted Numbers': predictions, 'Predicted Accuracy (%)': mean_accuracy}
        
        # Update progress bar
        progress_bar.progress(i / num_models)
        time.sleep(0.1)  # Simulate time delay for demonstration
    
    status_text.text('Done!')
    time.sleep(0.5)  # Show 'Done!' message briefly
    status_text.empty()  # Remove 'Please wait...' text
    progress_bar.empty()  # Remove progress bar
    
    return model_predictions

# Button to predict winning lotto numbers and display analysis
if st.button('Predict 5 sets Winning Lotto Numbers by 5 ML Models'):
    predictions = predict_numbers_and_accuracy(models)
    predictions_df = pd.DataFrame(predictions).T.reset_index()
    predictions_df.columns = ['Model', 'Predicted Numbers', 'Predicted Accuracy (%)']
    st.table(predictions_df)

    model_descriptions = {
    'Random Forest': 'An ensemble method that uses multiple decision trees to improve prediction accuracy.',
    'AdaBoost': 'An adaptive boosting algorithm that combines multiple weak learners to create a strong learner.',
    'Stacking': 'Combines predictions from multiple models and uses another model to compute the final prediction.',
    'SVM': 'Support Vector Machine is a powerful classifier that works well on a wide range of classification problems.',
    'CatBoost': 'A gradient boosting algorithm that can handle categorical features directly and is robust to overfitting.'
    }

    # 딕셔너리를 데이터프레임으로 변환
    model_descriptions_df = pd.DataFrame(list(model_descriptions.items()), columns=['Model', 'Description'])

    # 스트림릿을 사용하여 테이블로 표시
    st.table(model_descriptions_df)
    
    visualize_most_frequent(y)  # Call the visualization function here
