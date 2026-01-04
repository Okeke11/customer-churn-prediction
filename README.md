📉 Customer Churn Prediction Model
Project Overview
This project is a Machine Learning solution designed to predict customer churn (the likelihood of a customer canceling their subscription). By analyzing historical data—such as contract type, monthly charges, and tenure—the model identifies "at-risk" customers, enabling businesses to take proactive retention measures.

Tech Stack
Language: Python 3.x
Data Manipulation: Pandas, NumPy
Machine Learning: Scikit-Learn (Random Forest Classifier)
Visualization: Matplotlib, Seaborn

Key Features
Data Preprocessing: Handles missing values (e.g., in TotalCharges) and removes non-predictive features.
Feature Engineering: Implements One-Hot Encoding and Label Encoding to convert categorical text data into machine-readable numeric formats.
Model Training: Utilizes a Random Forest Classifier for robust prediction and reduced overfitting compared to single Decision Trees.
Evaluation: Includes a Confusion Matrix and Classification Report (Precision, Recall, F1-Score) to assess performance.
Feature Importance: Visualizes the top factors driving customer churn (e.g., "Month-to-month contracts").

Dataset
The project uses the Telco Customer Churn Dataset from Kaggle.
Rows: ~7,000 customers
Features: 21 (including Tenure, Payment Method, Paperless Billing, etc.)
Target: Churn (Yes/No)

Results
Accuracy: Achieved ~80% accuracy on the test set.
Insights: The model identified that Contract Type (specifically Month-to-Month) and Electronic Check payment methods are the strongest indicators of churn.

Future Improvements
Hyperparameter Tuning: Use GridSearchCV to optimize the Random Forest parameters.
Handling Imbalance: Implement SMOTE (Synthetic Minority Over-sampling Technique) to better balance the ratio of Churn vs. Non-Churn samples.
Deployment: Wrap the model in a FastAPI or Flask endpoint to serve real-time predictions.

Author: Okeke Chiagoziem Michael
