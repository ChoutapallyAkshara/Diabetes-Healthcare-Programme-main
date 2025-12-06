import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

@st.cache_data()
def load_data():
    # Make dataframe writeable (fix Copy-on-Write issue)
    df = pd.read_csv('diabetes.csv').copy()

    # Extract X and y as WRITEABLE numpy arrays
    X = df[['HbA1c_level','Pregnancies','Glucose','BloodPressure','SkinThickness',
            'Insulin','BMI','DiabetesPedigreeFunction','Age']].to_numpy().copy()
    
    y = df['Outcome'].to_numpy().copy()

    return df, X, y


@st.cache_data()
def train_model(X, y):
    # Ensure X and y are writeable numpy arrays
    X = np.array(X, copy=True)
    y = np.array(y, copy=True)

    model = DecisionTreeClassifier(
        ccp_alpha=0.0,
        class_weight=None,
        criterion='entropy',
        max_depth=4,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        random_state=42,
        splitter='best'
    )

    model.fit(X, y)
    score = model.score(X, y)

    return model, score


def predict(X, y, features):
    # Train model
    model, score = train_model(X, y)

    # Ensure features are writeable
    features = np.array(features, dtype=float, copy=True).reshape(1, -1)

    prediction = model.predict(features)

    return prediction, score



