
"""
Gaussian Naive Bayes model for Bank Marketing classification.

Preprocessing:
- StandardScaler for numerical features
- OneHotEncoder for categorical features
- Dense encoded output for Gaussian Naive Bayes

Classifier:
- Gaussian Naive Bayes
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB


# Numerical features used by the model
NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

# Categorical features used by the model
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome"
]


def create_model():
    """
    Create the Gaussian Naive Bayes preprocessing
    and classification pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                CATEGORICAL_FEATURES
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GaussianNB())
        ]
    )

    return model
