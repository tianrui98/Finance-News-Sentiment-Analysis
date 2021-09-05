
"""
The Classifier class builds a pipeline, conducts grid search,
and contains the functions to train the model, evaluate the model, 
and make predictions with XGboost.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost.sklearn import XGBClassifier

class Classifier:
    def __init__(self, k_folds, vectorizer_max_feat, gridsearch_params={
            # Best parameters are chosen based on our experiments
            'classifier__n_estimators': [60], #tried [50, 60, 70]
            'classifier__max_depth': [60], #tried [40, 50, 60]
            'classifier__learning_rate': [0.5],#tried [ 0.4, 0.5, 0.6]
        }, scoring=make_scorer(f1_score)):

        # Create sklearn pipeline & gridsearch objects
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=vectorizer_max_feat)),
            ('classifier', XGBClassifier(use_label_encoder=False, verbosity=0))
        ])
        self.search = GridSearchCV(self.pipeline, gridsearch_params, cv=k_folds, n_jobs=-1, scoring=scoring)

    # Helpful methods for interacting with the model
    def train(self, x, y):
        train_results = self.search.fit(x, y)
        self.training_score = self.search.best_score_
        return train_results

    def evaluate(self, x, y):
        return self.search.score(x, y)

    def predict(self, x):
        return self.search.predict(x)

    #print the best parameters when we train the model for the first time
    def print_best_param(self):
        print(self.search.best_params_)