"""
YSC4228 Group Assignment 2 for Krzystof Lis, Zhu Tianrui and Julian Chong
"""
#Imports
import argparse
from pathlib import Path
from classifier import *
import numpy as np
import pandas as pd

def enable_parsing():
    """
    Adds the arguments to our parser

    Returns: None

    """
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument("--train", help="Training Dataset",
                        required=True)
    parser.add_argument("--test", help="Testing Dataset",
                        required=True)
    parser.add_argument("--pred", help="New Prediction Dataset")
    parser.add_argument("--max_feat", help="Maximum number of features used by TFIDF Vectoriser",
                        type=int, required=True)
    parser.add_argument("--num_folds", help="Number of Folds used for cross validation",
                        type=int, required=True)
    return parser

def get_arguments (parser):
    """
    Get CLI arguments and raises error messages if inputs are invalid

    """
    args = parser.parse_args()

    train_data_path = Path(args.train)
    if not train_data_path.exists():
        print("Training data file does not exists!")
        return

    test_data_path = Path(args.test)
    if not test_data_path.exists():
        print("Test data file does not exists!")
        return

    pred_data_path = Path(args.pred)
    if pred_data_path.suffix != '.csv':
        print("The prediction path must be a path to a new csv file!")
        return
    pred_data_path.parents[0].mkdir(exist_ok=True, parents=True)

    try:
        train_data = pd.read_csv(args.train)
    except:
        print("Training data is in an invalid format!")
        return

    try:
        test_data = pd.read_csv(args.test)
    except:
        print("Test data is in an invalid format!")
        return

    k_folds = args.num_folds
    max_feat = args.max_feat

    return train_data,test_data,k_folds,max_feat,pred_data_path

def build_model (train_data,test_data,k_folds,max_feat,pred_data_path):
    """
    splits the dataset and performs classification

    Then evaluates results and exports predictions as .csv

    """
    #Split dataset
    train_matrix = train_data.to_numpy()
    train_corpus = train_matrix[:, 0]
    train_labels = train_matrix[:, 1].astype(np.float32)

    test_matrix = test_data.to_numpy()
    test_corpus = test_matrix[:, 0]
    test_labels = test_matrix[:, 1].astype(np.float32)

    # Train & evaluate the model
    print("Training...")
    classifier = Classifier(k_folds=k_folds, vectorizer_max_feat=max_feat)
    classifier.train(train_corpus, train_labels)
    test_score = classifier.evaluate(test_corpus, test_labels)

    # Print the best parameters when we train the model for the first time
    # classifier.print_best_param()

    print("Done!")

    print(f"\nTraining f1-score: {round(classifier.training_score, 4)}")
    print(f"Test f1-score: {round(test_score, 4)}\n")

    # Predict and Export
    preds = classifier.predict(test_corpus)
    preds_df = pd.DataFrame({
        "Sentence": test_data.Sentence,
        "Predicted Bad Sentence": preds
    })
    preds_df.to_csv(pred_data_path, index=False)
    print(f"Predictions saved to {pred_data_path}!\n")

    return None

def main():
    parser = enable_parsing()
    train_data,test_data,k_folds,max_feat,pred_data_path =  get_arguments(parser)
    build_model (train_data,test_data,k_folds,max_feat,pred_data_path )

if __name__ == "__main__":
    main()