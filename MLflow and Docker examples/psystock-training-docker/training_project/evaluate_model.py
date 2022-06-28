import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score,average_precision_score, f1_score


def classification_metrics(df:None):
    metrics={}
    metrics["accuracy_score"]=accuracy_score(df["y_pred"], df["y_test"]  )
    metrics["average_precision_score"]=average_precision_score( df["y_pred"], df["y_test"]  )
    metrics["f1_score"]=f1_score( df["y_pred"], df["y_test"]  )
    return metrics
    
if __name__ == "__main__":

    with mlflow.start_run(run_name="evaluate_model") as run:
        mlflow.set_tag("mlflow.runName", "evaluate_model")
        df=pd.read_csv("data/predictions/test_predictions.csv")
        metrics = classification_metrics(df)
        mlflow.log_metrics(metrics)
