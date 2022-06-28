import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.4)

def evaluate(actual,pred):
    return accuracy_score(actual,pred)

mlflow.set_experiment("Multi_Model_Tracking")

# mlflow.sklearn.autolog()
with mlflow.start_run(run_name='lr_model_baseline'):
 clf = LogisticRegression()
 clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
mlflow.log_metric("accuracy", evaluate(y_test, y_pred))
mlflow.sklearn.log_model(clf, "model")