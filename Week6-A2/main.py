import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset(file_path):
    """
    this function load wine dataset from wine folder
    """

    # column names from wine.names file
    columns = [
        "Class",
        "Alcohol",
        "Malic_Acid",
        "Ash",
        "Alcalinity_of_Ash",
        "Magnesium",
        "Total_Phenols",
        "Flavanoids",
        "Nonflavanoid_Phenols",
        "Proanthocyanins",
        "Color_Intensity",
        "Hue",
        "OD280_OD315",
        "Proline"
    ]

    # read wine.data file
    df = pd.read_csv(file_path, header=None, names=columns)

    return df


def clean_dataset(df):
    """
    this function clean dataset
    """

    # check missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # remove duplicate rows if have duplicate
    df = df.drop_duplicates()

    # remove rows with empty values
    df = df.dropna()

    return df


def train_svm_model(X_train, y_train):
    """
    this function train svm model using linear kernel
    """

    # use scaler because wine columns have different value ranges
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear"))
    ])

    # train model using training data
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    this function predict and evaluate svm model
    """

    # predict wine class using testing data
    y_pred = model.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percentage = accuracy * 100

    # generate report and confusion matrix
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy Score:")
    print(f"{accuracy_percentage:.2f}%")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)


def main():
    """
    main function of the program
    """

    # read from wine folder, not zip file
    file_path = os.path.join("wine", "wine.data")

    print("Loading Wine dataset...")
    df = load_dataset(file_path)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nCleaning dataset...")
    df = clean_dataset(df)

    # separate input features and target class
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # split dataset into training and testing
    # 20% testing and 80% training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTraining Data Size:", len(X_train))
    print("Testing Data Size:", len(X_test))

    print("\nTraining SVM model...")
    model = train_svm_model(X_train, y_train)

    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()