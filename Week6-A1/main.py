import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_dataset(file_path):
    """
    this function load csv dataset file
    """

    # read csv file
    df = pd.read_csv(file_path)

    return df


def clean_dataset(df):
    """
    this function clean dataset
    remove unwanted data and null values
    """

    # remove Id column because not needed
    if "Id" in df.columns:
        df = df.drop("Id", axis=1)

    # remove duplicate rows
    df = df.drop_duplicates()

    # check if there is missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # remove rows with null values
    df = df.dropna()

    return df


def visualize_dataset(df):
    """
    this function create 2 scatter plot graphs
    one for sepal and one for petal
    """

    # -----------------------------
    # Scatter Plot 1: Sepal
    # -----------------------------
    plt.figure(figsize=(8, 6))

    # scatter plot for sepal length and sepal width
    sns.scatterplot(
        data=df,
        x="SepalLengthCm",
        y="SepalWidthCm",
        hue="Species"
    )

    plt.title("Sepal Scatter Plot")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")

    # save sepal graph image
    plt.savefig("sepal_scatter_plot.png")

    plt.show()

    # -----------------------------
    # Scatter Plot 2: Petal
    # -----------------------------
    plt.figure(figsize=(8, 6))

    # scatter plot for petal length and petal width
    sns.scatterplot(
        data=df,
        x="PetalLengthCm",
        y="PetalWidthCm",
        hue="Species"
    )

    plt.title("Petal Scatter Plot")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")

    # save petal graph image
    plt.savefig("petal_scatter_plot.png")

    plt.show()


def train_svm_model(X_train, y_train):
    """
    this function train svm model
    using linear kernel
    """

    # create svm classifier model
    model = SVC(kernel="linear")

    # train model using training dataset
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, encoder):
    """
    this function test and evaluate model
    """

    # predict testing dataset
    y_pred = model.predict(X_test)

    # calculate model accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # convert accuracy to percentage
    accuracy_percentage = accuracy * 100

    # create classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )

    # create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nAccuracy Score:")
    print(f"{accuracy_percentage:.2f}%")

    print("\nClassification Report:")
    print(report)

    print("\nConfusion Matrix:")
    print(cm)

    return cm


def visualize_confusion_matrix(cm, encoder):
    """
    this function create confusion matrix graph
    """

    plt.figure(figsize=(6, 5))

    # heatmap for confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # save graph image
    plt.savefig("confusion_matrix.png")

    plt.show()


def main():
    """
    main function of program
    """

    # dataset file path
    file_path = "Iris.csv"

    print("Loading IRIS dataset...")

    # load dataset
    df = load_dataset(file_path)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nCleaning dataset...")

    # clean dataset
    df = clean_dataset(df)

    print("\nVisualising dataset...")

    # create sepal and petal graph visualization
    visualize_dataset(df)

    # separate features and target column
    X = df.drop("Species", axis=1)
    y = df["Species"]

    # convert flower names to numbers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # split dataset into training and testing
    # test_size=0.2 means 20% testing and 80% training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # show how many records used for training and testing
    print("\nTraining Data Size:", len(X_train))
    print("Testing Data Size:", len(X_test))

    print("\nTraining SVM model using linear kernel...")

    # train svm model
    model = train_svm_model(X_train, y_train)

    print("\nEvaluating model...")

    # evaluate trained model
    cm = evaluate_model(
        model,
        X_test,
        y_test,
        encoder
    )

    print("\nGenerating confusion matrix...")

    # display confusion matrix graph
    visualize_confusion_matrix(cm, encoder)

    print("\nProgram finished successfully.")


# start program here
if __name__ == "__main__":
    main()