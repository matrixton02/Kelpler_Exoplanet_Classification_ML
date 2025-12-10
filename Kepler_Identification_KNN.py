import numpy as np
import K_nearest_neighbour as knn
import Dataset_preprocessor as dp
import matplotlib.pyplot as plt

def plot(metrics,accuracies):                                                   # This fucntions plots the accuracies of different distance fuction KNN for our dataset
    plt.figure(figsize=(12, 6))
    bars = plt.bar(metrics, accuracies, color="skyblue", edgecolor="black")

    plt.title("Accuracy Comparison Across Distance Metrics (Kepler Exoplanet Dataset)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)

    # Label bars with accuracy values
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 1,
             f"{acc:.1f}%", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig("Knn_result.png", dpi=300)
    plt.show()

if __name__=="__main__":
    filepath="data.csv"
    X_train,X_test,y_train,y_test,X_train_scaled,X_test_scaled=dp.prepare_dataset(filepath)         # getting our training and tetsing dataset properly scaled by the data preprocessor

    metrics=[]
    accuracies=[]
    for dist in ["euclidean","manhattan","mahalanobis","rbf","rbf_normalized"]:                     # for each each type of distance fucntion we create object and do the training and testing (for KNN training is just memorizing the dataset)
        model=knn.KNN(k=7,dist_type=dist)
        model.fit(X_train,y_train)
        preds=model.predict(X_test)
        acc=np.mean(preds==y_test)
        print(dist,acc)
        acc=acc*100
        metrics.append(dist)
        accuracies.append(acc)
    
    plot(metrics,accuracies)                                                                         # plotting the accuracies

