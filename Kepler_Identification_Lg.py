import numpy as np
import Logistic_regression as lg
import Dataset_preprocessor as dp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score,roc_curve


def plot_training_results(y_true, y_prob, cost_history):                                        # this fucntion plots our ROC_AUC cuvre and cost history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})', color='#2E86AB')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.6)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve - Exoplanet Detection', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    iterations = np.arange(len(cost_history)) * 100
    ax2.plot(iterations, cost_history, linewidth=2.5, color='#A23B72')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.set_title('Training Cost History', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    final_cost = cost_history[-1]
    ax2.annotate(f'Final: {final_cost:.4f}', 
                 xy=(iterations[-1], final_cost),
                 xytext=(-60, 20), textcoords='offset points',
                 fontsize=10, color='#A23B72',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                 arrowprops=dict(arrowstyle='->', color='#A23B72', lw=1.5))
    
    plt.tight_layout()
    plt.savefig("Lg_result.png", dpi=300)
    plt.show()

def train_model(filepath,alpha=0.5,num_iter=10000,lambda_reg=0.01):                         # this fucntiosn trais our logistic regression model

    X_train,X_test,Y_train,Y_test,X_train_scaled,X_test_scaled=dp.prepare_dataset(filepath)

    print("\nTraining the model.....")
    theta,cost_history=lg.gradient_descent(X_train_scaled,Y_train,alpha,num_iter,lambda_reg,verbose=True)
    Y_train_pred=lg.predict(X_train_scaled,theta)
    Y_test_pred=lg.predict(X_test_scaled,theta)

    Y_train_prob=lg.predict_prob(X_train_scaled,theta)
    Y_test_prob=lg.predict_prob(X_test_scaled,theta)

    print("\n\n")
    print("TRAINING SET PERFORMANCE")
    print("\n")
    print(f"Accuracy: {np.mean(Y_train_pred == Y_train):.4f}")
    print(f"ROC-AUC: {roc_auc_score(Y_train, Y_train_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_train, Y_train_pred))
    
    print("\n\n")
    print("TEST SET PERFORMANCE")
    print("\n")
    print(f"Accuracy: {np.mean(Y_test_pred == Y_test):.4f}")
    print(f"ROC-AUC: {roc_auc_score(Y_test, Y_test_prob):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_test, Y_test_pred))
    
    plot_training_results(Y_test,Y_test_prob,cost_history)
    return theta, scaler, features

if __name__=="__main__":
    filepath="data.csv"
    try:
        theta,scaler,features=train_model(filepath,alpha=0.5,num_iter=100000,lambda_reg=0.01)
        print("\n\n")
        print("MODEL TRAINING COMPLETE!")
        print("\n")
        print(f"Model parameters saved: theta with {len(theta)} coefficients")

    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        print("Download the Kepler data from NASA Exoplanet Archive first.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


