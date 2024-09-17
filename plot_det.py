import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import DetCurveDisplay
from scipy import stats

sample_size = 100

# Function to calculate FAR and FRR for a given threshold
def calculate_frr_far(genuine_scores, impostor_scores, threshold):
    # False Rejection Rate (FRR) = False Non-Match Rate
    FRR = np.sum(genuine_scores > threshold) / len(genuine_scores)
    
    # False Acceptance Rate (FAR) = False Match Rate
    FAR = np.sum(impostor_scores <= threshold) / len(impostor_scores)

    # True Positive Rate (TPR) = 1 - FRR (True Accept Rate)
    TPR = 1 - FRR

    # True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
    TP = np.sum(genuine_scores <= threshold)
    FP = np.sum(impostor_scores <= threshold)
    TN = np.sum(impostor_scores > threshold)
    FN = np.sum(genuine_scores > threshold)

    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Precision = TP / (TP + FP)
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall = TP / (TP + FN)
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return FRR, FAR, TPR, Accuracy, Precision, Recall

# Function to compute FAR, FRR, and plot DET curve
def compute_and_plot_det(genuine_scores, impostor_scores):

    plt.figure(figsize=(12, 6))

    
    for genuine_scores, impostor_scores, label in zip(genuine_scores_list, impostor_scores_list, labels):
        # Sort the scores to establish potential thresholds
        all_scores = np.concatenate((genuine_scores, impostor_scores))
        thresholds = np.sort(all_scores)
        
        frr_list = []
        far_list = []
        tpr_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []

        # Calculate FRR, FAR, and TPR for each threshold
        for threshold in thresholds:
            FRR, FAR, TPR, Accuracy, Precision, Recall = calculate_frr_far(genuine_scores, impostor_scores, threshold)
            frr_list.append(FRR)
            far_list.append(FAR)
            tpr_list.append(TPR)
            accuracy_list.append(Accuracy)
            precision_list.append(Precision)
            recall_list.append(Recall)   

        eer_threshold, eer_value, eer_index = find_eer(thresholds, frr_list, far_list)
        print(f"Equal Error Rate (EER) threshold : {eer_threshold} for: {label}")
        print(f"Equal Error Rate (FAR = FRR): {eer_value} for: {label}")
        print(f"Accuracy: {accuracy_list[eer_index]}, Precission: {precision_list[eer_index]}, Recall: {recall_list[eer_index]} for {label}")
        # Reduce the number of points by sampling
        if len(thresholds) > sample_size:
            indices = np.linspace(0, len(thresholds) - 1, sample_size).astype(int)  # Evenly spaced sampling
            frr_list = np.array(frr_list)[indices]
            far_list = np.array(far_list)[indices]
            tpr_list = np.array(tpr_list)[indices]
            accuracy_list = np.array(accuracy_list)[indices]
            precision_list = np.array(precision_list)[indices]
            recall_list = np.array(recall_list)[indices]

        # Plot the DET curve
        plt.scatter(frr_list, far_list, label=f"{label}: ({eer_value:.2f} %)", marker='o')
        plt.plot(frr_list, far_list)

    plt.xlim(-0.05, 0.75)
    plt.ylim(-0.05, 0.75)
    plt.title('DET Curve (FRR vs FAR)', fontsize=17)
    plt.xlabel('False Rejection Rate (FRR)', fontsize=14)
    plt.ylabel('False Acceptance Rate (FAR)', fontsize=14)
    # Set the ticks for both axes to show numbers every 0.1
    plt.xticks([i/10 for i in range(8)], fontsize=12)  # X-axis: ticks from 0 to 0.7
    plt.yticks([i/10 for i in range(8)], fontsize=12)  # Y-axis: ticks from 0 to 0.7
    plt.grid(True)
    plt.legend()
    plt.show()

# Function to find the threshold where FAR ≈ FRR (Equal Error Rate)
def find_eer(thresholds, frr_list, far_list):
    # Find the index where the difference between FAR and FRR is minimized
    differences = np.abs(np.array(frr_list) - np.array(far_list))
    eer_index = np.argmin(differences)
    
    eer_threshold = thresholds[eer_index]
    eer_value = frr_list[eer_index]  # FAR ≈ FRR at this point
    
    return eer_threshold, eer_value, eer_index

# Example usage
if __name__ == "__main__":
    # Example genuine and impostor scores
    genuine_scores_euclid = np.load('euclidean_distances.npy')
    impostor_scores_euclid  = np.load('euclidean_distances_sus.npy')
    genuine_scores_man = np.load('manhattan_distances.npy')
    impostor_scores_man  = np.load('manhattan_distances_sus.npy')
    genuine_scores_cos = np.load('cosine_distances.npy')
    impostor_scores_cos  = np.load('cosine_distances_sus.npy')
    # List of all series
    genuine_scores_list = [genuine_scores_euclid , genuine_scores_man, genuine_scores_cos]
    impostor_scores_list = [impostor_scores_euclid, impostor_scores_man, impostor_scores_cos]
    
    # Labels for each series
    labels = ['Euclidean ', 'Manhattan', 'Cosine']
    # Compute and plot DET curve
    compute_and_plot_det(genuine_scores_list, impostor_scores_list)
