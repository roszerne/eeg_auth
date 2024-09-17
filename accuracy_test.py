import numpy as np
import h5py
import contextlib
from keras.models import load_model
from scipy.spatial.distance import euclidean, cityblock, cosine
import os
import random
import re
n = 110 # number of categories
m = 110 # number of categories for impostors
start_index = 1 # index of the file containing the first category
start_index_sus = 1 # index of the file containing the first category for impostors
samples_per_category = 225  # number of samples per category for genuine test
samples_per_category_sus = 2 # number of samples per category for impostor test
eta, T, C = 20, 256, 3  # replace with actual values for eta, T, and C
num_of_samples = 1000
data_path = "./data_3/rest2/alpha/"
impostor_data_path = "./data_3/rest2/alpha/"
model = load_model('fingerprinting.keras')

threshold = 2587.518
euclidean_distances = np.array([])  
manhattan_distances = np.array([])  
cosine_distances = np.array([])     
cosine_distances_sus = np.array([])
def format_filename(n, data_path):
    print(n)
    # Wzorzec wyrażenia regularnego do dopasowania pliku
    pattern = re.compile(rf"{n}_rest2_\d+_fp16\.npy")
    
    # Przechodzenie przez wszystkie pliki w danym folderze
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if pattern.match(file):
                # Zwracanie pełnej ścieżki do pliku
                #print(file)
                return file

# Function to calculate FAR and FRR for a given threshold
def calculate_frr_far(genuine_scores, impostor_scores):
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

# Load data from each .npy file
# Calculate the genuine score
fingeprintings = []
for category in range(start_index, start_index + n):
    file_path = os.path.join(data_path, format_filename(category, data_path))  # np. 'data/category_0.npy'
    data = np.load(file_path)  # Load the .npy file
    fingerprint = model.predict(np.expand_dims(data[0], axis=0)).flatten()
    fingeprintings.append(fingerprint)

    samples = random.sample(range(1, num_of_samples), samples_per_category)
    for i in range(0, samples_per_category):
        array1 = data[samples[i]]      
        array1 = np.expand_dims(array1, axis=0)
        # Make a prediction
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            prediction1 = model.predict(array1)
        prediction_flat1 = prediction1.flatten()
        cosine_distance = cityblock(prediction_flat1, fingerprint)   
        cosine_distances = np.append(cosine_distances, cosine_distance)


index = start_index_sus
for fingerprint in fingeprintings:
    for category in range(start_index_sus, start_index_sus + m):
        if category == index:
            continue
        print(f"Category: {category}, index: {index}")
        file_path = os.path.join(impostor_data_path, format_filename(category, impostor_data_path))  # np. 'data/category_0.npy'
        data = np.load(file_path)  # Load the .npy file
        
        samples = random.sample(range(1, num_of_samples), samples_per_category_sus)
        for i in range(0, samples_per_category_sus):
            array1 = data[samples[i]]      
            array1 = np.expand_dims(array1, axis=0)
            # Make a prediction
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                prediction1 = model.predict(array1)
            prediction_flat1 = prediction1.flatten()
            cosine_distance = cityblock(prediction_flat1, fingerprint)   
            cosine_distances_sus = np.append(cosine_distances_sus, cosine_distance)
    index += 1


print(cosine_distances)
print(cosine_distances_sus)
print(len(cosine_distances))
print(len(cosine_distances_sus))
FRR, FAR, TPR, Accuracy, Precision, Recall = calculate_frr_far(cosine_distances, cosine_distances_sus)
print(FRR)
print(FAR)
print(TPR)
print(Accuracy)
print(Precision)
print(Recall)
