import numpy as np
import h5py
import contextlib
from keras.models import load_model
from scipy.spatial.distance import euclidean, cityblock, cosine
import os
import re


n = 10  # number of categories / users
start_index = 101 # index of the file containing the first category
samples_per_category = 20  # number of samples per category
#eta, T, C = 20, 256, 3  
eta, T, C = 20, 160, 3 
data_path = "./data_32/rest2/test/"   # directory path for .npy files
model = load_model('fingerprinting.keras') # path to the fingerprinting model

euclidean_distances = np.array([])  # Pusta tablica numpy
manhattan_distances = np.array([])  # Pusta tablica numpy
cosine_distances = np.array([])     # Pusta tablica numpy

def format_filename(n):
    # Wzorzec wyrażenia regularnego do dopasowania pliku
    pattern = re.compile(rf"{n}_rest2_\d+_fp16\.npy")
    
    # Przechodzenie przez wszystkie pliki w danym folderze
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if pattern.match(file):
                # Zwracanie pełnej ścieżki do pliku
                #print(file)
                return file

# Load data from each .npy file
# Calculate the genuine score
for category in range(start_index, start_index + n):
    file_path = os.path.join(data_path, format_filename(category))  # np. 'data/category_0.npy'
    data = np.load(file_path)  # Load the .npy file

    for i in range(0, samples_per_category):
        for j in range(i, samples_per_category):
            array1 = data[i]
            array2 = data[j]         
            array1 = np.expand_dims(array1, axis=0)
            array2 = np.expand_dims(array2, axis=0)
            # Make a prediction
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                prediction1 = model.predict(array1)
            prediction_flat1 = prediction1.flatten()
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                prediction2 = model.predict(array2)
            prediction_flat2 = prediction2.flatten()
            
            # Compute distances
            euclidean_distance = euclidean(prediction_flat1, prediction_flat2)
            manhattan_distance = cityblock(prediction_flat1, prediction_flat2)
            cosine_distance = cosine(prediction_flat1, prediction_flat2)   
            # Store the distances in the arrays
            euclidean_distances = np.append(euclidean_distances, euclidean_distance)
            manhattan_distances = np.append(manhattan_distances, manhattan_distance)
            cosine_distances = np.append(cosine_distances, cosine_distance)

np.save('euclidean_distances.npy', euclidean_distances)
np.save('manhattan_distances.npy', manhattan_distances)
np.save('cosine_distances.npy', cosine_distances)

print("Distances have been computed and saved.")

data_array = np.zeros((n, 1100, eta, T, C))
print(data_array.shape)
index = 0
for category in range(start_index, start_index + n):
    file_path = os.path.join(data_path, format_filename(category))  # np. 'data/category_0.npy'
    data = np.load(file_path)  # Load the .npy file
    data_array[index] = data[:, :, :, :]
    print(index)
    index += 1

euclidean_distances_sus = np.array([])  
manhattan_distances_sus = np.array([])  
cosine_distances_sus = np.array([])     

for k in range (n):
    for l in range(k + 1, n):
        for i in range(0, samples_per_category):
            for j in range(i, samples_per_category):
                array1 = data_array[k, i, :, :, :]
                array2 = data_array[l, j, :, :, :]         
                array1 = np.expand_dims(array1, axis=0)
                array2 = np.expand_dims(array2, axis=0)
                # Make a prediction
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    prediction1 = model.predict(array1)
                prediction_flat1 = prediction1.flatten()
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    prediction2 = model.predict(array2)
                prediction_flat2 = prediction2.flatten()
                # Compute distances
                euclidean_distance = euclidean(prediction_flat1, prediction_flat2)
                manhattan_distance = cityblock(prediction_flat1, prediction_flat2)
                cosine_distance = cosine(prediction_flat1, prediction_flat2)   
                # Store the distances in the arrays
                euclidean_distances_sus = np.append(euclidean_distances_sus, euclidean_distance)
                manhattan_distances_sus = np.append(manhattan_distances_sus, manhattan_distance)
                cosine_distances_sus = np.append(cosine_distances_sus, cosine_distance)

np.save('euclidean_distances_sus.npy', euclidean_distances_sus )
np.save('manhattan_distances_sus.npy', manhattan_distances_sus )
np.save('cosine_distances_sus.npy', cosine_distances_sus )

print("Distances have been computed and saved.")