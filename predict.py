import gzip
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import process

K_MAX = 1000

def calculate_dist(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2) ** 2))

def gen_nearest_k_vectors(test_features_labels, train_features_labels, output_file, k=K_MAX):
    if os.path.exists(output_file):
        return

    nearest_neighbors = []
    # nearest_neighbors[i] = an array of nearest neighbors of i-th image in test

    for test_features, test_label in test_features_labels:
        distances = []
        for train_features, train_label in train_features_labels:
            dist = calculate_dist(test_features, train_features)
            distances.append((dist, train_label))
        distances.sort(key=lambda x: x[0]) # Sort by distance
        k_nearest_labels = [train_label for ignore, train_label in distances[:k]] # Save k nearest labels
        nearest_neighbors.append(k_nearest_labels)

    with open(output_file, 'wb') as f:
        pickle.dump(nearest_neighbors, f) # Save to file, don't have to calculate again

def load_binary(file_name):
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data

def predict_on_test_data(nearest_neighbors, k):
    # Use for preprocessed data (distance was calculated)
    k_nearest = nearest_neighbors[:k]
    return np.bincount(k_nearest).argmax()

def predict_label(vectorized_image, k, comparing_features_labels):
    # Use for input data (distance was not calculated)

    distances = []
    for comparing_feature, comparing_label in comparing_features_labels:
        dist = calculate_dist(vectorized_image, comparing_feature)
        distances.append([dist, comparing_label])
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for ignore, label in distances[:k]]
    predict = np.bincount(np.array(k_nearest_labels)).argmax()
    return predict

def predict_with_methods(image, k, extract_methods, *methods_data):
    #Return predictions with different extract methods

    results = []
    image = image.reshape((1, 28, 28))
    image_features = process.extract_features(image)
    for i in range(len(extract_methods)):
        # Append [Method name, predict]
        results.append([extract_methods[i], predict_label(image_features[i][0], k, methods_data[i])])

    return results

def graph_accuracy_in_range(test_features_labels, nearest_neighbors, test_range):
    accuracies = []
    true_labels = np.array([label for ignore, label in test_features_labels])
    for k in test_range:
        predictions = []
        for i in range(len(nearest_neighbors)):
            predict = predict_on_test_data(nearest_neighbors[i], k)
            predictions.append(predict)
        predictions = np.array(predictions)
        correct_predictions = np.sum(predictions == true_labels)
        accuracy = correct_predictions / len(test_features_labels)
        accuracies.append(accuracy)

    plt.plot(test_range, accuracies, color="green")
    plt.xlim(min(test_range)-1, max(test_range)+1)
    plt.ylim(0, 1)

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("K (Number of neighbors)")
    plt.grid()

    plt.savefig("Accuracy.png")
    plt.show()
