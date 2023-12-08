from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier  # MLP is an NN
from sklearn.datasets import load_digits

from sklearn import svm
import numpy as np
import argparse
import imutils  # If you are unable to install this library, ask the TA; we only need this in extract_hsv_histogram.
import cv2
import os
import random
from sklearn.model_selection import train_test_split


random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)

classifiers = {
    'SVM': svm.LinearSVC(random_state=random_seed),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'NN': MLPClassifier(solver='sgd', random_state=random_seed, hidden_layer_sizes=(500,), max_iter=20, verbose=1)
}



class Classifier:

    def __init__(self, image, path_to_dataset, target_img_size   = (32,32)):
        self.path_to_dataset=  path_to_dataset
        self.target_img_size = target_img_size  # fix image size because classification algorithms THAT WE WILL USE HERE expect that


    def execute(self):
        self.run_experiment('hog')
        self.run_experiment('hsv')
        self.run_experiment('raw')

        # # Example
        test_img_path = r'test2.jpg'
        img = cv2.imread(test_img_path)
        features = self.extract_features(img, 'raw')  # be careful of the choice of feature set
        nn = classifiers['NN']
        nn.predict_proba([features])



    def extract_hsv_histogram(self,img):
 
        img = cv2.resize(img, self.target_img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        
        if imutils.is_cv2():
            hist = cv2.normalize(hist)
        else:
            cv2.normalize(hist, hist)
        return hist.flatten()   

    def extract_hog_features(self,img):
        img = cv2.resize(img, self.target_img_size)
        win_size = (32, 32)
        cell_size = (4, 4)
        block_size_in_cells = (2, 2)
        
        block_size = (block_size_in_cells[1] * cell_size[1], block_size_in_cells[0] * cell_size[0])
        block_stride = (cell_size[1], cell_size[0])
        nbins = 9  # Number of orientation bins
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        h = hog.compute(img)
        h = h.flatten()
        return h.flatten()
    
    def extract_raw_pixels(self,img):
        img = cv2.resize(img, self.target_img_size)
        img = img.flatten()
        return img
        
    def extract_features(self,img, feature_set='hog'):
        if feature_set == 'hsv_hist':
            return self.extract_hsv_histogram(img)
        elif feature_set == 'hog':
            return self.extract_hog_features(img)
        elif feature_set == 'raw':
            return self.extract_raw_pixels(img)
        else:
            raise ValueError('Invalid feature set passed')
    

    def load_dataset(self,feature_set='hog'):
        features = []
        labels = []
        img_filenames = os.listdir(self.path_to_dataset)

        for i, fn in enumerate(img_filenames):
            if fn.split('.')[-1] != 'jpg':
                continue

            label = fn.split('.')[0]
            labels.append(label)

            path = os.path.join(self.path_to_dataset, fn)
            img = cv2.imread(path)
            features.append(self.extract_features(img, feature_set))
            
            # show an update every 1,000 images
            if i > 0 and i % 1000 == 0:
                print("[INFO] processed {}/{}".format(i, len(img_filenames)))
            
        return features, labels  

    
    # This function will test all our classifiers on a specific feature set
    def run_experiment(self,feature_set):
        
        # Load dataset with extracted features
        print('Loading dataset. This will take time ...')
        features, labels = self.load_dataset(feature_set)
        print('Finished loading dataset.')
        
        # Since we don't want to know the performance of our classifier on images it has seen before
        # we are going to withhold some images that we will test the classifier on after training 
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=random_seed)
        
        for model_name, model in classifiers.items():
            print('############## Training', model_name, "##############")
            # Train the model only on the training features
            model.fit(train_features, train_labels)
            
            # Test the model on images it hasn't seen before
            accuracy = model.score(test_features, test_labels)
            
            print(model_name, 'accuracy:', accuracy*100, '%')




