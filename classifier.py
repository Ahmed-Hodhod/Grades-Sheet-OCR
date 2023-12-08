from sklearn.datasets import load_digits
load_digits()

from sklearn import svm
import numpy as np
import cv2
import os
import random
from sklearn.model_selection import train_test_split


random_seed = 42  
random.seed(random_seed)
np.random.seed(random_seed)


class Classifier:

    def __init__(self, path_to_dataset, target_img_size   = (32,32)):

        self.path_to_dataset=  path_to_dataset
        self.target_img_size = target_img_size  # fix image size because classification algorithms THAT WE WILL USE HERE expect that
        self.classifier_algorithm = svm.LinearSVC(random_state=random_seed)


    def classify(self, img):
        features = self.extract_hog_features(img)
        return self.classifier_algorithm.predict([features])


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
            features.append(self.extract_hog_features(img))
            
            # show an update every 1,000 images
            if i > 0 and i % 1000 == 0:
                print("[INFO] processed {}/{}".format(i, len(img_filenames)))
            
        return features, labels  

    def train_model(self):
        
        # Load dataset with extracted features
        print('Loading dataset. This will take time ...')
        features, labels = self.load_dataset('hog')
        print('Finished loading dataset.')
        
        # Since we don't want to know the performance of our classifier on images it has seen before
        # we are going to withhold some images that we will test the classifier on after training 
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=random_seed)
        
        
        print("############## Training SVM ##############")
        # Train the model only on the training features
        self.classifier_algorithm.fit(train_features, train_labels)
        
        # Test the model on images it hasn't seen before
        accuracy = self.classifier_algorithm.score(test_features, test_labels)
        
        print("SVM model accuracy: ", accuracy*100, '%')
        return accuracy * 100 


classifier = Classifier(path_to_dataset="./digits_dataset/")

# Example
# test_img_path = r'test5.jpg'
# img = cv2.imread(test_img_path)
# classifier.train_model()
# img = cv2.resize(img, (32, 32))
# print(classifier.classify(img) ) 