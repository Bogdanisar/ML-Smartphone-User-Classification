import pdb
import numpy as np
import math

class NaiveBayes:
  
    def __init__(self, num_bins, max_value): 
        self.num_bins = num_bins  
        self.bins = np.linspace(0, max_value, num=num_bins)  
        
    def values_to_bins(self, x):
        num_samples = x.shape[0]  
        x = np.digitize(x, self.bins)    
        return x - 1
        
    def fit(self, train_images, train_labels):
        self.train_images = self.values_to_bins(train_images)
        self.train_labels = train_labels
        # calculam P(c) - probabilitatea claselor
        self.num_classes = max(train_labels) + 1 
        prob_classes = []
        for c in range(self.num_classes):
            prob_classes.append(np.sum(train_labels == c)/ len(train_labels))
        
        self.prob_classes = np.array(prob_classes)
        print('probabilitatea claselor ', self.prob_classes)
        
        # calculam P(x|c) - probabilitata unui bin pe fiecare pozitie, in functie de clasa 
        self.num_features = train_images.shape[1]  
        position_bin_class_prob = np.zeros((self.num_features, self.num_bins, self.num_classes))
        for pos in range(self.num_features):
            for idx_bin in range(self.num_bins):
                for class_id in range(self.num_classes):   
                    train_images_class = self.train_images[self.train_labels == class_id, :]  # luam doar imaginile care au clasa class_id
                    position_bin_class_prob[pos][idx_bin][class_id] = np.sum(train_images_class[:, pos] == idx_bin)/train_images_class[:, pos].shape[0]
                    
        self.position_bin_class_prob = position_bin_class_prob + 1e-10 
         
    def predict(self, test_images):
        num_images = test_images.shape[0]
        test_images = self.values_to_bins(test_images)
        probs = np.zeros((num_images, self.num_classes))
        for class_id in range(self.num_classes):   
            for idx_img in range(num_images):
                probs[idx_img][class_id] = np.log(self.prob_classes[class_id])  
                for pos in range(self.num_features):   
                    probs[idx_img][class_id] += np.log(self.position_bin_class_prob[pos][int(test_images[idx_img][pos])][class_id])  
                  
        return np.argmax(probs, 1)
        
    def score(self, test_images, test_labels):
        predicted = self.predict(test_images)
        
        return (predicted == test_labels).mean()
         
classifier = NaiveBayes(5, 255)
train_images = np.loadtxt('data/train_images.txt')
train_labels = np.loadtxt('data/train_labels.txt', 'int')
test_images = np.loadtxt('data/test_images.txt')
test_labels = np.loadtxt('data/test_labels.txt', 'int')

classifier.fit(train_images.copy(), train_labels)    
accuracy = classifier.score(test_images.copy(), test_labels)
print('acuratetea clasificatorul este ', accuracy)

