# LungCancerDetectionDeepLearning

4 Lung CT scans gathered which are labelled by doctors. We used 3 of patient scans to train our neural network and remaining one to test our network on a scans that our network has never seen. Then applied connected components algorithm to segment lungs to reduce Region Of Interest. After obtaining segmented lung images to generate data set to train neural network we applied sliding window to our ROI. We tried varios of window sizes. For each window sizes we tried different core sizes. Core sizes are used to determine label of current window.

After creating our data set we used Keras framework to train our neural network. Our neural network structure can be seen at neural_network.py file. We split our data set into 2 groups one for training, other one for testing our network's accuracy. Training data set has %85 of total data set and test data set has the remaining of data set. We achieved %80 of test accuracy.

For showing usefull data to user we implemented a GUI. Our GUI simply creates png images from Dicom scans. Then apply the same pre processing steps to create data set which is mentioned earlier. Then for the most successful window/core size which is 21x21 window size 7x7 core size we applied sliding windows on segmented lung images. Each cropped frames generated from sliding window tested at our neural network. The output of neural network determined the class of core size current window. 

After obtaining label matrix we colored areas based on their classes. Red colors represents Ground Glass, green represents honeycomb, blue represents healthy lung areas. 

You can see our result pictures at results folder on main structure. I' m gladly happy to achieve such a accurate results and contribute medical image processing and deep learning community something.
