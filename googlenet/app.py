import numpy as np
import cv2 as cv
 
# read names of classes
with open('synset_words.txt') as f:
	classes = [x[x.find(' ') + 1:] for x in f]
 
image = cv.imread('goldfish.jpg')
cv.imshow("image", image)
# create tensor with 224x224 spatial size and subtract mean values (104, 117, 123) 
# from corresponding channels (R, G, B)
input = cv.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
 
# load model from caffe
net = cv.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')
# feed input tensor to the model
net.setInput(input)
# perform inference and get output
out = net.forward() 
# get indices with the highest probability
indexes = np.argsort(out[0])[-5:]
# define font style
font = cv.FONT_HERSHEY_COMPLEX
#print the last five most likelihold result
for i in reversed(indexes):
	print('class:', classes[i], ' probability:', out[0][i])
#add txet into the result image
cv.putText(image, classes[indexes[-1]], (20, 20), font, 1.0, (0, 0, 255), 2)
cv.imshow("result", image)
#save the result image
cv.imwrite("./goldfish_res.jpg", image)
cv.waitKey(0)

