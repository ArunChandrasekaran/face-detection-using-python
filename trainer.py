import cv2,os
import numpy as np
from PIL import Image 

recognizer = cv2.createLBPHFaceRecognizer();
path = 'dataset'

def getImagesWithID(path):
     imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
     # images will contains face images
     faces= []
     # labels will contains the label that is assigned to the image
     IDs= []
     for imagePath in imagePaths:
         # Read the image and convert to grayscale
         faceImg = Image.open(imagePath).convert('L')
         # Convert the image format into numpy array
         faceNp = np.array(faceImg, 'uint8')
         # Get the label of the image
         ID = int(os.path.split(imagePath)[1].split(".")[1])
         faces.append(faceNp)
         IDs.append(ID)
         cv2.imshow("training",faceNp)
         cv2.waitKey(10)
     # return the images list and labels list
     return IDs, faces


Ids,faces= getImagesWithID(path)

recognizer.train(faces,np.array(Ids))
recognizer.save('trainer/trainer.yml')
cv2.destroyAllWindows()
