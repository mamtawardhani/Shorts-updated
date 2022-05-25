from PIL import Image
import PIL.ImageOps
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

label, data = fetch_openml('mnist_784', version=1, return_X_y=True)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    label, data,  train_size=7000, test_size=3000)

ScaledXtrain = Xtrain / 255.0
ScaledXtest = Xtest / 255.0

model = LogisticRegression(solver="saga", multi_class="multinomial")
model.fit(ScaledXtrain, Ytrain)


def Predict(img):
    imgOpened = Image.open(img)
    BnW = imgOpened.convert("L")

    reSized = BnW.resize((28, 28), Image.ANTIALIAS)

    pixelSize = 15

    pixelProgression = np.percentile(reSized, pixelSize)
    #changed the clip
    invertedImg = np.clip(reSized - pixelProgression, 0, 255)
    
    maxPixel = np.max(reSized)
    imgArray = np.asarray(invertedImg)

    sample = np.array(imgArray).reshape(1, 784)
    #stored it in a variable
    sample_predict = model.predict(sample)

    return sample_predict[0]
