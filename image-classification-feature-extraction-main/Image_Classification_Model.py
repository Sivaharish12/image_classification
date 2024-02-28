import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image


with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = r"C:\Users\haris\Downloads\rdr21.webp"

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])
for i in range(3):
    print()

print(r"C:\Users\haris\AppData\Local\Programs\Python\Python311\python.exe D:\image-classification-feature-extraction-main\image-classification-feature-extraction-main\Image_Classification_Model.py")
print()
print("The Category of the Given Image is : ",*pred)