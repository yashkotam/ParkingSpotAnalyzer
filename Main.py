import os
import cv2
import configparser
import requests
import subprocess
from VehicleImagePlateAnalyzer import VehicleImagePlateAnalyzer

config = configparser.ConfigParser()
config.read(r'config.txt')

pyTesseractpath = config.get('PyTesseract', 'pyTesseractpath')
pyTesseractConfig = config.get('PyTesseract', 'pyTesseractConfig')

unitId = config.get('Default', 'unitId')
apiUri = config.get('Default', 'apiUri')

print("Parking Spot Analyzer. Unit Id : "+unitId)

vehicleImagePlateAnalyzer = VehicleImagePlateAnalyzer(pyTesseractpath, pyTesseractConfig)

# open camera module and click image and save to Images folder as image.jpeg
img_path = "Images/image.jpeg"
subprocess.run(["libcamera-jpeg","-o",img_path])

# read image in folder Images  
img = cv2.imread(img_path)
print("processing image : "+img_path)

# process image and get vehicle numbers
numbersList = vehicleImagePlateAnalyzer.getVehicleNumbers(img)
print("processed image : "+img_path)
print("number Plates List : ")
print(numbersList)

payload = {}
payload['id'] = unitId
payload['numbersList'] = numbersList

print("making POST call to upload payload : ")
print(payload)
response = requests.post(apiUri, json=payload)
print("received response : ")
print(response)

# delete image after processing
subprocess.run(["rm",img_path])
print("deleted image : "+img_path)
