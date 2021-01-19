from flask import Flask
from mnist_classifier import predict_single
app = Flask(__name__)
import cv2
from flask import request
from flask import render_template
import os
import numpy as np
from PIL import Image
from mnist_classifier import Net
import torch
from torchvision import datasets, transforms
UPLOAD_FOLDER = './static/uploaded_images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_single(data):
    data = transform(data)
    data = data.to(device)
    data = data.unsqueeze(1)
    output_probs = model(data)
    return torch.argmax(output_probs, dim=1)


@app.route("/", methods=["GET", "POST"])
def predict():
    print('request.method: ', request.method)
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = UPLOAD_FOLDER + '/' + image_file.filename
            image_file.save(image_location)
            data = cv2.imread(image_location)
            data = Image.fromarray(data)
            pred = predict_single(data)
            class_name_map = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
            pred = class_name_map[pred[0].item()]
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == '__main__':
    model = Net().to(device)
    model.to(device)
    model.load_state_dict(torch.load('./mnist_cnn.pt'))  # load trained model
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    app.run(debug=True)