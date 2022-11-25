from flask import Flask, render_template, jsonify, request, Markup
from model import predict_image
import utils
from flask_cors import CORS
from urllib.request import urlopen
from flask import jsonify
import urllib
app = Flask(__name__)

CORS(app)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST','PUT','PATCH'])
def predict():
    if request.method == 'POST':
        try:
            print(request.files['file'])
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            res = Markup(utils.disease_dic[prediction])
            
            # return render_template('display.html', status=200, result=res)  
            return res 
        except:
            pass
    if request.method == 'PUT':
        try:
            print(request.files['file'])
            file = request.files['file']
            img = file.read()
            prediction = predict_image(img)
            print(prediction)
            
            return prediction
        except:
            pass
    if request.method == 'PATCH':
        try:
            content = request.json
            data=content['disease']
            print(data)
            res = Markup(utils.disease_dic[data])
            return res
        except:
            pass
    return render_template('index.html', status=500, res="Internal Server Error")


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')
