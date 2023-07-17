from flask import Flask, render_template, jsonify, request, Markup
# import utils
from flask_cors import CORS
from urllib.request import urlopen
from flask import jsonify
from model import new_data
from model import preprocess_new_data
from model import model
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
CORS(app)

# @app.route('/', methods=['GET'])
# def home():
#   res=model.predict(new_data)
#   print(res)
#   return render_template('index.html')
from model import preprocess_new_data
@app.route('/predict', methods=['GET', 'POST','PUT','PATCH'])
def predict():
    if request.method == 'POST':
        # Get JSON data from the request
        json_data = request.get_json()

        # Extract data from the JSON object
        type_ = json_data['type']
        bedrooms = int(json_data['bedrooms'])
        category = json_data['category']
        state = json_data['state']
        locality = json_data['locality']
        bathrooms = int(json_data['bathrooms'])
        toilets = int(json_data['toilets'])
        furnished = int(json_data['furnished'])
        serviced = int(json_data['serviced'])
        shared = int(json_data['shared'])
        parking = int(json_data['parking'])
        sub_type = json_data['sub_type']
        listmonth = float(json_data['listmonth'])
        listyear = float(json_data['listyear'])
        label_encoder = LabelEncoder()


        # Preprocess the new data using the function from model.py
        new_data = preprocess_new_data(type_, bedrooms, category, state, locality, bathrooms, toilets, furnished, serviced, shared, parking, sub_type, listmonth, listyear)
        
        label_encoder = LabelEncoder()
        categorical_columns = ['category', 'sub_type', 'locality', 'type', 'state']

        for column in categorical_columns:
         new_data[column] = label_encoder.fit_transform(new_data[column])

        scaler = Normalizer()
        new_data = scaler.fit_transform(new_data)

        # Call the prediction function with the new_data DataFrame
        prediction_result = model.predict(new_data)

        print(prediction_result)


        # Convert the prediction result to JSON and return it as the response
        return jsonify({'prediction': str(prediction_result)})

if __name__ == "__main__":
    app.run(debug=True)
# if __name__ == "__main__":
#     app.run(debug=False,host='0.0.0.0')

