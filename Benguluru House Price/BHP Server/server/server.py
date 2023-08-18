from flask import Flask , request , jsonify
import util
from flask_cors import CORS , cross_origin

app = Flask(__name__)

app.config['CORS_SUPPORTS_CREDENTIALS'] = True

cors = CORS(app,support_credentials=True , resources={r"/predict_price": {"origins": "*"}})

@app.route('/get_location_name')

def get_location_name():
    response = jsonify( {
        'locations' : util.get_location_names() 
    })
    response.headers.add('Access-Control-Allow-Origin' ,  '*' )

    return response

@app.route('/predict_price' , methods = ['POST'])

# @cross_origin(origin='localhost',headers=['Content- Type','Authorization']   , support_credentials=True )

def predict_price():

    request_data = request.get_json()
    print(request_data)

    sqft =request_data['total_sqft']
    bhk = request_data['bhk']
    bath = request_data['bath']
    location = request_data['location']

    predict_price = util.get_predicted_price(bhk,sqft , bath,location )

    response = jsonify({
        'predicted_price' :predict_price
    })

    return response


if __name__ == '__main__':
    print("starting Pytho Flask Server for House Prediction......")
    util.load_saved_columns()
    app.run()#debug=True, port=5000