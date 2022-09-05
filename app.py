from flask import Flask, request, jsonify
import pickle
import numpy as np
import xgboost as xg
from predict import check_language

#GAmodel = pickle.load(open('GAmodel.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return  "hello world"

@app.route('/predict', methods =['POST'])
def predict():
    SMS_text = request.form.get('SMS_text')
# print('my sms:',GAmodel)
   # input_query = np.array([[SMS_text]])
   # print('my query:',input_query)
  #  result = GAmodel.predict(input_query)[0]
    result=check_language(SMS_text)
    #print('my result'+check_language(input_query))



    return jsonify({'spam': str(result)},timeout=1200)
   #return jsonify({'spam': str(result)})

if __name__ == '__main__':
    app.run(debug=True)