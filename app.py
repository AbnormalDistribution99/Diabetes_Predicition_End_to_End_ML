from distutils.log import debug
from flask import Flask, render_template, request
import pickle
import numpy as np

filename = 'Diabetes_Prediction.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form.get('pregnancies'))
        glucose = int(request.form.get('glucose'))
        bp = int(request.form.get('bloodpressure'))
        st = int(request.form.get('skinthickness'))
        insulin = int(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        dpf = float(request.form.get('dpf'))
        age = int(request.form.get('age'))
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug= True)