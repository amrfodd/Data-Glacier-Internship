#importing libraries
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

model = pickle.load(open('models/iri.pkl', 'rb'))



#creating instance of the class
app = Flask(__name__,template_folder="templates")



#to tell flask what url shoud trigger the function index()
@app.route('/home')
def index():
    return flask.render_template('home.html')



# Route 'classify' accepts GET request
@app.route('/predict',methods = ['POST'])
def predict():
    sepal_len = request.form['seplen'] # Get parameters for sepal length
    sepal_wid = request.form['sepwid'] # Get parameters for sepal width
    petal_len = request.form['Petlen'] # Get parameters for petal length
    petal_wid = request.form['Petwid'] # Get parameters for petal width
        
    arr = np.array([sepal_len, sepal_wid, petal_len, petal_wid]) # Convert to numpy array
    arr = arr.reshape(1,-1)
    
    
    # Get the output from the classification model
    result = model.predict(arr)

    # Render the output in the prediction page
    return render_template('predict.html', data=result)
      
    
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)







'''


try:
        sepal_len = request.form['seplen'] # Get parameters for sepal length
        sepal_wid = request.form['sepwid'] # Get parameters for sepal width
        petal_len = request.form['petlen'] # Get parameters for petal length
        petal_wid = request.form['petwid'] # Get parameters for petal width
        
        arr = np.array([sepal_len, sepal_wid, petal_len, petal_wid]) # Convert to numpy array
        
        # Get the output from the classification model
        result = model.predict(arr)

        # Render the output in the prediction page
        return render_template('predict.html', data=result)
    except:
        return 'Error'
        
'''