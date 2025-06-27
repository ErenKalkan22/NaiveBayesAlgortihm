from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load pickle files using absolute paths
cv = pickle.load(open(os.path.join(script_dir, "cv.pkl"), "rb"))
loaded_model = pickle.load(open(os.path.join(script_dir, "review.pkl"), "rb"))
print(type(loaded_model))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and 'message' in request.form:
        review = str(request.form.get('message'))
        if len(review) == 0:
            return render_template('index.html', prediction_text="Please try again")
        else:
            data = [review]
            vect = cv.transform(data).toarray()
            pred = loaded_model.predict(vect)

            if pred[0] == 1:
                return render_template('index.html', prediction_text="This comment expresses a positive opinion")
            else:
                return render_template('index.html', prediction_text="This comment expresses a negative opinion")
    else:
        return render_template('index.html', prediction_text="Please try again")


if __name__ == "__main__":
    app.run(debug=True)
