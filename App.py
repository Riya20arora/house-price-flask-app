from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

features = ['house age', 'distance to the nearest MRT station', 
            'number of convenience stores', 'latitude', 'longitude']

@app.route('/')
def home():
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = {col: float(request.form[col]) for col in features}
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)
        predicted_price = round(prediction[0], 2)
        return render_template('result.html', price=predicted_price)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)