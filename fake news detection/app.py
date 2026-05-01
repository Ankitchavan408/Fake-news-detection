from flask import Flask, render_template, request
import pickle

# 1️⃣ Create Flask app FIRST
app = Flask(__name__)

# 2️⃣ Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 3️⃣ Home page
@app.route('/')
def home():
    return render_template('index.html')

# 4️⃣ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news_vec = vectorizer.transform([news])

    prediction = model.predict(news_vec)[0]
    prob = model.predict_proba(news_vec)[0]

    confidence = max(prob) * 100

    if prediction == 0:
        result = f"FAKE NEWS ❌ ({confidence:.2f}%)"
    else:
        result = f"REAL NEWS ✅ ({confidence:.2f}%)"

    print("RAW:", prediction, "PROB:", prob)

    return render_template('result.html', prediction=result)

# 5️⃣ Run app
if __name__ == '__main__':
    app.run(debug=True)
