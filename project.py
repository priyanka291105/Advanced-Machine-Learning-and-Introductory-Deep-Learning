import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# -------------------------
# LOAD DATASET
# -------------------------
data = pd.read_csv(r"C:\Users\PC 06\Desktop\like\backend\fooood_nutrition.csv")

# ✅ FIX: REMOVE MISSING VALUES
data = data.dropna()

# -------------------------
# FEATURES & LABELS
# -------------------------
X_text = data["food_name"]

y = data[["calories", "protein", "carbs", "fat", "fiber", "sugar"]]

# -------------------------
# TEXT VECTORIZATION
# -------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# -------------------------
# MODEL TRAINING
# -------------------------
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=100, random_state=42)
)

model.fit(X, y)

# -------------------------
# HTML TEMPLATE
# -------------------------
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Food Nutrition Predictor</title>
</head>
<body style="background-color: lightblue; text-align:center; font-family: Arial;">

    <h1>🍎 Food Nutrition Predictor</h1>

    <form method="POST" action="/predict">
        <input type="text" name="food" placeholder="Enter food name" required>
        <button type="submit">Predict</button>
    </form>

    <h2>Result</h2>

    <p><b>Calories:</b> {{calories}}</p>
    <p><b>Protein:</b> {{protein}}</p>
    <p><b>Carbs:</b> {{carbs}}</p>
    <p><b>Fat:</b> {{fat}}</p>
    <p><b>Fiber:</b> {{fiber}}</p>
    <p><b>Sugar:</b> {{sugar}}</p>

</body>
</html>
"""

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template_string(html,
        calories="", protein="", carbs="", fat="", fiber="", sugar=""
    )

@app.route("/predict", methods=["POST"])
def predict():
    food = request.form["food"]

    vec = vectorizer.transform([food])
    result = model.predict(vec)[0]

    return render_template_string(html,
        calories=round(result[0], 2),
        protein=round(result[1], 2),
        carbs=round(result[2], 2),
        fat=round(result[3], 2),
        fiber=round(result[4], 2),
        sugar=round(result[5], 2)
    )

# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)