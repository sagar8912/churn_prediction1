from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model(
    r"D:\python_notes\Churn_prediction_project\best_model.keras"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Manual encoding (MUST match training)
    geography_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Male": 1, "Female": 0}

    CreditScore = int(request.form["CreditScore"])
    Geography = geography_map[request.form["Geography"]]
    Gender = gender_map[request.form["Gender"]]
    Age = int(request.form["Age"])
    Tenure = int(request.form["Tenure"])
    Balance = float(request.form["Balance"])
    NumOfProducts = int(request.form["NumOfProducts"])
    HasCrCard = int(request.form["HasCrCard"])
    IsActiveMember = int(request.form["IsActiveMember"])
    EstimatedSalary = float(request.form["EstimatedSalary"])

    # ✅ EXACTLY 12 FEATURES (MATCH MODEL)
    input_data = np.array([[
        CreditScore,
        Geography,
        Gender,
        Age,
        Tenure,
        Balance,
        NumOfProducts,
        HasCrCard,
        IsActiveMember,
        EstimatedSalary,
        0, 0   # ONLY if your training had 2 derived features
    ]], dtype=np.float32)

    # Predict
    prob = model.predict(input_data)[0][0]

    result = "Customer Will Exit ❌" if prob > 0.5 else "Customer Will Stay ✅"

    return render_template(
        "result.html",
        prediction=result,
        probability=round(prob * 100, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)





