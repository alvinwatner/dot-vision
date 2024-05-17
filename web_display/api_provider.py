from flask import Flask, jsonify, render_template
from dot_vision import ensemble_model
import random
from dot_vision import ensemble_model

app = Flask(__name__)

"""
steps:
1. image height (h) and width (w)
2. screen height (h') and width (w')
3. divide (h') and (h) to get scale (p)
4. divide (w') and (w) to get scale (q)
5. use (p) and (q) to calculate (x) and (y) on the client side
6. where (x) = (p \cdot w) and (y) = (q \cdot h)
"""


@app.route("/data")
def get_position():
    transformed_points = ensemble_model()
    print(f"transformed_points = {transformed_points}")
    position = {
        "x": x,
        "y": y,
    }
    return jsonify(position=position)


@app.route("/")
def home():
    return render_template("api_index.html")


if __name__ == '__main__':
    app.run(debug=True, port=3001)
