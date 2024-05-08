from flask import Flask, jsonify, render_template
import random

app = Flask(__name__)

"""
steps:
1. image height (h) and width (w)
2. screen height (y) and width (x)
3. divide y and h to get scale (p)
4. divide x and w to get scale (q)
5. use p and q to calculate dot on the client side
"""

@app.route("/data")
def get_position():
    position = {
        "x": random.randint(0, 1000),
        "y": random.randint(0, 1000),
        "initial_width": random.randint(0, 1000),
        "initial_height": random.randint(0, 1000),
    }
    return jsonify(position=position)


@app.route("/")
def home():
    return render_template("api_index.html")


if __name__ == '__main__':
    app.run(debug=True)
