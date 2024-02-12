from flask import Flask, render_template, request, redirect, flash, url_for
app = Flask(__name__)

from app.routes.perry import perry
from app.routes.nicole import nicole
from app.routes.naveen import naveen
from app.routes.yewteck import yewteck

app.register_blueprint(perry)
app.register_blueprint(nicole)
app.register_blueprint(naveen)
app.register_blueprint(yewteck)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == "__main__":
    app.run(debug=True, port=80)

