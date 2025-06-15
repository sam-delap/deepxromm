"""
This module implements a Flask webapp GUI for DeepXROMM
"""
from flask import Flask, render_template, request
import deepxromm

app = Flask(__name__)

@app.route("/")
def return_homepage():
    return render_template('index.html')

@app.get("/create-project")
def create_project():
    return render_template('create_project.html')

@app.post("/create-project")
def create_project_now():
    experimenter = request.form['experimenter']
    mode = request.form['mode']
    working_dir = request.form['working_dir']

    return f"{experimenter}, {mode}, {working_dir}"

@app.route("/import-data")
def import_xmalab_data():
    return "Import xmalab data page"

@app.route("/export-data")
def export_dlc_data():
    return "Export DLC data page"

@app.route("/train-network")
def train_network():
    return "Train network page"

@app.route("/track-new-trial")
def track_new_trial():
    return "Track new trial page"

@app.route("/autocorrect")
def autocorrect_points():
    return "Autocorrect page"

@app.route("/find-highest-variation")
def find_highest_variation():
    return "Find highest variation page"

if __name__ == "__main__":
    app.run(debug=True)
