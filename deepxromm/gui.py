"""
This module implements a Flask webapp GUI for DeepXROMM
"""

import os
from flask import Flask, render_template, request, session, send_file
from deepxromm import DeepXROMM
from .logging_utils import LOG_FILE

app = Flask(__name__)
app.secret_key = "super-secret-key"  # required to use sessions


@app.context_processor
def inject_project():
    return {
        "current_project": session.get("current_project"),
    }


@app.errorhandler(500)
def internal_error(error):
    return render_template("errors/500.html", error=error), 500


@app.route("/")
def return_homepage():
    return render_template("index.html")


@app.get("/logs")
def get_logs():
    return send_file(LOG_FILE, mimetype="text/plain")


@app.get("/create-project")
def create_project():
    return render_template("create_project.html")


@app.post("/create-project")
def create_project_now():
    experimenter = request.form["experimenter"]
    mode = request.form.get("mode", None)
    working_dir = request.form["working_dir"]

    try:
        if mode:
            DeepXROMM.create_new_project(working_dir, experimenter, mode=mode)
        else:
            DeepXROMM.create_new_project(working_dir, experimenter)
    except Exception as e:
        return render_template("errors/500.html", error=e), 500

    return f"✅ Project created in: {working_dir}"


@app.get("/load-project")
def load_project():
    return render_template("load_project.html")


@app.post("/load-project")
def load_new_project():
    working_dir = request.form["working_dir"]
    try:
        deepxromm = DeepXROMM.load_project(working_dir)
    except Exception as e:
        return render_template("errors/500.html", error=e), 500

    session["current_project"] = deepxromm.config["task"]
    return f"✅ Project loaded in: {working_dir}"


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
