import os
from flask import Flask, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, FileAllowed
from wtforms import (
    StringField,
    IntegerField,
    DecimalField,
    TextAreaField,
    SelectField,
    SubmitField,
    FileField,
)
from wtforms.validators import DataRequired
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"
app.config["UPLOAD_FOLDER"] = "static/uploads/"
app.config["ALLOWED_EXTENSIONS"] = set(["png", "jpg", "jpeg"])
bootstrap = Bootstrap5(app)


@app.route("/")
def index():
    return render_template("index.html")


class MammogramForm(FlaskForm):
    name = StringField(
        "Patient Name",
        validators=[DataRequired()],
    )
    age = IntegerField(
        "Age",
        validators=[DataRequired()],
    )
    address = TextAreaField(
        "Address",
        validators=[DataRequired()],
    )
    city = StringField(
        "City",
        validators=[DataRequired()],
    )
    mammogram = FileField(
        "Mammogram",
        validators=[
            FileRequired(),
            FileAllowed(["jpg", "png"], "Select only jpg/png image files"),
        ],
    )
    algorithm = SelectField(
        "Algorithm",
        choices=["Inception", "VGG16", "VGG19", "Mob"],
        default="Inception",
        validators=[DataRequired()],
    )
    submit = SubmitField(
        "Submit",
    )


@app.route("/diagnose", methods=["GET", "POST"])
def diagnose():
    form = MammogramForm()
    if form.validate_on_submit():
        name = form.name.data
        age = form.age.data
        address = form.address.data
        city = form.city.data
        mammogram = form.mammogram.data
        algorithm = form.algorithm.data

        filename = secure_filename(mammogram.filename)
        mammogram.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    

        out = predict(filepath, algorithm)
        out = np.argmax(out)
        class_names = np.array(["benign", "malignant"])
        result = class_names[out]

        session["name"] = name
        session["age"] = age
        session["address"] = address
        session["city"] = city
        session["result"] = result
        session["image"] = filepath
        session["algorithm"] = algorithm
        return redirect(url_for("result"))
    return render_template("diagnose.html", form=form)


@app.route("/result")
def result():
    name = session["name"]
    age = session["age"]
    address = session["address"]
    city = session["city"]
    result = session["result"]
    image= session["image"]
    algorithm= session["algorithm"]
    return render_template(
        "result.html",
        name=name,
        age=age,
        address=address,
        city=city,
        result=result.upper(),
        image=image,
        algorithm=algorithm,
    )


def predict(filepath, algorithm):
    image = cv2.imread(filepath)
    image = cv2.resize(image, (128, 128))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    image = np.reshape(image, (1, 128, 128, 3))
    predictions = None
    if algorithm == "Inception":
        inception_model = load_model("prediction_models/inception_model_95.h5")
        predictions = inception_model.predict(image)
    elif algorithm == "VGG16":
        vgg16_model = load_model("prediction_models/inception_model_95.h5")
        predictions = vgg16_model.predict(image)
    elif algorithm == "Mob":
        mob_model = load_model("prediction_models/inception_model_95.h5")
        predictions = mob_model.predict(image)
    elif algorithm == "VGG19":
        vgg19_model = load_model("prediction_models/inception_model_95.h5")
        predictions = vgg19_model.predict(image)
    return predictions


if __name__ == "__main__":
    app.run(debug=True)
