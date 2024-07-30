from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
    image = FileField('Image', validators=[FileRequired()])
    submit = SubmitField('Predict')