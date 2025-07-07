from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import (
    DataRequired, Length, EqualTo, ValidationError
)

def username_unique(form, field):
    from models import User
    if User.query.filter_by(username=field.data).first():
        raise ValidationError('Пользователь с таким именем уже существует')

class LoginForm(FlaskForm):
    username = StringField(
        'Имя пользователя',
        validators=[DataRequired(message='Поле обязательно'), Length(3, 80, message='От 3 до 80 символов')]
    )
    password = PasswordField(
        'Пароль',
        validators=[DataRequired(message='Поле обязательно')]
    )
    submit   = SubmitField('Войти')

class RegisterForm(FlaskForm):
    username = StringField(
        'Имя пользователя',
        validators=[
            DataRequired(message='Поле обязательно'),
            Length(3, 80, message='От 3 до 80 символов'),
            username_unique
        ]
    )
    password = PasswordField(
        'Пароль',
        validators=[
            DataRequired(message='Поле обязательно'),
            Length(6, 128, message='От 6 до 128 символов')
        ]
    )
    password2 = PasswordField(
        'Повторить пароль',
        validators=[
            DataRequired(message='Поле обязательно'),
            EqualTo('password', message='Пароли должны совпадать')
        ]
    )
    submit   = SubmitField('Зарегистрироваться')
