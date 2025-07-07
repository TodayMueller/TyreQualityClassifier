import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'classifier_key')
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:123@localhost/classifierdb'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
