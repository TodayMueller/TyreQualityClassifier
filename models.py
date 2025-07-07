from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    requests      = db.relationship('ImageRequest', backref='user', lazy=True)
    
class ImageRequest(db.Model):
    __tablename__ = 'image_requests'
    id              = db.Column(db.Integer, primary_key=True)
    timestamp       = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    filename        = db.Column(db.String(200), nullable=False)
    verdict         = db.Column(db.String(50), nullable=False)
    probability     = db.Column(db.Float,  nullable=True)
    processing_time = db.Column(db.Float,  nullable=True) 
    image_data      = db.Column(db.LargeBinary, nullable=False)
    user_id         = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
