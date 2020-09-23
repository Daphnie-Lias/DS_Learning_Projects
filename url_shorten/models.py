from .extensions import db
from datetime import datetime
from dataclasses import dataclass



@dataclass
class Url(db.Model):
    """ This class represents the Url table."""
    __tablename__ = 'Url'
    id = db.Column(db.Integer,autoincrement=True, primary_key=True)
    original_url = db.Column(db.String(512))
    short_url = db.Column(db.String(6), unique=True)
    visits = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
    last_access = db.Column(db.DateTime, default=datetime.utcnow())

    def __init__(self, *args, **kwargs):
        super(Url, self).__init__(*args, **kwargs)

    def __repr__(self):
        return '<URL %s>' % self.original_url
