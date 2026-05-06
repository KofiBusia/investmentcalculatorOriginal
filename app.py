# Investment Calculator Flask Application
# File: app.py
# Description: A Flask web application for financial calculations including DCF, FCFE, FCFF, credit risk,
#              leverage ratios, and more, with SQLAlchemy for database integration and WTForms for input validation.

# --- STANDARD LIBRARY IMPORTS ---
import json
import logging
import math
import os
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

# --- THIRD-PARTY IMPORTS ---
from flask import Flask, jsonify, render_template, request, send_from_directory, send_file, session, redirect, url_for, make_response, flash, abort
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_session import Session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_mail import Mail, Message
from dotenv import load_dotenv
import statistics
import csv
import io
import hashlib
import requests as http_requests
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# --- ENVIRONMENT CONFIGURATION ---
load_dotenv()
logger = logging.getLogger(__name__)

# --- FLASK APPLICATION INITIALIZATION ---
app = Flask(__name__)

# --- CONFIGURATION SETTINGS ---
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'books')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use PostgreSQL on Render (DATABASE_URL env var); fall back to SQLite locally.
_raw_db_url = os.getenv('DATABASE_URL', 'sqlite:///site.db')
# Render provides postgres:// scheme; SQLAlchemy requires postgresql://
if _raw_db_url.startswith('postgres://'):
    _raw_db_url = _raw_db_url.replace('postgres://', 'postgresql://', 1)

app.config.update(
    SECRET_KEY=os.environ['SECRET_KEY'],
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50 MB limit for book uploads
    SESSION_TYPE='filesystem',
    SESSION_FILE_THRESHOLD=500,
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=86400,
    WTF_CSRF_TIME_LIMIT=7200,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV', 'production') == 'production',
    SESSION_COOKIE_HTTPONLY=True,
    SQLALCHEMY_DATABASE_URI=_raw_db_url,
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_FILE_DIR=os.path.join(os.path.dirname(__file__), 'instance', 'sessions'),
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    # Flask-Mail (Gmail SMTP)
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv('MAIL_USERNAME', ''),
    MAIL_PASSWORD=os.getenv('MAIL_PASSWORD', ''),
    MAIL_DEFAULT_SENDER=os.getenv('MAIL_USERNAME', 'noreply@investiq.com'),
    ADMIN_EMAIL=os.getenv('ADMIN_EMAIL', 'kyeikofi@gmail.com'),
    # Google OAuth
    GOOGLE_CLIENT_ID=os.getenv('GOOGLE_CLIENT_ID', ''),
    GOOGLE_CLIENT_SECRET=os.getenv('GOOGLE_CLIENT_SECRET', ''),
)

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize Flask-Session
Session(app)

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO if os.getenv('FLASK_ENV') == 'production' else logging.DEBUG,
    filename='app.log' if os.getenv('FLASK_ENV') == 'production' else None,
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
logger.info("Application initialized")

# --- SQLALCHEMY & EXTENSIONS INITIALIZATION ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'user_login'
login_manager.login_message = 'Please log in to access this page.'

# Google OAuth via Authlib
from authlib.integrations.flask_client import OAuth as AuthlibOAuth
_oauth = AuthlibOAuth(app)
google_oauth = _oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID', ''),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

def send_email_safe(subject, recipients, body_html, body_text=''):
    """Send email; silently log on failure so the app never crashes."""
    try:
        msg = Message(subject, recipients=recipients, html=body_html, body=body_text or body_html)
        mail.send(msg)
    except Exception as e:
        logger.error(f'Email send error: {e}')

# --- HR PLATFORM MODELS ---
class JobListing(db.Model):
    __tablename__ = 'job_listings'
    id           = db.Column(db.Integer, primary_key=True)
    title        = db.Column(db.String(200), nullable=False)
    company      = db.Column(db.String(200), default='InvestIQ / Partner')
    location     = db.Column(db.String(200), default='')
    job_type     = db.Column(db.String(50), default='Full-Time')
    sector       = db.Column(db.String(100), default='Finance')
    description  = db.Column(db.Text, default='')
    requirements = db.Column(db.Text, default='')
    salary_range = db.Column(db.String(100), default='')
    is_active    = db.Column(db.Boolean, default=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


class JobApplication(db.Model):
    __tablename__ = 'job_applications'
    id           = db.Column(db.Integer, primary_key=True)
    job_id       = db.Column(db.Integer, db.ForeignKey('job_listings.id'), nullable=True)
    full_name    = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(200), nullable=False)
    phone        = db.Column(db.String(50), default='')
    cover_letter = db.Column(db.Text, default='')
    cv_data      = db.Column(db.Text, default='')  # JSON
    status       = db.Column(db.String(50), default='New')
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    job          = db.relationship('JobListing', backref='applications', lazy=True)


class TrainingBooking(db.Model):
    __tablename__ = 'training_bookings'
    id              = db.Column(db.Integer, primary_key=True)
    booking_type    = db.Column(db.String(20), default='individual')  # individual | corporate
    full_name       = db.Column(db.String(200), nullable=False)
    email           = db.Column(db.String(200), nullable=False)
    phone           = db.Column(db.String(50), default='')
    organization    = db.Column(db.String(200), default='')
    participants    = db.Column(db.Integer, default=1)
    category        = db.Column(db.String(100), default='')
    preferred_date      = db.Column(db.String(100), default='')
    notes               = db.Column(db.Text, default='')
    other_program       = db.Column(db.String(200), default='')   # when category == 'Other'
    referral_code_used  = db.Column(db.String(20),  default='')   # code entered at booking
    status              = db.Column(db.String(50), default='Pending')
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)


# --- DATABASE MODELS ---
class Article(db.Model):
    __tablename__ = 'articles'
    id             = db.Column(db.Integer, primary_key=True)
    title          = db.Column(db.String(200), nullable=False)
    slug           = db.Column(db.String(220), unique=True, nullable=False)
    summary        = db.Column(db.String(500), nullable=False)
    body           = db.Column(db.Text, nullable=False)
    category       = db.Column(db.String(80), default='General')
    thumbnail_url  = db.Column(db.String(500), default='')
    is_published   = db.Column(db.Boolean, default=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at     = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'<Article {self.title}>'


class Video(db.Model):
    __tablename__ = 'videos'
    id            = db.Column(db.Integer, primary_key=True)
    title         = db.Column(db.String(200), nullable=False)
    youtube_url   = db.Column(db.String(500), nullable=False)
    description   = db.Column(db.String(500), default='')
    is_featured   = db.Column(db.Boolean, default=False)
    is_published  = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def youtube_id(self):
        """Extract YouTube video ID from any YouTube URL format."""
        import re
        patterns = [
            r'(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})',
            r'^([A-Za-z0-9_-]{11})$',
        ]
        for pattern in patterns:
            m = re.search(pattern, self.youtube_url)
            if m:
                return m.group(1)
        return ''

    @property
    def embed_url(self):
        vid_id = self.youtube_id
        return f'https://www.youtube.com/embed/{vid_id}' if vid_id else ''

    @property
    def thumbnail_url(self):
        vid_id = self.youtube_id
        return f'https://img.youtube.com/vi/{vid_id}/hqdefault.jpg' if vid_id else ''

    def __repr__(self):
        return f'<Video {self.title}>'


# --- USER AUTHENTICATION MODEL ---
class SiteUser(UserMixin, db.Model):
    __tablename__ = 'site_users'
    id           = db.Column(db.Integer, primary_key=True)
    full_name    = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(200), unique=True, nullable=False)
    phone        = db.Column(db.String(50), default='')
    password_hash= db.Column(db.String(256), default='')
    google_id    = db.Column(db.String(200), default='')
    is_active    = db.Column(db.Boolean, default=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

@login_manager.user_loader
def load_user(user_id):
    return SiteUser.query.get(int(user_id))


# --- BOOK MODELS ---
class Book(db.Model):
    __tablename__ = 'books'
    id           = db.Column(db.Integer, primary_key=True)
    title        = db.Column(db.String(300), nullable=False)
    author       = db.Column(db.String(200), default='')
    description  = db.Column(db.Text, default='')
    cover_url    = db.Column(db.String(500), default='')
    file_path    = db.Column(db.String(500), default='')
    requires_donation = db.Column(db.Boolean, default=False)
    donation_amount   = db.Column(db.Float, default=0.0)
    is_active    = db.Column(db.Boolean, default=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)

class BookRequest(db.Model):
    __tablename__ = 'book_requests'
    id           = db.Column(db.Integer, primary_key=True)
    book_id      = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=False)
    full_name    = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(200), nullable=False)
    phone        = db.Column(db.String(50), default='')
    request_type = db.Column(db.String(20), default='access')  # access | donation
    donation_ref = db.Column(db.String(200), default='')
    message      = db.Column(db.Text, default='')
    status       = db.Column(db.String(30), default='Pending')
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    book         = db.relationship('Book', backref='requests', lazy=True)


# --- DONATION MODEL ---
class Donation(db.Model):
    __tablename__ = 'donations'
    id           = db.Column(db.Integer, primary_key=True)
    full_name    = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(200), nullable=False)
    phone        = db.Column(db.String(50), default='')
    amount       = db.Column(db.Float, nullable=False)
    currency     = db.Column(db.String(10), default='GHS')
    purpose      = db.Column(db.String(200), default='General Support')
    reference    = db.Column(db.String(200), default='')
    message      = db.Column(db.Text, default='')
    status       = db.Column(db.String(30), default='Pending')
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


# --- GISI EXAM PAYMENT MODEL ---
class GISIPayment(db.Model):
    __tablename__ = 'gisi_payments'
    id            = db.Column(db.Integer, primary_key=True)
    full_name     = db.Column(db.String(200), nullable=False)
    email         = db.Column(db.String(200), nullable=False)
    phone         = db.Column(db.String(50),  default='')
    amount        = db.Column(db.Float,       nullable=False)
    plan          = db.Column(db.String(30),  default='single')  # single | bundle
    section       = db.Column(db.Integer,     default=0)
    reference     = db.Column(db.String(200), default='', unique=True)
    status        = db.Column(db.String(30),  default='Pending')  # Pending | Approved | Rejected
    admin_token   = db.Column(db.String(64),  default='', unique=True)  # one-time admin approve link token
    access_code   = db.Column(db.String(20),  default='', unique=True)  # code sent to user after approval
    approved_at   = db.Column(db.DateTime,    nullable=True)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)


# --- CONTACT MESSAGE MODEL ---
class ContactMessage(db.Model):
    __tablename__ = 'contact_messages'
    id           = db.Column(db.Integer, primary_key=True)
    full_name    = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(200), nullable=False)
    phone        = db.Column(db.String(50), default='')
    subject      = db.Column(db.String(300), default='')
    message      = db.Column(db.Text, nullable=False)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


# --- EMPLOYERS' CORNER: CANDIDATE PROFILES ---
class CandidateProfile(db.Model):
    __tablename__    = 'candidate_profiles'
    id               = db.Column(db.Integer, primary_key=True)
    full_name        = db.Column(db.String(200), nullable=False)
    email            = db.Column(db.String(200), nullable=False)
    phone            = db.Column(db.String(50),  default='')
    location         = db.Column(db.String(200), default='')
    desired_role     = db.Column(db.String(200), default='')
    desired_sector   = db.Column(db.String(100), default='')
    current_title    = db.Column(db.String(200), default='')
    skills_summary   = db.Column(db.String(500), default='')
    profile_summary  = db.Column(db.Text,        default='')
    linkedin         = db.Column(db.String(500), default='')
    years_exp        = db.Column(db.String(30),  default='')
    availability     = db.Column(db.String(50),  default='')
    is_visible       = db.Column(db.Boolean,     default=True)
    created_at       = db.Column(db.DateTime,    default=datetime.utcnow)

    @property
    def skills_list(self):
        return [s.strip() for s in (self.skills_summary or '').split(',') if s.strip()]

    @property
    def availability_urgency(self):
        """Return 0 (most urgent) to 5 for sorting."""
        order = {
            'Immediately available': 0,
            'Immediately': 0,
            'Within 1 week': 1,
            'Within 2 weeks': 2,
            'Within 1 month': 3,
            'Within 3 months': 4,
        }
        return order.get(self.availability, 5)

    @property
    def inquiry_count(self):
        return len(self.inquiries) if hasattr(self, 'inquiries') else 0


class EmployerAccount(db.Model):
    __tablename__    = 'employer_accounts'
    id               = db.Column(db.Integer, primary_key=True)
    company_name     = db.Column(db.String(200), nullable=False, unique=True)
    contact_name     = db.Column(db.String(200), nullable=False)
    email            = db.Column(db.String(200), nullable=False, unique=True)
    password_hash    = db.Column(db.String(256), nullable=False)
    phone            = db.Column(db.String(50),  default='')
    industry         = db.Column(db.String(100), default='')
    company_size     = db.Column(db.String(50),  default='')
    website          = db.Column(db.String(500), default='')
    hiring_for       = db.Column(db.String(300), default='')
    is_verified      = db.Column(db.Boolean, default=False)
    is_active        = db.Column(db.Boolean, default=True)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    shortlists       = db.relationship('EmployerShortlist', backref='employer', lazy=True, cascade='all, delete-orphan')
    sent_inquiries   = db.relationship('EmployerInquiry', backref='employer_account', lazy=True, foreign_keys='EmployerInquiry.employer_account_id')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    @property
    def initials(self):
        words = self.company_name.split()
        return (words[0][0] + (words[1][0] if len(words) > 1 else '')).upper()

    @property
    def shortlist_count(self):
        return len(self.shortlists)

    @property
    def inquiry_count(self):
        return len(self.sent_inquiries)


class EmployerShortlist(db.Model):
    __tablename__  = 'employer_shortlists'
    id             = db.Column(db.Integer, primary_key=True)
    employer_id    = db.Column(db.Integer, db.ForeignKey('employer_accounts.id'), nullable=False)
    candidate_id   = db.Column(db.Integer, db.ForeignKey('candidate_profiles.id'), nullable=False)
    notes          = db.Column(db.String(500), default='')
    stage          = db.Column(db.String(50),  default='Saved')  # Saved, Contacted, Interviewing, Offer, Hired
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    candidate      = db.relationship('CandidateProfile', backref='shortlisted_by', lazy=True)


class EmployerInquiry(db.Model):
    __tablename__        = 'employer_inquiries'
    id                   = db.Column(db.Integer, primary_key=True)
    candidate_id         = db.Column(db.Integer, db.ForeignKey('candidate_profiles.id'), nullable=False)
    employer_account_id  = db.Column(db.Integer, db.ForeignKey('employer_accounts.id'), nullable=True)
    employer_company     = db.Column(db.String(200), default='')
    employer_name        = db.Column(db.String(200), nullable=False)
    employer_email       = db.Column(db.String(200), nullable=False)
    employer_phone       = db.Column(db.String(50),  default='')
    inquiry_type         = db.Column(db.String(80),  default='Interview Request')
    role_offering        = db.Column(db.String(200), default='')
    message              = db.Column(db.Text,        default='')
    created_at           = db.Column(db.DateTime,    default=datetime.utcnow)
    candidate            = db.relationship('CandidateProfile', backref='inquiries', lazy=True)


class Referral(db.Model):
    __tablename__    = 'referrals'
    id               = db.Column(db.Integer, primary_key=True)
    referrer_name    = db.Column(db.String(200), default='')
    referrer_email   = db.Column(db.String(200), nullable=False, index=True)
    referral_code    = db.Column(db.String(20),  nullable=False, index=True)
    referred_name    = db.Column(db.String(200), default='')
    referred_email   = db.Column(db.String(200), default='')
    booking_id       = db.Column(db.Integer, db.ForeignKey('training_bookings.id'), nullable=True)
    status           = db.Column(db.String(20),  default='Pending')   # Pending / Successful
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    booking          = db.relationship('TrainingBooking', backref='referral_record', lazy=True)


class CVSurveyResponse(db.Model):
    __tablename__         = 'cv_survey_responses'
    id                    = db.Column(db.Integer, primary_key=True)
    full_name             = db.Column(db.String(200), default='')
    email                 = db.Column(db.String(200), nullable=False, index=True)
    yin_member            = db.Column(db.String(3),   default='No')   # Yes / No
    stock_pitch           = db.Column(db.String(3),   default='No')
    want_internship       = db.Column(db.String(3),   default='No')
    want_national_service = db.Column(db.String(3),   default='No')
    created_at            = db.Column(db.DateTime,    default=datetime.utcnow)


class MentorshipApplication(db.Model):
    __tablename__   = 'mentorship_applications'
    id              = db.Column(db.Integer, primary_key=True)
    full_name       = db.Column(db.String(200), nullable=False)
    email           = db.Column(db.String(200), nullable=False, index=True)
    phone           = db.Column(db.String(50),  default='')
    institution     = db.Column(db.String(200), default='')
    program         = db.Column(db.String(200), default='')   # e.g. BSc Finance
    year_of_study   = db.Column(db.String(30),  default='')
    interest_area   = db.Column(db.String(100), default='')   # Finance, IB, AM, etc.
    availability    = db.Column(db.String(50),  default='')   # Weekdays, Weekends, etc.
    why_mentorship  = db.Column(db.Text,        default='')
    linkedin        = db.Column(db.String(500), default='')
    status          = db.Column(db.String(30),  default='Pending')  # Pending / Matched / Active
    created_at      = db.Column(db.DateTime,    default=datetime.utcnow)


class PasswordResetToken(db.Model):
    __tablename__ = 'password_reset_tokens'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('site_users.id'), nullable=False)
    token      = db.Column(db.String(100), unique=True, nullable=False, index=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    used       = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CVSurveyExtra(db.Model):
    __tablename__ = 'cv_survey_extra'
    id                  = db.Column(db.Integer, primary_key=True)
    email               = db.Column(db.String(200), nullable=False, index=True)
    yin_join_date       = db.Column(db.String(20), default='')   # YYYY-MM-DD
    years_with_yin      = db.Column(db.String(10), default='')   # computed
    seeking_full_time   = db.Column(db.String(3), default='No')  # Yes / No
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)


class YINProgram(db.Model):
    __tablename__ = 'yin_programs'
    id          = db.Column(db.Integer, primary_key=True)
    name        = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, default='')
    is_active   = db.Column(db.Boolean, default=True)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class YINRegistration(db.Model):
    __tablename__ = 'yin_registrations'
    id               = db.Column(db.Integer, primary_key=True)
    program_id       = db.Column(db.Integer, db.ForeignKey('yin_programs.id'), nullable=False)
    program_name     = db.Column(db.String(200), default='')
    full_name        = db.Column(db.String(200), nullable=False)
    phone            = db.Column(db.String(50), default='')
    email            = db.Column(db.String(200), nullable=False, index=True)
    institution      = db.Column(db.String(200), default='')
    institution_type = db.Column(db.String(20), default='')   # tertiary / non-tertiary
    how_heard        = db.Column(db.String(200), default='')
    is_existing_member = db.Column(db.Boolean, default=False)
    yin_code         = db.Column(db.String(20), default='', index=True)  # existing or generated
    confirmed        = db.Column(db.Boolean, default=False)
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)


class ContentAdmin(db.Model):
    __tablename__ = 'content_admins'
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(200), nullable=False)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_active     = db.Column(db.Boolean, default=True)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


def _validate_company_name(name):
    """Return (ok: bool, error: str). Enforces real registered company name."""
    import re
    name = (name or '').strip()
    _BANNED = {
        'company','test','abc','n/a','na','none','null','business','firm',
        'corp','inc','llc','ltd','enterprise','enterprises','organization',
        'organisation','my company','your company','example','demo',
        'placeholder','company name','employer','hr','unknown','anonymous',
        'private','acme','foo','bar','baz','xyz','123','000',
    }
    if len(name) < 3:
        return False, 'Company name must be at least 3 characters.'
    if name.lower() in _BANNED:
        return False, f'"{name}" is not a valid registered company name. Please enter your official company name exactly as registered.'
    if re.match(r'^\d+$', name):
        return False, 'Company name cannot be numbers only.'
    if not re.match(r"^[A-Za-z0-9&\-\.\(\)\s']+$", name):
        return False, 'Company name contains invalid characters. Only letters, numbers, spaces, &, -, . and () are allowed.'
    if len(name) > 200:
        return False, 'Company name is too long (max 200 characters).'
    return True, ''


# --- CREATE TABLES (runs on every import, idempotent) ---
with app.app_context():
    try:
        db.create_all()
    except Exception as _db_err:
        logger.warning(f"db.create_all() skipped: {_db_err}")
    # Add any missing columns that were added after initial table creation
    try:
        _migrate_sql = [
            "ALTER TABLE gisi_payments ADD COLUMN IF NOT EXISTS admin_token VARCHAR(64)",
            "ALTER TABLE gisi_payments ADD COLUMN IF NOT EXISTS access_code VARCHAR(20)",
            "ALTER TABLE gisi_payments ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP",
        ]
        for _sql in _migrate_sql:
            try:
                db.session.execute(db.text(_sql))
            except Exception:
                db.session.rollback()
        db.session.commit()
    except Exception as _col_err:
        db.session.rollback()
        logger.warning(f"Column migration skipped: {_col_err}")

# --- DATACLASSES ---
from dataclasses import dataclass

@dataclass
class DCFResult:
    total_pv: float
    pv_cash_flows: list
    terminal_value: float
    pv_terminal: float
    total_dcf: float
    intrinsic_per_share: float = None

# --- FORMS ---
class PeriodForm(FlaskForm):
    period_name = StringField('Period Name', validators=[DataRequired(), Length(max=10)])
    currency = SelectField('Currency', choices=[
        ('USD', 'USD - US Dollar'),
        ('GHS', 'GHS - Ghanaian Cedi'),
        ('EUR', 'EUR - Euro'),
        ('GBP', 'GBP - British Pound'),
        ('JPY', 'JPY - Japanese Yen')
    ], validators=[DataRequired()])
    weight_scenario = SelectField('Weighting Scenario', choices=[
        ('balanced', 'Balanced (25% each P/B, P/TBV, P/E, DDM)'),
        ('conservative', 'Conservative (40% P/B, 30% P/TBV, 20% P/E, 10% DDM)'),
        ('growth', 'Growth (40% P/E, 30% P/B, 20% P/TBV, 10% DDM)')
    ], validators=[DataRequired()])
    current_price = FloatField('Current Stock Price', validators=[DataRequired(), NumberRange(min=0)])
    required_return = FloatField('Required Return (%)', validators=[DataRequired(), NumberRange(min=0)])
    book_value_per_share = FloatField('Book Value per Share', validators=[DataRequired(), NumberRange(min=0)])
    pb_multiple = FloatField('P/B Multiple', validators=[DataRequired(), NumberRange(min=0)])
    tangible_book_value_per_share = FloatField('Tangible Book Value per Share', validators=[DataRequired(), NumberRange(min=0)])
    ptbv_multiple = FloatField('P/TBV Multiple', validators=[DataRequired(), NumberRange(min=0)])
    eps = FloatField('Current EPS', validators=[DataRequired()])
    pe_multiple = FloatField('P/E Multiple', validators=[DataRequired(), NumberRange(min=0)])
    eps_growth = FloatField('EPS Growth Rate (%)', validators=[DataRequired()])
    pe_years = IntegerField('Projection Years for P/E', validators=[DataRequired(), NumberRange(min=1, max=5)])
    dividend_per_share = FloatField('Dividend per Share', validators=[DataRequired(), NumberRange(min=0)])
    dividend_growth = FloatField('Dividend Growth Rate (%)', validators=[DataRequired()])
    roe = FloatField('Return on Equity (%)', validators=[DataRequired()])

class FCFFForm(FlaskForm):
    currency = SelectField('Currency', choices=[
        ('USD', 'USD - US Dollar'),
        ('GHS', 'GHS - Ghanaian Cedi'),
        ('EUR', 'EUR - Euro'),
        ('GBP', 'GBP - British Pound'),
        ('JPY', 'JPY - Japanese Yen')
    ], validators=[DataRequired()])
    ebit = FloatField('EBIT', validators=[DataRequired(), NumberRange(min=0)])
    tax_rate = FloatField('Tax Rate (%)', validators=[DataRequired(), NumberRange(min=0, max=100)])
    depreciation = FloatField('Depreciation', validators=[DataRequired(), NumberRange(min=0)])
    capex = FloatField('Capital Expenditures (CapEx)', validators=[DataRequired(), NumberRange(min=0)])
    delta_nwc = FloatField('Change in Net Working Capital (ΔNWC)', validators=[DataRequired()])
    net_income = FloatField('Net Income', validators=[DataRequired(), NumberRange(min=0)])
    interest = FloatField('Interest Expense', validators=[DataRequired(), NumberRange(min=0)])
    ebitda = FloatField('EBITDA', validators=[DataRequired(), NumberRange(min=0)])
    taxes = FloatField('Taxes', validators=[DataRequired(), NumberRange(min=0)])

# --- JINJA FILTERS ---
def format_number(value, decimal_places=2, is_percentage=False, is_currency=False):
    if isinstance(value, (int, float)):
        if is_currency:
            return f"GHS {value:,.{decimal_places}f}"
        elif is_percentage:
            return f"{value:.{decimal_places}f}%"
        else:
            return f"{value:,.{decimal_places}f}" if abs(value) >= 1000 else f"{value:.{decimal_places}f}"
    return value

app.jinja_env.filters['format_number'] = format_number

def format_currency(value, currency='GHS'):
    try:
        value = round(float(value), 2)
        currency_symbols = {
            'USD': '$',
            'GHS': '₵',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥'
        }
        symbol = currency_symbols.get(currency, '')
        return f"{symbol}{value:,.2f}"
    except (ValueError, TypeError):
        return "0.00"

app.jinja_env.filters['currency'] = format_currency
app.jinja_env.filters['format_currency'] = format_currency

# early_exit route (legacy — redirects to home)
@app.route('/early-exit')
def early_exit():
    return redirect(url_for('index'))

# --- FINANCIAL CALCULATION FUNCTIONS ---
def calculate_cca(pe_ratio, earnings):
    """Calculate Comparable Company Analysis (CCA) valuation."""
    return pe_ratio * earnings

def calculate_nav(assets, liabilities):
    """Calculate Net Asset Value (NAV)."""
    return assets - liabilities

def calculate_market_cap(share_price, shares_outstanding):
    """Calculate Market Capitalization."""
    return share_price * shares_outstanding

def calculate_ev(market_cap, debt, cash):
    """Calculate Enterprise Value (EV)."""
    return market_cap + debt - cash

def calculate_replacement_cost(tangible_assets, intangible_assets, adjustment_factor):
    """Calculate Replacement Cost."""
    return (tangible_assets + intangible_assets) * adjustment_factor

def calculate_risk_adjusted_return(returns, risk_free_rate, beta, market_return):
    """Calculate Risk-Adjusted Return using CAPM."""
    return risk_free_rate + beta * (market_return - risk_free_rate)

def calculate_two_stage_ddm(dividend, g_high, years_high, g_terminal, r):
    """Calculate Two-Stage Dividend Discount Model (DDM)."""
    g_high = g_high / 100
    g_terminal = g_terminal / 100
    r = r / 100
    pv_dividends = 0
    current_dividend = dividend
    for t in range(1, years_high + 1):
        current_dividend *= (1 + g_high)
        pv_dividends += current_dividend / (1 + r)**t
    terminal_dividend = current_dividend * (1 + g_terminal)
    if r <= g_terminal:
        raise ValueError("Discount rate must exceed terminal growth rate")
    terminal_value = terminal_dividend / (r - g_terminal)
    pv_terminal = terminal_value / (1 + r)**years_high
    return pv_dividends + pv_terminal

def calculate_two_stage_dcf(fcfe, g_high, years_high, g_terminal, r):
    """Calculate Two-Stage Discounted Cash Flow (DCF)."""
    g_high = g_high / 100
    g_terminal = g_terminal / 100
    r = r / 100
    pv_fcfes = 0
    current_fcf = fcfe
    for t in range(1, years_high + 1):
        current_fcf *= (1 + g_high)
        pv_fcfes += current_fcf / (1 + r)**t
    terminal_fcf = current_fcf * (1 + g_terminal)
    if r <= g_terminal:
        raise ValueError("Discount rate must exceed terminal growth rate")
    terminal_value = terminal_fcf / (r - g_terminal)
    pv_terminal = terminal_value / (1 + r)**years_high
    return pv_fcfes + pv_terminal

def calculate_pe_target(eps, g, years, pe):
    """Calculate P/E Target Price."""
    g = g / 100
    projected_eps = eps * (1 + g)**years
    return projected_eps * pe

def calculate_beta(stock_returns, market_returns):
    """Calculate Beta using covariance and variance."""
    if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
        raise ValueError("Stock and market returns must have equal length and at least 2 data points.")
    cov = statistics.covariance(stock_returns, market_returns)
    var = statistics.variance(market_returns)
    if var == 0:
        raise ValueError("Market returns variance cannot be zero.")
    return cov / var

def calculate_tbills_rediscount(face_value, discount_rate, days_to_maturity):
    """Calculate T-Bill Rediscount Value."""
    discount_rate = discount_rate / 100
    discount_amount = face_value * discount_rate * (days_to_maturity / 365)
    return face_value - discount_amount

# ── FREEMIUM ACCESS CONTROL ───────────────────────────────────────────────────
# 5 calculators available free (no login); all others require authentication.
FREE_CALC_ROUTES = {
    '/dcf', '/bonds', '/mortgage',
    '/portfolio-return', '/portfolio_return',
    '/tbill', '/treasury-bill',
}

# Every calculator/tool route that requires login.
# Public pages (home, about, contact, jobs listing, training, donate, books, etc.) are always open.
PROTECTED_CALC_ROUTES = {
    '/fcff', '/fcfe', '/asset-allocation', '/leverage_ratios', '/cost_sustainability',
    '/capital-structure', '/multiples-master-valuation', '/calculate-beta',
    '/bank-intrinsic-value', '/calculate-cost-of-equity', '/target-price',
    '/tbills-rediscount', '/credit_risk', '/bond_risk_help', '/portfolio_risk_help',
    '/bond_risk', '/bond-risk', '/portfolio-risk', '/portfolio_risks',
    '/portfolio-diversification', '/portfolio_diversification',
    '/volatility', '/risk-calculator', '/risk_calculator',
    '/risk-assessment', '/risk_assessment', '/expected-return', '/duration',
    '/dvm', '/intrinsic-value', '/valuation-methods', '/multi-method-valuation',
    '/specialized-industry-multiples', '/Specialized_Industry_Multiples',
    '/valuation-performance-multiples', '/Valuation_Performance_Multiples',
    '/bond-calculator', '/private-equity', '/private-debt', '/startup-valuation',
    '/informal-sector', '/options', '/monte-carlo', '/real-estate',
    '/fx-forward', '/commodity-futures', '/yield-curve', '/tips',
    '/convertible-bond', '/cds', '/esop', '/drip', '/pension', '/microfinance',
    '/esg', '/early-exit',
}

@app.before_request
def enforce_freemium():
    path = request.path.rstrip('/')
    if not path:
        path = '/'
    # Calculator routes that are not free
    if path in PROTECTED_CALC_ROUTES and path not in FREE_CALC_ROUTES:
        if not current_user.is_authenticated:
            return redirect(url_for('user_login', next=request.url))
    # Articles and videos require login
    if path in ('/articles', '/videos') or path.startswith('/articles/'):
        if not current_user.is_authenticated:
            return redirect(url_for('user_login', next=request.url))
    # Job application requires login (job listings remain public)
    if request.endpoint == 'job_apply' and not current_user.is_authenticated:
        return redirect(url_for('user_login', next=request.url))


# --- ROUTES ---
@app.route('/')
def index():
    """Render the homepage."""
    logger.debug("Rendering index page")
    books = Book.query.filter_by(is_active=True).order_by(Book.created_at.desc()).limit(4).all()
    return render_template('index.html', featured_books=books)


# ── USER AUTH ROUTES ──────────────────────────────────────────────────────────
@app.route('/signup', methods=['GET', 'POST'])
def user_signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip().lower()
        phone     = request.form.get('phone', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')
        if not full_name or not email or not password:
            error = 'Name, email and password are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif SiteUser.query.filter_by(email=email).first():
            error = 'An account with that email already exists.'
        else:
            user = SiteUser(full_name=full_name, email=email, phone=phone)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            login_user(user)
            send_email_safe(
                'Welcome to InvestIQ!',
                [email],
                f'<h2>Welcome, {full_name}!</h2><p>Your InvestIQ account is now active. '
                f'Explore our <a href="https://investiq.com">financial calculators</a> and professional tools.</p>'
            )
            return redirect(url_for('index'))
    return render_template('signup.html', error=error)


@app.route('/login', methods=['GET', 'POST'])
def user_login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = SiteUser.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=request.form.get('remember') == 'on')
            return redirect(request.args.get('next') or url_for('index'))
        error = 'Invalid email or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def user_logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    import secrets
    sent = False
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if not email:
            error = 'Please enter your email address.'
        else:
            u = SiteUser.query.filter(db.func.lower(SiteUser.email) == email).first()
            if u:
                token = secrets.token_urlsafe(32)
                expires = datetime.utcnow() + timedelta(hours=1)
                db.session.add(PasswordResetToken(user_id=u.id, token=token, expires_at=expires))
                db.session.commit()
                reset_url = url_for('reset_password', token=token, _external=True)
                send_email_safe(
                    subject='Reset your InvestIQ password',
                    recipients=[u.email],
                    body_html=f'''<p>Hi {u.full_name},</p>
<p>Click the link below to reset your password. This link expires in 1 hour.</p>
<p><a href="{reset_url}" style="background:#2563eb;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:700;">Reset Password</a></p>
<p>If you did not request this, ignore this email.</p>''',
                    body_text=f'Reset your InvestIQ password: {reset_url}'
                )
            sent = True
    return render_template('forgot_password.html', sent=sent, error=error)


@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    rec = PasswordResetToken.query.filter_by(token=token, used=False).first()
    if not rec or rec.expires_at < datetime.utcnow():
        return render_template('reset_password.html', invalid=True)
    error = None
    if request.method == 'POST':
        pw  = request.form.get('password', '')
        pw2 = request.form.get('password2', '')
        if len(pw) < 6:
            error = 'Password must be at least 6 characters.'
        elif pw != pw2:
            error = 'Passwords do not match.'
        else:
            u = SiteUser.query.get(rec.user_id)
            u.set_password(pw)
            rec.used = True
            db.session.commit()
            flash('Password updated. You can now sign in.', 'success')
            return redirect(url_for('user_login'))
    return render_template('reset_password.html', invalid=False, error=error, token=token)


# ── GOOGLE OAUTH ──────────────────────────────────────────────────────────────
@app.route('/auth/google')
def google_login():
    if not app.config.get('GOOGLE_CLIENT_ID'):
        flash('Google login is not configured yet. Please use email/password.', 'warning')
        return redirect(url_for('user_login'))
    # Save the 'next' destination so we can redirect after login
    next_url = request.args.get('next') or request.referrer or url_for('index')
    session['oauth_next'] = next_url
    redirect_uri = url_for('google_callback', _external=True)
    return google_oauth.authorize_redirect(redirect_uri)


@app.route('/auth/google/callback')
def google_callback():
    try:
        token     = google_oauth.authorize_access_token()
        user_info = token.get('userinfo') or google_oauth.userinfo()
        g_email   = (user_info.get('email') or '').lower().strip()
        g_name    = user_info.get('name') or user_info.get('given_name', '')
        g_id      = user_info.get('sub', '')
        if not g_email:
            flash('Google did not return an email address. Try again.', 'error')
            return redirect(url_for('user_login'))
        # Find existing user by Google ID or email
        user = SiteUser.query.filter_by(google_id=g_id).first()
        if not user:
            user = SiteUser.query.filter_by(email=g_email).first()
        if user:
            # Update google_id if missing
            if not user.google_id:
                user.google_id = g_id
                db.session.commit()
        else:
            # Create new account
            user = SiteUser(full_name=g_name, email=g_email, google_id=g_id)
            db.session.add(user)
            db.session.commit()
            send_email_safe(
                'Welcome to InvestIQ!',
                [g_email],
                f'<h2>Welcome, {g_name}!</h2>'
                f'<p>Your InvestIQ account is now active via Google. '
                f'Explore our 51+ professional calculators at investiq.com.</p>'
            )
        login_user(user, remember=True)
        next_url = session.pop('oauth_next', None) or url_for('index')
        return redirect(next_url)
    except Exception as e:
        logger.error(f'Google OAuth callback error: {e}')
        flash('Google sign-in failed. Please try again or use email/password.', 'error')
        return redirect(url_for('user_login'))


# ── ADMIN: USER CSV EXPORT ────────────────────────────────────────────────────
@app.route('/admin/users')
def admin_users():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    users = SiteUser.query.order_by(SiteUser.created_at.desc()).all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/users/export')
def admin_users_export():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    users = SiteUser.query.order_by(SiteUser.created_at.desc()).all()
    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(['ID', 'Full Name', 'Email', 'Phone', 'Registered'])
    for u in users:
        w.writerow([u.id, u.full_name, u.email, u.phone, u.created_at.strftime('%Y-%m-%d %H:%M')])
    output = make_response(si.getvalue())
    output.headers['Content-Disposition'] = 'attachment; filename=users.csv'
    output.headers['Content-type'] = 'text/csv'
    return output


# ── BOOKS ROUTES ──────────────────────────────────────────────────────────────
@app.route('/books')
def books_page():
    books = Book.query.filter_by(is_active=True).order_by(Book.created_at.desc()).all()
    return render_template('books.html', books=books)


@app.route('/books/<int:book_id>/request', methods=['GET', 'POST'])
def book_request(book_id):
    book  = Book.query.get_or_404(book_id)
    success = False
    error   = None
    if request.method == 'POST':
        full_name    = request.form.get('full_name', '').strip()
        email        = request.form.get('email', '').strip()
        phone        = request.form.get('phone', '').strip()
        request_type = request.form.get('request_type', 'access')
        donation_ref = request.form.get('donation_ref', '').strip()
        message      = request.form.get('message', '').strip()
        if not full_name or not email:
            error = 'Name and email are required.'
        else:
            br = BookRequest(book_id=book.id, full_name=full_name, email=email, phone=phone,
                             request_type=request_type, donation_ref=donation_ref, message=message)
            db.session.add(br)
            db.session.commit()
            send_email_safe(
                f'Book Request: {book.title}',
                [app.config['ADMIN_EMAIL']],
                f'''<h3>Book Request</h3>
<p><b>Book:</b> {book.title}</p>
<p><b>Type:</b> {request_type}</p>
<p><b>From:</b> {full_name} ({email}) — {phone}</p>
<p><b>Donation Ref:</b> {donation_ref or "N/A"}</p>
<p><b>Message:</b> {message or "None"}</p>'''
            )
            success = True
    return render_template('book_request.html', book=book, success=success, error=error)


@app.route('/admin/books')
def admin_books():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    books = Book.query.order_by(Book.created_at.desc()).all()
    return render_template('admin_books.html', books=books)


@app.route('/admin/books/new', methods=['GET', 'POST'])
def admin_book_new():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    error = None
    if request.method == 'POST':
        title       = request.form.get('title', '').strip()
        author      = request.form.get('author', '').strip()
        description = request.form.get('description', '').strip()
        cover_url   = request.form.get('cover_url', '').strip()
        req_donation= request.form.get('requires_donation') == 'on'
        don_amount  = float(request.form.get('donation_amount', 0) or 0)
        file_path   = ''
        if 'book_file' in request.files:
            f = request.files['book_file']
            if f and f.filename:
                fname = secure_filename(f.filename)
                dest  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                f.save(dest)
                file_path = fname
        if not title:
            error = 'Title is required.'
        else:
            book = Book(title=title, author=author, description=description, cover_url=cover_url,
                        file_path=file_path, requires_donation=req_donation, donation_amount=don_amount)
            db.session.add(book)
            db.session.commit()
            return redirect(url_for('admin_books'))
    return render_template('admin_book_form.html', book=None, error=error)


@app.route('/admin/books/<int:book_id>/edit', methods=['GET', 'POST'])
def admin_book_edit(book_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    book  = Book.query.get_or_404(book_id)
    error = None
    if request.method == 'POST':
        book.title       = request.form.get('title', '').strip()
        book.author      = request.form.get('author', '').strip()
        book.description = request.form.get('description', '').strip()
        book.cover_url   = request.form.get('cover_url', '').strip()
        book.requires_donation = request.form.get('requires_donation') == 'on'
        book.donation_amount   = float(request.form.get('donation_amount', 0) or 0)
        book.is_active   = request.form.get('is_active') == 'on'
        if 'book_file' in request.files:
            f = request.files['book_file']
            if f and f.filename:
                fname = secure_filename(f.filename)
                dest  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                f.save(dest)
                book.file_path = fname
        db.session.commit()
        return redirect(url_for('admin_books'))
    return render_template('admin_book_form.html', book=book, error=error)


@app.route('/admin/books/<int:book_id>/delete', methods=['POST'])
def admin_book_delete(book_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    book = Book.query.get_or_404(book_id)
    db.session.delete(book)
    db.session.commit()
    return redirect(url_for('admin_books'))


@app.route('/admin/book-requests')
def admin_book_requests():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    requests_list = BookRequest.query.order_by(BookRequest.created_at.desc()).all()
    return render_template('admin_book_requests.html', requests=requests_list)


# ── DONATION ROUTES ───────────────────────────────────────────────────────────
@app.route('/donate', methods=['GET', 'POST'])
def donate_page():
    success = False
    error   = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip()
        phone     = request.form.get('phone', '').strip()
        amount    = request.form.get('amount', '').strip()
        currency  = request.form.get('currency', 'GHS')
        purpose   = request.form.get('purpose', 'General Support')
        reference = request.form.get('reference', '').strip()
        message   = request.form.get('message', '').strip()
        if not full_name or not email or not amount:
            error = 'Name, email and amount are required.'
        else:
            try:
                don = Donation(full_name=full_name, email=email, phone=phone,
                               amount=float(amount), currency=currency,
                               purpose=purpose, reference=reference, message=message)
                db.session.add(don)
                db.session.commit()
                send_email_safe(
                    f'Donation Received: {currency} {amount} — {full_name}',
                    [app.config['ADMIN_EMAIL']],
                    f'''<h3>New Donation</h3>
<p><b>From:</b> {full_name} ({email}) — {phone}</p>
<p><b>Amount:</b> {currency} {amount}</p>
<p><b>Purpose:</b> {purpose}</p>
<p><b>Reference:</b> {reference or "None"}</p>
<p><b>Message:</b> {message or "None"}</p>'''
                )
                success = True
            except Exception as e:
                logger.error(f'Donation error: {e}')
                error = 'Could not process. Please try again.'
    return render_template('donate.html', success=success, error=error)


@app.route('/admin/donations')
def admin_donations():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    donations = Donation.query.order_by(Donation.created_at.desc()).all()
    return render_template('admin_donations.html', donations=donations)

@app.route('/asset-allocation')
def asset_allocation():
    """Render the asset allocation page."""
    logger.debug("Rendering asset allocation page")
    return render_template('asset_allocation_npra.html')

@app.route('/bond_risk_help')
def bond_risk_help():
    """Render the bond risk help page."""
    logger.debug("Rendering bond risk help page")
    return render_template('bond_risk_help.html')

@app.route('/portfolio_risk_help')
def portfolio_risk_help():
    """Render the portfolio risk help page."""
    logger.debug("Rendering portfolio risk help page")
    return render_template('portfolio_risk_help.html')

@app.route('/non_portfolio_risk_help')
def non_portfolio_risk_help():
    """Render the non-portfolio risk help page."""
    logger.debug("Rendering non-portfolio risk help page")
    return render_template('non_portfolio_risk_help.html')

@app.route('/dcf', methods=['GET', 'POST'])
def dcf_calculator():
    """Handle DCF calculator form and calculations."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    error = None
    results = None
    if request.method == 'POST':
        try:
            years = int(request.form.get('years', 0))
            if not 1 <= years <= 10:
                raise ValueError("Years must be between 1 and 10")
            discount_rate = float(request.form['discount_rate']) / 100
            terminal_growth = float(request.form['terminal_growth']) / 100
            shares_outstanding = float(request.form.get('shares_outstanding', 0))
            if discount_rate <= terminal_growth:
                raise ValueError("Discount rate must exceed terminal growth rate")
            if shares_outstanding <= 0:
                raise ValueError("Shares outstanding must be a positive number")
            cash_flows = [float(request.form[f'cash_flow_{i}']) for i in range(1, years + 1)]
            pv_cash_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, 1)]
            total_pv = sum(pv_cash_flows)
            terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** years
            total_dcf = total_pv + pv_terminal
            intrinsic_per_share = total_dcf / shares_outstanding if shares_outstanding > 0 else None
            results = DCFResult(total_pv, pv_cash_flows, terminal_value, pv_terminal, total_dcf, intrinsic_per_share)
            logger.info("DCF calculation successful")
        except ValueError as e:
            error = str(e)
            logger.error(f"DCF calculation error: {e}")
    return render_template('dcf.html', error=error, results=results, form_data=form_data)

@app.route('/fcff', methods=['GET', 'POST'])
def fcff_calculator():
    """Handle FCFF calculator form and calculations."""
    form = FCFFForm()
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    error = None
    result = None
    currency = 'GHS'  # Default currency

    if request.method == 'POST' and form.validate_on_submit():
        try:
            # Extract and convert form data
            currency = form.currency.data
            ebit = form.ebit.data
            tax_rate = form.tax_rate.data / 100
            depreciation = form.depreciation.data
            capex = form.capex.data
            delta_nwc = form.delta_nwc.data
            net_income = form.net_income.data
            interest = form.interest.data
            ebitda = form.ebitda.data
            taxes = form.taxes.data

            # Calculate FCFF using all three approaches
            fcff_ebit = ebit * (1 - tax_rate) + depreciation - capex - delta_nwc
            fcff_net_income = net_income + (interest * (1 - tax_rate)) + depreciation - capex - delta_nwc
            fcff_ebitda = ebitda - taxes - capex - delta_nwc

            # Prepare results
            result = {
                'ebit_approach': round(fcff_ebit, 2),
                'net_income_approach': round(fcff_net_income, 2),
                'ebitda_approach': round(fcff_ebitda, 2),
                'currency': currency,
                'interpretation': (
                    f"FCFF calculated using EBIT: {format_currency(fcff_ebit, currency)}, "
                    f"Net Income: {format_currency(fcff_net_income, currency)}, "
                    f"EBITDA: {format_currency(fcff_ebitda, currency)}. "
                    "All methods should yield consistent results if inputs are accurate."
                )
            }
            logger.info("FCFF calculation successful")
        except ValueError as e:
            error = f"Calculation error: {str(e)}"
            logger.error(f"FCFF calculation error: {e}")
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            logger.error(f"Unexpected FCFF calculation error: {e}")

    return render_template(
        'fcff.html',
        form=form,
        form_data=form_data,
        error=error,
        result=result,
        currency_symbol=format_currency(0, currency).replace('0.00', '')
    )

@app.route('/fcfe', methods=['GET', 'POST'])
def fcfe_calculator():
    """Handle FCFE calculator form and calculations."""
    current_year = datetime.now().year
    form_data = {}
    error = None
    results = None

    if request.method == 'POST':
        form_data = request.form
        try:
            # Input data from form
            num_years = int(form_data.get('num_years', 5))
            if num_years < 3 or num_years > 5:
                raise ValueError("Number of years must be between 3 and 5.")
            cost_of_equity = float(form_data.get('cost_equity', 23.08)) / 100
            perpetual_growth_rate = float(form_data.get('growth_rate', 4.0)) / 100
            shares_outstanding = float(form_data.get('shares_outstanding', 265000))
            current_price = float(form_data.get('current_price', 0))
            start_year = int(form_data.get('start_year', current_year))
            valuation_year = int(form_data.get('valuation_year', current_year))

            if cost_of_equity <= 0:
                raise ValueError("Cost of equity must be positive.")
            if perpetual_growth_rate >= cost_of_equity:
                raise ValueError("Perpetual growth rate must be less than cost of equity.")
            if shares_outstanding <= 0:
                raise ValueError("Shares outstanding must be positive.")
            if valuation_year < start_year:
                raise ValueError("Valuation year must be greater than or equal to start year.")
            if (valuation_year - start_year) > 10:
                raise ValueError("Projection period cannot exceed 10 years.")

            # Collect data for available years
            net_incomes = []
            non_cash_charges = []
            capexes = []
            changes_wc = []
            net_borrowings = []
            fcfe_results = []

            for i in range(1, num_years + 1):
                ni_key = f'net_income_{i}'
                nc_key = f'non_cash_{i}'
                cap_key = f'capex_{i}'
                cwc_key = f'change_wc_{i}'
                nb_key = f'net_borrowing_{i}'

                if ni_key in form_data and nc_key in form_data and cap_key in form_data and cwc_key in form_data and nb_key in form_data:
                    ni = float(form_data[ni_key])
                    nc = float(form_data[nc_key])
                    cap = float(form_data[cap_key])
                    cwc = float(form_data[cwc_key])
                    nb = float(form_data[nb_key])

                    fcfe = ni + nc - cap - cwc + nb

                    net_incomes.append(ni)
                    non_cash_charges.append(nc)
                    capexes.append(cap)
                    changes_wc.append(cwc)
                    net_borrowings.append(nb)
                    fcfe_results.append(fcfe)
                else:
                    raise ValueError(f"Missing data for Year {i}.")

            # Calculate average FCFE
            avg_fcfe = sum(fcfe_results) / num_years if fcfe_results else 0

            # Calculate average growth rate of past FCFE
            growth_rates = []
            for i in range(1, num_years):
                if fcfe_results[i-1] != 0:
                    growth_rates.append((fcfe_results[i] - fcfe_results[i-1]) / abs(fcfe_results[i-1]))
            avg_growth_rate = sum(growth_rates) / len(growth_rates) if growth_rates else 0

            # Forecast FV FCFE using the last FCFE as base and user-input perpetual growth rate
            last_fcfe = fcfe_results[-1] if fcfe_results else 0
            fv_fcfe = [last_fcfe * (1 + perpetual_growth_rate) ** t for t in range(1, num_years + 2)]

            # Total FV FCFE
            total_fv_fcfe = sum(fv_fcfe)

            # Calculate PV of FCFE
            pv_fcfe = [
                fv_fcfe[i] / (1 + cost_of_equity) ** (i + 1) for i in range(num_years)
            ]

            # Total PV FCFE
            total_pv_fcfe = sum(pv_fcfe)

            # Calculate Terminal Value at the end of the forecast period
            terminal_value = fv_fcfe[-1] * (1 + perpetual_growth_rate) / (cost_of_equity - perpetual_growth_rate)

            # PV of Terminal Value
            pv_terminal_value = terminal_value / (1 + cost_of_equity) ** (num_years + 1)

            # Total Present Value
            total_pv = total_pv_fcfe + pv_terminal_value

            # Intrinsic Value Per Share
            intrinsic_value_per_share = round(total_pv / shares_outstanding, 2)

            # Prepare DCF table
            dcf_table = []
            for i in range(num_years):
                actual_year = start_year + num_years + i
                time_period = i + 1
                dcf_table.append({
                    'forecast_year': i + 1,
                    'actual_year': actual_year,
                    'time_period': time_period,
                    'fv_fcfe': round(fv_fcfe[i], 2),
                    'pv_fcfe': round(pv_fcfe[i], 2)
                })

            # Add Terminal Value to DCF table
            dcf_table.append({
                'forecast_year': 'Terminal Value',
                'actual_year': start_year + num_years * 2,
                'time_period': num_years + 1,
                'fv_fcfe': round(terminal_value, 2),
                'pv_fcfe': round(pv_terminal_value, 2)
            })

            # Prepare results for template
            results = {
                'avg_fcfe': round(avg_fcfe, 2),
                'avg_growth_rate': round(avg_growth_rate * 100, 2),
                'intrinsic_value_per_share': intrinsic_value_per_share,
                'terminal_value': round(terminal_value, 2),
                'pv_terminal_value': round(pv_terminal_value, 2),
                'total_fv_fcfe': round(total_fv_fcfe, 2),
                'total_pv_fcfe': round(total_pv_fcfe, 2),
                'total_value': round(total_pv, 2),
                'current_price': current_price,
                'interpretation': (
                    f"The company’s Free Cash Flow to Equity averages GHS {avg_fcfe:,.2f} over {num_years} year(s). "
                    f"The estimated intrinsic value per share is GHS {intrinsic_value_per_share:,.2f}. Compare to current stock price."
                ),
                'dcf_table': dcf_table,
                'fcfe_data': [
                    {
                        'year': i + 1,
                        'net_income': net_incomes[i],
                        'non_cash_charges': non_cash_charges[i],
                        'capex': capexes[i],
                        'change_wc': changes_wc[i],
                        'net_borrowing': net_borrowings[i],
                        'fcfe': round(fcfe_results[i], 2)
                    } for i in range(num_years)
                ]
            }

            logger.info("FCFE calculation successful")
            return render_template(
                'intrinsic_value_fcfe.html',
                form_data=form_data,
                error=error,
                results=results,
                num_years=num_years,
                start_year=start_year,
                valuation_year=valuation_year,
                currency_symbol='GHS '
            )

        except (KeyError, ValueError) as e:
            error = str(e)
            logger.error(f"FCFE calculation error: {e}")

    return render_template(
        'intrinsic_value_fcfe.html',
        form_data=form_data,
        error=error,
        results=results,
        num_years=5,
        start_year=current_year,
        valuation_year=current_year,
        currency_symbol='GHS '
    )

@app.route('/ads.txt')
def ads_txt():
    """Serve ads.txt file from static directory."""
    return send_from_directory('static', 'ads.txt')

@app.route('/leverage_ratios')
def leverage_ratios():
    """Render the leverage ratios page."""
    logger.debug("Rendering leverage ratios page")
    return render_template('leverage_ratios.html')

@app.route('/cost_sustainability')
def cost_sustainability():
    """Render the cost sustainability page."""
    logger.debug("Rendering cost sustainability page")
    return render_template('cost_sustainability.html')

@app.route('/capital-structure', methods=['GET', 'POST'])
def capital_structure():
    """Handle capital structure form and WACC calculations."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    error = None
    result = None
    debug = None
    if request.method == 'POST':
        try:
            input_method = request.form.get('input_method', 'direct')
            if input_method == 'direct':
                market_cap = float(request.form.get('market_cap', 0))
                if market_cap <= 0:
                    raise ValueError("Market capitalization must be positive.")
            else:
                share_price = float(request.form.get('share_price', 0))
                outstanding_shares = float(request.form.get('outstanding_shares', 0))
                if share_price <= 0 or outstanding_shares <= 0:
                    raise ValueError("Share price and outstanding shares must be positive.")
                market_cap = share_price * outstanding_shares

            total_debt = float(request.form.get('total_debt', 0))
            cash_and_equivalents = float(request.form.get('cash_and_equivalents', 0))
            risk_free_rate = float(request.form.get('risk_free_rate', 0)) / 100
            beta = float(request.form.get('beta', 0))
            market_return = float(request.form.get('market_return', 0)) / 100
            interest_rate = float(request.form.get('interest_rate', 0)) / 100
            tax_rate = float(request.form.get('tax_rate', 0)) / 100

            if any(x < 0 for x in [total_debt, cash_and_equivalents, risk_free_rate, beta, market_return, interest_rate]):
                raise ValueError("Inputs cannot be negative.")
            if not (0 <= tax_rate <= 1):
                raise ValueError("Tax rate must be between 0% and 100%.")

            net_debt = max(total_debt - cash_and_equivalents, 0)
            total_capital = market_cap + net_debt
            if total_capital <= 0:
                raise ValueError("Total capital must be positive.")

            equity_weight = market_cap / total_capital
            debt_weight = net_debt / total_capital
            cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
            cost_of_debt = interest_rate * (1 - tax_rate)
            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt)

            sensitivity = {'values': [], 'share_prices': [], 'market_caps': []}
            base_share_price = share_price if input_method == 'shares' else market_cap / outstanding_shares if outstanding_shares > 0 else 0
            base_market_cap = market_cap
            price_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]
            for multiplier in price_multipliers:
                adjusted_market_cap = base_market_cap * multiplier
                adjusted_total_capital = adjusted_market_cap + net_debt
                if adjusted_total_capital > 0:
                    eq_weight = adjusted_market_cap / adjusted_total_capital
                    dt_weight = net_debt / adjusted_total_capital
                    adjusted_wacc = (eq_weight * cost_of_equity) + (dt_weight * cost_of_debt)
                    sensitivity['values'].append([round(eq_weight * 100, 2), round(dt_weight * 100, 2), round(adjusted_wacc * 100, 2)])
                else:
                    sensitivity['values'].append(['N/A', 'N/A', 'N/A'])
                if input_method == 'shares' and base_share_price > 0:
                    sensitivity['share_prices'].append(round(base_share_price * multiplier, 2))
                else:
                    sensitivity['market_caps'].append(round(adjusted_market_cap, 2))

            debug = {
                'market_cap': market_cap,
                'total_debt': total_debt,
                'cash_and_equivalents': cash_and_equivalents,
                'net_debt': net_debt,
                'total_capital': total_capital,
                'equity_weight': equity_weight,
                'debt_weight': debt_weight,
                'cost_of_equity': cost_of_equity,
                'cost_of_debt': cost_of_debt,
                'wacc': wacc,
                'sensitivity': sensitivity
            }

            if debt_weight > 0.7:
                debug['warning'] = "High debt weight detected (>70%). This may indicate significant financial leverage."

            result = {
                'equity_weight': round(equity_weight * 100, 2),
                'debt_weight': round(debt_weight * 100, 2),
                'total_capital': round(total_capital, 2),
                'cost_of_equity': round(cost_of_equity * 100, 2),
                'cost_of_debt': round(cost_of_debt * 100, 2),
                'wacc': round(wacc * 100, 2)
            }
            logger.info("Capital structure calculation successful")
            return render_template('capital_structure.html', result=result, debug=debug, form_data=form_data)
        except ValueError as e:
            logger.error(f"Capital structure error: {e}")
            return render_template('capital_structure.html', error=str(e), form_data=form_data)
        except Exception as e:
            logger.error(f"Unexpected error in capital structure: {e}")
            return render_template('capital_structure.html', error=f"An error occurred: {str(e)}", form_data=form_data)
    return render_template('capital_structure.html', form_data=form_data)

@app.route('/credit_risk', methods=['GET', 'POST'])
def credit_risk():
    """Handle credit risk calculator form and calculations."""
    current_year = datetime.now().year
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    error = None
    results = None

    if request.method == 'POST':
        try:
            # Input data from form
            num_years = int(form_data.get('num_years', 5))
            if num_years < 3 or num_years > 5:
                raise ValueError("Number of years must be between 3 and 5.")
            start_year = int(form_data.get('start_year', current_year))
            valuation_year = int(form_data.get('valuation_year', current_year))

            if valuation_year < start_year:
                raise ValueError("Valuation year must be greater than or equal to start year.")
            if (valuation_year - start_year) > 10:
                raise ValueError("Projection period cannot exceed 10 years.")

            # Collect data for available years
            credit_risk_data = []
            for i in range(1, num_years + 1):
                data = {}
                # Validate all required fields
                required_fields = [
                    'cash_equivalents', 'trading_assets', 'loans_customers', 'advances_banks',
                    'other_assets', 'deposits_banks', 'deposits_customers', 'tax_liabilities',
                    'other_liabilities', 'retained_earnings', 'ebit', 'operating_income',
                    'total_assets', 'total_liabilities', 'market_value_equity', 'impairment_loss',
                    'interest_income', 'interest_expense', 'investment_securities',
                    'personnel_expenses', 'depreciation', 'other_expenses', 'shareholders_equity'
                ]
                for field in required_fields:
                    key = f'{field}_{i}'
                    if key not in form_data or form_data[key] == '':
                        raise ValueError(f"Missing or invalid data for {field} in Year {i}.")
                    try:
                        data[field] = float(form_data[key])
                        if data[field] < 0:
                            raise ValueError(f"{field.replace('_', ' ').title()} must be non-negative for Year {i}.")
                    except ValueError:
                        raise ValueError(f"Invalid number for {field.replace('_', ' ').title()} in Year {i}.")

                # Calculate Working Capital
                working_capital = (data['cash_equivalents'] + data['trading_assets'] + 
                                 data['loans_customers'] + data['advances_banks'] + 
                                 data['other_assets']) - (data['deposits_banks'] + 
                                 data['deposits_customers'] + data['tax_liabilities'] + 
                                 data['other_liabilities'])
                
                # Calculate Earning Assets
                earning_assets = (data['loans_customers'] + data['advances_banks'] + 
                                data['investment_securities'] + data['cash_equivalents'])
                
                # Calculate Non-Interest Expenses
                non_interest_expenses = (data['personnel_expenses'] + data['depreciation'] + 
                                      data['other_expenses'])

                # Calculate Ratios
                z_score = (1.2 * (working_capital / data['total_assets']) +
                          1.4 * (data['retained_earnings'] / data['total_assets']) +
                          3.3 * (data['ebit'] / data['total_assets']) +
                          0.6 * (data['market_value_equity'] / data['total_liabilities']) +
                          1.0 * (data['operating_income'] / data['total_assets']))
                
                pcl_ratio = abs(data['impairment_loss']) / data['loans_customers'] if data['loans_customers'] != 0 else 0
                nim = ((data['interest_income'] - data['interest_expense']) / earning_assets 
                      if earning_assets != 0 else 0)
                efficiency_ratio = (non_interest_expenses / data['operating_income'] 
                                  if data['operating_income'] != 0 else 0)
                leverage_ratio = (data['shareholders_equity'] / data['total_assets'] 
                                if data['total_assets'] != 0 else 0)
                debt_to_assets = (data['total_liabilities'] / data['total_assets'] 
                                if data['total_assets'] != 0 else 0)

                credit_risk_data.append({
                    'year': i,
                    'working_capital': working_capital,
                    'earning_assets': earning_assets,
                    'non_interest_expenses': non_interest_expenses,
                    'z_score': z_score,
                    'pcl_ratio': pcl_ratio,
                    'nim': nim,
                    'efficiency_ratio': efficiency_ratio,
                    'leverage_ratio': leverage_ratio,
                    'debt_to_assets': debt_to_assets,
                    **data
                })

            # Calculate averages
            avg_z_score = sum(item['z_score'] for item in credit_risk_data) / num_years
            avg_pcl_ratio = sum(item['pcl_ratio'] for item in credit_risk_data) / num_years
            avg_nim = sum(item['nim'] for item in credit_risk_data) / num_years
            avg_efficiency_ratio = sum(item['efficiency_ratio'] for item in credit_risk_data) / num_years
            avg_leverage_ratio = sum(item['leverage_ratio'] for item in credit_risk_data) / num_years
            avg_debt_to_assets = sum(item['debt_to_assets'] for item in credit_risk_data) / num_years

            # Interpretation
            interpretation = []
            if avg_z_score > 2.99:
                interpretation.append('The bank exhibits <strong>low credit risk</strong> (Z-Score > 2.99), indicating strong financial stability.')
            elif avg_z_score >= 1.81:
                interpretation.append('The bank is in the <strong>grey zone</strong> (Z-Score 1.81–2.99), suggesting moderate credit risk and potential financial uncertainty.')
            else:
                interpretation.append('The bank shows <strong>high credit risk</strong> (Z-Score < 1.81), indicating a high likelihood of financial distress.')
            interpretation.append(f'Average PCL Ratio ({(avg_pcl_ratio * 100):.2f}%) reflects loan quality; higher values indicate riskier loans.')
            interpretation.append(f'Average NIM ({(avg_nim * 100):.2f}%) shows profitability from lending; declining NIM may signal credit risk issues.')
            interpretation.append(f'Average Efficiency Ratio ({(avg_efficiency_ratio * 100):.2f}%) indicates operational efficiency; higher ratios suggest lower loss absorption capacity.')
            interpretation.append(f'Average Leverage Ratio ({(avg_leverage_ratio * 100):.2f}%) shows capital adequacy; higher ratios indicate stronger buffers against losses.')
            interpretation.append(f'Average Debt to Assets Ratio ({(avg_debt_to_assets * 100):.2f}%) reflects leverage; higher ratios increase risk if assets underperform.')

            # Warning for high risk
            warning = None
            if (avg_z_score < 1.81 or avg_pcl_ratio > 0.05 or avg_nim < 0.03 or 
                avg_efficiency_ratio > 0.7 or avg_debt_to_assets > 0.9):
                warning = ('Warning: High credit risk detected. Low Z-Score (<1.81), high PCL Ratio (>5%), '
                          'low NIM (<3%), high Efficiency Ratio (>70%), or high Debt to Assets Ratio (>90%) '
                          'suggest potential financial distress.')

            # Prepare results for template
            results = {
                'avg_z_score': round(avg_z_score, 2),
                'avg_pcl_ratio': round(avg_pcl_ratio * 100, 2),
                'avg_nim': round(avg_nim * 100, 2),
                'avg_efficiency_ratio': round(avg_efficiency_ratio * 100, 2),
                'avg_leverage_ratio': round(avg_leverage_ratio * 100, 2),
                'avg_debt_to_assets': round(avg_debt_to_assets * 100, 2),
                'interpretation': '<br>'.join(interpretation),
                'warning': warning,
                'credit_risk_data': credit_risk_data,
                'scenario': 'Past Data Analysis' if start_year < valuation_year else 'Current/Future Analysis',
                'valuation_year': valuation_year,
                'start_year': start_year,
                'num_years': num_years
            }

            logger.info("Credit risk calculation successful")
            return render_template(
                'credit_risk.html',
                form_data=form_data,
                error=error,
                results=results,
                num_years=num_years,
                start_year=start_year,
                valuation_year=valuation_year
            )

        except (KeyError, ValueError) as e:
            error = str(e)
            logger.error(f"Credit risk calculation error: {e}")

    return render_template(
        'credit_risk.html',
        form_data=form_data,
        error=error,
        results=results,
        num_years=5,
        start_year=current_year,
        valuation_year=current_year
    )

@app.route('/multiples-master-valuation', methods=['GET', 'POST'])
def multiples_master_valuation():
    """Handle multiples master valuation form and calculations."""
    form = PeriodForm()
    form_data = {}
    periods = []
    if request.method == 'POST':
        try:
            for period in ['Q1', 'Q2', 'H1', 'FY']:
                try:
                    currency = request.form[f'{period}_currency']
                    weight_scenario = request.form[f'{period}_weight_scenario']
                    current_price = float(request.form[f'{period}_current_price'])
                    pb_multiple = float(request.form[f'{period}_pb_multiple'])
                    book_value_per_share = float(request.form[f'{period}_book_value_per_share'])
                    ptbv_multiple = float(request.form[f'{period}_ptbv_multiple'])
                    tangible_book_value = float(request.form[f'{period}_tangible_book_value'])
                    pe_multiple = float(request.form[f'{period}_pe_multiple'])
                    eps = float(request.form[f'{period}_eps'])
                    dividend_per_share = float(request.form[f'{period}_dividend_per_share'])
                    dividend_growth = float(request.form[f'{period}_dividend_growth'])
                    required_return = float(request.form[f'{period}_required_return'])

                    if any(x <= 0 for x in [current_price, pb_multiple, book_value_per_share, ptbv_multiple, tangible_book_value, pe_multiple, eps, dividend_per_share, required_return]):
                        raise ValueError("All monetary values and rates must be positive and non-zero")
                    if any(x < 0 or x > 100 for x in [dividend_growth]) or required_return > 50:
                        raise ValueError("Growth rates must be between 0 and 100%, discount rate between 0.01 and 50")
                    if weight_scenario not in ['conservative', 'balanced', 'growth']:
                        raise ValueError("Invalid weighting scenario")
                    if currency not in ['USD', 'GHS', 'EUR', 'GBP', 'JPY']:
                        raise ValueError("Invalid currency selected")

                    pb_value = pb_multiple * book_value_per_share
                    ptbv_value = ptbv_multiple * tangible_book_value
                    pe_value = pe_multiple * eps
                    ddm_value = calculate_two_stage_ddm(dividend_per_share, dividend_growth, 5, dividend_growth * 0.5, required_return)

                    weights = {
                        'conservative': [30, 20, 30, 20],
                        'balanced': [25, 25, 25, 25],
                        'growth': [20, 20, 30, 30]
                    }[weight_scenario]
                    values = [pb_value, ptbv_value, pe_value, ddm_value]
                    weighted_average = sum(v * w / 100 for v, w in zip(values, weights))
                    over_under_valuation = (current_price / weighted_average - 1) * 100 if weighted_average > 0 else float('inf')

                    period_data = {
                        'period_name': period,
                        'currency': currency,
                        'current_price': round(current_price, 2),
                        'pb_value': round(pb_value, 2),
                        'ptbv_value': round(ptbv_value, 2),
                        'pe_value': round(pe_value, 2),
                        'ddm_value': round(ddm_value, 2),
                        'weighted_average': round(weighted_average, 2),
                        'over_under_valuation': round(over_under_valuation, 2),
                        'weights': weights,
                        'weight_scenario': weight_scenario
                    }
                    periods.append(period_data)

                except ValueError as e:
                    logger.error(f"Calculation error for period {period}: {e}")
                    periods.append({'period_name': period, 'error': str(e)})
                    continue
            logger.info("Multiples master valuation calculation successful")
            return render_template('multiples_master.html', periods=periods, form=form, form_data=request.form.to_dict())
        except Exception as e:
            logger.error(f"Unexpected error in multiples master valuation: {e}")
            return render_template('multiples_master.html', error=f"An unexpected error occurred: {str(e)}", form=form, form_data=request.form.to_dict())
    return render_template('multiples_master.html', form=form, form_data=form_data)

@app.route('/calculate-beta', methods=['GET', 'POST'])
def calculate_beta_route():
    """Handle Beta calculation form and results."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            stock_returns = [float(x) for x in request.form.get('stock_returns', '').split(',') if x.strip()]
            market_returns = [float(x) for x in request.form.get('market_returns', '').split(',') if x.strip()]
            beta = calculate_beta(stock_returns, market_returns)
            logger.info("Beta calculation successful")
            return render_template('calculate_beta.html', result={'beta': round(beta, 2)}, form_data=form_data)
        except ValueError as e:
            logger.error(f"Beta calculation error: {e}")
            return render_template('calculate_beta.html', error=str(e), form_data=form_data)
    return render_template('calculate_beta.html', form_data=form_data)

@app.route('/tbills-rediscount', methods=['GET', 'POST'])
def tbills_rediscount():
    """Handle T-Bill rediscount calculation form and results."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            face_value = float(request.form.get('initial_fv', 0))
            discount_rate = float(request.form.get('rate', 0))
            days_to_maturity = int(request.form.get('days_to_maturity', 0))
            if face_value <= 0 or discount_rate <= 0 or days_to_maturity <= 0:
                raise ValueError("All inputs must be positive numbers.")
            rediscount_value = calculate_tbills_rediscount(face_value, discount_rate, days_to_maturity)
            logger.info("T-Bill rediscount calculation successful")
            return render_template('tbills_rediscount.html', result={'settlement_fv': round(rediscount_value, 2), 'face_value_after_rediscount': round(face_value, 2)}, form_data=form_data)
        except ValueError as e:
            logger.error(f"T-Bill rediscount error: {e}")
            return render_template('tbills_rediscount.html', error=str(e), form_data=form_data)
    return render_template('tbills_rediscount.html', form_data=form_data)

@app.route('/bank-intrinsic-value', methods=['GET', 'POST'])
def bank_intrinsic_value():
    """Handle bank intrinsic value calculation."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    result = None
    model = form_data.get('model', 'DDM') if request.method == 'POST' else 'DDM'
    num_years = int(form_data.get('num_years', 5)) if request.method == 'POST' else 5
    error = None
    valuation_comment = ""

    if request.method == 'POST':
        try:
            for key in form_data:
                if key not in ['model', 'num_years']:
                    form_data[key] = float(form_data[key]) if form_data[key] else 0.0

            if model == 'DDM':
                dividends = [form_data.get(f'dividend_{i}', 0.0) for i in range(1, num_years + 1)]
                if len(dividends) != num_years:
                    raise ValueError(f"Exactly {num_years} dividend forecasts required")
            elif model == 'RIM':
                eps_list = [form_data.get(f'eps_{i}', 0.0) for i in range(1, num_years + 1)]
                if len(eps_list) != num_years:
                    raise ValueError(f"Exactly {num_years} EPS forecasts required")

            risk_free_rate = form_data.get('risk_free_rate', 0.0) / 100
            market_return = form_data.get('market_return', 0.0) / 100
            beta = form_data.get('beta', 0.0)
            discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)
            terminal_growth_rate = form_data.get('terminal_growth_rate', 0.0) / 100

            if model == 'DDM':
                pv_dividends = sum(dividends[i] / ((1 + discount_rate) ** (i + 1)) for i in range(num_years))
                terminal_dividend = dividends[-1] * (1 + terminal_growth_rate)
                terminal_value = terminal_dividend / (discount_rate - terminal_growth_rate)
                pv_terminal_value = terminal_value / ((1 + discount_rate) ** num_years)
                result = pv_dividends + pv_terminal_value
            elif model == 'RIM':
                book_value = form_data.get('book_value', 0.0)
                pv_residual_income = 0.0
                current_book_value = book_value
                for i in range(num_years):
                    residual_income = eps_list[i] - (discount_rate * current_book_value)
                    pv_residual_income += residual_income / ((1 + discount_rate) ** (i + 1))
                    current_book_value += eps_list[i]
                terminal_eps = eps_list[-1] * (1 + terminal_growth_rate)
                terminal_residual_income = terminal_eps - (discount_rate * current_book_value)
                terminal_value = terminal_residual_income / (discount_rate - terminal_growth_rate)
                pv_terminal_value = terminal_value / ((1 + discount_rate) ** num_years)
                result = book_value + pv_residual_income + pv_terminal_value

            result = max(float(result), 0.0)
            if form_data.get('market_price'):
                market_price = form_data['market_price']
                valuation_comment = (
                    "The stock may be <span class='font-bold text-green-600'>undervalued</span>."
                    if market_price < result else
                    "The stock may be <span class='font-bold text-red-600'>overvalued</span>."
                    if market_price > result else
                    "The stock is priced at its intrinsic value."
                )
            logger.info(f"Bank intrinsic value calculated using {model} model")
        except (ValueError, ZeroDivisionError) as e:
            error = str(e) if "forecasts required" in str(e) else "Invalid input or calculation error. Ensure all inputs are valid numbers and discount rate is greater than growth rate."
            logger.error(f"Bank intrinsic value error: {e}")
            result = None

    return render_template(
        'bank_intrinsic_value.html',
        result=result,
        model=model,
        num_years=num_years,
        form_data=form_data,
        error=error,
        valuation_comment=valuation_comment
    )

@app.route('/calculate-cost-of-equity', methods=['GET', 'POST'])
def calculate_cost_of_equity():
    """Handle cost of equity calculation."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            risk_free_rate = float(request.form.get('risk_free_rate', 0))
            beta = float(request.form.get('beta', 0))
            market_return = float(request.form.get('market_return', 0))
            dividend_per_share = float(request.form.get('dividend_per_share', 0))
            stock_price = float(request.form.get('stock_price', 0))
            dividend_growth_rate = float(request.form.get('dividend_growth_rate', 0))
            capm_weight = float(request.form.get('capm_weight', 0))
            ddm_weight = float(request.form.get('ddm_weight', 0))

            if any(x < 0 for x in [risk_free_rate, market_return, dividend_per_share, stock_price, dividend_growth_rate, capm_weight, ddm_weight]):
                raise ValueError("All values must be positive")
            if beta <= 0:
                raise ValueError("Beta must be greater than zero")
            if capm_weight + ddm_weight != 100:
                raise ValueError("CAPM and DDM weights must sum to 100%")

            capm_cost = risk_free_rate / 100 + beta * (market_return / 100 - risk_free_rate / 100)
            ddm_cost = (dividend_per_share / stock_price) + (dividend_growth_rate / 100) if stock_price > 0 else 0
            weighted_cost = (capm_cost * capm_weight / 100) + (ddm_cost * ddm_weight / 100)

            result = {
                'capm_cost': round(capm_cost * 100, 2),
                'ddm_cost': round(ddm_cost * 100, 2),
                'weighted_cost': round(weighted_cost * 100, 2)
            }
            logger.info("Cost of equity calculation successful")
            return render_template('cost_of_equity.html', result=result, form_data=form_data)
        except ValueError as e:
            logger.error(f"Cost of Equity calculation error: {e}")
            return render_template('cost_of_equity.html', error=str(e), form_data=form_data)
    return render_template('cost_of_equity.html', form_data=form_data)

@app.route('/target-price')
def target_price():
    return render_template('target_price.html')

# --- STATIC / REDIRECT ROUTES ---
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/delete-account', methods=['GET', 'POST'])
def delete_account():
    if request.method == 'POST':
        return jsonify({'ok': True})
    return render_template('delete_account.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    success = False
    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip()
        phone     = request.form.get('phone', '').strip()
        subject   = request.form.get('subject', '').strip()
        message   = request.form.get('message', '').strip()
        if not full_name or not email or not message:
            error = 'Name, email, and message are required.'
        else:
            try:
                msg_obj = ContactMessage(full_name=full_name, email=email, phone=phone, subject=subject, message=message)
                db.session.add(msg_obj)
                db.session.commit()
                send_email_safe(
                    f'InvestIQ Contact: {subject or "New Message"}',
                    [app.config['ADMIN_EMAIL']],
                    f'''<h3>New Contact Message</h3>
<p><b>From:</b> {full_name} ({email})</p>
<p><b>Phone:</b> {phone or "Not provided"}</p>
<p><b>Subject:</b> {subject or "No subject"}</p>
<hr><p>{message.replace(chr(10),"<br>")}</p>'''
                )
                success = True
            except Exception as e:
                logger.error(f'Contact form error: {e}')
                error = 'Message could not be sent. Please try again.'
    return render_template('contact.html', success=success, error=error)

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/careers')
def careers():
    return render_template('careers.html')

@app.route('/press')
def press():
    return render_template('press.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/privacy')
def privacy_redirect():
    return redirect(url_for('privacy_policy'))

@app.route('/capital-markets-ghana')
def capital_markets_ghana():
    return render_template('capital_markets_ghana.html')

@app.route('/terms')
def terms():
    return render_template('terms_conditions.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/cookies')
def cookies():
    return render_template('cookies.html')

@app.route('/security')
def security():
    return render_template('security.html')

@app.route('/help')
def help_page():
    try:
        with open('calculators.json') as f:
            calculators = json.load(f)
    except FileNotFoundError:
        calculators = []
    return render_template('help.html', calculators=calculators)

@app.route('/faq')
def faq():
    return redirect(url_for('help_page'))

@app.route('/bond_risk')
@app.route('/bond-risk')
def bond_risk():
    return render_template('bond_risk.html', form_data=request.form or {}, result=None, error=None)

# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RISK (NPRA-compliant, multiple risk metrics)
# ──────────────────────────────────────────────────────────────────────────────
_PORT_RISK_METRICS = [
    'Sharpe Ratio', 'Sortino Ratio', 'Maximum Drawdown',
    'Information Ratio', 'Beta', "Treynor Ratio", "Jensen's Alpha",
    'VaR (95%)', 'Correlation',
]
# Ghana-specific assumed annual returns and volatilities per asset class
_ASSET_RETURNS = dict(gov=0.18, local_gov=0.16, equities=0.20, bank=0.16,
                      corp=0.15, coll=0.14, alt=0.12, foreign=0.10)
_ASSET_VOLS    = dict(gov=0.03, local_gov=0.04, equities=0.25, bank=0.02,
                      corp=0.05, coll=0.08, alt=0.15, foreign=0.20)
_GHANA_RF      = 0.16  # Ghana T-Bill proxy risk-free rate

@app.route('/portfolio-risk', methods=['GET', 'POST'])
@app.route('/portfolio_risks', methods=['GET', 'POST'])
def portfolio_risks():
    import math as _m
    form_data = None
    result    = None
    npra_alerts = []
    error       = None

    if request.method == 'POST':
        form_data = request.form
        try:
            gov   = float(form_data.get('gov_securities', 0))
            lgs   = float(form_data.get('local_gov_securities', 0))
            eq    = float(form_data.get('equities', 0))
            bk    = float(form_data.get('bank_securities', 0))
            corp  = float(form_data.get('corporate_debt', 0))
            coll  = float(form_data.get('collective_schemes', 0))
            alt   = float(form_data.get('alternatives', 0))
            frn   = float(form_data.get('foreign', 0))
            grn   = float(form_data.get('green_bonds', 0))
            pv    = float(form_data.get('portfolio_value', 0))
            mret  = float(form_data.get('market_return', 0)) / 100
            mvol  = float(form_data.get('market_volatility', 0)) / 100
            bret  = float(form_data.get('benchmark_return', 0)) / 100
            bvol  = float(form_data.get('benchmark_volatility', 0)) / 100
            dvol  = float(form_data.get('downside_volatility', 0)) / 100
            peak  = float(form_data.get('peak_value', 0))
            trough = float(form_data.get('trough_value', 0))
            metric = form_data.get('risk_metric', 'Sharpe Ratio')

            # NPRA compliance alerts
            for name, (alloc, limit) in [
                ('Government Securities', (gov, 75)),
                ('Local Gov Securities', (lgs, 25)),
                ('Equities', (eq, 20)),
                ('Bank Securities', (bk, 35)),
                ('Corporate Debt', (corp, 35)),
                ('Collective Schemes', (coll, 15)),
                ('Alternatives', (alt, 25)),
                ('Foreign Assets', (frn, 5)),
            ]:
                t = 'warning' if alloc > limit else 'success'
                npra_alerts.append({'type': t,
                    'message': f'{name}: {alloc:.1f}% — NPRA limit {limit}% '
                               f'({"EXCEEDED" if t == "warning" else "compliant"})'})

            # Portfolio weights and expected return
            w = dict(gov=gov/100, local_gov=lgs/100, equities=eq/100, bank=bk/100,
                     corp=corp/100, coll=coll/100, alt=alt/100, foreign=frn/100)
            Ep = sum(w[k] * _ASSET_RETURNS[k] for k in w)

            # Portfolio volatility (diagonal covariance — zero inter-asset correlation)
            var_p = sum((w[k] * _ASSET_VOLS[k])**2 for k in w)
            sig_p = _m.sqrt(var_p) if var_p > 0 else 1e-9

            if metric == 'Sharpe Ratio':
                val  = (Ep - _GHANA_RF) / sig_p
                desc = (f'(Rp={Ep:.2%} − Rf={_GHANA_RF:.2%}) / σp={sig_p:.2%} = {val:.4f}')
                interp = 'Higher value = better risk-adjusted return. >1 is acceptable, >2 is good.'
            elif metric == 'Sortino Ratio':
                ds   = dvol if dvol > 0 else sig_p
                val  = (Ep - _GHANA_RF) / ds
                desc = f'(Rp − Rf) / Downside σ = {val:.4f}'
                interp = 'Like Sharpe but penalises only downside volatility.'
            elif metric == 'Maximum Drawdown':
                val  = (peak - trough) / peak * 100 if peak > 0 else 0
                desc = f'(Peak {peak:,.2f} − Trough {trough:,.2f}) / Peak = {val:.2f}%'
                interp = 'Worst peak-to-trough decline. Lower is better.'
            elif metric == 'Information Ratio':
                te   = abs(sig_p - bvol) if bvol else sig_p
                te   = te if te > 0 else 1e-9
                val  = (Ep - bret) / te
                desc = f'(Rp={Ep:.2%} − Rb={bret:.2%}) / Tracking Error={te:.4f} = {val:.4f}'
                interp = 'Measures excess return per unit of active risk. >0.5 is good.'
            elif metric == 'Beta':
                val  = sig_p / mvol if mvol > 0 else 0
                desc = f'σp={sig_p:.4f} / σm={mvol:.4f} = {val:.4f}'
                interp = 'β=1 moves with market; β<1 lower risk; β>1 higher risk.'
            elif metric == 'Treynor Ratio':
                beta = sig_p / mvol if mvol > 0 else 1
                val  = (Ep - _GHANA_RF) / beta if beta else 0
                desc = f'(Rp − Rf) / β = ({Ep:.2%} − {_GHANA_RF:.2%}) / {beta:.4f} = {val:.4f}'
                interp = 'Reward per unit of systematic risk.'
            elif metric == "Jensen's Alpha":
                beta  = sig_p / mvol if mvol > 0 else 1
                capm  = _GHANA_RF + beta * (mret - _GHANA_RF)
                val   = Ep - capm
                desc  = f'α = Rp − [Rf + β(Rm−Rf)] = {Ep:.2%} − {capm:.2%} = {val:.4f}'
                interp = 'Positive α = outperformance vs CAPM prediction.'
            elif metric == 'VaR (95%)':
                z    = 1.6449  # 95% one-tailed z-score
                val  = pv * (Ep - z * sig_p)
                desc = f'VaR = {pv:,.2f} × ({Ep:.2%} − 1.6449 × {sig_p:.2%}) = GHS {val:,.2f}'
                interp = 'Maximum expected loss over one year at 95% confidence.'
            elif metric == 'Correlation':
                pair = form_data.get('correlation_pair', 'equities-foreign')
                k1, k2 = pair.split('-')[0], pair.split('-')[-1]
                v1 = _ASSET_VOLS.get(k1, _ASSET_VOLS.get('equities'))
                v2 = _ASSET_VOLS.get(k2, _ASSET_VOLS.get('foreign'))
                val  = 0.15  # assumed moderate positive correlation as proxy
                desc = f'Estimated correlation between {k1} and {k2}: {val:.2f}'
                interp = 'Values near -1 offer maximum diversification benefit.'
            else:
                val = 0; desc = 'Unknown metric.'; interp = ''

            result = dict(metric=metric,
                          value=f'{val:.4f}<br><small class="text-gray-500">{desc}</small>',
                          description=interp)
        except Exception as exc:
            error = str(exc)

    return render_template('portfolio_risks.html', result=result,
                           form_data=form_data or {}, risk_metrics=_PORT_RISK_METRICS,
                           npra_alerts=npra_alerts, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO RETURN (9 globally-accepted methods, GIPS-aligned)
# ──────────────────────────────────────────────────────────────────────────────
def _irr(cash_flows, max_iter=1000, tol=1e-9):
    """Newton-Raphson IRR. Raises ValueError if convergence fails."""
    rate = 0.1
    for _ in range(max_iter):
        npv  = sum(cf / (1 + rate)**t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + rate)**(t + 1) for t, cf in enumerate(cash_flows))
        if abs(dnpv) < 1e-12:
            break
        rate -= npv / dnpv
        if abs(npv) < tol:
            break
    return rate

@app.route('/portfolio-return', methods=['GET', 'POST'])
@app.route('/portfolio_return', methods=['GET', 'POST'])
def portfolio_return():
    import math as _m
    result = None
    error  = None

    if request.method == 'POST':
        try:
            method    = request.form.get('method', 'twr')
            raw       = request.form.get('data', '')
            avg_infl  = float(request.form.get('average_inflation', 0)) / 100
            mi_raw    = request.form.get('monthly_inflation', '').strip()
            nums      = [float(x.strip()) for x in raw.split(',') if x.strip()]

            if method == 'twr':
                if len(nums) < 1:
                    raise ValueError('Provide at least one sub-period return.')
                nominal = 1.0
                for r in nums:
                    nominal *= (1 + r)
                nominal -= 1
            elif method == 'mwr':
                if len(nums) < 2:
                    raise ValueError('MWR needs at least 2 cash flows.')
                nominal = _irr(nums)
            elif method == 'modified psa_dietz':
                if len(nums) != 4:
                    raise ValueError('Modified Dietz: initial_value, final_value, cash_flow, weight (0–1)')
                v0, v1, cf, w = nums
                denom = v0 + cf * w
                if abs(denom) < 1e-9:
                    raise ValueError('Denominator is zero — check inputs.')
                nominal = (v1 - v0 - cf) / denom
            elif method == 'simple_dietz':
                if len(nums) != 3:
                    raise ValueError('Simple Dietz: initial_value, final_value, cash_flow')
                v0, v1, cf = nums
                denom = v0 + cf / 2
                if abs(denom) < 1e-9:
                    raise ValueError('Denominator is zero — check inputs.')
                nominal = (v1 - v0 - cf) / denom
            elif method == 'irr':
                if len(nums) < 2:
                    raise ValueError('IRR needs at least 2 cash flows.')
                nominal = _irr(nums)
            elif method == 'hpr':
                if len(nums) != 3:
                    raise ValueError('HPR: initial_price, final_price, dividend')
                p0, p1, d = nums
                if abs(p0) < 1e-9:
                    raise ValueError('Initial price cannot be zero.')
                nominal = (p1 - p0 + d) / p0
            elif method == 'annualized':
                if len(nums) != 2:
                    raise ValueError('Annualized: total_return (decimal), number_of_years')
                r_total, yrs = nums
                if yrs <= 0:
                    raise ValueError('Years must be positive.')
                nominal = (1 + r_total) ** (1 / yrs) - 1
            elif method == 'geometric_mean':
                if not nums:
                    raise ValueError('Provide at least one return.')
                product = 1.0
                for r in nums:
                    product *= (1 + r)
                nominal = product ** (1 / len(nums)) - 1
            elif method == 'arithmetic_mean':
                if not nums:
                    raise ValueError('Provide at least one return.')
                nominal = sum(nums) / len(nums)
            else:
                raise ValueError(f'Unknown method: {method}')

            # Real return — Fisher equation: (1+Rn)/(1+Ri) − 1
            real_avg = (1 + nominal) / (1 + avg_infl) - 1 if (1 + avg_infl) != 0 else 0

            # Time-weighted inflation from monthly rates
            if mi_raw:
                mi_rates = [float(x.strip()) / 100 for x in mi_raw.split(',') if x.strip()]
                tw_infl  = 1.0
                for i in mi_rates:
                    tw_infl *= (1 + i)
                tw_infl = tw_infl ** (1 / len(mi_rates)) - 1 if mi_rates else avg_infl
            else:
                tw_infl = avg_infl
            real_tw = (1 + nominal) / (1 + tw_infl) - 1 if (1 + tw_infl) != 0 else 0

            result = dict(
                method=method,
                nominal_return=f'{nominal:.4%}',
                real_return_avg=f'{real_avg:.4%}',
                real_return_tw=f'{real_tw:.4%}',
            )
        except Exception as exc:
            error = str(exc)

    return render_template('portfolio_return.html', result=result, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO VOLATILITY (Markowitz covariance matrix)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/volatility', methods=['GET', 'POST'])
def volatility():
    import math as _m
    result    = None
    error     = None
    form_data = None

    if request.method == 'POST':
        form_data = request.form
        try:
            n = int(form_data.get('num_assets', 0))
            if n < 1 or n > 10:
                raise ValueError('Number of assets must be between 1 and 10.')
            weights = [float(form_data.get(f'weight_{i}', 0)) for i in range(1, n + 1)]
            if abs(sum(weights) - 1) > 0.05:
                raise ValueError(f'Weights must sum to 1 (current sum: {sum(weights):.4f}).')
            cov = [[float(form_data.get(f'cov_{i}_{j}', 0)) for j in range(1, n + 1)]
                   for i in range(1, n + 1)]

            # Portfolio variance: w^T * C * w
            var_p = sum(weights[i] * weights[j] * cov[i][j]
                        for i in range(n) for j in range(n))
            if var_p < 0:
                raise ValueError('Covariance matrix produces negative variance — check symmetry/PSD.')
            vol_p = _m.sqrt(var_p)

            rows = ['<table class="results-table"><tr><th>Asset</th><th>Weight</th>'
                    '<th>Own Variance (cov_ii)</th><th>Contribution to Variance</th></tr>']
            for i in range(n):
                contrib = sum(weights[i] * weights[j] * cov[i][j] for j in range(n))
                rows.append(f'<tr><td>Asset {i+1}</td><td>{weights[i]:.4f}</td>'
                             f'<td>{cov[i][i]:.6f}</td><td>{contrib:.6f}</td></tr>')
            rows.append('</table>')
            rows.append(f'<p><strong>Portfolio Variance:</strong> {var_p:.6f}</p>')
            rows.append(f'<p><strong>Portfolio Volatility (σ):</strong> {vol_p:.4%}</p>')
            result = ''.join(rows)
        except Exception as exc:
            error = str(exc)

    return render_template('volatility.html', result=result, error=error, form_data=form_data or {})


# ──────────────────────────────────────────────────────────────────────────────
# NON-PORTFOLIO RISK CALCULATOR (viewer page)
# ──────────────────────────────────────────────────────────────────────────────
_NP_RISK_METRICS = [
    'Credit Spread', 'Expected Loss', 'Modified Duration',
    'Price Change (Duration)', 'Bid-Ask Spread',
]

@app.route('/risk-calculator')
@app.route('/risk_calculator')
def risk_calculator():
    return render_template('risk_calculator.html', risk_metrics=_NP_RISK_METRICS,
                           form_data={}, result=None, alerts=[])


@app.route('/non-portfolio-risk', methods=['GET', 'POST'])
def non_portfolio_risk_calc():
    """Non-portfolio single-asset risk metrics."""
    result    = None
    error     = None
    form_data = {}
    alerts    = []

    if request.method == 'POST':
        form_data = request.form
        try:
            metric = form_data.get('risk_metric', 'Credit Spread')
            corp_y  = float(form_data.get('corporate_yield', 0)) / 100
            rf_y    = float(form_data.get('risk_free_yield', 0)) / 100
            pd_pct  = float(form_data.get('probability_default', 0)) / 100
            lgd_pct = float(form_data.get('loss_given_default', 0)) / 100
            ead     = float(form_data.get('exposure_at_default', 0))
            bp      = float(form_data.get('bond_price', 0))
            mac_d   = float(form_data.get('macaulay_duration', 0))
            ytm     = float(form_data.get('yield_to_maturity', 0)) / 100
            comp    = float(form_data.get('compounding_periods', 1))
            dy      = float(form_data.get('yield_change', 0)) / 100
            bid     = float(form_data.get('bid_price', 0))
            ask     = float(form_data.get('ask_price', 0))

            if metric == 'Credit Spread':
                val   = (corp_y - rf_y) * 10000  # in basis points
                desc  = f'CS = Corp Yield − Risk-Free Yield = ({corp_y:.2%} − {rf_y:.2%}) = {val:.1f} bps'
                interp = 'Credit spread above 200 bps signals elevated credit risk.'
            elif metric == 'Expected Loss':
                val   = pd_pct * lgd_pct * ead
                desc  = f'EL = PD × LGD × EAD = {pd_pct:.2%} × {lgd_pct:.2%} × {ead:,.2f} = GHS {val:,.2f}'
                interp = 'The expected monetary loss if the issuer defaults.'
            elif metric == 'Modified Duration':
                if comp <= 0:
                    raise ValueError('Compounding periods must be positive.')
                val   = mac_d / (1 + ytm / comp)
                desc  = f'D_mod = D_mac / (1 + YTM/m) = {mac_d:.4f} / (1 + {ytm:.2%}/{comp:.0f}) = {val:.4f}'
                interp = ('Modified Duration measures % price change per 1% yield move. '
                          'Higher values mean greater interest-rate sensitivity.')
            elif metric == 'Price Change (Duration)':
                if comp <= 0:
                    raise ValueError('Compounding periods must be positive.')
                d_mod = mac_d / (1 + ytm / comp)
                val   = -d_mod * dy * bp
                pct   = -d_mod * dy
                desc  = (f'ΔP ≈ −D_mod × Δy × P = −{d_mod:.4f} × {dy:.2%} × {bp:,.2f} = '
                         f'GHS {val:,.2f} ({pct:.2%})')
                interp = ('Estimated bond price change for the given yield shift. '
                          'Negative Δy (rate drop) → price rise.')
            elif metric == 'Bid-Ask Spread':
                val   = ask - bid
                pct   = val / bid * 100 if bid else 0
                desc  = f'Spread = Ask − Bid = {ask:.4f} − {bid:.4f} = {val:.4f} ({pct:.2f}%)'
                interp = 'Narrower spreads indicate greater liquidity.'
            else:
                val = 0; desc = ''; interp = ''

            result = dict(metric=metric,
                          value=f'{val:,.4f}<br><small class="text-gray-500">{desc}</small>',
                          description=interp)
        except Exception as exc:
            error = str(exc)
    else:
        form_data = {}

    return render_template('risk_calculator.html', risk_metrics=_NP_RISK_METRICS,
                           form_data=form_data, result=result, alerts=alerts, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# RISK ASSESSMENT (NPRA portfolio compliance + expected return/volatility)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/risk-assessment', methods=['GET', 'POST'])
@app.route('/risk_assessment', methods=['GET', 'POST'])
def risk_assessment():
    import math as _m
    result      = None
    form_data   = None
    npra_alerts = []
    error       = None

    if request.method == 'POST':
        form_data = request.form
        try:
            gov   = float(form_data.get('gov_securities', 0))
            lgs   = float(form_data.get('local_gov_securities', 0))
            eq    = float(form_data.get('equities', 0))
            bk    = float(form_data.get('bank_securities', 0))
            corp  = float(form_data.get('corporate_debt', 0))
            coll  = float(form_data.get('collective_schemes', 0))
            alt   = float(form_data.get('alternatives', 0))
            frn   = float(form_data.get('foreign', 0))
            pv    = float(form_data.get('portfolio_value', 0))

            NPRA_LIMITS = [
                ('Government Securities', gov, 75),
                ('Local Gov Securities', lgs, 25),
                ('Equities', eq, 20),
                ('Bank Securities', bk, 35),
                ('Corporate Debt', corp, 35),
                ('Collective Schemes', coll, 15),
                ('Alternatives', alt, 25),
                ('Foreign Assets', frn, 5),
            ]
            for name, alloc, limit in NPRA_LIMITS:
                t = 'warning' if alloc > limit else 'success'
                npra_alerts.append({'type': t,
                    'message': f'{name}: {alloc:.1f}% vs NPRA max {limit}% '
                               f'— {"⚠ EXCEEDED" if t=="warning" else "✓ compliant"}'})

            w = dict(gov=gov/100, local_gov=lgs/100, equities=eq/100, bank=bk/100,
                     corp=corp/100, coll=coll/100, alt=alt/100, foreign=frn/100)
            Ep  = sum(w[k] * _ASSET_RETURNS[k] for k in w)
            var = sum((w[k] * _ASSET_VOLS[k])**2 for k in w)
            sig = _m.sqrt(var) if var > 0 else 0

            # 10% market stress test — equity + alternatives are most sensitive
            stress_pct  = (w['equities'] * 0.10 + w['alt'] * 0.07 + w['foreign'] * 0.08)
            stress_loss = round(pv * stress_pct, 2)

            result = dict(
                expected_return=round(Ep * 100, 2),
                volatility=round(sig * 100, 2),
                stress_loss=f'{stress_loss:,.2f}',
            )
        except Exception as exc:
            error = str(exc)

    return render_template('risk_assessment.html', result=result, form_data=form_data or {},
                           npra_alerts=npra_alerts, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO DIVERSIFICATION
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/portfolio-diversification', methods=['GET', 'POST'])
@app.route('/portfolio_diversification', methods=['GET', 'POST'])
def portfolio_diversification():
    import math as _m
    result    = None
    error     = None
    form_data = None

    if request.method == 'POST':
        form_data = request.form
        try:
            n = int(form_data.get('num_assets', 0))
            if n < 1 or n > 10:
                raise ValueError('Number of assets must be 1–10.')
            weights = [float(form_data.get(f'weight_{i}', 0)) for i in range(1, n + 1)]
            returns = [float(form_data.get(f'return_{i}', 0)) for i in range(1, n + 1)]
            vols    = [float(form_data.get(f'volatility_{i}', 0)) for i in range(1, n + 1)]

            if abs(sum(weights) - 1) > 0.05:
                raise ValueError(f'Weights must sum to 1 (current: {sum(weights):.4f}).')

            port_return = sum(weights[i] * returns[i] for i in range(n))
            # Diagonal variance (zero cross-correlations simplification)
            port_var    = sum((weights[i] * vols[i])**2 for i in range(n))
            port_vol    = _m.sqrt(port_var) if port_var > 0 else 0

            rows = ['<table class="results-table"><tr><th>Asset</th><th>Weight</th>'
                    '<th>Expected Return (%)</th><th>Volatility (%)</th>'
                    '<th>Weighted Return (%)</th></tr>']
            for i in range(n):
                rows.append(f'<tr><td>Asset {i+1}</td><td>{weights[i]:.4f}</td>'
                             f'<td>{returns[i]:.2f}</td><td>{vols[i]:.2f}</td>'
                             f'<td>{weights[i]*returns[i]:.4f}</td></tr>')
            rows.append('</table>')
            rows.append(f'<p><strong>Portfolio Expected Return:</strong> {port_return:.4f}%</p>')
            rows.append(f'<p><strong>Portfolio Volatility (σ):</strong> {port_vol:.4f}%</p>')
            rows.append(f'<p><strong>Sharpe-proxy (Rp/σp):</strong> '
                        f'{port_return/port_vol:.4f}' if port_vol > 0 else '')
            result = ''.join(rows)
        except Exception as exc:
            error = str(exc)

    return render_template('portfolio_diversification.html', result=result, error=error,
                           form_data=form_data or {})


# ──────────────────────────────────────────────────────────────────────────────
# EXPECTED RETURN (weighted sum, CAPM cross-check)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/expected-return', methods=['GET', 'POST'])
@app.route('/expected_return', methods=['GET', 'POST'])
def expected_return():
    result    = None
    error     = None
    form_data = None

    if request.method == 'POST':
        form_data = request.form
        try:
            n = int(form_data.get('num_assets', 0))
            if n < 1 or n > 10:
                raise ValueError('Number of assets must be 1–10.')
            weights = [float(form_data.get(f'weight_{i}', 0)) for i in range(1, n + 1)]
            returns = [float(form_data.get(f'return_{i}', 0)) for i in range(1, n + 1)]

            if abs(sum(weights) - 1) > 0.05:
                raise ValueError(f'Weights must sum to 1 (current: {sum(weights):.4f}).')

            Ep = sum(weights[i] * returns[i] for i in range(n))

            rows = ['<table class="results-table"><tr><th>Asset</th><th>Weight</th>'
                    '<th>Expected Return</th><th>Contribution</th></tr>']
            for i in range(n):
                rows.append(f'<tr><td>Asset {i+1}</td><td>{weights[i]:.4f}</td>'
                             f'<td>{returns[i]:.4f}</td>'
                             f'<td>{weights[i]*returns[i]:.6f}</td></tr>')
            rows.append('</table>')
            rows.append(f'<p><strong>Portfolio Expected Return:</strong> {Ep:.4f} '
                        f'({Ep*100:.2f}%)</p>')
            result = ''.join(rows)
        except Exception as exc:
            error = str(exc)

    return render_template('expected_return.html', result=result, error=error,
                           form_data=form_data or {})


# ──────────────────────────────────────────────────────────────────────────────
# BOND DURATION (Macaulay, Modified, Effective — CFA/ICMA standard)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/duration', methods=['GET', 'POST'])
def duration():
    import math as _m
    result = None
    error  = None

    if request.method == 'POST':
        try:
            n     = int(request.form.get('num_periods', 1))
            cfs   = [float(request.form.get(f'cf_{i}', 0)) for i in range(1, n + 1)]
            ytm   = float(request.form.get('yield', 0)) / 100      # annual
            comp  = int(request.form.get('compounding', 1))
            p0    = float(request.form.get('initial_price', 0))
            p_dn  = float(request.form.get('price_drop', 0))        # yield −1%
            p_up  = float(request.form.get('price_rise', 0))        # yield +1%

            if any(cf < 0 for cf in cfs):
                raise ValueError('Cash flows must be non-negative.')
            if ytm < 0:
                raise ValueError('YTM must be non-negative.')
            if comp < 1:
                raise ValueError('Compounding periods must be ≥ 1.')
            if p0 <= 0:
                raise ValueError('Initial bond price must be positive.')

            r_per   = ytm / comp   # periodic yield
            pv_cfs  = [cfs[t] / (1 + r_per)**(t + 1) for t in range(n)]
            total_pv = sum(pv_cfs)
            if total_pv <= 0:
                raise ValueError('Sum of discounted cash flows is zero — check inputs.')

            # Macaulay Duration (in years)
            d_mac = sum((t + 1) * pv_cfs[t] for t in range(n)) / (total_pv * comp)
            # Modified Duration: D_mac / (1 + y/m)
            d_mod = d_mac / (1 + ytm / comp)
            # Effective Duration: (P↓ − P↑) / (2 × P₀ × Δy) where Δy = 0.01
            dy = 0.01
            d_eff = (p_dn - p_up) / (2 * p0 * dy) if p0 > 0 else 0

            result = dict(
                macaulay_duration=round(d_mac, 4),
                modified_duration=round(d_mod, 4),
                effective_duration=round(d_eff, 4),
            )
        except Exception as exc:
            error = str(exc)

    return render_template('duration.html', result=result, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# BONDS CALCULATOR (simple yield & maturity — Ghana fixed-income standard)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/bonds', methods=['GET', 'POST'])
def bonds():
    result = None
    error  = None

    if request.method == 'POST':
        try:
            principal    = float(request.form.get('principal', 0))
            tenor        = float(request.form.get('tenor', 0))
            rate         = float(request.form.get('rate', 0)) / 100      # annual coupon %
            total_coupons = float(request.form.get('total_coupons', 0))

            if principal <= 0:
                raise ValueError('Principal must be positive.')
            if tenor <= 0:
                raise ValueError('Tenor (days) must be positive.')
            if rate < 0:
                raise ValueError('Coupon rate cannot be negative.')
            if total_coupons < 0:
                raise ValueError('Total coupons cannot be negative.')

            # Maturity amount = face value + total coupon income
            maturity_amount = principal + total_coupons
            # Flat/simple yield annualised: (total_coupons / principal) × (365 / tenor)
            bond_yield = (total_coupons / principal) * (365 / tenor) * 100 if principal > 0 else 0
            # Current yield (using annual coupon rate provided):
            annual_coupon  = principal * rate
            current_yield  = (annual_coupon / principal) * 100 if principal > 0 else rate * 100
            # Price if par = 100: capital gain/loss = 0 since we assume par pricing
            price_per_100  = 100.0   # par-priced bond

            result = dict(
                maturity_amount=round(maturity_amount, 2),
                bond_yield=round(bond_yield, 4),
                annual_coupon=round(annual_coupon, 2),
                current_yield=round(current_yield, 4),
                principal=principal,
                tenor=int(tenor),
                rate=rate * 100,
            )
        except Exception as exc:
            error = str(exc)

    return render_template('bonds.html', result=result, error=error)


# Note: /cds is handled by cds_calculator() at line ~2369 (GET + POST, ISDA-standard).
# The duplicate GET-only stub has been removed to prevent Flask endpoint conflict.


# ──────────────────────────────────────────────────────────────────────────────
# DIVIDEND VALUATION MODEL (Gordon Growth, Multi-Stage, No-Growth)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/dvm', methods=['GET', 'POST'])
@app.route('/dvm-calculator', methods=['GET', 'POST'])
def dvm_calculator():
    results    = None
    model_type = 'gordon_growth'
    error      = None

    if request.method == 'POST':
        model_type = request.form.get('model_type', 'gordon_growth')
        try:
            r = float(request.form.get('r', 0)) / 100  # discount rate
            if r <= 0:
                raise ValueError('Discount rate must be positive.')

            if model_type == 'gordon_growth':
                d1 = float(request.form.get('d1', 0))
                g  = float(request.form.get('g', 0)) / 100
                if r <= g:
                    raise ValueError('Discount rate must exceed growth rate (r > g).')
                iv = d1 / (r - g)
                results = dict(intrinsic_value=round(iv, 4),
                               formula=f'P₀ = D₁/(r−g) = {d1}/({r:.4f}−{g:.4f}) = {iv:.4f}')

            elif model_type == 'multi_stage':
                periods  = int(request.form.get('periods', 1))
                divs     = [float(request.form.get(f'dividend_{i}', 0))
                            for i in range(1, periods + 1)]
                g_term   = float(request.form.get('terminal_growth', 0)) / 100
                if r <= g_term:
                    raise ValueError('Discount rate must exceed terminal growth rate.')
                pv_divs  = [d / (1 + r)**t for t, d in enumerate(divs, 1)]
                last_div = divs[-1] if divs else 0
                tv       = last_div * (1 + g_term) / (r - g_term)
                pv_tv    = tv / (1 + r)**periods
                iv       = sum(pv_divs) + pv_tv
                results  = dict(intrinsic_value=round(iv, 4),
                                pv_dividends=[round(p, 4) for p in pv_divs],
                                terminal_value=round(tv, 4),
                                pv_terminal=round(pv_tv, 4))

            elif model_type == 'no_growth':
                d = float(request.form.get('d', 0))
                iv = d / r
                results = dict(intrinsic_value=round(iv, 4),
                               formula=f'P₀ = D/r = {d}/{r:.4f} = {iv:.4f}')
            else:
                raise ValueError(f'Unknown model: {model_type}')
        except Exception as exc:
            error = str(exc)

    return render_template('dvm.html', results=results, model_type=model_type, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# INTRINSIC VALUE — Full DCF (CAPM / WACC / Manual, 1-stage or 2-stage)
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/intrinsic-value', methods=['GET', 'POST'])
@app.route('/intrinsic_value', methods=['GET', 'POST'])
def intrinsic_value():
    result = None
    debug  = None
    error  = None
    form   = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        try:
            import math as _m
            num_y    = int(form.get('num_fcf_years', 3))
            fcf_base = [float(form.get(f'fcf_{i}', 0)) for i in range(1, num_y + 1)]
            growth_m = form.get('growth_model', 'single_stage')
            term_m   = form.get('terminal_method', 'gordon_growth')
            disc_m   = form.get('discount_rate_method', 'capm')

            # ── Discount rate ──
            rf   = float(form.get('risk_free_rate', 5)) / 100
            rm   = float(form.get('market_return', 10)) / 100
            beta = float(form.get('beta', 1.0))
            ke   = rf + beta * (rm - rf)  # CAPM cost of equity

            if disc_m == 'capm':
                r = ke
            elif disc_m == 'wacc':
                we = float(form.get('equity_weight', 50)) / 100
                wd = float(form.get('debt_weight', 50)) / 100
                kd = float(form.get('cost_of_debt', 5)) / 100
                tc = float(form.get('tax_rate', 30)) / 100
                r  = we * ke + wd * kd * (1 - tc)
            else:
                r = float(form.get('manual_discount_rate', 10)) / 100

            if r <= 0:
                raise ValueError('Discount rate must be positive.')

            # ── Projected FCFs ──
            if growth_m == 'single_stage':
                proj_fcfs = fcf_base[:]
            else:
                hg   = float(form.get('high_growth_rate', 10)) / 100
                hy   = int(form.get('high_growth_years', 5))
                base = fcf_base[-1] if fcf_base else 0
                proj_fcfs = [base * (1 + hg)**t for t in range(1, hy + 1)]

            n         = len(proj_fcfs)
            pv_fcfs   = [cf / (1 + r)**t for t, cf in enumerate(proj_fcfs, 1)]
            ev        = sum(pv_fcfs)
            last_fcf  = proj_fcfs[-1] if proj_fcfs else 0

            # ── Terminal value ──
            if term_m == 'gordon_growth':
                g_t = float(form.get('perpetual_growth_rate', 2)) / 100
                if r <= g_t:
                    raise ValueError('Discount rate must exceed perpetual growth rate.')
                tv   = last_fcf * (1 + g_t) / (r - g_t)
                tv_g = g_t; tv_x = None
            else:
                ex   = float(form.get('exit_multiple', 8))
                tv   = last_fcf * ex
                tv_g = None; tv_x = ex

            pv_tv = tv / (1 + r)**n
            ev   += pv_tv

            debt  = float(form.get('total_debt', 0) or 0)
            cash_ = float(form.get('cash_and_equivalents', 0) or 0)
            shares = float(form.get('outstanding_shares', 1))
            if shares <= 0:
                raise ValueError('Shares outstanding must be positive.')

            equity_val = ev - debt + cash_
            result     = equity_val / shares

            # ── Sensitivity grid (3×3) ──
            if term_m == 'gordon_growth':
                g_vals   = [round(g_t - 0.01, 4), round(g_t, 4), round(g_t + 0.01, 4)]
                g_labels = [round(g * 100, 1) for g in g_vals]
            else:
                g_vals   = [ex - 1, ex, ex + 1]
                g_labels = g_vals
            r_vals   = [round(r - 0.01, 4), round(r, 4), round(r + 0.01, 4)]
            r_labels = [round(rv * 100, 1) for rv in r_vals]
            sens     = []
            for gv in g_vals:
                row = []
                for rv in r_vals:
                    try:
                        if term_m == 'gordon_growth':
                            if rv <= gv:
                                row.append('N/A'); continue
                            tv_s = last_fcf * (1 + gv) / (rv - gv)
                        else:
                            tv_s = last_fcf * gv
                        ev_s = sum(cf / (1 + rv)**t for t, cf in enumerate(proj_fcfs, 1))
                        ev_s += tv_s / (1 + rv)**n
                        iv_s  = (ev_s - debt + cash_) / shares
                        row.append(f'GHS {iv_s:,.2f}')
                    except Exception:
                        row.append('N/A')
                sens.append(row)

            debug = dict(enterprise_value=round(ev, 2), equity_value=round(equity_val, 2),
                         discount_rate=r, terminal_method=term_m,
                         terminal_growth_rate=tv_g, exit_multiple=tv_x,
                         sensitivity=dict(g_rates=g_labels, r_rates=r_labels, values=sens))
        except Exception as exc:
            error = str(exc)

    return render_template('intrinsic_value.html', form=form, result=result,
                           debug=debug, error=error)


# ──────────────────────────────────────────────────────────────────────────────
# STATIC INFO PAGES
# ──────────────────────────────────────────────────────────────────────────────
@app.route('/valuation-methods')
@app.route('/valuation_methods')
def valuation_methods():
    return render_template('valuation_methods.html')


@app.route('/multi-method-valuation')
@app.route('/multi_method_valuation')
def multi_method_valuation():
    return render_template('multi_method_valuation.html')


# ──────────────────────────────────────────────────────────────────────────────
# SPECIALIZED INDUSTRY MULTIPLES
# ──────────────────────────────────────────────────────────────────────────────
_SPEC_INPUT_FIELDS = {
    'net_debt':       ['total_debt', 'cash'],
    'net_debt_ebitda':['total_debt', 'cash', 'ebitda'],
    'revenue_growth': ['current_revenue', 'prior_revenue'],
    'eps_growth':     ['current_eps', 'prior_eps'],
    'ltm_ebitda':     [],
    'ntm_ebitda':     [],
    'unlevered_pe':   ['ev', 'ebiat'],
    'tev_ebitdar':    ['ev', 'ebitda', 'rent_expense'],
    'ev_subscribers': ['ev', 'subscribers'],
    'ev_boe':         ['ev', 'boe'],
    'ev_square_foot': ['ev', 'square_footage'],
    'ev_mau':         ['ev', 'mau'],
    'p_ffo':          ['share_price', 'ffo_per_share'],
    'p_tbv':          ['share_price', 'tangible_bvps'],
}
_SPEC_FIELD_LABELS = {
    'total_debt':      {'label': 'Total Debt', 'placeholder': 'e.g., 500000000'},
    'cash':            {'label': 'Cash & Equivalents', 'placeholder': 'e.g., 200000000'},
    'ebitda':          {'label': 'EBITDA', 'placeholder': 'e.g., 200000000'},
    'current_revenue': {'label': 'Current Revenue', 'placeholder': 'e.g., 1000000000'},
    'prior_revenue':   {'label': 'Prior Revenue', 'placeholder': 'e.g., 900000000'},
    'current_eps':     {'label': 'Current EPS', 'placeholder': 'e.g., 1.50'},
    'prior_eps':       {'label': 'Prior EPS', 'placeholder': 'e.g., 1.20'},
    'ev':              {'label': 'Enterprise Value', 'placeholder': 'e.g., 2000000000'},
    'ebiat':           {'label': 'EBIAT', 'placeholder': 'e.g., 150000000'},
    'rent_expense':    {'label': 'Rent Expense', 'placeholder': 'e.g., 50000000'},
    'subscribers':     {'label': 'Subscribers', 'placeholder': 'e.g., 500000'},
    'boe':             {'label': 'BOE (barrels)', 'placeholder': 'e.g., 10000000'},
    'square_footage':  {'label': 'Square Footage', 'placeholder': 'e.g., 100000'},
    'mau':             {'label': 'Monthly Active Users', 'placeholder': 'e.g., 5000000'},
    'share_price':     {'label': 'Share Price', 'placeholder': 'e.g., 10.50'},
    'ffo_per_share':   {'label': 'FFO per Share', 'placeholder': 'e.g., 1.20'},
    'tangible_bvps':   {'label': 'Tangible BV per Share', 'placeholder': 'e.g., 8.00'},
}
_SPEC_TITLES = {
    'net_debt': 'Net Debt', 'net_debt_ebitda': 'Net Debt/EBITDA',
    'revenue_growth': 'Revenue Growth', 'eps_growth': 'EPS Growth',
    'ltm_ebitda': 'LTM EBITDA', 'ntm_ebitda': 'NTM EBITDA',
    'unlevered_pe': 'Unlevered P/E', 'tev_ebitdar': 'TEV/EBITDAR',
    'ev_subscribers': 'EV/Subscribers', 'ev_boe': 'EV/BOE',
    'ev_square_foot': 'EV/Sq Ft', 'ev_mau': 'EV/MAU',
    'p_ffo': 'P/FFO', 'p_tbv': 'P/TBV',
}


def _calc_spec(formula, data):
    """Calculate one period of a specialized industry multiple."""
    if formula == 'net_debt':
        return data['total_debt'] - data['cash']
    elif formula == 'net_debt_ebitda':
        return (data['total_debt'] - data['cash']) / data['ebitda']
    elif formula == 'revenue_growth':
        return (data['current_revenue'] - data['prior_revenue']) / data['prior_revenue'] * 100
    elif formula == 'eps_growth':
        return (data['current_eps'] - data['prior_eps']) / data['prior_eps'] * 100
    elif formula == 'unlevered_pe':
        return data['ev'] / data['ebiat']
    elif formula == 'tev_ebitdar':
        return data['ev'] / (data['ebitda'] + data['rent_expense'])
    elif formula == 'ev_subscribers':
        return data['ev'] / data['subscribers']
    elif formula == 'ev_boe':
        return data['ev'] / data['boe']
    elif formula == 'ev_square_foot':
        return data['ev'] / data['square_footage']
    elif formula == 'ev_mau':
        return data['ev'] / data['mau']
    elif formula == 'p_ffo':
        return data['share_price'] / data['ffo_per_share']
    elif formula == 'p_tbv':
        return data['share_price'] / data['tangible_bvps']
    raise ValueError(f'Unknown formula: {formula}')


@app.route('/specialized-industry-multiples')
@app.route('/Specialized_Industry_Multiples')
def specialized_industry_multiples():
    return render_template('Specialized_Industry_Multiples.html',
                           form_data={}, results=None, error=None,
                           currency_symbol='GHS ',
                           input_fields=_SPEC_INPUT_FIELDS,
                           field_labels=_SPEC_FIELD_LABELS,
                           formula_titles=_SPEC_TITLES)


@app.route('/specialized', methods=['GET', 'POST'])
def specialized_calc():
    form_data = request.form if request.method == 'POST' else {}
    results   = None
    error     = None

    if request.method == 'POST':
        try:
            formula    = form_data.get('formula', 'net_debt')
            currency   = form_data.get('currency', 'GHS')
            cur_sym    = {'GHS': 'GHS ', 'USD': '$', 'EUR': '€', 'GBP': '£'}.get(currency, 'GHS ')
            num_p      = int(form_data.get('num_periods', 1))

            if formula == 'ltm_ebitda':
                qs = [float(form_data.get(f'ebitda_q{i}', 0)) for i in range(1, 5)]
                ltm = sum(qs)
                results = [{'period': 'LTM', 'result': ltm}]
            elif formula == 'ntm_ebitda':
                cfy  = float(form_data.get('current_fy_ebitda', 0))
                nfy  = float(form_data.get('next_fy_ebitda', 0))
                mr   = float(form_data.get('months_remaining', 6))
                mp   = float(form_data.get('months_passed', 6))
                ntm  = cfy * (mr / 12) + nfy * (mp / 12)
                results = [{'period': 'NTM', 'result': ntm}]
            else:
                fields  = _SPEC_INPUT_FIELDS.get(formula, [])
                results = []
                for i in range(1, num_p + 1):
                    data = {f: float(form_data.get(f'{f}_{i}', form_data.get(f, 0)))
                            for f in fields}
                    res  = _calc_spec(formula, data)
                    row  = {'period': i, 'result': round(res, 4)}
                    row.update({f: data[f] for f in fields})
                    results.append(row)
        except Exception as exc:
            error = str(exc)
            results = None

    return render_template('Specialized_Industry_Multiples.html',
                           form_data=form_data, results=results, error=error,
                           currency_symbol=form_data.get('currency_symbol', 'GHS '),
                           input_fields=_SPEC_INPUT_FIELDS,
                           field_labels=_SPEC_FIELD_LABELS,
                           formula_titles=_SPEC_TITLES)


# ──────────────────────────────────────────────────────────────────────────────
# VALUATION PERFORMANCE MULTIPLES
# ──────────────────────────────────────────────────────────────────────────────
_VP_INPUT_FIELDS = {
    'ev':           ['market_cap', 'total_debt', 'preferred_stock', 'minority_interest', 'cash', 'non_operating_assets'],
    'ev_ebitda':    ['market_cap', 'total_debt', 'preferred_stock', 'minority_interest', 'cash', 'non_operating_assets', 'ebitda'],
    'ev_ebit':      ['market_cap', 'total_debt', 'preferred_stock', 'minority_interest', 'cash', 'non_operating_assets', 'ebit'],
    'ev_sales':     ['market_cap', 'total_debt', 'preferred_stock', 'minority_interest', 'cash', 'non_operating_assets', 'revenue'],
    'pe':           ['share_price', 'eps'],
    'pb':           ['share_price', 'bvps'],
    'peg':          ['share_price', 'eps', 'eps_growth'],
    'ebitda_margin':['ebitda', 'revenue'],
    'ebit_margin':  ['ebit', 'revenue'],
    'net_margin':   ['net_income', 'revenue'],
    'roe':          ['net_income', 'equity'],
    'roa':          ['net_income', 'total_assets'],
    'roic':         ['ebit', 'tax_rate', 'total_debt', 'equity', 'cash', 'non_operating_assets'],
    'roa_fin':      ['net_income', 'avg_total_assets'],
    'roe_fin':      ['net_income', 'avg_equity'],
}
_VP_TITLES = {
    'ev': 'Enterprise Value', 'ev_ebitda': 'EV/EBITDA', 'ev_ebit': 'EV/EBIT',
    'ev_sales': 'EV/Sales', 'pe': 'P/E', 'pb': 'P/B', 'peg': 'PEG',
    'ebitda_margin': 'EBITDA Margin', 'ebit_margin': 'EBIT Margin',
    'net_margin': 'Net Margin', 'roe': 'ROE', 'roa': 'ROA', 'roic': 'ROIC',
    'roa_fin': 'ROA (Fin)', 'roe_fin': 'ROE (Fin)',
}
_VP_FIELD_LABELS = {
    'market_cap':          {'label': 'Market Cap (GHS)',             'placeholder': 'e.g., 1000000000'},
    'total_debt':          {'label': 'Total Debt (GHS)',             'placeholder': 'e.g., 500000000'},
    'preferred_stock':     {'label': 'Preferred Stock (GHS)',        'placeholder': 'e.g., 0'},
    'minority_interest':   {'label': 'Minority Interest (GHS)',      'placeholder': 'e.g., 0'},
    'cash':                {'label': 'Cash & Equivalents (GHS)',     'placeholder': 'e.g., 200000000'},
    'non_operating_assets':{'label': 'Non-Op Assets (GHS)',          'placeholder': 'e.g., 0'},
    'ebitda':              {'label': 'EBITDA (GHS)',                 'placeholder': 'e.g., 200000000'},
    'ebit':                {'label': 'EBIT (GHS)',                   'placeholder': 'e.g., 150000000'},
    'revenue':             {'label': 'Revenue (GHS)',                'placeholder': 'e.g., 500000000'},
    'net_income':          {'label': 'Net Income (GHS)',             'placeholder': 'e.g., 100000000'},
    'equity':              {'label': "Shareholders' Equity (GHS)",   'placeholder': 'e.g., 800000000'},
    'total_assets':        {'label': 'Total Assets (GHS)',           'placeholder': 'e.g., 2000000000'},
    'avg_total_assets':    {'label': 'Avg Total Assets (GHS)',       'placeholder': 'e.g., 1900000000'},
    'avg_equity':          {'label': 'Average Equity (GHS)',         'placeholder': 'e.g., 790000000'},
    'share_price':         {'label': 'Share Price (GHS)',            'placeholder': 'e.g., 10.50'},
    'eps':                 {'label': 'EPS (GHS)',                    'placeholder': 'e.g., 1.25'},
    'bvps':                {'label': 'Book Value per Share (GHS)',   'placeholder': 'e.g., 5.00'},
    'eps_growth':          {'label': 'EPS Growth Rate (%)',          'placeholder': 'e.g., 20'},
    'tax_rate':            {'label': 'Tax Rate (%)',                 'placeholder': 'e.g., 25'},
}


def _calc_vp(formula, data):
    """Calculate one year of a valuation/performance multiple."""
    ev_val = (data.get('market_cap', 0) + data.get('total_debt', 0) +
              data.get('preferred_stock', 0) + data.get('minority_interest', 0) -
              data.get('cash', 0) - data.get('non_operating_assets', 0))
    if formula == 'ev':
        return ev_val
    elif formula == 'ev_ebitda':
        return ev_val / data['ebitda']
    elif formula == 'ev_ebit':
        return ev_val / data['ebit']
    elif formula == 'ev_sales':
        return ev_val / data['revenue']
    elif formula == 'pe':
        return data['share_price'] / data['eps']
    elif formula == 'pb':
        return data['share_price'] / data['bvps']
    elif formula == 'peg':
        pe = data['share_price'] / data['eps']
        return pe / data['eps_growth']
    elif formula == 'ebitda_margin':
        return data['ebitda'] / data['revenue'] * 100
    elif formula == 'ebit_margin':
        return data['ebit'] / data['revenue'] * 100
    elif formula == 'net_margin':
        return data['net_income'] / data['revenue'] * 100
    elif formula == 'roe':
        return data['net_income'] / data['equity'] * 100
    elif formula == 'roa':
        return data['net_income'] / data['total_assets'] * 100
    elif formula == 'roic':
        nopat = data['ebit'] * (1 - data.get('tax_rate', 25) / 100)
        ic    = (data['total_debt'] + data['equity'] - data.get('cash', 0) -
                 data.get('non_operating_assets', 0))
        return nopat / ic * 100 if ic else 0
    elif formula == 'roa_fin':
        return data['net_income'] / data['avg_total_assets'] * 100
    elif formula == 'roe_fin':
        return data['net_income'] / data['avg_equity'] * 100
    raise ValueError(f'Unknown formula: {formula}')


@app.route('/valuation-performance-multiples')
@app.route('/Valuation_Performance_Multiples')
def valuation_performance_multiples():
    return render_template('Valuation_Performance_Multiples.html',
                           results=None, error=None, average_result=0,
                           unit='x', selected_formula='ev',
                           formulaTitles=_VP_TITLES,
                           inputFields=_VP_INPUT_FIELDS,
                           fieldLabels=_VP_FIELD_LABELS)


@app.route('/valuation-performance', methods=['GET', 'POST'])
def valuation_performance_calc():
    form_data       = request.form if request.method == 'POST' else {}
    results         = None
    error           = None
    average_result  = 0
    selected_formula = form_data.get('formula', 'ev')
    # Determine unit
    pct_formulas = {'ebitda_margin', 'ebit_margin', 'net_margin', 'roe', 'roa', 'roic',
                    'roa_fin', 'roe_fin', 'peg'}
    ghs_formulas = {'ev'}
    unit = ('%' if selected_formula in pct_formulas else
            'GHS' if selected_formula in ghs_formulas else 'x')

    if request.method == 'POST':
        try:
            fields   = _VP_INPUT_FIELDS.get(selected_formula, [])
            num_yrs  = int(form_data.get('num_years', 1))
            results  = []
            for yr in range(1, num_yrs + 1):
                data = {f: float(form_data.get(f'{f}_{yr}', form_data.get(f, 0)))
                        for f in fields}
                res  = _calc_vp(selected_formula, data)
                row  = {'year': yr, 'result': round(res, 4)}
                row.update({f: data[f] for f in fields})
                results.append(row)
            average_result = sum(r['result'] for r in results) / len(results) if results else 0
        except Exception as exc:
            error = str(exc)
            results = None

    return render_template('Valuation_Performance_Multiples.html',
                           results=results, error=error,
                           average_result=round(average_result, 4),
                           unit=unit, selected_formula=selected_formula,
                           formulaTitles=_VP_TITLES,
                           inputFields=_VP_INPUT_FIELDS,
                           fieldLabels=_VP_FIELD_LABELS)

# ================================================================
# --- PUBLIC ARTICLES & VIDEOS ---
# ================================================================

@app.route('/articles')
def articles():
    """Public articles listing."""
    category = request.args.get('category', '')
    q = Article.query.filter_by(is_published=True)
    if category:
        q = q.filter_by(category=category)
    posts = q.order_by(Article.created_at.desc()).all()
    categories = db.session.query(Article.category).filter_by(is_published=True).distinct().all()
    categories = [c[0] for c in categories]
    featured_video = Video.query.filter_by(is_featured=True, is_published=True).first()
    return render_template('articles.html', posts=posts, categories=categories,
                           selected_category=category, featured_video=featured_video)


@app.route('/articles/<int:article_id>')
def article_detail(article_id):
    """Public single-article view."""
    post = Article.query.filter_by(id=article_id, is_published=True).first_or_404()
    return render_template('article_detail.html', post=post)


@app.route('/videos')
def videos():
    """Public videos listing."""
    vids = Video.query.filter_by(is_published=True).order_by(Video.created_at.desc()).all()
    return render_template('videos.html', videos=vids)


# ================================================================
# --- ADMIN SECTION ---
# ================================================================
ADMIN_PASSWORD       = os.environ['ADMIN_PASSWORD']
SUPER_ADMIN_PASSWORD = os.environ['SUPER_ADMIN_PASSWORD']


def admin_required(f):
    """Decorator: redirect to /admin/login if not authenticated."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


def content_admin_required(f):
    """Decorator: allows full admin OR content admin (articles/videos only)."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in') and not session.get('content_admin_logged_in'):
            return redirect(url_for('content_admin_login'))
        return f(*args, **kwargs)
    return decorated


def _is_full_admin():
    return bool(session.get('admin_logged_in'))


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        if not email:
            # Master password login
            if password == ADMIN_PASSWORD or password == SUPER_ADMIN_PASSWORD:
                session['admin_logged_in']       = True
                session['super_admin_logged_in'] = True
                session.permanent = True
                return redirect(url_for('admin_dashboard'))
            error = 'Incorrect password. Please try again.'
        else:
            # Content admin (email + password)
            ca = ContentAdmin.query.filter_by(email=email, is_active=True).first()
            if ca and ca.check_password(password):
                session['content_admin_logged_in'] = True
                session['content_admin_id'] = ca.id
                session.permanent = True
                return redirect(url_for('admin_articles'))
            error = 'Invalid email or password.'
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    session.pop('super_admin_logged_in', None)
    session.pop('content_admin_logged_in', None)
    session.pop('content_admin_id', None)
    return redirect(url_for('admin_login'))


# ---- LEGACY REDIRECTS (old content-admin and super-admin URLs) ----

@app.route('/content-admin/login', methods=['GET', 'POST'])
def content_admin_login():
    return redirect(url_for('admin_login'))


@app.route('/content-admin/logout')
def content_admin_logout():
    return redirect(url_for('admin_logout'))


@app.route('/admin/content-admins', methods=['GET', 'POST'])
@admin_required
def admin_content_admins():
    error = None
    success = None
    if request.method == 'POST':
        action = request.form.get('action', 'create')
        if action == 'create':
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            if not name or not email or not password:
                error = 'Name, email, and password are all required.'
            elif ContentAdmin.query.filter_by(email=email).first():
                error = 'An account with that email already exists.'
            else:
                ca = ContentAdmin(name=name, email=email)
                ca.set_password(password)
                db.session.add(ca)
                db.session.commit()
                success = f'Content admin "{name}" created successfully.'
    admins = ContentAdmin.query.order_by(ContentAdmin.created_at.desc()).all()
    return render_template('admin_content_admins.html', admins=admins, error=error, success=success)


@app.route('/admin/content-admins/<int:ca_id>/toggle', methods=['POST'])
@admin_required
def admin_content_admin_toggle(ca_id):
    ca = ContentAdmin.query.get_or_404(ca_id)
    ca.is_active = not ca.is_active
    db.session.commit()
    return redirect(url_for('admin_content_admins'))


@app.route('/admin/content-admins/<int:ca_id>/delete', methods=['POST'])
@admin_required
def admin_content_admin_delete(ca_id):
    ca = ContentAdmin.query.get_or_404(ca_id)
    db.session.delete(ca)
    db.session.commit()
    return redirect(url_for('admin_content_admins'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    article_count = Article.query.count()
    video_count   = Video.query.count()
    published     = Article.query.filter_by(is_published=True).count()
    recent        = Article.query.order_by(Article.created_at.desc()).limit(5).all()
    job_count     = JobListing.query.filter_by(is_active=True).count()
    app_count     = JobApplication.query.count()

    # Super-admin data exports section
    is_super_admin = bool(session.get('super_admin_logged_in') or session.get('admin_logged_in'))
    training_count  = TrainingBooking.query.count()          if is_super_admin else 0
    employer_count  = EmployerAccount.query.count()          if is_super_admin else 0
    candidate_count = CandidateProfile.query.count()         if is_super_admin else 0
    yin_reg_count   = YINRegistration.query.count()          if is_super_admin else 0
    referral_count  = db.session.query(Referral.referrer_email).distinct().count() if is_super_admin else 0
    successful_refs = Referral.query.filter_by(status='Successful').count()        if is_super_admin else 0

    linked_count = 0
    total_signups = 0
    if is_super_admin:
        linked_ids = set(
            [r[0] for r in db.session.query(EmployerShortlist.candidate_id).distinct().all()] +
            [r[0] for r in db.session.query(EmployerInquiry.candidate_id).distinct().all()]
        )
        linked_count = len(linked_ids)
        all_emails = set()
        for u in SiteUser.query.with_entities(SiteUser.email).all():               all_emails.add(u[0].strip().lower())
        for e in EmployerAccount.query.with_entities(EmployerAccount.email).all(): all_emails.add(e[0].strip().lower())
        for c in CandidateProfile.query.with_entities(CandidateProfile.email).all(): all_emails.add(c[0].strip().lower())
        for b in TrainingBooking.query.with_entities(TrainingBooking.email).all(): all_emails.add(b[0].strip().lower())
        for a in JobApplication.query.with_entities(JobApplication.email).all():   all_emails.add(a[0].strip().lower())
        total_signups = len(all_emails)

    return render_template('admin_dashboard.html',
                           article_count=article_count, video_count=video_count,
                           published=published, recent=recent,
                           job_count=job_count, app_count=app_count,
                           is_super_admin=is_super_admin,
                           training_count=training_count, employer_count=employer_count,
                           candidate_count=candidate_count, yin_reg_count=yin_reg_count,
                           referral_count=referral_count, successful_refs=successful_refs,
                           linked_count=linked_count, total_signups=total_signups)


# ---- ARTICLE CRUD ----

@app.route('/admin/articles')
@content_admin_required
def admin_articles():
    posts = Article.query.order_by(Article.created_at.desc()).all()
    return render_template('admin_articles.html', posts=posts, is_full_admin=_is_full_admin())


@app.route('/admin/articles/new', methods=['GET', 'POST'])
@content_admin_required
def admin_article_new():
    fa = _is_full_admin()
    if request.method == 'POST':
        title   = request.form.get('title', '').strip()
        summary = request.form.get('summary', '').strip()
        body    = request.form.get('body', '').strip()
        category = request.form.get('category', 'General').strip()
        thumbnail_url = request.form.get('thumbnail_url', '').strip()
        is_published  = request.form.get('is_published') == '1'

        if not title or not body:
            return render_template('admin_article_form.html', error='Title and body are required.',
                                   action='New', article=None, is_full_admin=fa)

        # Auto-generate slug from title
        import re
        slug_base = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        slug = slug_base
        counter = 1
        while Article.query.filter_by(slug=slug).first():
            slug = f'{slug_base}-{counter}'
            counter += 1

        article = Article(title=title, slug=slug, summary=summary, body=body,
                          category=category, thumbnail_url=thumbnail_url,
                          is_published=is_published)
        db.session.add(article)
        db.session.commit()
        logger.info(f'Admin created article: {title}')
        return redirect(url_for('admin_articles'))

    return render_template('admin_article_form.html', action='New', article=None, error=None, is_full_admin=fa)


@app.route('/admin/articles/<int:article_id>/edit', methods=['GET', 'POST'])
@content_admin_required
def admin_article_edit(article_id):
    fa = _is_full_admin()
    article = Article.query.get_or_404(article_id)
    if request.method == 'POST':
        article.title         = request.form.get('title', '').strip()
        article.summary       = request.form.get('summary', '').strip()
        article.body          = request.form.get('body', '').strip()
        article.category      = request.form.get('category', 'General').strip()
        article.thumbnail_url = request.form.get('thumbnail_url', '').strip()
        article.is_published  = request.form.get('is_published') == '1'
        article.updated_at    = datetime.utcnow()
        db.session.commit()
        logger.info(f'Admin updated article {article_id}')
        return redirect(url_for('admin_articles'))
    return render_template('admin_article_form.html', action='Edit', article=article, error=None, is_full_admin=fa)


@app.route('/admin/articles/<int:article_id>/delete', methods=['POST'])
@content_admin_required
def admin_article_delete(article_id):
    article = Article.query.get_or_404(article_id)
    db.session.delete(article)
    db.session.commit()
    logger.info(f'Admin deleted article {article_id}')
    return redirect(url_for('admin_articles'))


# ---- VIDEO CRUD ----

@app.route('/admin/videos')
@content_admin_required
def admin_videos():
    vids = Video.query.order_by(Video.created_at.desc()).all()
    return render_template('admin_videos.html', videos=vids, is_full_admin=_is_full_admin())


@app.route('/admin/videos/new', methods=['GET', 'POST'])
@content_admin_required
def admin_video_new():
    fa = _is_full_admin()
    if request.method == 'POST':
        title        = request.form.get('title', '').strip()
        youtube_url  = request.form.get('youtube_url', '').strip()
        description  = request.form.get('description', '').strip()
        is_featured  = request.form.get('is_featured') == '1'
        is_published = request.form.get('is_published') == '1'

        if not title or not youtube_url:
            return render_template('admin_video_form.html', action='New', video=None,
                                   error='Title and YouTube URL are required.', is_full_admin=fa)

        # If featuring this video, un-feature others
        if is_featured:
            Video.query.update({'is_featured': False}, synchronize_session=False)

        video = Video(title=title, youtube_url=youtube_url, description=description,
                      is_featured=is_featured, is_published=is_published)
        db.session.add(video)
        db.session.commit()
        logger.info(f'Admin added video: {title}')
        return redirect(url_for('admin_videos'))

    return render_template('admin_video_form.html', action='New', video=None, error=None, is_full_admin=fa)


@app.route('/admin/videos/<int:video_id>/edit', methods=['GET', 'POST'])
@content_admin_required
def admin_video_edit(video_id):
    fa = _is_full_admin()
    video = Video.query.get_or_404(video_id)
    if request.method == 'POST':
        video.title        = request.form.get('title', '').strip()
        video.youtube_url  = request.form.get('youtube_url', '').strip()
        video.description  = request.form.get('description', '').strip()
        video.is_featured  = request.form.get('is_featured') == '1'
        video.is_published = request.form.get('is_published') == '1'

        if video.is_featured:
            Video.query.filter(Video.id != video_id).update({'is_featured': False}, synchronize_session=False)

        db.session.commit()
        logger.info(f'Admin updated video {video_id}')
        return redirect(url_for('admin_videos'))
    return render_template('admin_video_form.html', action='Edit', video=video, error=None, is_full_admin=fa)


@app.route('/admin/videos/<int:video_id>/delete', methods=['POST'])
@content_admin_required
def admin_video_delete(video_id):
    video = Video.query.get_or_404(video_id)
    db.session.delete(video)
    db.session.commit()
    logger.info(f'Admin deleted video {video_id}')
    return redirect(url_for('admin_videos'))


# API: homepage data (latest 3 articles + featured video for AJAX)
@app.route('/api/homepage-content')
def api_homepage_content():
    posts = Article.query.filter_by(is_published=True).order_by(Article.created_at.desc()).limit(3).all()
    featured = Video.query.filter_by(is_featured=True, is_published=True).first()
    return jsonify({
        'articles': [{'id': p.id, 'title': p.title, 'summary': p.summary,
                      'category': p.category, 'thumbnail_url': p.thumbnail_url,
                      'created_at': p.created_at.strftime('%b %d, %Y')} for p in posts],
        'featured_video': {
            'id': featured.id, 'title': featured.title,
            'embed_url': featured.embed_url, 'description': featured.description,
            'thumbnail_url': featured.thumbnail_url
        } if featured else None
    })


# ================================================================
# --- THE REMAINING 15 WORLD-CLASS CALCULATORS ---
# ================================================================

import math

@app.route('/options', methods=['GET','POST'])
def options_calculator():
    """Black-Scholes Options Pricing — CFA/CBOE standard."""
    result = None; error = None
    if request.method == 'POST':
        try:
            S  = float(request.form['spot_price'])
            K  = float(request.form['strike_price'])
            T  = float(request.form['time_to_expiry_years'])
            r  = float(request.form['risk_free_rate']) / 100
            sigma = float(request.form['volatility']) / 100
            opt   = request.form.get('option_type', 'call')
            currency = request.form.get('currency', 'USD')

            from math import log, sqrt, exp
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / sqrt(2)))

            d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
            d2 = d1 - sigma*sqrt(T)
            Nd1, Nd2 = norm_cdf(d1), norm_cdf(d2)
            Nm_d1, Nm_d2 = norm_cdf(-d1), norm_cdf(-d2)
            pdf_d1 = exp(-0.5*d1**2) / sqrt(2*math.pi)

            if opt == 'call':
                price = S*Nd1 - K*exp(-r*T)*Nd2
                delta = Nd1
                rho   = K*T*exp(-r*T)*Nd2 / 100
            else:
                price = K*exp(-r*T)*Nm_d2 - S*Nm_d1
                delta = Nd1 - 1
                rho   = -K*T*exp(-r*T)*Nm_d2 / 100

            gamma = pdf_d1 / (S*sigma*sqrt(T))
            vega  = S*pdf_d1*sqrt(T) / 100
            theta = (-(S*pdf_d1*sigma)/(2*sqrt(T)) - r*K*exp(-r*T)*(Nd2 if opt=='call' else Nm_d2)) / 365

            intrinsic = max(S-K, 0) if opt=='call' else max(K-S, 0)
            time_value = price - intrinsic

            result = dict(
                spot_price=S, strike_price=K, T=T, r_pct=r*100, sigma_pct=sigma*100,
                option_type=opt.title(), currency=currency,
                price=round(price,4), delta=round(delta,4), gamma=round(gamma,6),
                vega=round(vega,4), theta=round(theta,4), rho=round(rho,4),
                d1=round(d1,4), d2=round(d2,4),
                intrinsic_value=round(intrinsic,4), time_value=round(time_value,4),
                itm='In the Money' if intrinsic>0 else ('At the Money' if intrinsic==0 else 'Out of the Money')
            )
        except Exception as e: error = str(e)
    return render_template('options.html', result=result, error=error)


@app.route('/monte-carlo', methods=['GET','POST'])
def monte_carlo():
    """Monte Carlo Portfolio Simulation — GBM model."""
    result = None; error = None
    if request.method == 'POST':
        try:
            import random
            S0    = float(request.form['initial_price'])
            mu    = float(request.form['expected_return']) / 100
            sigma = float(request.form['volatility']) / 100
            T     = float(request.form['time_horizon_years'])
            n_sim = min(int(request.form.get('n_simulations', 1000)), 5000)
            currency = request.form.get('currency', 'USD')
            steps = int(T * 252)

            finals = []
            for _ in range(n_sim):
                S = S0
                for _ in range(steps):
                    z = random.gauss(0, 1)
                    S *= math.exp((mu - 0.5*sigma**2)*(1/252) + sigma*math.sqrt(1/252)*z)
                finals.append(round(S, 4))

            finals_sorted = sorted(finals)
            mean_final = sum(finals) / n_sim
            var_95 = finals_sorted[int(0.05 * n_sim)]
            var_99 = finals_sorted[int(0.01 * n_sim)]
            cvar_95 = sum(finals_sorted[:int(0.05*n_sim)]) / max(int(0.05*n_sim), 1)
            prob_gain = sum(1 for x in finals if x > S0) / n_sim * 100

            # Percentile distribution for chart (10 buckets)
            bucket_size = len(finals_sorted) // 10
            distribution = [{'range': f'{i*10}-{(i+1)*10}%', 'value': round(finals_sorted[min(i*bucket_size, len(finals_sorted)-1)],2)} for i in range(10)]

            result = dict(
                initial_price=S0, mu_pct=mu*100, sigma_pct=sigma*100,
                T=T, n_sim=n_sim, currency=currency,
                mean_final=round(mean_final,2),
                median_final=round(finals_sorted[n_sim//2],2),
                min_final=round(min(finals),2), max_final=round(max(finals),2),
                var_95=round(var_95,2), var_99=round(var_99,2),
                cvar_95=round(cvar_95,2),
                prob_gain=round(prob_gain,1),
                expected_return_pct=round((mean_final/S0-1)*100,2),
                distribution=distribution,
            )
        except Exception as e: error = str(e)
    return render_template('monte_carlo.html', result=result, error=error)


@app.route('/real-estate', methods=['GET','POST'])
def real_estate():
    """Real Estate / Property Valuation — Cap Rate, GRM, NPV, DSCR."""
    result = None; error = None
    if request.method == 'POST':
        try:
            purchase_price   = float(request.form['purchase_price'])
            annual_rent      = float(request.form['annual_rental_income'])
            op_expenses      = float(request.form['operating_expenses'])
            vacancy_rate     = float(request.form.get('vacancy_rate', 5)) / 100
            cap_rate_market  = float(request.form.get('cap_rate_market', 6)) / 100
            mortgage_rate    = float(request.form.get('mortgage_rate', 7)) / 100
            ltv              = float(request.form.get('ltv', 70)) / 100
            term_years       = int(request.form.get('term_years', 25))
            currency         = request.form.get('currency', 'USD')

            eff_gross_income = annual_rent * (1 - vacancy_rate)
            noi = eff_gross_income - op_expenses
            cap_rate_actual = noi / purchase_price
            grm = purchase_price / annual_rent
            property_value_cap = noi / cap_rate_market if cap_rate_market > 0 else 0

            # Mortgage
            loan = purchase_price * ltv
            equity = purchase_price * (1 - ltv)
            n = term_years * 12
            r_monthly = mortgage_rate / 12
            if r_monthly > 0:
                monthly_payment = loan * r_monthly * (1+r_monthly)**n / ((1+r_monthly)**n - 1)
            else:
                monthly_payment = loan / n
            annual_debt_service = monthly_payment * 12
            dscr = noi / annual_debt_service if annual_debt_service > 0 else 0

            # Cash-on-cash return
            annual_cash_flow = noi - annual_debt_service
            coc_return = annual_cash_flow / equity * 100 if equity > 0 else 0

            # 5-year NPV (simple, assuming 3% appreciation)
            appreciation = 0.03
            wacc = mortgage_rate
            pv_total = sum(noi*(1+appreciation)**yr / (1+wacc)**yr for yr in range(1,6))
            terminal = purchase_price*(1+appreciation)**5
            npv = pv_total + terminal/(1+wacc)**5 - purchase_price

            result = dict(
                purchase_price=purchase_price, annual_rent=annual_rent, op_expenses=op_expenses,
                vacancy_rate_pct=vacancy_rate*100, currency=currency,
                eff_gross_income=round(eff_gross_income,2), noi=round(noi,2),
                cap_rate_actual_pct=round(cap_rate_actual*100,4),
                cap_rate_market_pct=cap_rate_market*100,
                property_value_cap=round(property_value_cap,2),
                grm=round(grm,2), loan=round(loan,2), equity=round(equity,2),
                monthly_payment=round(monthly_payment,2),
                annual_debt_service=round(annual_debt_service,2),
                dscr=round(dscr,2), annual_cash_flow=round(annual_cash_flow,2),
                coc_return=round(coc_return,2), npv_5yr=round(npv,2),
                mortgage_rate_pct=mortgage_rate*100, ltv_pct=ltv*100, term_years=term_years,
            )
        except Exception as e: error = str(e)
    return render_template('real_estate.html', result=result, error=error)


@app.route('/fx-forward', methods=['GET','POST'])
def fx_forward():
    """FX Forward & Swap Pricing — Interest Rate Parity (CFA/ACCA/ACI standard)."""
    result = None; error = None
    if request.method == 'POST':
        try:
            spot        = float(request.form['spot_rate'])
            r_d         = float(request.form['domestic_rate']) / 100
            r_f         = float(request.form['foreign_rate']) / 100
            days        = int(request.form['days'])
            notional    = float(request.form.get('notional', 1000000))
            base_ccy    = request.form.get('base_currency', 'USD')
            quote_ccy   = request.form.get('quote_currency', 'GBP')

            T = days / 360
            # Covered Interest Rate Parity
            forward = spot * (1 + r_d*T) / (1 + r_f*T)
            forward_points = (forward - spot) * 10000
            swap_cost_pct  = (forward - spot) / spot * 100 / T * 365

            # P&L of forward vs spot
            notional_base = notional
            notional_quote = notional * spot

            result = dict(
                spot=spot, r_d_pct=r_d*100, r_f_pct=r_f*100, days=days,
                notional=notional, base_ccy=base_ccy, quote_ccy=quote_ccy,
                forward=round(forward,6),
                forward_points=round(forward_points,2),
                swap_cost_annualised_pct=round(swap_cost_pct,4),
                premium_discount='Premium' if forward>spot else 'Discount',
                notional_base=notional_base,
                notional_quote=round(notional_quote,2),
                forward_value_base=round(notional,2),
                forward_value_quote=round(notional*forward,2),
                T=round(T,4),
            )
        except Exception as e: error = str(e)
    return render_template('fx_forward.html', result=result, error=error)


@app.route('/commodity-futures', methods=['GET','POST'])
def commodity_futures():
    """Commodity Futures Pricing — Cost-of-Carry model (Hull/CFA)."""
    result = None; error = None
    if request.method == 'POST':
        try:
            spot     = float(request.form['spot_price'])
            rf       = float(request.form['risk_free_rate']) / 100
            storage  = float(request.form['storage_cost_pct']) / 100
            cy       = float(request.form['convenience_yield_pct']) / 100
            T        = float(request.form['time_to_maturity_years'])
            currency = request.form.get('currency', 'USD')
            commodity= request.form.get('commodity', 'Commodity')

            # F = S × e^((r + u - y)T)  — continuous compounding (Hull)
            futures_continuous = spot * math.exp((rf + storage - cy) * T)
            # Simple compounding version
            futures_simple = spot * (1 + rf + storage - cy) ** T
            basis = spot - futures_simple
            annualised_basis_pct = basis / spot / T * 100 if T > 0 else 0

            result = dict(
                spot=spot, rf_pct=rf*100, storage_pct=storage*100, cy_pct=cy*100,
                T=T, currency=currency, commodity=commodity,
                futures_continuous=round(futures_continuous,4),
                futures_simple=round(futures_simple,4),
                basis=round(basis,4), basis_pct=round(basis/spot*100,4),
                annualised_basis_pct=round(annualised_basis_pct,4),
                net_carry=round((rf+storage-cy)*100,4),
                contango='Contango (F>S)' if futures_simple>spot else 'Backwardation (F<S)',
            )
        except Exception as e: error = str(e)
    return render_template('commodity_futures.html', result=result, error=error)


@app.route('/yield-curve', methods=['GET','POST'])
def yield_curve():
    """Yield Curve Builder — Bootstrap zero/spot rates from coupon bonds (CFA Level 1-2)."""
    result = None; error = None
    if request.method == 'POST':
        try:
            # Accept multiple bonds from form
            maturities   = request.form.getlist('maturity_years')
            coupon_rates = request.form.getlist('coupon_rate')
            prices       = request.form.getlist('price')
            face_values  = request.form.getlist('face_value')
            currency     = request.form.get('currency', 'USD')

            bonds = []
            for i in range(len(maturities)):
                try:
                    m = float(maturities[i]); c = float(coupon_rates[i])
                    p = float(prices[i]);     f = float(face_values[i]) if i < len(face_values) else 100
                    bonds.append({'maturity': m, 'coupon_rate': c, 'price': p, 'face': f})
                except: pass

            if not bonds:
                raise ValueError('Enter at least one bond')

            bonds.sort(key=lambda x: x['maturity'])

            # Bootstrap zero rates
            zero_rates = {}
            spot_rates = []
            for bond in bonds:
                m = bond['maturity']; c_rate = bond['coupon_rate']/100; p = bond['price']; f = bond['face']
                periods = int(m * 2)  # semi-annual
                c = c_rate * f / 2
                if periods <= 1:
                    z = (f + c) / p - 1
                    z = (1 + z) ** 2 - 1  # annualise semi-annual
                else:
                    pv_known = sum(c / (1 + zero_rates.get(t/2, 0.05))**t for t in range(1, periods))
                    last_cf = c + f
                    z_semi = ((last_cf) / (p - pv_known)) ** (1/periods) - 1
                    z = (1 + z_semi)**2 - 1

                zero_rates[m] = z
                # YTM (rough)
                ytm_approx = (c_rate*f + (f-p)/m) / ((f+p)/2)
                spot_rates.append({
                    'maturity': m, 'coupon_rate': c_rate*100,
                    'price': p, 'face': f,
                    'zero_rate': round(z*100, 4),
                    'ytm_approx': round(ytm_approx*100, 4),
                })

            result = dict(currency=currency, spot_rates=spot_rates, bond_count=len(bonds))
        except Exception as e: error = str(e)
    return render_template('yield_curve.html', result=result, error=error)


@app.route('/cds', methods=['GET','POST'])
def cds_calculator():
    """Credit Default Swap (CDS) Pricing — ISDA standard."""
    result = None; error = None
    if request.method == 'POST':
        try:
            notional     = float(request.form['notional'])
            spread_bps   = float(request.form['spread_bps'])
            recovery     = float(request.form.get('recovery_rate', 40)) / 100
            maturity     = float(request.form['maturity_years'])
            rf           = float(request.form.get('risk_free_rate', 5)) / 100
            currency     = request.form.get('currency', 'USD')

            spread = spread_bps / 10000
            lgd    = 1 - recovery

            # Simplified CDS pricing using constant hazard rate
            # Hazard rate: lambda = spread / LGD
            hazard = spread / lgd if lgd > 0 else 0

            # Survival probability at maturity
            surv_prob = math.exp(-hazard * maturity)

            # Premium leg PV (quarterly payments assumed)
            periods = int(maturity * 4)
            dt = 0.25
            premium_leg = sum(spread * notional * dt * math.exp(-hazard*i*dt) * math.exp(-rf*i*dt)
                              for i in range(1, periods+1))

            # Protection leg PV
            protection_leg = sum(lgd * notional * hazard * dt *
                                  math.exp(-hazard*i*dt) * math.exp(-rf*i*dt)
                                  for i in range(1, periods+1))

            fair_spread_bps = round(protection_leg / (premium_leg / spread / notional) / notional * 10000, 2) if premium_leg else 0
            mtm = protection_leg - premium_leg  # positive = protection buyer benefits
            annual_premium = spread * notional

            result = dict(
                notional=notional, spread_bps=spread_bps, recovery_pct=recovery*100,
                maturity=maturity, rf_pct=rf*100, currency=currency,
                lgd_pct=lgd*100, hazard_rate_pct=round(hazard*100,4),
                survival_prob_pct=round(surv_prob*100,2),
                premium_leg=round(premium_leg,2), protection_leg=round(protection_leg,2),
                fair_spread_bps=fair_spread_bps, mtm=round(mtm,2),
                annual_premium=round(annual_premium,2),
                breakeven='Protection Buyer Profitable' if mtm>0 else 'Protection Seller Profitable',
            )
        except Exception as e: error = str(e)
    return render_template('cds.html', result=result, error=error)


@app.route('/tips', methods=['GET','POST'])
def tips_calculator():
    """TIPS (Inflation-Linked Bond) Calculator — US Treasury / UK IL Gilt standard."""
    result = None; error = None
    if request.method == 'POST':
        try:
            face_value   = float(request.form['face_value'])
            real_coupon  = float(request.form['real_coupon_rate']) / 100
            inflation    = float(request.form['inflation_rate']) / 100
            ytm_real     = float(request.form['ytm_real']) / 100
            years        = int(request.form['years'])
            currency     = request.form.get('currency', 'USD')

            cash_flows = []
            pv_total = 0
            for yr in range(1, years+1):
                index_ratio = (1 + inflation) ** yr
                adj_face    = face_value * index_ratio
                adj_coupon  = adj_face * real_coupon
                cf = adj_coupon + (adj_face if yr == years else 0)
                pv = cf / (1 + ytm_real) ** yr
                pv_total += pv
                cash_flows.append({'year': yr, 'index_ratio': round(index_ratio,4),
                                   'adj_face': round(adj_face,2), 'coupon': round(adj_coupon,2),
                                   'total_cf': round(cf,2), 'pv': round(pv,2)})

            nominal_ytm_approx = (1+ytm_real)*(1+inflation)-1
            real_return = ytm_real * 100
            inflation_breakeven = inflation * 100

            result = dict(
                face_value=face_value, real_coupon_pct=real_coupon*100,
                inflation_pct=inflation*100, ytm_real_pct=ytm_real*100,
                years=years, currency=currency,
                price=round(pv_total,4),
                nominal_ytm_pct=round(nominal_ytm_approx*100,4),
                inflation_breakeven_pct=inflation_breakeven,
                real_return_pct=real_return,
                cash_flows=cash_flows,
                total_adj_face=round(face_value*(1+inflation)**years,2),
            )
        except Exception as e: error = str(e)
    return render_template('tips.html', result=result, error=error)


@app.route('/convertible-bond', methods=['GET','POST'])
def convertible_bond():
    """Convertible Bond Valuation — Straight Bond + Call Option (Black-Scholes)."""
    result = None; error = None
    if request.method == 'POST':
        try:
            from math import log, sqrt, exp, erf
            def norm_cdf(x): return 0.5*(1+erf(x/sqrt(2)))

            face_value = float(request.form['face_value'])
            coupon_rate= float(request.form['coupon_rate'])/100
            ytm        = float(request.form['ytm'])/100
            years      = int(request.form['years'])
            conv_ratio = float(request.form['conversion_ratio'])
            stock_price= float(request.form['stock_price'])
            sigma      = float(request.form['volatility'])/100
            rf         = float(request.form['risk_free_rate'])/100
            currency   = request.form.get('currency','USD')

            # Straight bond value
            c = coupon_rate * face_value
            r = ytm
            pv_coupons = c * (1-(1+r)**-years)/r if r>0 else c*years
            straight_value = pv_coupons + face_value/(1+r)**years

            # Conversion value
            conv_value = conv_ratio * stock_price
            conv_premium_pct = (face_value/conv_value - 1)*100 if conv_value>0 else 0

            # Call option on conversion (Black-Scholes, T=years, K=face_value/conv_ratio)
            K_opt = face_value / conv_ratio if conv_ratio>0 else face_value
            T = years
            if sigma*sqrt(T) > 0:
                d1 = (log(stock_price/K_opt)+(rf+0.5*sigma**2)*T)/(sigma*sqrt(T))
                d2 = d1 - sigma*sqrt(T)
                call_val = stock_price*norm_cdf(d1) - K_opt*exp(-rf*T)*norm_cdf(d2)
            else:
                call_val = max(stock_price-K_opt,0)
            option_value = call_val * conv_ratio

            total_value = straight_value + option_value
            delta = total_value / face_value  # conversion parity

            result = dict(
                face_value=face_value, coupon_rate_pct=coupon_rate*100, ytm_pct=ytm*100,
                years=years, conv_ratio=conv_ratio, stock_price=stock_price,
                sigma_pct=sigma*100, rf_pct=rf*100, currency=currency,
                straight_value=round(straight_value,4),
                conv_value=round(conv_value,4),
                option_value=round(option_value,4),
                total_value=round(total_value,4),
                conv_premium_pct=round(conv_premium_pct,4),
                delta=round(delta,4),
                investment_value_floor=round(straight_value,4),
            )
        except Exception as e: error = str(e)
    return render_template('convertible_bond.html', result=result, error=error)


@app.route('/esop', methods=['GET','POST'])
def esop_calculator():
    """ESOP / Employee Stock Option Calculator — Black-Scholes, dilution, vesting schedule."""
    result = None; error = None
    if request.method == 'POST':
        try:
            from math import log, sqrt, exp, erf
            def norm_cdf(x): return 0.5*(1+erf(x/sqrt(2)))

            options     = float(request.form['options_granted'])
            K           = float(request.form['strike_price'])
            S           = float(request.form['current_stock_price'])
            vesting_yrs = float(request.form.get('vesting_years',4))
            cliff_yrs   = float(request.form.get('cliff_years',1))
            T           = float(request.form['time_to_expiry'])
            sigma       = float(request.form['volatility'])/100
            rf          = float(request.form['risk_free_rate'])/100
            tax_rate    = float(request.form.get('tax_rate',25))/100
            shares_out  = float(request.form['shares_outstanding'])
            currency    = request.form.get('currency','USD')

            d1 = (log(S/K)+(rf+0.5*sigma**2)*T)/(sigma*sqrt(T)) if sigma*sqrt(T)>0 else 0
            d2 = d1 - sigma*sqrt(T)
            bs_value = S*norm_cdf(d1) - K*exp(-rf*T)*norm_cdf(d2)
            total_bs_value = bs_value * options

            intrinsic = max(S-K,0)*options
            dilution_pct = options/(shares_out+options)*100

            # After-tax proceeds at exercise
            spread = max(S-K,0)
            tax_on_exercise = spread * options * tax_rate
            net_proceeds = spread * options - tax_on_exercise

            # Vesting schedule
            vesting = []
            for yr in range(1, int(vesting_yrs)+1):
                if yr < cliff_yrs: vested = 0
                elif yr == int(cliff_yrs): vested = options * (cliff_yrs/vesting_yrs)
                else: vested = options / vesting_yrs
                vesting.append({'year': yr, 'vested_this_year': round(vested,0),
                                'cumulative_pct': round(min(yr/vesting_yrs,1)*100,1)})

            result = dict(
                options=options, strike=K, stock_price=S,
                vesting_years=vesting_yrs, cliff_years=cliff_yrs, T=T,
                sigma_pct=sigma*100, rf_pct=rf*100, tax_rate_pct=tax_rate*100,
                shares_out=shares_out, currency=currency,
                bs_per_option=round(bs_value,4), total_bs_value=round(total_bs_value,2),
                intrinsic_total=round(intrinsic,2), time_premium=round(total_bs_value-intrinsic,2),
                dilution_pct=round(dilution_pct,4),
                tax_on_exercise=round(tax_on_exercise,2),
                net_proceeds=round(net_proceeds,2),
                itm='In the Money' if S>K else ('At the Money' if S==K else 'Out of the Money'),
                vesting_schedule=vesting,
            )
        except Exception as e: error = str(e)
    return render_template('esop.html', result=result, error=error)


@app.route('/drip', methods=['GET','POST'])
def drip_calculator():
    """DRIP — Dividend Reinvestment / Compound Growth Calculator."""
    result = None; error = None
    if request.method == 'POST':
        try:
            initial_inv   = float(request.form['initial_investment'])
            share_price   = float(request.form['share_price'])
            div_per_share = float(request.form['annual_dividend_per_share'])
            div_growth    = float(request.form.get('dividend_growth_rate',5))/100
            price_growth  = float(request.form.get('share_price_growth_rate',7))/100
            years         = int(request.form['years'])
            currency      = request.form.get('currency','USD')

            shares = initial_inv / share_price
            schedule = []
            total_dividends = 0

            for yr in range(1, years+1):
                cur_price = share_price * (1+price_growth)**yr
                cur_div   = div_per_share * (1+div_growth)**yr
                div_income= shares * cur_div
                new_shares= div_income / cur_price
                shares   += new_shares
                total_dividends += div_income
                portfolio_value = shares * cur_price
                schedule.append({'year': yr, 'share_price': round(cur_price,2),
                                  'dividends': round(div_income,2), 'new_shares': round(new_shares,4),
                                  'total_shares': round(shares,4), 'portfolio_value': round(portfolio_value,2)})

            final_value = shares * share_price*(1+price_growth)**years
            cagr = (final_value/initial_inv)**(1/years)-1

            result = dict(
                initial_investment=initial_inv, share_price=share_price,
                div_per_share=div_per_share, div_growth_pct=div_growth*100,
                price_growth_pct=price_growth*100, years=years, currency=currency,
                final_shares=round(shares,4), final_portfolio_value=round(final_value,2),
                total_dividends_reinvested=round(total_dividends,2),
                total_gain=round(final_value-initial_inv,2),
                cagr_pct=round(cagr*100,4),
                without_drip=round(initial_inv*(1+price_growth)**years,2),
                schedule=schedule[:20],
            )
        except Exception as e: error = str(e)
    return render_template('drip.html', result=result, error=error)


@app.route('/mortgage', methods=['GET','POST'])
def mortgage_calculator():
    """Loan Amortisation / Mortgage Calculator — global standard."""
    result = None; error = None
    if request.method == 'POST':
        try:
            principal  = float(request.form['principal'])
            annual_rate= float(request.form['annual_rate'])/100
            term_years = int(request.form['term_years'])
            frequency  = int(request.form.get('frequency',12))  # payments per year
            currency   = request.form.get('currency','USD')

            n = term_years * frequency
            r = annual_rate / frequency
            if r > 0:
                payment = principal * r*(1+r)**n / ((1+r)**n-1)
            else:
                payment = principal / n

            total_paid     = payment * n
            total_interest = total_paid - principal

            # Amortisation schedule (first 24 periods)
            balance = principal
            schedule = []
            for i in range(1, n+1):
                interest_pmt = balance * r
                principal_pmt = payment - interest_pmt
                balance = max(balance - principal_pmt, 0)
                if i <= 24 or i == n:
                    schedule.append({'period': i, 'payment': round(payment,2),
                                     'principal': round(principal_pmt,2),
                                     'interest': round(interest_pmt,2),
                                     'balance': round(balance,2)})

            result = dict(
                principal=principal, annual_rate_pct=annual_rate*100,
                term_years=term_years, frequency=frequency, currency=currency,
                periodic_payment=round(payment,2),
                total_paid=round(total_paid,2),
                total_interest=round(total_interest,2),
                interest_as_pct_of_principal=round(total_interest/principal*100,2),
                schedule=schedule,
            )
        except Exception as e: error = str(e)
    return render_template('mortgage.html', result=result, error=error)


@app.route('/pension', methods=['GET','POST'])
def pension_calculator():
    """Pension & Retirement Fund Adequacy Calculator — global standard."""
    result = None; error = None
    if request.method == 'POST':
        try:
            current_age    = int(request.form['current_age'])
            retirement_age = int(request.form['retirement_age'])
            life_exp       = int(request.form.get('life_expectancy',85))
            current_savings= float(request.form['current_savings'])
            monthly_contrib= float(request.form['monthly_contribution'])
            return_rate    = float(request.form['expected_return'])/100
            inflation_rate = float(request.form['inflation_rate'])/100
            currency       = request.form.get('currency','USD')

            accum_years = retirement_age - current_age
            dist_years  = life_exp - retirement_age
            real_return = (1+return_rate)/(1+inflation_rate)-1
            monthly_r   = return_rate/12
            monthly_r_real = real_return/12

            # Accumulation phase — FV of current savings + FV of contributions
            fv_savings = current_savings * (1+return_rate)**accum_years
            n_accum = accum_years * 12
            if monthly_r > 0:
                fv_contrib = monthly_contrib * ((1+monthly_r)**n_accum - 1) / monthly_r * (1+monthly_r)
            else:
                fv_contrib = monthly_contrib * n_accum
            corpus = fv_savings + fv_contrib

            # Distribution phase — monthly income (real annuity)
            n_dist = dist_years * 12
            if monthly_r_real > 0:
                monthly_income = corpus * monthly_r_real * (1+monthly_r_real)**n_dist / ((1+monthly_r_real)**n_dist-1)
            else:
                monthly_income = corpus / n_dist

            # Required corpus for desired income (reverse)
            desired_income = monthly_income  # show achievable

            # Milestones
            milestones = []
            for age in range(current_age+5, retirement_age+1, 5):
                yrs = age - current_age
                fv_s = current_savings*(1+return_rate)**yrs
                n_y  = yrs*12
                fv_c = monthly_contrib*((1+monthly_r)**n_y-1)/monthly_r*(1+monthly_r) if monthly_r>0 else monthly_contrib*n_y
                milestones.append({'age':age,'projected_corpus':round(fv_s+fv_c,2)})

            result = dict(
                current_age=current_age, retirement_age=retirement_age, life_exp=life_exp,
                accum_years=accum_years, dist_years=dist_years,
                current_savings=current_savings, monthly_contrib=monthly_contrib,
                return_rate_pct=return_rate*100, inflation_pct=inflation_rate*100,
                real_return_pct=round(real_return*100,4), currency=currency,
                corpus=round(corpus,2), fv_savings=round(fv_savings,2),
                fv_contrib=round(fv_contrib,2),
                monthly_income=round(monthly_income,2),
                annual_income=round(monthly_income*12,2),
                milestones=milestones,
            )
        except Exception as e: error = str(e)
    return render_template('pension.html', result=result, error=error)


@app.route('/microfinance', methods=['GET','POST'])
def microfinance_calculator():
    """SME / Microfinance Loan Pricing — CGAP/IFC standard, effective APR disclosure."""
    result = None; error = None
    if request.method == 'POST':
        try:
            principal   = float(request.form['loan_amount'])
            annual_rate = float(request.form['annual_rate'])/100
            term_months = int(request.form['term_months'])
            orig_fee    = float(request.form.get('origination_fee_pct',2))/100
            insurance   = float(request.form.get('insurance_pct',0.5))/100
            currency    = request.form.get('currency','GHS')

            r = annual_rate / 12
            n = term_months

            if r > 0:
                emi = principal * r * (1+r)**n / ((1+r)**n-1)
            else:
                emi = principal / n

            total_paid     = emi * n
            total_interest = total_paid - principal
            fee_amount     = principal * orig_fee
            insurance_pm   = principal * insurance / 12
            total_insurance= insurance_pm * n

            # True APR (flat rate equivalent)
            net_proceeds   = principal - fee_amount
            true_monthly_rate = r  # approximate; use IRR for exact
            apr = ((total_paid + fee_amount + total_insurance) / net_proceeds - 1) / (n/12) * 100

            schedule = []
            bal = principal
            for i in range(1, min(n+1,13)):
                int_pmt  = bal * r
                prin_pmt = emi - int_pmt
                bal      = max(bal - prin_pmt, 0)
                schedule.append({'month':i,'emi':round(emi,2),'principal':round(prin_pmt,2),
                                  'interest':round(int_pmt,2),'insurance':round(insurance_pm,2),
                                  'total_payment':round(emi+insurance_pm,2),'balance':round(bal,2)})

            result = dict(
                principal=principal, annual_rate_pct=annual_rate*100,
                term_months=term_months, currency=currency,
                orig_fee_pct=orig_fee*100, insurance_pct=insurance*100,
                emi=round(emi,2), total_monthly=round(emi+insurance_pm,2),
                total_paid=round(total_paid,2), total_interest=round(total_interest,2),
                fee_amount=round(fee_amount,2), total_insurance=round(total_insurance,2),
                net_proceeds=round(net_proceeds,2), apr=round(apr,2),
                cost_of_credit=round((total_interest+fee_amount+total_insurance)/principal*100,2),
                schedule=schedule,
            )
        except Exception as e: error = str(e)
    return render_template('microfinance.html', result=result, error=error)


@app.route('/esg', methods=['GET','POST'])
def esg_calculator():
    """ESG / Responsible Investment Scoring — GRI/SASB/TCFD-aligned framework."""
    result = None; error = None
    if request.method == 'POST':
        try:
            companies = request.form.getlist('company_name')
            env_scores= [float(x) for x in request.form.getlist('environmental_score')]
            soc_scores= [float(x) for x in request.form.getlist('social_score')]
            gov_scores= [float(x) for x in request.form.getlist('governance_score')]
            e_w = float(request.form.get('e_weight',40))/100
            s_w = float(request.form.get('s_weight',30))/100
            g_w = float(request.form.get('g_weight',30))/100
            sector   = request.form.get('sector','General')
            currency = request.form.get('currency','USD')

            def rating(score):
                if score>=80: return 'AAA'
                if score>=70: return 'AA'
                if score>=60: return 'A'
                if score>=50: return 'BBB'
                if score>=40: return 'BB'
                if score>=30: return 'B'
                return 'CCC'

            results_list = []
            for i in range(len(companies)):
                e = env_scores[i] if i<len(env_scores) else 0
                s = soc_scores[i] if i<len(soc_scores) else 0
                g = gov_scores[i] if i<len(gov_scores) else 0
                weighted = e*e_w + s*s_w + g*g_w
                results_list.append({
                    'company': companies[i], 'e': e, 's': s, 'g': g,
                    'weighted': round(weighted,2), 'rating': rating(weighted),
                })

            results_list.sort(key=lambda x: -x['weighted'])

            result = dict(
                e_weight=e_w*100, s_weight=s_w*100, g_weight=g_w*100,
                sector=sector, currency=currency,
                companies=results_list,
                top_performer=results_list[0] if results_list else None,
            )
        except Exception as e: error = str(e)
    return render_template('esg.html', result=result, error=error)


# ================================================================
# --- CSV UPLOAD TEMPLATES FOR ALL CALCULATORS ---
# ================================================================

# Each entry: 'calculator_key': {'filename': '...', 'headers': [...], 'example': {...}}
CALCULATOR_TEMPLATES = {
    'dcf':           {'filename':'dcf_template.csv',           'headers':['initial_fcf','growth_rate_high','years_high','growth_rate_terminal','wacc','current_debt','cash','shares_outstanding','currency'],'example':{'initial_fcf':5000000,'growth_rate_high':15,'years_high':5,'growth_rate_terminal':3,'wacc':10,'current_debt':2000000,'cash':500000,'shares_outstanding':1000000,'currency':'USD'}},
    'tbill':         {'filename':'tbill_template.csv',         'headers':['face_value','discount_rate','tenor_days','currency'],'example':{'face_value':10000,'discount_rate':28.5,'tenor_days':364,'currency':'GHS'}},
    'bond':          {'filename':'bond_template.csv',          'headers':['face_value','coupon_rate','ytm','years','frequency','bond_type','currency'],'example':{'face_value':1000,'coupon_rate':8.5,'ytm':9.2,'years':10,'frequency':2,'bond_type':'government','currency':'USD'}},
    'private_equity':{'filename':'pe_template.csv',            'headers':['method','exit_revenue','exit_multiple','investment','ownership_pct','hold_years','required_return','currency'],'example':{'method':'vc','exit_revenue':5000000,'exit_multiple':8,'investment':500000,'ownership_pct':20,'hold_years':5,'required_return':30,'currency':'USD'}},
    'private_debt':  {'filename':'private_debt_template.csv',  'headers':['principal','base_rate','spread','pik_rate','arrangement_fee','term_years','debt_type','currency'],'example':{'principal':5000000,'base_rate':5.3,'spread':4.5,'pik_rate':2,'arrangement_fee':1.5,'term_years':5,'debt_type':'senior','currency':'USD'}},
    'startup':       {'filename':'startup_template.csv',       'headers':['company_name','method','annual_revenue','net_margin','growth_rate','discount_rate','informality_discount','currency'],'example':{'company_name':'My Company','method':'informal_dcf','annual_revenue':500000,'net_margin':12,'growth_rate':15,'discount_rate':25,'informality_discount':30,'currency':'GHS'}},
    'options':       {'filename':'options_template.csv',       'headers':['spot_price','strike_price','time_to_expiry_years','risk_free_rate','volatility','option_type','currency'],'example':{'spot_price':100,'strike_price':105,'time_to_expiry_years':0.5,'risk_free_rate':5,'volatility':20,'option_type':'call','currency':'USD'}},
    'monte_carlo':   {'filename':'monte_carlo_template.csv',   'headers':['initial_price','expected_return','volatility','time_horizon_years','n_simulations','currency'],'example':{'initial_price':100,'expected_return':8,'volatility':20,'time_horizon_years':1,'n_simulations':1000,'currency':'USD'}},
    'real_estate':   {'filename':'real_estate_template.csv',   'headers':['purchase_price','annual_rental_income','operating_expenses','vacancy_rate','cap_rate_market','mortgage_rate','ltv','term_years','currency'],'example':{'purchase_price':500000,'annual_rental_income':48000,'operating_expenses':12000,'vacancy_rate':5,'cap_rate_market':6,'mortgage_rate':7,'ltv':70,'term_years':25,'currency':'USD'}},
    'fx_forward':    {'filename':'fx_forward_template.csv',    'headers':['spot_rate','domestic_rate','foreign_rate','days','notional','base_currency','quote_currency'],'example':{'spot_rate':1.25,'domestic_rate':5.25,'foreign_rate':4.0,'days':90,'notional':1000000,'base_currency':'USD','quote_currency':'GBP'}},
    'commodity':     {'filename':'commodity_template.csv',     'headers':['spot_price','risk_free_rate','storage_cost_pct','convenience_yield_pct','time_to_maturity_years','currency','commodity'],'example':{'spot_price':80,'risk_free_rate':5,'storage_cost_pct':1.5,'convenience_yield_pct':0.5,'time_to_maturity_years':0.5,'currency':'USD','commodity':'Crude Oil'}},
    'yield_curve':   {'filename':'yield_curve_template.csv',   'headers':['maturity_years','coupon_rate','price','face_value'],'example':{'maturity_years':2,'coupon_rate':4.5,'price':99.5,'face_value':100}},
    'cds':           {'filename':'cds_template.csv',           'headers':['notional','spread_bps','recovery_rate','maturity_years','risk_free_rate','currency'],'example':{'notional':10000000,'spread_bps':150,'recovery_rate':40,'maturity_years':5,'risk_free_rate':5,'currency':'USD'}},
    'tips':          {'filename':'tips_template.csv',          'headers':['face_value','real_coupon_rate','inflation_rate','ytm_real','years','currency'],'example':{'face_value':1000,'real_coupon_rate':1.5,'inflation_rate':3.0,'ytm_real':1.2,'years':10,'currency':'USD'}},
    'convertible':   {'filename':'convertible_template.csv',   'headers':['face_value','coupon_rate','ytm','years','conversion_ratio','stock_price','volatility','risk_free_rate','currency'],'example':{'face_value':1000,'coupon_rate':3,'ytm':6,'years':5,'conversion_ratio':20,'stock_price':45,'volatility':30,'risk_free_rate':5,'currency':'USD'}},
    'esop':          {'filename':'esop_template.csv',          'headers':['options_granted','strike_price','current_stock_price','vesting_years','cliff_years','time_to_expiry','volatility','risk_free_rate','tax_rate','shares_outstanding','currency'],'example':{'options_granted':10000,'strike_price':10,'current_stock_price':15,'vesting_years':4,'cliff_years':1,'time_to_expiry':7,'volatility':35,'risk_free_rate':5,'tax_rate':25,'shares_outstanding':1000000,'currency':'USD'}},
    'drip':          {'filename':'drip_template.csv',          'headers':['initial_investment','share_price','annual_dividend_per_share','dividend_growth_rate','share_price_growth_rate','years','currency'],'example':{'initial_investment':10000,'share_price':50,'annual_dividend_per_share':2,'dividend_growth_rate':5,'share_price_growth_rate':7,'years':20,'currency':'USD'}},
    'mortgage':      {'filename':'mortgage_template.csv',      'headers':['principal','annual_rate','term_years','frequency','currency'],'example':{'principal':250000,'annual_rate':7.5,'term_years':30,'frequency':12,'currency':'USD'}},
    'pension':       {'filename':'pension_template.csv',       'headers':['current_age','retirement_age','life_expectancy','current_savings','monthly_contribution','expected_return','inflation_rate','currency'],'example':{'current_age':30,'retirement_age':65,'life_expectancy':85,'current_savings':50000,'monthly_contribution':500,'expected_return':8,'inflation_rate':3,'currency':'USD'}},
    'microfinance':  {'filename':'microfinance_template.csv',  'headers':['loan_amount','annual_rate','term_months','origination_fee_pct','insurance_pct','currency'],'example':{'loan_amount':5000,'annual_rate':24,'term_months':12,'origination_fee_pct':2,'insurance_pct':0.5,'currency':'GHS'}},
    'esg':           {'filename':'esg_template.csv',           'headers':['company_name','environmental_score','social_score','governance_score','e_weight','s_weight','g_weight','sector','currency'],'example':{'company_name':'Company A','environmental_score':72,'social_score':65,'governance_score':80,'e_weight':40,'s_weight':30,'g_weight':30,'sector':'Energy','currency':'USD'}},
    'fcff':          {'filename':'fcff_template.csv',          'headers':['revenue','ebit_margin','tax_rate','depreciation','capex','change_in_wc','wacc','growth_rate','terminal_growth','years','currency'],'example':{'revenue':10000000,'ebit_margin':15,'tax_rate':25,'depreciation':500000,'capex':800000,'change_in_wc':200000,'wacc':10,'growth_rate':8,'terminal_growth':3,'years':5,'currency':'USD'}},
    'credit_risk':   {'filename':'credit_risk_template.csv',   'headers':['working_capital','total_assets','retained_earnings','ebit','market_equity','total_liabilities','revenue','currency'],'example':{'working_capital':2000000,'total_assets':10000000,'retained_earnings':3000000,'ebit':1500000,'market_equity':8000000,'total_liabilities':5000000,'revenue':12000000,'currency':'USD'}},
}


@app.route('/api/template/<calculator_key>')
def download_template(calculator_key):
    """Download a CSV input template for a given calculator."""
    tpl = CALCULATOR_TEMPLATES.get(calculator_key)
    if not tpl:
        return jsonify({'error': 'Template not found'}), 404
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=tpl['headers'])
    writer.writeheader()
    writer.writerow(tpl['example'])
    resp = make_response(output.getvalue())
    resp.headers['Content-Type'] = 'text/csv'
    resp.headers['Content-Disposition'] = f'attachment; filename="{tpl["filename"]}"'
    return resp


@app.route('/api/upload/<calculator_key>', methods=['POST'])
def upload_calculator_data(calculator_key):
    """Parse uploaded CSV and return parsed field values as JSON."""
    tpl = CALCULATOR_TEMPLATES.get(calculator_key)
    if not tpl:
        return jsonify({'error': 'Calculator not supported'}), 404
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file uploaded'}), 400
    try:
        content = f.read().decode('utf-8-sig')
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return jsonify({'error': 'CSV is empty'}), 400
        return jsonify({'rows': rows, 'count': len(rows)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================================================================
# --- STOCK DATA APIs ---
# ================================================================

GLOBAL_TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA',
    'JPM','GS','BRK-B','V','XOM','JNJ','NFLX','TSM',
]

_stock_cache = {'global': [], 'gse': [], 'ts': 0}

@app.route('/api/stocks/ticker')
def api_stocks_ticker():
    """Return rolling ticker data for global + GSE stocks (cached 5 min)."""
    import time
    now = time.time()
    if now - _stock_cache['ts'] < 300 and (_stock_cache['global'] or _stock_cache['gse']):
        return jsonify({'global': _stock_cache['global'], 'gse': _stock_cache['gse']})

    # GSE stocks
    gse_data = []
    try:
        r = http_requests.get('https://dev.kwayisi.org/apis/gse/live', timeout=8)
        if r.ok:
            for s in r.json():
                gse_data.append({
                    'symbol': s.get('name', s.get('code', '')),
                    'price':  s.get('price', 0),
                    'change': s.get('change', 0),
                    'change_pct': s.get('change_percent', s.get('pct', 0)),
                    'market': 'GSE',
                })
    except Exception:
        pass

    # Global stocks — single batch download (one HTTP request for all tickers)
    global_data = []
    try:
        import yfinance as yf, pandas as pd
        raw = yf.download(
            tickers=GLOBAL_TICKERS,
            period='5d',
            interval='1d',
            progress=False,
            threads=True,
            auto_adjust=True,
        )
        if not raw.empty:
            close = raw['Close']
            # close is DataFrame[date x ticker] for multi-ticker downloads
            close = close.dropna(how='all')
            if len(close) >= 2:
                last_row = close.iloc[-1]
                prev_row = close.iloc[-2]
            elif len(close) == 1:
                last_row = close.iloc[-1]
                prev_row = close.iloc[-1]
            else:
                last_row = prev_row = pd.Series(dtype=float)
            for sym in GLOBAL_TICKERS:
                try:
                    price = float(last_row[sym])
                    prev  = float(prev_row[sym])
                    if pd.isna(price) or pd.isna(prev) or prev == 0:
                        continue
                    chg = round(price - prev, 2)
                    pct = round(chg / prev * 100, 2)
                    global_data.append({
                        'symbol': sym,
                        'price':  round(price, 2),
                        'change': chg,
                        'change_pct': pct,
                        'market': 'GLOBAL',
                    })
                except Exception:
                    pass
    except Exception:
        pass

    # Only overwrite cache if we actually got data; otherwise serve stale
    if global_data or gse_data:
        _stock_cache.update({'global': global_data, 'gse': gse_data, 'ts': now})

    return jsonify({'global': _stock_cache['global'], 'gse': _stock_cache['gse']})


@app.route('/api/stocks/lookup')
def api_stock_lookup():
    """Return key fundamentals for a given ticker symbol."""
    symbol = request.args.get('symbol', '').upper().strip()
    market = request.args.get('market', 'global').lower()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400

    if market == 'gse':
        # Step 1: live endpoint — always works, has price/change/volume
        live_row = {}
        try:
            live_r = http_requests.get('https://dev.kwayisi.org/apis/gse/live', timeout=6)
            if live_r.ok:
                for s in live_r.json():
                    if s.get('name', '').upper() == symbol or s.get('code', '').upper() == symbol:
                        live_row = s
                        break
        except Exception:
            pass

        # Step 2: detail endpoint — richer data but often slow; short timeout, fully optional
        detail = {}
        try:
            dr = http_requests.get(
                f'https://dev.kwayisi.org/apis/gse/equities/{symbol}',
                timeout=4,
            )
            if dr.ok:
                detail = dr.json()
        except Exception:
            pass

        # Nothing found at all
        if not live_row and not detail:
            return jsonify({'error': f'GSE stock "{symbol}" not found on the exchange'}), 404

        # Merge: live_row wins for real-time price/volume; detail wins for fundamentals
        company    = (detail.get('company') or {})
        price      = live_row.get('price') or detail.get('price') or 0
        change     = live_row.get('change', '')
        change_pct = live_row.get('change_percent', live_row.get('pct', ''))
        volume     = live_row.get('volume', '')
        shares     = detail.get('shares', '')
        eps        = detail.get('eps', '')
        dps        = detail.get('dps', '')
        capital    = detail.get('capital', '')
        market_cap = capital if capital else (round(shares * price, 2) if shares and price else '')
        pe_ratio   = round(price / eps, 2) if eps and price else ''
        name       = (company.get('name') or detail.get('name')
                      or live_row.get('name') or symbol)
        sector     = company.get('sector') or company.get('industry') or ''

        return jsonify({
            'symbol':            symbol,
            'name':              name,
            'current_price':     price,
            'change':            change,
            'change_pct':        change_pct,
            'shares_outstanding': shares,
            'traded_volume':     volume,
            'market_cap':        market_cap,
            'pe_ratio':          pe_ratio,
            'eps':               eps,
            'dps':               dps,
            'sector':            sector,
            'currency':          'GHS',
            'exchange':          'GSE',
        })
    else:
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            info = t.info
            fi = t.fast_info
            return jsonify({
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'current_price': round(float(fi.last_price), 2) if fi.last_price else info.get('currentPrice', ''),
                'shares_outstanding': info.get('sharesOutstanding', ''),
                'market_cap': info.get('marketCap', ''),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', '')),
                'eps': info.get('trailingEps', ''),
                'beta': info.get('beta', ''),
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else '',
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', ''),
                'book_value': info.get('bookValue', ''),
                'revenue': info.get('totalRevenue', ''),
                'ebitda': info.get('ebitda', ''),
                'total_debt': info.get('totalDebt', ''),
                'total_cash': info.get('totalCash', ''),
                'roe': round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else '',
                'roa': round(info.get('returnOnAssets', 0) * 100, 2) if info.get('returnOnAssets') else '',
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500


# ================================================================
# --- CSV / EXCEL EXPORT ---
# ================================================================

@app.route('/api/export/csv', methods=['POST'])
def export_csv():
    """Generic CSV export — receives JSON rows + filename from client."""
    data = request.get_json(force=True, silent=True) or {}
    rows = data.get('rows', [])
    filename = data.get('filename', 'investiq_result') + '.csv'
    if not rows:
        return jsonify({'error': 'No data'}), 400
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    resp = make_response(output.getvalue())
    resp.headers['Content-Type'] = 'text/csv'
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp


# ================================================================
# --- NEW CALCULATORS ---
# ================================================================

@app.route('/tbill', methods=['GET', 'POST'])
@app.route('/treasury-bill', methods=['GET', 'POST'])
def tbill_calculator():
    result = None
    error = None
    if request.method == 'POST':
        try:
            face_value = float(request.form['face_value'])
            discount_rate = float(request.form['discount_rate']) / 100
            tenor_days = int(request.form.get('tenor_days', 364))
            currency = request.form.get('currency', 'GHS')

            # T-Bill price = Face Value / (1 + discount_rate × days/364)
            price = face_value / (1 + discount_rate * tenor_days / 364)
            discount_amount = face_value - price
            # Effective yield (actual/365)
            effective_yield = (discount_amount / price) * (365 / tenor_days) * 100
            # Holding period return
            hpr = (discount_amount / price) * 100

            result = {
                'face_value': face_value,
                'price': round(price, 4),
                'discount_amount': round(discount_amount, 4),
                'discount_rate_pct': discount_rate * 100,
                'effective_yield': round(effective_yield, 4),
                'hpr': round(hpr, 4),
                'tenor_days': tenor_days,
                'currency': currency,
            }
        except Exception as e:
            error = str(e)
    return render_template('tbill.html', result=result, error=error)


@app.route('/bond-calculator', methods=['GET', 'POST'])
def bond_calculator():
    result = None
    error = None
    if request.method == 'POST':
        try:
            face_value = float(request.form['face_value'])
            coupon_rate = float(request.form['coupon_rate']) / 100
            ytm = float(request.form['ytm']) / 100
            years = int(request.form['years'])
            frequency = int(request.form.get('frequency', 2))  # semi-annual default
            bond_type = request.form.get('bond_type', 'government')
            currency = request.form.get('currency', 'USD')

            # Periods
            n = years * frequency
            c = (coupon_rate * face_value) / frequency
            r = ytm / frequency

            # Bond price (PV of coupons + PV of par)
            if r == 0:
                price = c * n + face_value
            else:
                pv_coupons = c * (1 - (1 + r) ** -n) / r
                pv_par = face_value / (1 + r) ** n
                price = pv_coupons + pv_par

            premium_discount = price - face_value

            # Duration (Macaulay)
            if r == 0:
                mac_duration = sum([(t / frequency) * c / price for t in range(1, n + 1)]) + (n / frequency) * face_value / price
            else:
                weighted_cf = sum([(t / frequency) * c / (1 + r) ** t for t in range(1, n + 1)])
                weighted_cf += (n / frequency) * face_value / (1 + r) ** n
                mac_duration = weighted_cf / price

            modified_duration = mac_duration / (1 + r)
            # DV01
            dv01 = modified_duration * price * 0.0001

            # Accrued interest (assume clean price = dirty price for now)
            annual_coupon = coupon_rate * face_value
            coupon_payments = [{'period': t, 'cash_flow': round(c, 2),
                                 'pv': round(c / (1 + r) ** t, 2)} for t in range(1, n + 1)]
            coupon_payments[-1]['cash_flow'] = round(c + face_value, 2)
            coupon_payments[-1]['pv'] = round((c + face_value) / (1 + r) ** n, 2)

            result = {
                'face_value': face_value,
                'coupon_rate_pct': coupon_rate * 100,
                'ytm_pct': ytm * 100,
                'years': years,
                'frequency': frequency,
                'bond_type': bond_type.title(),
                'currency': currency,
                'price': round(price, 4),
                'premium_discount': round(premium_discount, 4),
                'premium_discount_pct': round(premium_discount / face_value * 100, 4),
                'annual_coupon': round(annual_coupon, 4),
                'mac_duration': round(mac_duration, 4),
                'modified_duration': round(modified_duration, 4),
                'dv01': round(dv01, 6),
                'coupon_payments': coupon_payments[:20],  # cap for display
                'total_periods': n,
            }
        except Exception as e:
            error = str(e)
    return render_template('bond_calculator.html', result=result, error=error)


@app.route('/private-equity', methods=['GET', 'POST'])
def private_equity():
    result = None
    error = None
    if request.method == 'POST':
        try:
            method = request.form.get('method', 'vc')
            currency = request.form.get('currency', 'USD')

            if method == 'vc':
                # VC Method
                exit_revenue = float(request.form['exit_revenue'])
                exit_multiple = float(request.form['exit_multiple'])
                investment = float(request.form['investment'])
                ownership_pct = float(request.form['ownership_pct']) / 100
                hold_years = float(request.form['hold_years'])
                required_return = float(request.form['required_return']) / 100

                exit_value = exit_revenue * exit_multiple
                investor_exit_value = exit_value * ownership_pct
                moic = investor_exit_value / investment
                irr = (moic ** (1 / hold_years) - 1) * 100
                pre_money = investment / ownership_pct - investment
                post_money = investment / ownership_pct

                result = {
                    'method': 'Venture Capital (VC) Method',
                    'currency': currency,
                    'exit_value': round(exit_value, 2),
                    'investor_exit_value': round(investor_exit_value, 2),
                    'moic': round(moic, 2),
                    'irr': round(irr, 2),
                    'pre_money_valuation': round(pre_money, 2),
                    'post_money_valuation': round(post_money, 2),
                    'hold_years': hold_years,
                    'ownership_pct': ownership_pct * 100,
                }
            else:
                # LBO Method
                entry_ebitda = float(request.form['entry_ebitda'])
                entry_multiple = float(request.form['entry_multiple'])
                exit_ebitda = float(request.form['exit_ebitda'])
                exit_multiple_lbo = float(request.form['exit_multiple_lbo'])
                debt_pct = float(request.form['debt_pct']) / 100
                hold_years = float(request.form['hold_years'])
                currency = request.form.get('currency', 'USD')

                entry_ev = entry_ebitda * entry_multiple
                debt = entry_ev * debt_pct
                equity = entry_ev * (1 - debt_pct)
                exit_ev = exit_ebitda * exit_multiple_lbo
                exit_equity = max(exit_ev - debt * 0.7, 0)  # assume 30% debt paydown
                moic = exit_equity / equity if equity > 0 else 0
                irr = (moic ** (1 / hold_years) - 1) * 100 if moic > 0 else 0

                result = {
                    'method': 'Leveraged Buyout (LBO) Method',
                    'currency': currency,
                    'entry_ev': round(entry_ev, 2),
                    'equity_invested': round(equity, 2),
                    'debt': round(debt, 2),
                    'debt_pct': debt_pct * 100,
                    'exit_ev': round(exit_ev, 2),
                    'exit_equity': round(exit_equity, 2),
                    'moic': round(moic, 2),
                    'irr': round(irr, 2),
                    'hold_years': hold_years,
                }

        except Exception as e:
            error = str(e)
    return render_template('private_equity.html', result=result, error=error)


@app.route('/private-debt', methods=['GET', 'POST'])
def private_debt():
    result = None
    error = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            base_rate = float(request.form['base_rate']) / 100
            spread = float(request.form['spread']) / 100
            arrangement_fee = float(request.form.get('arrangement_fee', 0)) / 100
            term_years = float(request.form['term_years'])
            pik_rate = float(request.form.get('pik_rate', 0)) / 100  # Payment-in-kind
            currency = request.form.get('currency', 'USD')
            debt_type = request.form.get('debt_type', 'senior')

            all_in_rate = base_rate + spread
            annual_cash_interest = principal * all_in_rate
            annual_pik = principal * pik_rate
            total_annual_yield = all_in_rate + pik_rate

            # Upfront fee economics
            net_proceeds = principal * (1 - arrangement_fee)
            # Yield-to-maturity including fee
            fee_adjusted_ytm = (annual_cash_interest + principal * arrangement_fee / term_years) / net_proceeds * 100

            # Total return over term
            total_cash_interest = annual_cash_interest * term_years
            total_pik = annual_pik * term_years
            total_fees = principal * arrangement_fee
            total_return = total_cash_interest + total_pik + total_fees

            result = {
                'principal': principal,
                'currency': currency,
                'debt_type': debt_type.title(),
                'base_rate_pct': base_rate * 100,
                'spread_pct': spread * 100,
                'all_in_rate_pct': round(all_in_rate * 100, 4),
                'pik_rate_pct': pik_rate * 100,
                'total_annual_yield_pct': round(total_annual_yield * 100, 4),
                'annual_cash_interest': round(annual_cash_interest, 2),
                'annual_pik': round(annual_pik, 2),
                'arrangement_fee_pct': arrangement_fee * 100,
                'net_proceeds': round(net_proceeds, 2),
                'fee_adjusted_ytm': round(fee_adjusted_ytm, 4),
                'total_cash_interest': round(total_cash_interest, 2),
                'total_pik': round(total_pik, 2),
                'total_fees': round(total_fees, 2),
                'total_return': round(total_return, 2),
                'term_years': term_years,
            }
        except Exception as e:
            error = str(e)
    return render_template('private_debt.html', result=result, error=error)


@app.route('/startup-valuation', methods=['GET', 'POST'])
def startup_valuation():
    """Startup & Informal Sector Enterprise Valuation — NEW."""
    result = None
    error = None
    if request.method == 'POST':
        try:
            method = request.form.get('method', 'scorecard')
            currency = request.form.get('currency', 'USD')
            company_name = request.form.get('company_name', 'Company')

            if method == 'scorecard':
                # Berkus / Scorecard Method
                base_valuation = float(request.form['base_valuation'])
                team_score = float(request.form.get('team_score', 0)) / 100
                opportunity_score = float(request.form.get('opportunity_score', 0)) / 100
                product_score = float(request.form.get('product_score', 0)) / 100
                sales_score = float(request.form.get('sales_score', 0)) / 100
                competition_score = float(request.form.get('competition_score', 0)) / 100

                # Scorecard weights (standard: team 30%, opportunity 25%, product 15%, sales 20%, competition 10%)
                weighted = (team_score * 0.30 + opportunity_score * 0.25 +
                            product_score * 0.15 + sales_score * 0.20 + competition_score * 0.10)
                valuation = base_valuation * weighted * 2  # multiply by 2 relative to median

                result = {
                    'method': 'Scorecard Method',
                    'company_name': company_name,
                    'currency': currency,
                    'base_valuation': base_valuation,
                    'valuation': round(valuation, 2),
                    'team_score': team_score * 100,
                    'opportunity_score': opportunity_score * 100,
                    'product_score': product_score * 100,
                    'sales_score': sales_score * 100,
                    'competition_score': competition_score * 100,
                    'composite_score': round(weighted * 100, 1),
                }

            elif method == 'revenue_multiple':
                revenue = float(request.form['revenue'])
                revenue_multiple = float(request.form['revenue_multiple'])
                growth_rate = float(request.form.get('growth_rate', 0)) / 100
                stage_discount = float(request.form.get('stage_discount', 0)) / 100

                base_valuation = revenue * revenue_multiple
                growth_premium = base_valuation * growth_rate
                valuation = (base_valuation + growth_premium) * (1 - stage_discount)

                result = {
                    'method': 'Revenue Multiple Method',
                    'company_name': company_name,
                    'currency': currency,
                    'revenue': revenue,
                    'revenue_multiple': revenue_multiple,
                    'growth_rate_pct': growth_rate * 100,
                    'stage_discount_pct': stage_discount * 100,
                    'base_valuation': round(base_valuation, 2),
                    'growth_premium': round(growth_premium, 2),
                    'valuation': round(valuation, 2),
                }

            elif method == 'berkus':
                # Berkus Method — for pre-revenue startups
                sound_idea = float(request.form.get('sound_idea', 0))
                prototype = float(request.form.get('prototype', 0))
                quality_team = float(request.form.get('quality_team', 0))
                strategic_relations = float(request.form.get('strategic_relations', 0))
                product_rollout = float(request.form.get('product_rollout', 0))

                valuation = sound_idea + prototype + quality_team + strategic_relations + product_rollout

                result = {
                    'method': 'Berkus Method (Pre-Revenue)',
                    'company_name': company_name,
                    'currency': currency,
                    'sound_idea': sound_idea,
                    'prototype': prototype,
                    'quality_team': quality_team,
                    'strategic_relations': strategic_relations,
                    'product_rollout': product_rollout,
                    'valuation': round(valuation, 2),
                }

            elif method == 'informal_dcf':
                # Informal Sector Adapted DCF — for third-world informal enterprises
                annual_revenue = float(request.form['annual_revenue'])
                net_margin = float(request.form['net_margin']) / 100
                growth_rate = float(request.form['growth_rate']) / 100
                terminal_growth = float(request.form.get('terminal_growth', 3)) / 100
                discount_rate = float(request.form['discount_rate']) / 100
                years = int(request.form.get('years', 5))
                informality_discount = float(request.form.get('informality_discount', 30)) / 100

                net_income = annual_revenue * net_margin
                cash_flows = []
                pv_total = 0
                for yr in range(1, years + 1):
                    cf = net_income * ((1 + growth_rate) ** yr)
                    pv = cf / ((1 + discount_rate) ** yr)
                    pv_total += pv
                    cash_flows.append({'year': yr, 'cash_flow': round(cf, 2), 'pv': round(pv, 2)})

                terminal_value = (net_income * (1 + growth_rate) ** years * (1 + terminal_growth)) / (discount_rate - terminal_growth)
                pv_terminal = terminal_value / ((1 + discount_rate) ** years)
                enterprise_value_formal = pv_total + pv_terminal
                # Apply informality discount (governance risk, regulatory risk, succession risk)
                enterprise_value = enterprise_value_formal * (1 - informality_discount)

                result = {
                    'method': 'Informal Sector DCF (Adjusted)',
                    'company_name': company_name,
                    'currency': currency,
                    'annual_revenue': annual_revenue,
                    'net_margin_pct': net_margin * 100,
                    'growth_rate_pct': growth_rate * 100,
                    'discount_rate_pct': discount_rate * 100,
                    'informality_discount_pct': informality_discount * 100,
                    'enterprise_value_formal': round(enterprise_value_formal, 2),
                    'enterprise_value': round(enterprise_value, 2),
                    'pv_cash_flows': round(pv_total, 2),
                    'pv_terminal': round(pv_terminal, 2),
                    'cash_flows': cash_flows,
                }

        except Exception as e:
            error = str(e)
    return render_template('startup_valuation.html', result=result, error=error)


@app.route('/informal-sector', methods=['GET', 'POST'])
def informal_sector():
    """Informal Sector Restructuring Roadmap — NEW."""
    result = None
    if request.method == 'POST':
        try:
            sector = request.form.get('sector', 'retail')
            employees = int(request.form.get('employees', 5))
            annual_revenue = float(request.form.get('annual_revenue', 0))
            registered = request.form.get('registered', 'no') == 'yes'
            has_accounts = request.form.get('has_accounts', 'no') == 'yes'
            has_banking = request.form.get('has_banking', 'no') == 'yes'
            currency = request.form.get('currency', 'USD')

            steps = []
            score = 0

            if registered:
                score += 20
            else:
                steps.append({
                    'priority': 'Critical',
                    'step': 'Business Registration',
                    'action': 'Register with national business registry / companies commission',
                    'benefit': 'Legal existence, access to contracts, bank credit, government grants',
                    'timeline': '1-3 months',
                })
            if has_accounts:
                score += 20
            else:
                steps.append({
                    'priority': 'Critical',
                    'step': 'Formal Bookkeeping',
                    'action': 'Adopt double-entry bookkeeping; hire part-time accountant or use free accounting software (Wave, GnuCash)',
                    'benefit': 'Tax compliance, audit readiness, investor confidence',
                    'timeline': '1 month',
                })
            if has_banking:
                score += 15
            else:
                steps.append({
                    'priority': 'High',
                    'step': 'Open Business Bank Account',
                    'action': 'Separate personal and business finances via SME bank account',
                    'benefit': 'Credit history, mobile payments, payroll, loan access',
                    'timeline': '2 weeks',
                })

            steps += [
                {
                    'priority': 'High',
                    'step': 'Tax Identification Number (TIN)',
                    'action': 'Obtain TIN from revenue authority; file simple tax returns annually',
                    'benefit': 'Eligibility for government contracts, tender participation',
                    'timeline': '2-4 weeks',
                },
                {
                    'priority': 'Medium',
                    'step': 'Corporate Governance Basics',
                    'action': 'Define ownership structure, appoint a manager, create simple board of 3',
                    'benefit': 'Succession planning, investor readiness, reduced key-person risk',
                    'timeline': '1-2 months',
                },
                {
                    'priority': 'Medium',
                    'step': 'Financial Statements',
                    'action': 'Prepare annual Income Statement, Balance Sheet, and Cash Flow Statement',
                    'benefit': 'Bankability, private equity/debt eligibility, grant applications',
                    'timeline': 'Ongoing (annually)',
                },
                {
                    'priority': 'Medium',
                    'step': 'Digitise Operations',
                    'action': 'Adopt point-of-sale (POS), mobile money, inventory management tools',
                    'benefit': 'Audit trail, faster collections, reduced cash leakage',
                    'timeline': '1-3 months',
                },
                {
                    'priority': 'Low',
                    'step': 'Access Finance',
                    'action': 'Apply for SME loans, development finance (DFI), microfinance, or equity crowdfunding after formalisation',
                    'benefit': 'Growth capital at lower cost than informal lenders',
                    'timeline': 'After 6-12 months of formal operation',
                },
            ]

            formality_score = score
            formality_label = 'Fully Informal' if score < 20 else ('Partially Formal' if score < 50 else 'Mostly Formal')

            result = {
                'sector': sector.title(),
                'employees': employees,
                'annual_revenue': annual_revenue,
                'currency': currency,
                'formality_score': formality_score,
                'formality_label': formality_label,
                'steps': steps,
            }
        except Exception as e:
            result = {'error': str(e)}
    return render_template('informal_sector.html', result=result)


# ================================================================
# --- GUIDE DOWNLOAD ---
# ================================================================

@app.route('/download-guide')
@app.route('/download_guide')
def download_guide():
    return redirect(url_for('help_page'))


@app.route('/api/download-guide-pdf')
def download_guide_pdf():
    """Generate and serve a plain-text/HTML guide as downloadable file."""
    guide_content = """InvestIQ — Investment Calculator Guide
======================================
Version 2026 | investiq.app

CALCULATORS INCLUDED
--------------------
1. DCF (Discounted Cash Flow) — /dcf
   Formula: IV = Σ FCF_t/(1+WACC)^t + TV/(1+WACC)^n
   Terminal Value = FCF_n × (1+g) / (WACC-g)

2. FCFF (Free Cash Flow to Firm) — /fcff
   FCFF = EBIT(1-T) + D&A - ΔWC - CapEx

3. FCFE (Free Cash Flow to Equity) — /intrinsic_value_fcfe
   FCFE = Net Income + D&A - ΔWC - CapEx + Net Borrowing

4. Treasury Bill Calculator — /tbill
   Price = FV / (1 + r × d/364)
   Effective Yield = (Discount/Price) × (365/d)

5. Bond Pricing Calculator — /bond-calculator
   Price = Σ C/(1+r)^t + F/(1+r)^n
   Macaulay Duration, Modified Duration, DV01

6. Private Equity Calculator — /private-equity
   VC Method: Pre-money = Investment/Ownership% - Investment
   LBO Method: MOIC = Exit Equity / Equity Invested
   IRR = MOIC^(1/years) - 1

7. Private Debt Calculator — /private-debt
   All-in Rate = Base Rate + Spread
   Fee-Adjusted YTM = (Interest + Fee/Term) / Net Proceeds

8. Startup & Enterprise Valuation — /startup-valuation
   Scorecard, Revenue Multiple, Berkus, Informal DCF methods

9. Informal Sector Restructuring — /informal-sector
   Step-by-step formalisation roadmap

10. CAPM (Cost of Equity) — /cost_of_equity
    Ke = Rf + β(Rm - Rf)

11. Altman Z-Score (Banking) — /credit_risk
    Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5

12. Beta Calculator — /calculate_beta
13. Target Price — /target-price
14. DDM — /dvm
15. Multi-Method Valuation — /multi_method_valuation
16. Bond Risk — /bond-risk
17. Portfolio Risks — /portfolio-risk
18. Duration — /duration
19. Asset Allocation — /asset-allocation
20. Capital Structure — /capital_structure

HOW TO USE VIDEO TUTORIALS
---------------------------
Visit /videos on InvestIQ to watch step-by-step video guides.

DISCLAIMER
----------
All calculators are for educational and informational purposes only.
They do not constitute financial advice.
Always consult a qualified financial advisor.

© 2026 InvestIQ. All rights reserved.
"""
    resp = make_response(guide_content)
    resp.headers['Content-Type'] = 'text/plain; charset=utf-8'
    resp.headers['Content-Disposition'] = 'attachment; filename="InvestIQ_Calculator_Guide.txt"'
    return resp


# ============================================================
# HR PLATFORM ROUTES
# ============================================================

@app.route('/jobs')
def jobs_page():
    listings = JobListing.query.filter_by(is_active=True).order_by(JobListing.created_at.desc()).all()
    return render_template('hr_jobs.html', listings=listings)


@app.route('/stock-pitch')
def stock_pitch():
    return render_template('hr_stock_pitch.html')


@app.route('/stock-pitch/download-pptx')
def download_stock_pitch_pptx():
    from pptx_generator import build_pptx
    buf = build_pptx()
    return send_file(
        buf,
        as_attachment=True,
        download_name='YIN_Stock_Pitch_Template.pptx',
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )


# ── GISI EXAMS ───────────────────────────────────────────────────────────────
@app.route('/gisi-exams')
def gisi_exams():
    paid_sections = []
    if current_user.is_authenticated:
        payments = GISIPayment.query.filter_by(email=current_user.email, status='Approved').all()
        for pay in payments:
            if pay.plan == 'bundle':
                paid_sections = list(set(paid_sections + [2, 3, 4, 5]))
            elif pay.section not in paid_sections:
                paid_sections.append(pay.section)
    return render_template('hr_gisi_exams.html', paid_sections=paid_sections)


@app.route('/gisi-exams/pay', methods=['POST'])
def gisi_exams_pay():
    import secrets as _sec
    full_name = request.form.get('full_name', '').strip()
    email     = request.form.get('email', '').strip()
    phone     = request.form.get('phone', '').strip()
    plan      = request.form.get('plan', 'single')
    section   = int(request.form.get('section', 2))
    reference = request.form.get('reference', '').strip()

    if not full_name or not email or not reference:
        return jsonify({'success': False, 'message': 'Please fill in all required fields.'}), 400

    # Prevent duplicate references
    existing = GISIPayment.query.filter_by(reference=reference).first()
    if existing:
        return jsonify({'success': False, 'message': 'This MoMo reference has already been submitted. Contact support if this is an error.'}), 400

    amount       = 200.0 if plan == 'bundle' else 60.0
    admin_token  = _sec.token_urlsafe(32)
    access_code  = 'SIPQ-' + _sec.token_hex(3).upper()

    pay = GISIPayment(
        full_name=full_name, email=email, phone=phone,
        amount=amount, plan=plan, section=section,
        reference=reference, status='Pending',
        admin_token=admin_token, access_code=access_code
    )
    db.session.add(pay)
    db.session.commit()

    base = request.host_url.rstrip('/')
    approve_url = f"{base}/gisi-exams/approve/{admin_token}"
    reject_url  = f"{base}/gisi-exams/reject/{admin_token}"
    section_label = f"Section {section}" if plan == 'single' else "Sections 2–5 (Bundle)"

    # Email admin with approve/reject buttons
    try:
        msg = Message(
            f'[ACTION REQUIRED] Practice Questions Payment — {full_name} — GHS{amount:.0f}',
            sender=app.config.get('MAIL_USERNAME', 'kyeikofi@gmail.com'),
            recipients=['kyeikofi@gmail.com']
        )
        msg.html = f'''
<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f8fafc;border-radius:12px;overflow:hidden;">
  <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:28px 32px;text-align:center;">
    <h1 style="color:#fff;margin:0;font-size:22px;">⚡ Securities Industry Practice Questions — Payment Approval Required</h1>
    <p style="color:#93c5fd;margin:8px 0 0;font-size:14px;">InvestIQ Talent Hub</p>
  </div>
  <div style="padding:28px 32px;background:#fff;">
    <table style="width:100%;border-collapse:collapse;font-size:15px;">
      <tr><td style="padding:8px 0;color:#64748b;width:140px;font-weight:600;">Name</td><td style="padding:8px 0;color:#1e293b;font-weight:700;">{full_name}</td></tr>
      <tr style="background:#f8fafc;"><td style="padding:8px 6px;color:#64748b;font-weight:600;">Email</td><td style="padding:8px 6px;color:#1e293b;">{email}</td></tr>
      <tr><td style="padding:8px 0;color:#64748b;font-weight:600;">Phone</td><td style="padding:8px 0;color:#1e293b;">{phone or "—"}</td></tr>
      <tr style="background:#f8fafc;"><td style="padding:8px 6px;color:#64748b;font-weight:600;">Plan</td><td style="padding:8px 6px;color:#1e293b;">{plan.title()} — <strong>GHS{amount:.0f}</strong></td></tr>
      <tr><td style="padding:8px 0;color:#64748b;font-weight:600;">Access</td><td style="padding:8px 0;color:#1e293b;">{section_label}</td></tr>
      <tr style="background:#f8fafc;"><td style="padding:8px 6px;color:#64748b;font-weight:600;">MoMo Ref</td><td style="padding:8px 6px;color:#1e293b;font-family:monospace;font-weight:700;font-size:16px;">{reference}</td></tr>
      <tr><td style="padding:8px 0;color:#64748b;font-weight:600;">Access Code</td><td style="padding:8px 0;color:#1e293b;font-family:monospace;font-weight:700;font-size:16px;letter-spacing:2px;">{access_code}</td></tr>
      <tr style="background:#f8fafc;"><td style="padding:8px 6px;color:#64748b;font-weight:600;">Submitted</td><td style="padding:8px 6px;color:#1e293b;">{datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")}</td></tr>
    </table>
    <p style="margin:20px 0 8px;color:#475569;font-size:14px;">Please verify the MoMo payment against your MTN MoMo records, then click the appropriate button:</p>
    <div style="text-align:center;margin:24px 0;">
      <a href="{approve_url}" style="display:inline-block;background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;text-decoration:none;padding:14px 36px;border-radius:10px;font-weight:700;font-size:16px;margin:0 8px;">✅ Approve &amp; Send Access Code</a>
      <a href="{reject_url}" style="display:inline-block;background:#dc2626;color:#fff;text-decoration:none;padding:14px 36px;border-radius:10px;font-weight:700;font-size:16px;margin:8px;">❌ Reject</a>
    </div>
    <p style="font-size:12px;color:#94a3b8;text-align:center;margin-top:16px;">These links are one-time use and expire after action. The user is currently waiting for approval.</p>
  </div>
</div>'''
        mail.send(msg)
        email_ok = True
        email_error = None
    except Exception as e:
        email_ok = False
        email_error = str(e)
        logger.error(f'GISI admin email failed: {e}')

    sections_unlocking = list(range(2, 6)) if plan == 'bundle' else [section]
    return jsonify({
        'success': True,
        'pending': True,
        'sections_unlocking': sections_unlocking,
        'email': email,
        'email_ok': email_ok,
        'email_error': email_error
    })


@app.route('/gisi-exams/approve/<admin_token>')
def gisi_exams_approve(admin_token):
    pay = GISIPayment.query.filter_by(admin_token=admin_token).first_or_404()
    if pay.status == 'Approved':
        return render_template('hr_gisi_approve_done.html', already=True, pay=pay)
    if pay.status == 'Rejected':
        return '<h2 style="font-family:sans-serif;color:#dc2626;text-align:center;margin-top:80px;">This payment was already rejected.</h2>', 400

    pay.status = 'Approved'
    pay.approved_at = datetime.utcnow()
    db.session.commit()

    section_label = f"Section {pay.section}" if pay.plan == 'single' else "Sections 2–5 (Bundle)"
    base = request.host_url.rstrip('/')
    exam_url = f"{base}/gisi-exams"

    # Email user with access code
    try:
        msg = Message(
            f'✅ Practice Questions Access Approved — Your Access Code Inside',
            sender=app.config.get('MAIL_USERNAME', 'kyeikofi@gmail.com'),
            recipients=[pay.email]
        )
        msg.html = f'''
<div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#f8fafc;border-radius:12px;overflow:hidden;">
  <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:32px;text-align:center;">
    <div style="width:64px;height:64px;background:#16a34a;border-radius:50%;margin:0 auto 16px;display:flex;align-items:center;justify-content:center;font-size:32px;">✅</div>
    <h1 style="color:#fff;margin:0;font-size:24px;">Payment Approved!</h1>
    <p style="color:#93c5fd;margin:8px 0 0;font-size:15px;">Your Securities Industry Practice Questions access is now ready</p>
  </div>
  <div style="padding:32px;background:#fff;text-align:center;">
    <p style="color:#475569;font-size:16px;margin-bottom:24px;">Hi <strong>{pay.full_name}</strong>, your payment of <strong>GHS{pay.amount:.0f}</strong> for <strong>{section_label}</strong> has been verified and approved.</p>
    <p style="color:#1e293b;font-size:15px;font-weight:600;margin-bottom:12px;">Your Personal Access Code</p>
    <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border:2px dashed #3b82f6;border-radius:14px;padding:20px 32px;display:inline-block;margin-bottom:24px;">
      <span style="font-family:monospace;font-size:32px;font-weight:900;color:#1d4ed8;letter-spacing:4px;">{pay.access_code}</span>
    </div>
    <p style="color:#64748b;font-size:14px;margin-bottom:24px;">Enter this code on the Practice Questions page to unlock your section(s). This code is unique to you — please keep it safe.</p>
    <a href="{exam_url}" style="display:inline-block;background:linear-gradient(135deg,#1e3a5f,#1d4ed8);color:#fff;text-decoration:none;padding:16px 40px;border-radius:12px;font-weight:700;font-size:17px;">Start Practising Now →</a>
    <p style="font-size:12px;color:#94a3b8;margin-top:24px;">Thank you for choosing InvestIQ for your examination preparation.<br/>InvestIQ Talent Hub — <a href="{base}" style="color:#3b82f6;">{base}</a></p>
  </div>
</div>'''
        mail.send(msg)
    except Exception as e:
        logger.error(f'GISI user approval email failed: {e}')

    return render_template('hr_gisi_approve_done.html', already=False, pay=pay)


@app.route('/gisi-exams/reject/<admin_token>')
def gisi_exams_reject(admin_token):
    pay = GISIPayment.query.filter_by(admin_token=admin_token).first_or_404()
    if pay.status != 'Pending':
        return f'<h2 style="font-family:sans-serif;text-align:center;margin-top:80px;">This payment has already been processed ({pay.status}).</h2>'
    pay.status = 'Rejected'
    db.session.commit()
    # Email user to notify rejection
    try:
        msg = Message(
            'Securities Industry Practice Questions — Payment Unable to Verify',
            sender=app.config.get('MAIL_USERNAME', 'kyeikofi@gmail.com'),
            recipients=[pay.email]
        )
        msg.html = f'''
<div style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;padding:40px 32px;background:#fff;border-radius:12px;border:1px solid #e2e8f0;">
  <h2 style="color:#dc2626;margin-top:0;">Payment Could Not Be Verified</h2>
  <p style="color:#475569;">Hi <strong>{pay.full_name}</strong>,</p>
  <p style="color:#475569;">We were unable to verify your MoMo payment reference <strong>{pay.reference}</strong> for GHS{pay.amount:.0f}.</p>
  <p style="color:#475569;">Please double-check the reference and resubmit, or contact us at <a href="mailto:kyeikofi@gmail.com">kyeikofi@gmail.com</a> for assistance.</p>
  <p style="color:#94a3b8;font-size:13px;margin-top:24px;">InvestIQ Talent Hub</p>
</div>'''
        mail.send(msg)
    except Exception:
        pass
    return '<div style="font-family:Arial,sans-serif;text-align:center;padding:80px 32px;"><h2 style="color:#dc2626;">Payment Rejected</h2><p style="color:#64748b;">The user has been notified by email.</p></div>'


@app.route('/gisi-exams/redeem', methods=['POST'])
def gisi_exams_redeem():
    if not current_user.is_authenticated:
        return jsonify({'success': False, 'message': 'Please log in to redeem your access code.', 'login_required': True}), 401

    code = request.form.get('code', '').strip().upper()
    if not code:
        return jsonify({'success': False, 'message': 'Please enter your access code.'}), 400

    # Check if the code exists at all (any status)
    pay_any = GISIPayment.query.filter_by(access_code=code).first()
    if not pay_any:
        return jsonify({'success': False, 'message': 'Access code not recognised. Please check the code in your approval email and try again.'}), 400
    if pay_any.status == 'Pending':
        return jsonify({'success': False, 'message': 'Your payment is still awaiting admin approval. You will receive your access code by email once confirmed — usually within 5 minutes.'}), 400
    if pay_any.status == 'Rejected':
        return jsonify({'success': False, 'message': 'This payment was rejected. Please contact kyeikofi@gmail.com for assistance.'}), 400
    pay = pay_any  # status is Approved

    if pay.email.lower() != current_user.email.lower():
        return jsonify({'success': False, 'message': 'This access code was issued to a different account. Please use the code sent to your registered email address.'}), 403

    # Reload all paid sections from DB for this user
    payments = GISIPayment.query.filter_by(email=current_user.email, status='Approved').all()
    paid = []
    for p in payments:
        if p.plan == 'bundle':
            paid = list(set(paid + [2, 3, 4, 5]))
        elif p.section not in paid:
            paid.append(p.section)

    return jsonify({'success': True, 'paid_sections': paid, 'name': pay.full_name})


@app.route('/jobs/<int:job_id>/apply', methods=['GET', 'POST'])
def job_apply(job_id):
    job = JobListing.query.get_or_404(job_id)
    error = None
    success = False
    if request.method == 'POST':
        try:
            appl = JobApplication(
                job_id=job.id,
                full_name=request.form.get('full_name', '').strip(),
                email=request.form.get('email', '').strip(),
                phone=request.form.get('phone', '').strip(),
                cover_letter=request.form.get('cover_letter', '').strip(),
            )
            if not appl.full_name or not appl.email:
                error = 'Full name and email are required.'
            else:
                db.session.add(appl)
                db.session.commit()
                success = True
        except Exception as e:
            error = 'Submission failed. Please try again.'
            logger.error(f'Job apply error: {e}')
    return render_template('hr_job_detail.html', job=job, error=error, success=success)


@app.route('/cv-builder', methods=['GET', 'POST'])
def cv_builder():
    jobs = JobListing.query.filter_by(is_active=True).order_by(JobListing.title).all()
    cv_data = None
    error = None
    saved_to_corner = False
    if request.method == 'POST':
        try:
            def _f(key): return request.form.get(key, '').strip()
            cv_data = {
                # Personal
                'full_name':          _f('full_name'),
                'headline':           _f('headline'),
                'email':              _f('email'),
                'phone':              _f('phone'),
                'city':               _f('city'),
                'state':              _f('state'),
                'zip_code':           _f('zip_code'),
                'country':            _f('country'),
                'linkedin':           _f('linkedin'),
                'github':             _f('github'),
                'website':            _f('website'),
                'nationality':        _f('nationality'),
                'work_authorization': _f('work_authorization'),
                # Backwards-compat location field
                'location': ', '.join(filter(None, [_f('city'), _f('state'), _f('country')])),
                # Summary & skills
                'summary':          _f('summary'),
                'technical_skills': _f('technical_skills'),
                'soft_skills':      _f('soft_skills'),
                'tools':            _f('tools'),
                'skills':           _f('skills'),
                # Additional sections (text)
                'awards':           _f('awards'),
                'memberships':      _f('memberships'),
                'volunteer':        _f('volunteer'),
                'publications':     _f('publications'),
                'references_type':  _f('references_type') or 'on_request',
                'references_text':  _f('references_text'),
                'interests':        _f('interests'),
                # Template
                'template': _f('template') or 'professional',
                # Employers' corner
                'desired_role':   _f('desired_role'),
                'desired_sector': _f('desired_sector'),
                'years_exp':      _f('years_exp'),
                'availability':   _f('availability'),
                # Dynamic sections
                'work_experiences': [],
                'educations':       [],
                'certifications':   [],
                'training':         [],
                'projects_list':    [],
                'languages':        [],
            }
            # Work experience
            for i in range(21):
                title = request.form.get(f'work_experiences-{i}-job_title', '')
                if not title and i > 0: break
                if title:
                    cv_data['work_experiences'].append({
                        'job_title':       title,
                        'company':         _f(f'work_experiences-{i}-company'),
                        'employment_type': _f(f'work_experiences-{i}-employment_type'),
                        'industry':        _f(f'work_experiences-{i}-industry'),
                        'location':        _f(f'work_experiences-{i}-location'),
                        'team_size':       _f(f'work_experiences-{i}-team_size'),
                        'start_date':      _f(f'work_experiences-{i}-start_date'),
                        'end_date':        _f(f'work_experiences-{i}-end_date'),
                        'responsibilities':_f(f'work_experiences-{i}-responsibilities'),
                    })
            # Education
            for i in range(11):
                degree = request.form.get(f'educations-{i}-degree', '')
                if not degree and i > 0: break
                if degree:
                    cv_data['educations'].append({
                        'degree':       degree,
                        'field':        _f(f'educations-{i}-field'),
                        'institution':  _f(f'educations-{i}-institution'),
                        'edu_location': _f(f'educations-{i}-edu_location'),
                        'start_date':   _f(f'educations-{i}-start_date'),
                        'end_date':     _f(f'educations-{i}-end_date'),
                        'gpa':          _f(f'educations-{i}-gpa'),
                        'honors':       _f(f'educations-{i}-honors'),
                        'thesis':       _f(f'educations-{i}-thesis'),
                    })
            # Certifications
            for i in range(15):
                name = request.form.get(f'certifications-{i}-name', '')
                if not name and i > 0: break
                if name:
                    cv_data['certifications'].append({
                        'name':          name,
                        'organization':  _f(f'certifications-{i}-organization'),
                        'issue_date':    _f(f'certifications-{i}-issue_date'),
                        'expiry_date':   _f(f'certifications-{i}-expiry_date'),
                        'credential_id': _f(f'certifications-{i}-credential_id'),
                    })
            # Professional training
            for i in range(15):
                tname = request.form.get(f'training-{i}-name', '')
                if not tname and i > 0: break
                if tname:
                    cv_data['training'].append({
                        'name':     tname,
                        'provider': _f(f'training-{i}-provider'),
                        'year':     _f(f'training-{i}-year'),
                        'duration': _f(f'training-{i}-duration'),
                    })
            # Projects
            for i in range(15):
                ptitle = request.form.get(f'projects_list-{i}-title', '')
                if not ptitle and i > 0: break
                if ptitle:
                    cv_data['projects_list'].append({
                        'title':       ptitle,
                        'role':        _f(f'projects_list-{i}-role'),
                        'description': _f(f'projects_list-{i}-description'),
                        'tech':        _f(f'projects_list-{i}-tech'),
                        'duration':    _f(f'projects_list-{i}-duration'),
                        'url':         _f(f'projects_list-{i}-url'),
                    })
            # Languages
            for i in range(15):
                lang = request.form.get(f'languages-{i}-language', '')
                if not lang and i > 0: break
                if lang:
                    cv_data['languages'].append({
                        'language':    lang,
                        'proficiency': _f(f'languages-{i}-proficiency'),
                    })
            if not cv_data['full_name'] or not cv_data['email']:
                error = 'Full name and email are required.'
                cv_data = None
            else:
                # Save survey response (always, regardless of employers corner consent)
                try:
                    survey = CVSurveyResponse(
                        full_name             = cv_data['full_name'],
                        email                 = cv_data['email'],
                        yin_member            = 'Yes' if request.form.get('yin_member') == 'yes' else 'No',
                        stock_pitch           = 'Yes' if request.form.get('stock_pitch') == 'yes' else 'No',
                        want_internship       = 'Yes' if request.form.get('want_internship') == 'yes' else 'No',
                        want_national_service = 'Yes' if request.form.get('want_national_service') == 'yes' else 'No',
                    )
                    db.session.add(survey)
                    # Save extra YIN survey fields
                    yin_join_date = request.form.get('yin_join_date', '').strip()
                    years_with_yin = ''
                    if yin_join_date:
                        try:
                            from datetime import date
                            join = date.fromisoformat(yin_join_date)
                            years_with_yin = str((date.today() - join).days // 365)
                        except Exception:
                            pass
                    seeking_ft = 'Yes' if request.form.get('seeking_full_time') == 'yes' else 'No'
                    if email:
                        db.session.add(CVSurveyExtra(
                            email=email,
                            yin_join_date=yin_join_date,
                            years_with_yin=years_with_yin,
                            seeking_full_time=seeking_ft,
                        ))
                    db.session.commit()
                except Exception as sv_err:
                    logger.error(f'Survey save error: {sv_err}')
            if cv_data and request.form.get('employers_corner_consent'):
                try:
                    current_title = (cv_data['work_experiences'][0]['job_title'] if cv_data['work_experiences'] else cv_data.get('headline', ''))
                    all_skills = ', '.join(filter(None, [cv_data['technical_skills'], cv_data['skills']]))[:500]
                    existing = CandidateProfile.query.filter_by(email=cv_data['email']).first()
                    if existing:
                        existing.full_name       = cv_data['full_name']
                        existing.phone           = cv_data['phone']
                        existing.location        = cv_data['location']
                        existing.desired_role    = cv_data['desired_role']
                        existing.desired_sector  = cv_data['desired_sector']
                        existing.current_title   = current_title
                        existing.skills_summary  = all_skills
                        existing.profile_summary = cv_data['summary']
                        existing.linkedin        = cv_data['linkedin']
                        existing.years_exp       = cv_data['years_exp']
                        existing.availability    = cv_data['availability']
                        existing.is_visible      = True
                    else:
                        db.session.add(CandidateProfile(
                            full_name       = cv_data['full_name'],
                            email           = cv_data['email'],
                            phone           = cv_data['phone'],
                            location        = cv_data['location'],
                            desired_role    = cv_data['desired_role'],
                            desired_sector  = cv_data['desired_sector'],
                            current_title   = current_title,
                            skills_summary  = all_skills,
                            profile_summary = cv_data['summary'],
                            linkedin        = cv_data['linkedin'],
                            years_exp       = cv_data['years_exp'],
                            availability    = cv_data['availability'],
                        ))
                    db.session.commit()
                    saved_to_corner = True
                except Exception as db_err:
                    logger.error(f'Employers Corner save error: {db_err}')
        except Exception as e:
            error = 'Error processing CV. Please check all fields.'
            logger.error(f'CV build error: {e}')
    return render_template('hr_cv_builder.html', cv_data=cv_data, error=error,
                           jobs=jobs, saved_to_corner=saved_to_corner,
                           now_date=datetime.utcnow().strftime('%Y-%m-%d'))


@app.route('/cv-preview')
def cv_preview():
    import json as _json
    cv_json = request.args.get('data', '')
    try:
        cv_data = _json.loads(cv_json) if cv_json else session.get('cv_data', {})
    except Exception:
        cv_data = session.get('cv_data', {})
    if not cv_data:
        return redirect(url_for('cv_builder'))
    template_name = cv_data.get('template', 'professional')
    return render_template(f'hr_cv_{template_name}.html', cv=cv_data)


@app.route('/training', methods=['GET', 'POST'])
def training_page():
    error = None
    import hashlib as _hashlib

    def _referral_code_for(email):
        """Generate a consistent 8-char uppercase referral code from an email."""
        return _hashlib.md5(email.strip().lower().encode()).hexdigest()[:8].upper()

    success        = False
    my_referral_code = None
    booking_type   = request.form.get('booking_type', 'individual')

    if request.method == 'POST':
        try:
            cat           = request.form.get('category', '').strip()
            other_prog    = request.form.get('other_program', '').strip()
            ref_code_used = request.form.get('referral_code', '').strip().upper()

            booking = TrainingBooking(
                booking_type=booking_type,
                full_name=request.form.get('full_name', '').strip(),
                email=request.form.get('email', '').strip().lower(),
                phone=request.form.get('phone', '').strip(),
                organization=request.form.get('organization', '').strip(),
                participants=int(request.form.get('participants', 1) or 1),
                category=cat if cat != 'Other' else f'Other: {other_prog}',
                other_program=other_prog,
                preferred_date=request.form.get('preferred_date', '').strip(),
                notes=request.form.get('notes', '').strip(),
                referral_code_used=ref_code_used,
            )
            if not booking.full_name or not booking.email:
                error = 'Full name and email are required.'
            else:
                db.session.add(booking)
                db.session.flush()   # get booking.id before commit

                # Process referral code if provided
                if ref_code_used:
                    # Find an existing referral record to identify the referrer
                    existing = Referral.query.filter_by(referral_code=ref_code_used).first()
                    if existing:
                        new_ref = Referral(
                            referrer_name=existing.referrer_name,
                            referrer_email=existing.referrer_email,
                            referral_code=ref_code_used,
                            referred_name=booking.full_name,
                            referred_email=booking.email,
                            booking_id=booking.id,
                            status='Successful',
                        )
                        db.session.add(new_ref)
                    # else: unrecognised code — silently ignore

                # Create a "seed" referral record for this booker if none exists
                my_code = _referral_code_for(booking.email)
                if not Referral.query.filter_by(referral_code=my_code).first():
                    db.session.add(Referral(
                        referrer_name=booking.full_name,
                        referrer_email=booking.email,
                        referral_code=my_code,
                        status='Pending',  # seed record — no referee yet
                    ))

                db.session.commit()
                my_referral_code = my_code

                send_email_safe(
                    f'New Training Booking: {booking.category}',
                    [app.config['ADMIN_EMAIL']],
                    f'''<h3>New Training Booking Request</h3>
<table>
<tr><td><b>Name:</b></td><td>{booking.full_name}</td></tr>
<tr><td><b>Email:</b></td><td>{booking.email}</td></tr>
<tr><td><b>Phone:</b></td><td>{booking.phone or "N/A"}</td></tr>
<tr><td><b>Type:</b></td><td>{booking.booking_type}</td></tr>
<tr><td><b>Organisation:</b></td><td>{booking.organization or "N/A"}</td></tr>
<tr><td><b>Participants:</b></td><td>{booking.participants}</td></tr>
<tr><td><b>Program:</b></td><td>{booking.category}</td></tr>
<tr><td><b>Preferred Date:</b></td><td>{booking.preferred_date or "Flexible"}</td></tr>
<tr><td><b>Referral Code Used:</b></td><td>{ref_code_used or "None"}</td></tr>
<tr><td><b>Notes:</b></td><td>{booking.notes or "None"}</td></tr>
</table>'''
                )
                success = True
        except Exception as e:
            error = 'Booking failed. Please try again.'
            logger.error(f'Training booking error: {e}')
    return render_template('hr_training.html', error=error, success=success,
                           booking_type=booking_type, my_referral_code=my_referral_code)


@app.route('/referral', methods=['GET', 'POST'])
def referral_page():
    success = False
    if request.method == 'POST':
        success = True
    return render_template('hr_referral.html', success=success)


@app.route('/employers-corner')
def employers_corner():
    emp = _current_employer()
    if not emp:
        return redirect(url_for('employer_login') + '?next=/employers-corner')
    sector    = request.args.get('sector',    '').strip()
    role_q    = request.args.get('role',      '').strip()
    skills_kw = request.args.get('skills',    '').strip()
    avail     = request.args.get('avail',     '').strip()
    exp_filter= request.args.get('exp',       '').strip()
    sort_by   = request.args.get('sort',      'recent')

    q = CandidateProfile.query.filter_by(is_visible=True)
    if sector:
        q = q.filter(CandidateProfile.desired_sector == sector)
    if role_q:
        q = q.filter(CandidateProfile.desired_role.ilike(f'%{role_q}%'))
    if skills_kw:
        q = q.filter(CandidateProfile.skills_summary.ilike(f'%{skills_kw}%'))
    if avail:
        q = q.filter(CandidateProfile.availability == avail)
    if exp_filter:
        q = q.filter(CandidateProfile.years_exp == exp_filter)

    candidates = q.order_by(CandidateProfile.created_at.desc()).all()

    # Apply Python-side sort
    if sort_by == 'availability':
        candidates = sorted(candidates, key=lambda c: c.availability_urgency)
    elif sort_by == 'experience':
        exp_order = {
            '20+ years (Senior Executive)': 0, '15–20 years': 1, '10–15 years': 2,
            '5–10 years': 3, '3–5 years': 4, '1–3 years': 5,
            '0–1 year (Graduate / Entry-level)': 6,
        }
        candidates = sorted(candidates, key=lambda c: exp_order.get(c.years_exp, 9))
    # 'recent' = default db order (already applied)

    # Stats
    immediate = [c for c in candidates if c.availability in ('Immediately available', 'Immediately', 'Within 1 week')]
    total_all  = CandidateProfile.query.filter_by(is_visible=True).count()

    sectors = [r[0] for r in
               db.session.query(CandidateProfile.desired_sector)
               .filter(CandidateProfile.is_visible == True,
                       CandidateProfile.desired_sector != '')
               .distinct().order_by(CandidateProfile.desired_sector).all()]

    exp_levels = [r[0] for r in
                  db.session.query(CandidateProfile.years_exp)
                  .filter(CandidateProfile.is_visible == True,
                          CandidateProfile.years_exp != '')
                  .distinct().order_by(CandidateProfile.years_exp).all()]

    availabilities = [r[0] for r in
                      db.session.query(CandidateProfile.availability)
                      .filter(CandidateProfile.is_visible == True,
                              CandidateProfile.availability != '')
                      .distinct().order_by(CandidateProfile.availability).all()]

    # Sector distribution for stats
    sector_counts = {}
    for s in sectors:
        sector_counts[s] = CandidateProfile.query.filter_by(is_visible=True, desired_sector=s).count()

    return render_template('hr_employers_corner.html',
                           candidates=candidates,
                           immediate=immediate,
                           sectors=sectors,
                           exp_levels=exp_levels,
                           availabilities=availabilities,
                           sector_counts=sector_counts,
                           current_sector=sector,
                           current_role=role_q,
                           current_skills=skills_kw,
                           current_avail=avail,
                           current_exp=exp_filter,
                           current_sort=sort_by,
                           total=len(candidates),
                           total_all=total_all,
                           immediate_count=len(immediate),
                           current_employer=_current_employer())


@app.route('/candidate/<int:cid>')
def candidate_detail(cid):
    c = CandidateProfile.query.filter_by(id=cid, is_visible=True).first_or_404()
    return render_template('hr_candidate_detail.html', c=c)


@app.route('/employer-inquiry', methods=['POST'])
def employer_inquiry():
    import json as _json
    try:
        data = request.get_json() or request.form
        cid  = int(data.get('candidate_id', 0))
        c    = CandidateProfile.query.filter_by(id=cid, is_visible=True).first()
        if not c:
            return _json.dumps({'ok': False, 'error': 'Candidate not found'}), 404

        emp_acct = _current_employer()
        inq = EmployerInquiry(
            candidate_id        = cid,
            employer_account_id = emp_acct.id if emp_acct else None,
            employer_company    = (emp_acct.company_name if emp_acct else str(data.get('employer_company', ''))).strip(),
            employer_name       = (emp_acct.contact_name if emp_acct else str(data.get('employer_name', ''))).strip(),
            employer_email      = (emp_acct.email if emp_acct else str(data.get('employer_email', ''))).strip(),
            employer_phone      = (emp_acct.phone if emp_acct else str(data.get('employer_phone', ''))).strip(),
            inquiry_type        = str(data.get('inquiry_type', 'Interview Request')).strip(),
            role_offering       = str(data.get('role_offering', '')).strip(),
            message             = str(data.get('message', '')).strip(),
        )
        if not inq.employer_name or not inq.employer_email:
            return _json.dumps({'ok': False, 'error': 'Name and email are required'}), 400

        db.session.add(inq)
        db.session.commit()

        # Email to candidate
        send_email_safe(
            f'New {inq.inquiry_type} from {inq.employer_company or inq.employer_name} — InvestIQ Talent Hub',
            [c.email],
            f'''<div style="font-family:Arial,sans-serif;max-width:580px;margin:0 auto;padding:20px;">
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:24px;border-radius:12px 12px 0 0;">
  <h1 style="color:#fff;font-size:20px;margin:0;">InvestIQ Talent Hub</h1>
  <p style="color:#93c5fd;margin:4px 0 0;font-size:13px;">You have a new employer inquiry</p>
</div>
<div style="background:#fff;border:1px solid #e2e8f0;border-top:none;padding:28px;border-radius:0 0 12px 12px;">
  <h2 style="color:#1e3a5f;font-size:18px;margin:0 0 16px;">Hello {c.full_name},</h2>
  <p style="color:#374151;font-size:14px;line-height:1.6;">An employer has expressed interest in your profile on the InvestIQ Employers' Corner. Here are their details:</p>
  <table style="width:100%;border-collapse:collapse;margin:18px 0;font-size:14px;">
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;border-radius:6px 0 0 0;width:38%;">Inquiry Type</td><td style="padding:10px;background:#f0f9ff;color:#0369a1;font-weight:700;border-radius:0 6px 0 0;">{inq.inquiry_type}</td></tr>
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;">Company</td><td style="padding:10px;">{inq.employer_company or '—'}</td></tr>
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;">Contact Person</td><td style="padding:10px;">{inq.employer_name}</td></tr>
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;">Email</td><td style="padding:10px;"><a href="mailto:{inq.employer_email}" style="color:#1d4ed8;">{inq.employer_email}</a></td></tr>
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;">Phone</td><td style="padding:10px;">{inq.employer_phone or '—'}</td></tr>
    <tr><td style="padding:10px;background:#f8fafc;font-weight:700;color:#1e3a5f;">Role Offered</td><td style="padding:10px;font-weight:600;color:#059669;">{inq.role_offering or '—'}</td></tr>
  </table>
  <div style="background:#f0f9ff;border-left:4px solid #0f4c81;padding:14px 18px;border-radius:0 8px 8px 0;margin-bottom:20px;">
    <p style="font-weight:700;color:#1e3a5f;margin:0 0 6px;font-size:13px;">MESSAGE FROM EMPLOYER:</p>
    <p style="color:#374151;font-size:14px;line-height:1.65;margin:0;">{inq.message}</p>
  </div>
  <p style="color:#374151;font-size:13px;">To respond, simply reply to this email or contact the employer directly using the details above.</p>
  <p style="color:#94a3b8;font-size:12px;margin-top:20px;">This message was sent via InvestIQ Talent Hub. To remove your profile, visit <a href="https://investright.onrender.com/cv-builder" style="color:#1d4ed8;">your CV Builder</a>.</p>
</div></div>'''
        )

        # Confirmation to employer
        send_email_safe(
            f'Your inquiry to {c.full_name} has been sent — InvestIQ',
            [inq.employer_email],
            f'''<div style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;padding:20px;">
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:24px;border-radius:12px 12px 0 0;">
  <h1 style="color:#fff;font-size:20px;margin:0;">InvestIQ Talent Hub</h1>
</div>
<div style="background:#fff;border:1px solid #e2e8f0;border-top:none;padding:28px;border-radius:0 0 12px 12px;">
  <p style="color:#374151;font-size:14px;">Hi {inq.employer_name}, your {inq.inquiry_type} to <strong>{c.full_name}</strong> has been delivered successfully. They will respond to you directly at <strong>{inq.employer_email}</strong>.</p>
  <p style="color:#94a3b8;font-size:12px;margin-top:16px;">Powered by InvestIQ Talent Hub — investright.onrender.com</p>
</div></div>'''
        )

        return _json.dumps({'ok': True, 'message': f'Inquiry sent to {c.full_name} successfully!'}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f'Employer inquiry error: {e}')
        return _json.dumps({'ok': False, 'error': 'An error occurred. Please try again.'}), 500, {'Content-Type': 'application/json'}


# ─── EMPLOYER ACCOUNT ROUTES ──────────────────────────────────────────────

def _current_employer():
    """Return logged-in EmployerAccount or None."""
    eid = session.get('employer_id')
    if eid:
        return EmployerAccount.query.filter_by(id=eid, is_active=True).first()
    return None


@app.route('/employer-register', methods=['GET', 'POST'])
def employer_register():
    emp = _current_employer()
    if emp:
        return redirect(url_for('employer_dashboard'))
    error = None
    form_data = {}
    if request.method == 'POST':
        form_data = {k: request.form.get(k, '').strip() for k in request.form}
        company_name  = form_data.get('company_name', '')
        contact_name  = form_data.get('contact_name', '')
        email         = form_data.get('email', '')
        password      = form_data.get('password', '')
        confirm_pw    = form_data.get('confirm_password', '')
        phone         = form_data.get('phone', '')
        industry      = form_data.get('industry', '')
        company_size  = form_data.get('company_size', '')
        website       = form_data.get('website', '')
        hiring_for    = form_data.get('hiring_for', '')

        # Company name strict validation
        ok, err = _validate_company_name(company_name)
        if not ok:
            error = err
        elif not contact_name or len(contact_name) < 2:
            error = 'Please enter your full name (contact person).'
        elif not email or '@' not in email:
            error = 'A valid work email address is required.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters.'
        elif password != confirm_pw:
            error = 'Passwords do not match.'
        elif EmployerAccount.query.filter_by(email=email).first():
            error = 'An account with this email already exists. Please log in.'
        elif EmployerAccount.query.filter(
                db.func.lower(EmployerAccount.company_name) == company_name.lower()
             ).first():
            error = f'A company named "{company_name}" is already registered. If this is your company, please log in or contact support.'
        else:
            try:
                acct = EmployerAccount(
                    company_name = company_name,
                    contact_name = contact_name,
                    email        = email,
                    phone        = phone,
                    industry     = industry,
                    company_size = company_size,
                    website      = website,
                    hiring_for   = hiring_for,
                )
                acct.set_password(password)
                db.session.add(acct)
                db.session.commit()
                session['employer_id'] = acct.id
                # Welcome email
                send_email_safe(
                    f'Welcome to InvestIQ Employers\' Corner — {company_name}',
                    [email],
                    f'''<div style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;">
<div style="background:linear-gradient(135deg,#0f172a,#0f4c81);padding:28px;border-radius:12px 12px 0 0;">
  <h1 style="color:#fff;font-size:22px;margin:0;">Welcome to InvestIQ Talent Hub</h1>
  <p style="color:#93c5fd;margin:5px 0 0;font-size:13px;">Your employer account is active</p>
</div>
<div style="background:#fff;border:1px solid #e2e8f0;border-top:none;padding:28px;border-radius:0 0 12px 12px;">
  <p style="color:#374151;font-size:15px;margin-bottom:16px;">Hi {contact_name},</p>
  <p style="color:#374151;font-size:14px;line-height:1.7;margin-bottom:16px;">
    <strong>{company_name}</strong> is now registered on the InvestIQ Employers\' Corner. You can now:
  </p>
  <ul style="color:#374151;font-size:14px;line-height:2;padding-left:20px;">
    <li>Browse and search verified finance &amp; investment professionals</li>
    <li>Save candidates to your persistent shortlist</li>
    <li>Send direct interview requests and job offers</li>
    <li>Track your full inquiry history</li>
    <li>Post job listings visible to all candidates</li>
  </ul>
  <a href="https://investright.onrender.com/employer-dashboard" style="display:inline-block;background:#0f4c81;color:#fff;padding:12px 24px;border-radius:10px;font-weight:700;text-decoration:none;margin-top:16px;">
    Go to Your Dashboard →
  </a>
  <p style="color:#94a3b8;font-size:12px;margin-top:20px;">InvestIQ Talent Hub · investright.onrender.com</p>
</div></div>'''
                )
                # Admin notification
                send_email_safe(
                    f'New Employer Registered: {company_name}',
                    [app.config['ADMIN_EMAIL']],
                    f'<p>New employer account: <strong>{company_name}</strong> | {contact_name} | {email} | {industry} | Size: {company_size}</p>'
                )
                return redirect(url_for('employer_dashboard'))
            except Exception as e:
                db.session.rollback()
                logger.error(f'Employer register error: {e}')
                error = 'Registration failed. Please try again.'
    return render_template('hr_employer_register.html', error=error, form_data=form_data)


@app.route('/employer-login', methods=['GET', 'POST'])
def employer_login():
    emp = _current_employer()
    if emp:
        return redirect(url_for('employer_dashboard'))
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        acct = EmployerAccount.query.filter_by(email=email, is_active=True).first()
        if acct and acct.check_password(password):
            session['employer_id'] = acct.id
            next_url = request.args.get('next') or url_for('employer_dashboard')
            return redirect(next_url)
        else:
            error = 'Invalid email or password. Please try again.'
    return render_template('hr_employer_login.html', error=error)


@app.route('/employer-logout')
def employer_logout():
    session.pop('employer_id', None)
    return redirect(url_for('employers_corner'))


@app.route('/employer-dashboard')
def employer_dashboard():
    emp = _current_employer()
    if not emp:
        return redirect(url_for('employer_login') + '?next=/employer-dashboard')

    # Persistent shortlist with candidate objects
    shortlist_entries = (EmployerShortlist.query
                         .filter_by(employer_id=emp.id)
                         .order_by(EmployerShortlist.created_at.desc())
                         .all())

    # Recent inquiries
    recent_inquiries = (EmployerInquiry.query
                        .filter_by(employer_account_id=emp.id)
                        .order_by(EmployerInquiry.created_at.desc())
                        .limit(20).all())

    # Recommended candidates — match employer's industry to candidate sector
    rec_q = CandidateProfile.query.filter_by(is_visible=True)
    if emp.industry:
        rec_q = rec_q.filter(CandidateProfile.desired_sector.ilike(f'%{emp.industry.split()[0]}%'))
    recommended = sorted(rec_q.limit(50).all(), key=lambda c: c.availability_urgency)[:12]

    # Stats
    total_candidates = CandidateProfile.query.filter_by(is_visible=True).count()
    immediate_count  = CandidateProfile.query.filter(
        CandidateProfile.is_visible == True,
        CandidateProfile.availability.in_(['Immediately available', 'Immediately', 'Within 1 week'])
    ).count()

    import json as _json
    cands_for_modal = CandidateProfile.query.filter_by(is_visible=True).all()
    candidates_json = _json.dumps({
        c.id: {'name': c.full_name, 'role': c.current_title or c.desired_sector or ''}
        for c in cands_for_modal
    })

    return render_template('hr_employer_dashboard.html',
                           emp=emp,
                           shortlist_entries=shortlist_entries,
                           recent_inquiries=recent_inquiries,
                           recommended=recommended,
                           total_candidates=total_candidates,
                           immediate_count=immediate_count,
                           candidates_json=candidates_json)


@app.route('/employer-shortlist/toggle', methods=['POST'])
def employer_shortlist_toggle():
    import json as _json
    emp = _current_employer()
    if not emp:
        return _json.dumps({'ok': False, 'error': 'Login required'}), 401, {'Content-Type': 'application/json'}
    cid = int(request.get_json().get('candidate_id', 0))
    c = CandidateProfile.query.filter_by(id=cid, is_visible=True).first()
    if not c:
        return _json.dumps({'ok': False, 'error': 'Candidate not found'}), 404, {'Content-Type': 'application/json'}
    existing = EmployerShortlist.query.filter_by(employer_id=emp.id, candidate_id=cid).first()
    if existing:
        db.session.delete(existing)
        db.session.commit()
        return _json.dumps({'ok': True, 'action': 'removed', 'count': emp.shortlist_count}), 200, {'Content-Type': 'application/json'}
    else:
        db.session.add(EmployerShortlist(employer_id=emp.id, candidate_id=cid))
        db.session.commit()
        return _json.dumps({'ok': True, 'action': 'added', 'count': emp.shortlist_count}), 200, {'Content-Type': 'application/json'}


@app.route('/employer-shortlist/stage', methods=['POST'])
def employer_shortlist_stage():
    import json as _json
    emp = _current_employer()
    if not emp:
        return _json.dumps({'ok': False}), 401, {'Content-Type': 'application/json'}
    data = request.get_json() or {}
    # Accept either shortlist_id (from dashboard) or candidate_id (legacy)
    sl_id = int(data.get('shortlist_id', 0))
    if sl_id:
        entry = EmployerShortlist.query.filter_by(id=sl_id, employer_id=emp.id).first()
    else:
        entry = EmployerShortlist.query.filter_by(
            employer_id=emp.id, candidate_id=int(data.get('candidate_id', 0))
        ).first()
    if entry:
        entry.stage = data.get('stage', 'Saved')
        db.session.commit()
    return _json.dumps({'ok': True}), 200, {'Content-Type': 'application/json'}


@app.route('/employer-shortlist/remove', methods=['POST'])
def employer_shortlist_remove():
    import json as _json
    emp = _current_employer()
    if not emp:
        return _json.dumps({'ok': False}), 401, {'Content-Type': 'application/json'}
    data = request.get_json() or {}
    sl_id = int(data.get('shortlist_id', 0))
    entry = EmployerShortlist.query.filter_by(id=sl_id, employer_id=emp.id).first()
    if entry:
        db.session.delete(entry)
        db.session.commit()
    return _json.dumps({'ok': True}), 200, {'Content-Type': 'application/json'}


@app.route('/employer-profile', methods=['GET', 'POST'])
def employer_profile():
    emp = _current_employer()
    if not emp:
        return redirect(url_for('employer_login'))
    error = success = None
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'update_profile':
            # company_name is read-only — never updated here
            emp.contact_name = request.form.get('contact_name', emp.contact_name).strip() or emp.contact_name
            emp.phone        = request.form.get('phone', '').strip()
            emp.industry     = request.form.get('industry', '').strip()
            emp.company_size = request.form.get('company_size', '').strip()
            emp.website      = request.form.get('website', '').strip()
            emp.hiring_for   = request.form.get('hiring_for', '').strip()
            db.session.commit()
            success = 'Profile updated successfully.'
        elif action == 'change_password':
            current_pw = request.form.get('current_password', '')
            new_pw     = request.form.get('new_password', '')
            confirm_pw = request.form.get('confirm_password', '')
            if not emp.check_password(current_pw):
                error = 'Current password is incorrect.'
            elif len(new_pw) < 8:
                error = 'New password must be at least 8 characters.'
            elif new_pw != confirm_pw:
                error = 'Passwords do not match.'
            else:
                emp.set_password(new_pw)
                db.session.commit()
                success = 'Password changed successfully.'
    return render_template('hr_employer_profile.html', emp=emp, error=error, success=success)


# Admin HR routes
@app.route('/admin/jobs')
def admin_jobs():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    listings = JobListing.query.order_by(JobListing.created_at.desc()).all()
    applications = JobApplication.query.order_by(JobApplication.created_at.desc()).limit(20).all()
    return render_template('hr_admin_jobs.html', listings=listings, applications=applications)


@app.route('/admin/jobs/new', methods=['GET', 'POST'])
def admin_jobs_new():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    error = None
    if request.method == 'POST':
        job = JobListing(
            title=request.form.get('title', '').strip(),
            company=request.form.get('company', 'InvestIQ').strip(),
            location=request.form.get('location', '').strip(),
            job_type=request.form.get('job_type', 'Full-Time'),
            sector=request.form.get('sector', 'Finance').strip(),
            description=request.form.get('description', '').strip(),
            requirements=request.form.get('requirements', '').strip(),
            salary_range=request.form.get('salary_range', '').strip(),
        )
        if not job.title:
            error = 'Title is required.'
        else:
            db.session.add(job)
            db.session.commit()
            return redirect(url_for('admin_jobs'))
    return render_template('hr_admin_job_form.html', job=None, error=error)


@app.route('/admin/jobs/<int:job_id>/edit', methods=['GET', 'POST'])
def admin_jobs_edit(job_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    job = JobListing.query.get_or_404(job_id)
    error = None
    if request.method == 'POST':
        job.title        = request.form.get('title', job.title).strip()
        job.company      = request.form.get('company', job.company).strip()
        job.location     = request.form.get('location', job.location).strip()
        job.job_type     = request.form.get('job_type', job.job_type)
        job.sector       = request.form.get('sector', job.sector).strip()
        job.description  = request.form.get('description', job.description).strip()
        job.requirements = request.form.get('requirements', job.requirements).strip()
        job.salary_range = request.form.get('salary_range', job.salary_range).strip()
        job.is_active    = bool(request.form.get('is_active'))
        if not job.title:
            error = 'Title is required.'
        else:
            db.session.commit()
            return redirect(url_for('admin_jobs'))
    return render_template('hr_admin_job_form.html', job=job, error=error)


@app.route('/admin/jobs/<int:job_id>/delete', methods=['POST'])
def admin_jobs_delete(job_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    job = JobListing.query.get_or_404(job_id)
    db.session.delete(job)
    db.session.commit()
    return redirect(url_for('admin_jobs'))


@app.route('/admin/training')
def admin_training():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    bookings = TrainingBooking.query.order_by(TrainingBooking.created_at.desc()).all()
    return render_template('hr_admin_training.html', bookings=bookings)


# ============================================================
# SUPER ADMIN — Training CSV + Linked Candidates CSV
# ============================================================

def _super_admin_required():
    """Return redirect if super admin not logged in, else None."""
    if not session.get('super_admin_logged_in') and not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    return None


@app.route('/super-admin/login', methods=['GET', 'POST'])
def super_admin_login():
    return redirect(url_for('admin_login'))


@app.route('/super-admin/logout')
def super_admin_logout():
    return redirect(url_for('admin_logout'))


@app.route('/super-admin')
def super_admin_dashboard():
    redir = _super_admin_required()
    if redir:
        return redir

    training_count  = TrainingBooking.query.count()
    employer_count  = EmployerAccount.query.count()
    candidate_count = CandidateProfile.query.count()

    linked_ids = set(
        [r[0] for r in db.session.query(EmployerShortlist.candidate_id).distinct().all()] +
        [r[0] for r in db.session.query(EmployerInquiry.candidate_id).distinct().all()]
    )
    linked_count = len(linked_ids)

    referral_count  = db.session.query(Referral.referrer_email).distinct().count()
    successful_refs = Referral.query.filter_by(status='Successful').count()

    # All signups (deduplicated by email)
    all_emails = set()
    for u in SiteUser.query.with_entities(SiteUser.email).all():          all_emails.add(u[0].strip().lower())
    for e in EmployerAccount.query.with_entities(EmployerAccount.email).all(): all_emails.add(e[0].strip().lower())
    for c in CandidateProfile.query.with_entities(CandidateProfile.email).all(): all_emails.add(c[0].strip().lower())
    for b in TrainingBooking.query.with_entities(TrainingBooking.email).all(): all_emails.add(b[0].strip().lower())
    for a in JobApplication.query.with_entities(JobApplication.email).all(): all_emails.add(a[0].strip().lower())
    total_signups = len(all_emails)

    return render_template('hr_super_admin.html',
                           training_count=training_count,
                           employer_count=employer_count,
                           candidate_count=candidate_count,
                           linked_count=linked_count,
                           referral_count=referral_count,
                           successful_refs=successful_refs,
                           total_signups=total_signups)


@app.route('/super-admin/training-csv')
def super_admin_training_csv():
    redir = _super_admin_required()
    if redir:
        return redir

    import csv, io
    bookings = TrainingBooking.query.order_by(TrainingBooking.created_at.desc()).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'ID', 'Booking Type', 'Full Name', 'Email', 'Phone',
        'Organisation', 'No. of Participants', 'Training Program',
        'Preferred Date / Format', 'Additional Notes', 'Status', 'Submitted At'
    ])
    for b in bookings:
        writer.writerow([
            b.id,
            b.booking_type.title(),
            b.full_name,
            b.email,
            b.phone or '',
            b.organization or '',
            b.participants or 1,
            b.category or '',
            b.preferred_date or '',
            (b.notes or '').replace('\n', ' '),
            b.status or 'Pending',
            b.created_at.strftime('%Y-%m-%d %H:%M') if b.created_at else '',
        ])

    output = buf.getvalue()
    return (
        output,
        200,
        {
            'Content-Type': 'text/csv; charset=utf-8',
            'Content-Disposition': 'attachment; filename="training_schedules.csv"',
        }
    )


@app.route('/super-admin/survey-csv')
def super_admin_survey_csv():
    redir = _super_admin_required()
    if redir:
        return redir
    import csv, io
    rows = CVSurveyResponse.query.order_by(CVSurveyResponse.created_at.desc()).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'ID', 'Full Name', 'Email',
        'YIN Member?', 'Stock Pitch Participant?',
        'Wants Internship?', 'Wants National Service?', 'Submitted At'
    ])
    for r in rows:
        writer.writerow([
            r.id, r.full_name, r.email,
            r.yin_member, r.stock_pitch,
            r.want_internship, r.want_national_service,
            r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else '',
        ])
    output = buf.getvalue()
    return (output, 200, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="cv_survey_responses.csv"',
    })


@app.route('/mentorship', methods=['GET', 'POST'])
def mentorship_page():
    success = False
    error = None
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email     = request.form.get('email', '').strip()
        if not full_name or not email:
            error = 'Full name and email are required.'
        else:
            try:
                app_entry = MentorshipApplication(
                    full_name     = full_name,
                    email         = email,
                    phone         = request.form.get('phone', '').strip(),
                    institution   = request.form.get('institution', '').strip(),
                    program       = request.form.get('program', '').strip(),
                    year_of_study = request.form.get('year_of_study', '').strip(),
                    interest_area = request.form.get('interest_area', '').strip(),
                    availability  = request.form.get('availability', '').strip(),
                    why_mentorship= request.form.get('why_mentorship', '').strip(),
                    linkedin      = request.form.get('linkedin', '').strip(),
                )
                db.session.add(app_entry)
                db.session.commit()
                success = True
            except Exception as e:
                logger.error(f'Mentorship application error: {e}')
                error = 'Something went wrong. Please try again.'
    return render_template('mentorship.html', success=success, error=error)


@app.route('/super-admin/mentorship-csv')
def super_admin_mentorship_csv():
    redir = _super_admin_required()
    if redir:
        return redir
    import csv, io
    rows = MentorshipApplication.query.order_by(MentorshipApplication.created_at.desc()).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'ID', 'Full Name', 'Email', 'Phone', 'Institution',
        'Program / Degree', 'Year of Study', 'Interest Area',
        'Availability', 'Why Mentorship', 'LinkedIn', 'Status', 'Applied At'
    ])
    for r in rows:
        writer.writerow([
            r.id, r.full_name, r.email, r.phone,
            r.institution, r.program, r.year_of_study,
            r.interest_area, r.availability,
            (r.why_mentorship or '').replace('\n', ' '),
            r.linkedin, r.status,
            r.created_at.strftime('%Y-%m-%d %H:%M') if r.created_at else '',
        ])
    output = buf.getvalue()
    return (output, 200, {
        'Content-Type': 'text/csv; charset=utf-8',
        'Content-Disposition': 'attachment; filename="mentorship_applications.csv"',
    })


@app.route('/super-admin/linked-candidates-csv')
def super_admin_linked_candidates_csv():
    redir = _super_admin_required()
    if redir:
        return redir

    import csv, io

    # All candidates with at least one shortlist or inquiry
    shortlisted_ids = {r[0] for r in db.session.query(EmployerShortlist.candidate_id).distinct().all()}
    inquired_ids    = {r[0] for r in db.session.query(EmployerInquiry.candidate_id).distinct().all()}
    linked_ids      = shortlisted_ids | inquired_ids

    candidates = CandidateProfile.query.filter(CandidateProfile.id.in_(linked_ids)).all()

    # Build employer name lookups
    def shortlisting_companies(cid):
        rows = (db.session.query(EmployerAccount.company_name)
                .join(EmployerShortlist, EmployerShortlist.employer_id == EmployerAccount.id)
                .filter(EmployerShortlist.candidate_id == cid)
                .all())
        return '; '.join(r[0] for r in rows)

    def inquiring_companies(cid):
        rows = (db.session.query(EmployerInquiry.employer_company)
                .filter(EmployerInquiry.candidate_id == cid,
                        EmployerInquiry.employer_company != '')
                .all())
        return '; '.join(r[0] for r in rows)

    def latest_activity(cid):
        sl = (EmployerShortlist.query
              .filter_by(candidate_id=cid)
              .order_by(EmployerShortlist.created_at.desc())
              .first())
        inq = (EmployerInquiry.query
               .filter_by(candidate_id=cid)
               .order_by(EmployerInquiry.created_at.desc())
               .first())
        dates = [d for d in [
            sl.created_at  if sl  else None,
            inq.created_at if inq else None,
        ] if d]
        return max(dates).strftime('%Y-%m-%d %H:%M') if dates else ''

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'ID', 'Full Name', 'Email', 'Phone', 'Location',
        'Current Title', 'Desired Role', 'Sector',
        'Years Experience', 'Availability', 'Skills Summary',
        'Profile Visible', 'Profile Created',
        'Shortlisted By (Companies)', 'Inquired By (Companies)',
        'Latest Employer Activity'
    ])
    for c in candidates:
        writer.writerow([
            c.id,
            c.full_name,
            c.email,
            c.phone or '',
            c.location or '',
            c.current_title or '',
            c.desired_role or '',
            c.desired_sector or '',
            c.years_exp or '',
            c.availability or '',
            (c.skills_summary or '').replace('\n', ' '),
            'Yes' if c.is_visible else 'No',
            c.created_at.strftime('%Y-%m-%d') if c.created_at else '',
            shortlisting_companies(c.id),
            inquiring_companies(c.id),
            latest_activity(c.id),
        ])

    output = buf.getvalue()
    return (
        output,
        200,
        {
            'Content-Type': 'text/csv; charset=utf-8',
            'Content-Disposition': 'attachment; filename="candidates_linked_to_employers.csv"',
        }
    )


@app.route('/super-admin/referrals-csv')
def super_admin_referrals_csv():
    redir = _super_admin_required()
    if redir:
        return redir

    import csv, io

    # Aggregate per referrer_email
    referrers = db.session.query(Referral.referrer_email).distinct().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        'Referrer Name', 'Referrer Email', 'Referral Code',
        'Successful Referrals', 'Pending Referrals',
        'Cumulative Discount %', 'Effective Price Factor',
        'Referred People (Name — Email)', 'Last Referral Date',
    ])

    for (email,) in referrers:
        rows = Referral.query.filter_by(referrer_email=email).all()
        if not rows:
            continue
        referrer_name = rows[0].referrer_name or ''
        referral_code = rows[0].referral_code or ''
        successful    = [r for r in rows if r.status == 'Successful']
        pending       = [r for r in rows if r.status == 'Pending']
        n             = len(successful)
        # Compound discount: each successful referral = 10% off remaining price
        price_factor  = round(0.9 ** n, 6)
        discount_pct  = round((1 - price_factor) * 100, 2)
        referred_list = '; '.join(
            f"{r.referred_name} — {r.referred_email}"
            for r in successful if r.referred_name or r.referred_email
        )
        dates = [r.created_at for r in successful if r.created_at]
        last_date = max(dates).strftime('%Y-%m-%d') if dates else ''
        writer.writerow([
            referrer_name, email, referral_code,
            n, len(pending),
            f'{discount_pct}%', price_factor,
            referred_list, last_date,
        ])

    output = buf.getvalue()
    return (
        output, 200,
        {
            'Content-Type': 'text/csv; charset=utf-8',
            'Content-Disposition': 'attachment; filename="referrals.csv"',
        }
    )


@app.route('/super-admin/all-signups-csv')
def super_admin_all_signups_csv():
    redir = _super_admin_required()
    if redir:
        return redir

    import csv, io

    # Collect from all 5 sources; deduplicate by email (keep earliest)
    seen   = {}   # email -> row dict

    def add(source, name, email, phone, signed_up):
        key = (email or '').strip().lower()
        if not key:
            return
        if key not in seen:
            seen[key] = {
                'source': source, 'name': name, 'email': email,
                'phone': phone or '', 'signed_up': signed_up,
            }
        else:
            # Add source if seen from multiple places
            if source not in seen[key]['source']:
                seen[key]['source'] += f' / {source}'

    for u in SiteUser.query.order_by(SiteUser.created_at).all():
        add('Site User', u.full_name, u.email, u.phone,
            u.created_at.strftime('%Y-%m-%d %H:%M') if u.created_at else '')

    for e in EmployerAccount.query.order_by(EmployerAccount.created_at).all():
        add('Employer', f'{e.company_name} ({e.contact_name})', e.email, e.phone,
            e.created_at.strftime('%Y-%m-%d %H:%M') if e.created_at else '')

    for c in CandidateProfile.query.order_by(CandidateProfile.created_at).all():
        add('Candidate (CV)', c.full_name, c.email, c.phone,
            c.created_at.strftime('%Y-%m-%d') if c.created_at else '')

    for b in TrainingBooking.query.order_by(TrainingBooking.created_at).all():
        add('Training Booking', b.full_name, b.email, b.phone,
            b.created_at.strftime('%Y-%m-%d %H:%M') if b.created_at else '')

    for a in JobApplication.query.order_by(JobApplication.created_at).all():
        add('Job Applicant', a.full_name, a.email, a.phone,
            a.created_at.strftime('%Y-%m-%d %H:%M') if a.created_at else '')

    # Sort by signup date
    rows = sorted(seen.values(), key=lambda r: r['signed_up'])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['Source', 'Full Name', 'Email', 'Phone', 'Signed Up At'])
    for r in rows:
        writer.writerow([r['source'], r['name'], r['email'], r['phone'], r['signed_up']])

    output = buf.getvalue()
    return (
        output, 200,
        {
            'Content-Type': 'text/csv; charset=utf-8',
            'Content-Disposition': 'attachment; filename="all_signups.csv"',
        }
    )


# ============================================================
# MOBILE JSON API  (consumed by InvestIQ Mobile app)
# ============================================================
import hashlib as _hashlib_api, secrets as _secrets_api

def _mobile_token(user):
    raw = f"{user.id}:{user.email}:{SUPER_ADMIN_PASSWORD}"
    return _hashlib_api.sha256(raw.encode()).hexdigest()

def _auth_from_request():
    """Return SiteUser or None from Authorization: Bearer <token> header."""
    hdr = request.headers.get('Authorization', '')
    if not hdr.startswith('Bearer '):
        return None
    token = hdr[7:]
    for u in SiteUser.query.all():
        if _mobile_token(u) == token:
            return u
    return None

def _j(data, status=200):
    import json as _j_json
    return app.response_class(_j_json.dumps(data), status=status,
                              mimetype='application/json')

@app.route('/api/auth/register', methods=['POST'])
def api_auth_register():
    data = request.get_json() or {}
    full_name = (data.get('full_name') or '').strip()
    email     = (data.get('email') or '').strip().lower()
    password  = data.get('password', '')
    if not full_name or not email or not password:
        return _j({'ok': False, 'error': 'Name, email and password are required.'}, 400)
    if len(password) < 6:
        return _j({'ok': False, 'error': 'Password must be at least 6 characters.'}, 400)
    if SiteUser.query.filter(db.func.lower(SiteUser.email) == email).first():
        return _j({'ok': False, 'error': 'An account with this email already exists.'}, 409)
    u = SiteUser(full_name=full_name, email=email)
    u.set_password(password)
    db.session.add(u)
    db.session.commit()
    return _j({'ok': True, 'token': _mobile_token(u),
               'user': {'id': u.id, 'full_name': u.full_name, 'email': u.email}})

@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    data     = request.get_json() or {}
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password', '')
    u = SiteUser.query.filter(db.func.lower(SiteUser.email) == email).first()
    if not u or not u.check_password(password):
        return _j({'ok': False, 'error': 'Invalid email or password.'}, 401)
    return _j({'ok': True, 'token': _mobile_token(u),
               'user': {'id': u.id, 'full_name': u.full_name, 'email': u.email}})


@app.route('/api/auth/forgot-password', methods=['POST'])
def api_auth_forgot_password():
    import secrets
    data  = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    if not email:
        return _j({'ok': False, 'error': 'Email is required.'}, 400)
    u = SiteUser.query.filter(db.func.lower(SiteUser.email) == email).first()
    if u:
        token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(hours=1)
        db.session.add(PasswordResetToken(user_id=u.id, token=token, expires_at=expires))
        db.session.commit()
        reset_url = f'https://investright.onrender.com/reset-password/{token}'
        send_email_safe(
            subject='Reset your InvestIQ password',
            recipients=[u.email],
            body_html=f'''<p>Hi {u.full_name},</p>
<p>Click the link below to reset your password. This link expires in 1 hour.</p>
<p><a href="{reset_url}" style="background:#2563eb;color:#fff;padding:12px 24px;border-radius:8px;text-decoration:none;font-weight:700;">Reset Password</a></p>
<p>If you did not request this, ignore this email.</p>''',
            body_text=f'Reset your InvestIQ password: {reset_url}'
        )
    return _j({'ok': True, 'message': 'If that email is registered, a reset link has been sent.'})

@app.route('/api/auth/me', methods=['GET'])
def api_auth_me():
    u = _auth_from_request()
    if not u:
        return _j({'ok': False, 'error': 'Unauthenticated.'}, 401)
    return _j({'ok': True, 'user': {'id': u.id, 'full_name': u.full_name, 'email': u.email}})

@app.route('/api/jobs', methods=['GET'])
def api_jobs():
    jobs = JobListing.query.filter_by(is_active=True).order_by(JobListing.created_at.desc()).all()
    return _j({'ok': True, 'jobs': [{
        'id': j.id, 'title': j.title, 'company': j.company,
        'location': j.location, 'job_type': j.job_type, 'sector': j.sector,
        'description': j.description, 'requirements': j.requirements,
        'salary_range': j.salary_range,
        'posted': j.created_at.strftime('%d %b %Y') if j.created_at else '',
    } for j in jobs]})

@app.route('/api/jobs/<int:job_id>/apply', methods=['POST'])
def api_apply_job(job_id):
    job  = JobListing.query.filter_by(id=job_id, is_active=True).first()
    if not job:
        return _j({'ok': False, 'error': 'Job not found.'}, 404)
    data = request.get_json() or {}
    if not data.get('full_name') or not data.get('email'):
        return _j({'ok': False, 'error': 'Name and email are required.'}, 400)
    app_obj = JobApplication(
        job_id=job_id,
        full_name=data.get('full_name', '').strip(),
        email=data.get('email', '').strip().lower(),
        phone=data.get('phone', '').strip(),
        cover_letter=data.get('cover_letter', '').strip(),
    )
    db.session.add(app_obj)
    db.session.commit()
    return _j({'ok': True, 'message': 'Application submitted successfully!'})

@app.route('/api/candidates', methods=['GET'])
def api_candidates():
    sector = request.args.get('sector', '')
    exp    = request.args.get('exp', '')
    avail  = request.args.get('avail', '')
    q = CandidateProfile.query.filter_by(is_visible=True)
    if sector: q = q.filter(CandidateProfile.desired_sector == sector)
    if exp:    q = q.filter(CandidateProfile.years_exp == exp)
    if avail:  q = q.filter(CandidateProfile.availability == avail)
    cands = q.order_by(CandidateProfile.created_at.desc()).limit(100).all()
    return _j({'ok': True, 'candidates': [{
        'id': c.id, 'full_name': c.full_name,
        'current_title': c.current_title, 'desired_role': c.desired_role,
        'desired_sector': c.desired_sector, 'years_exp': c.years_exp,
        'availability': c.availability, 'location': c.location,
        'skills_summary': c.skills_summary, 'profile_summary': c.profile_summary,
        'linkedin': c.linkedin,
        'initials': ''.join(w[0].upper() for w in (c.full_name or 'U').split()[:2]),
    } for c in cands]})

@app.route('/api/candidates/<int:cid>/inquire', methods=['POST'])
def api_candidate_inquire(cid):
    c = CandidateProfile.query.filter_by(id=cid, is_visible=True).first()
    if not c:
        return _j({'ok': False, 'error': 'Candidate not found.'}, 404)
    data = request.get_json() or {}
    if not data.get('employer_name') or not data.get('employer_email'):
        return _j({'ok': False, 'error': 'Your name and email are required.'}, 400)
    inq = EmployerInquiry(
        candidate_id=cid,
        employer_name=data.get('employer_name', '').strip(),
        employer_email=data.get('employer_email', '').strip(),
        employer_company=data.get('employer_company', '').strip(),
        employer_phone=data.get('employer_phone', '').strip(),
        inquiry_type=data.get('inquiry_type', 'Interview Request'),
        role_offering=data.get('role_offering', '').strip(),
        message=data.get('message', '').strip(),
    )
    db.session.add(inq)
    db.session.commit()
    return _j({'ok': True, 'message': f'Inquiry sent to {c.full_name}.'})

@app.route('/api/training/programs', methods=['GET'])
def api_training_programs():
    programs = [
        {'category': 'Finance', 'title': 'Financial Modelling & DCF', 'duration': '3 Days'},
        {'category': 'Finance', 'title': 'CFA Exam Preparation', 'duration': '12 Weeks'},
        {'category': 'Finance', 'title': 'Investment Banking', 'duration': '5 Days'},
        {'category': 'Finance', 'title': 'Risk Management', 'duration': '2 Days'},
        {'category': 'Finance', 'title': 'Portfolio Management', 'duration': '3 Days'},
        {'category': 'Finance', 'title': 'ACCA / ICAG Coaching', 'duration': 'Ongoing'},
        {'category': 'Technology', 'title': 'Python for Finance', 'duration': '4 Days'},
        {'category': 'Technology', 'title': 'SQL & Data Analytics', 'duration': '3 Days'},
        {'category': 'Technology', 'title': 'Machine Learning', 'duration': '5 Days'},
        {'category': 'Technology', 'title': 'Cloud Computing (AWS/Azure)', 'duration': '4 Days'},
        {'category': 'Technology', 'title': 'Cybersecurity Essentials', 'duration': '3 Days'},
        {'category': 'Healthcare', 'title': 'Clinical Research Methods', 'duration': '5 Days'},
        {'category': 'Healthcare', 'title': 'Healthcare Management', 'duration': '3 Days'},
        {'category': 'Legal', 'title': 'Corporate Law Fundamentals', 'duration': '3 Days'},
        {'category': 'Legal', 'title': 'GDPR & Data Privacy', 'duration': '1 Day'},
        {'category': 'HR', 'title': 'HR Management Professional', 'duration': '3 Days'},
        {'category': 'HR', 'title': 'Leadership & Management', 'duration': '3 Days'},
        {'category': 'Engineering', 'title': 'Project Management (PMP/PRINCE2)', 'duration': '5 Days'},
        {'category': 'Engineering', 'title': 'Lean Six Sigma', 'duration': '5 Days'},
        {'category': 'Marketing', 'title': 'Digital Marketing', 'duration': '3 Days'},
        {'category': 'Business', 'title': 'Business Strategy & Innovation', 'duration': '3 Days'},
        {'category': 'Business', 'title': 'Entrepreneurship & Startups', 'duration': '2 Days'},
        {'category': 'Other', 'title': 'Custom / Other', 'duration': 'Flexible'},
    ]
    return _j({'ok': True, 'programs': programs})

@app.route('/api/training/book', methods=['POST'])
def api_training_book():
    data = request.get_json() or {}
    if not data.get('full_name') or not data.get('email'):
        return _j({'ok': False, 'error': 'Name and email are required.'}, 400)
    import hashlib as _h
    def _code(email): return _h.md5(email.strip().lower().encode()).hexdigest()[:8].upper()

    b = TrainingBooking(
        booking_type=data.get('booking_type', 'individual'),
        full_name=data.get('full_name', '').strip(),
        email=data.get('email', '').strip().lower(),
        phone=data.get('phone', '').strip(),
        organization=data.get('organization', '').strip(),
        participants=int(data.get('participants', 1) or 1),
        category=data.get('category', '').strip(),
        other_program=data.get('other_program', '').strip(),
        preferred_date=data.get('preferred_date', '').strip(),
        notes=data.get('notes', '').strip(),
        referral_code_used=(data.get('referral_code') or '').strip().upper(),
    )
    db.session.add(b)
    db.session.flush()

    ref_used = b.referral_code_used
    if ref_used:
        existing = Referral.query.filter_by(referral_code=ref_used).first()
        if existing:
            db.session.add(Referral(
                referrer_name=existing.referrer_name, referrer_email=existing.referrer_email,
                referral_code=ref_used, referred_name=b.full_name, referred_email=b.email,
                booking_id=b.id, status='Successful',
            ))
    my_code = _code(b.email)
    if not Referral.query.filter_by(referral_code=my_code).first():
        db.session.add(Referral(referrer_name=b.full_name, referrer_email=b.email,
                                referral_code=my_code, status='Pending'))
    db.session.commit()
    return _j({'ok': True, 'message': 'Booking submitted! We\'ll contact you within 24 hours.',
               'referral_code': my_code})

@app.route('/api/cv/submit', methods=['POST'])
def api_cv_submit():
    data = request.get_json() or {}
    if not data.get('full_name') or not data.get('email'):
        return _j({'ok': False, 'error': 'Full name and email are required.'}, 400)
    # Upsert candidate profile
    email = data.get('email', '').strip().lower()
    c = CandidateProfile.query.filter(db.func.lower(CandidateProfile.email) == email).first()
    if not c:
        c = CandidateProfile(email=email, full_name=data.get('full_name', '').strip())
        db.session.add(c)
    c.full_name      = data.get('full_name', c.full_name).strip()
    c.phone          = data.get('phone', c.phone or '')
    c.location       = data.get('location', c.location or '')
    c.current_title  = data.get('current_title', c.current_title or '')
    c.desired_role   = data.get('desired_role', c.desired_role or '')
    c.desired_sector = data.get('desired_sector', c.desired_sector or '')
    c.years_exp      = data.get('years_exp', c.years_exp or '')
    c.availability   = data.get('availability', c.availability or '')
    c.skills_summary = data.get('skills_summary', c.skills_summary or '')
    c.profile_summary= data.get('profile_summary', c.profile_summary or '')
    c.linkedin       = data.get('linkedin', c.linkedin or '')
    c.is_visible     = bool(data.get('opt_in', True))
    db.session.commit()
    return _j({'ok': True, 'message': 'Profile saved and visible to employers!'})


# ── YIN PROGRAMS ──────────────────────────────────────────────────────────────

import re as _re

def _normalize_phone(phone):
    """Strip formatting and convert +233/233 prefix to leading 0."""
    p = _re.sub(r'[\s\-\(\)\.]', '', phone.strip())
    if p.startswith('+233'):
        p = '0' + p[4:]
    elif p.startswith('233') and len(p) == 12:
        p = '0' + p[3:]
    return p

def _validate_phone(phone):
    """Return (valid:bool, normalized:str). Must be 10 digits starting with 0."""
    p = _normalize_phone(phone)
    if _re.match(r'^0[2359]\d{8}$', p):
        return True, p
    return False, p

def _validate_email(email):
    """Basic structural email check."""
    return bool(_re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email.strip()))

def _yin_duplicate_check(email, phone_norm):
    """Return existing registration if email OR phone already registered."""
    q = YINRegistration.query.filter(
        db.or_(
            YINRegistration.email == email.lower().strip(),
            YINRegistration.phone == phone_norm
        )
    ).first()
    return q

def _generate_yin_code():
    """Return next unique YIN code.
    Reads every existing code, builds the full set of used numbers,
    then returns max+1 — guaranteed unique even after dedup deletes entries."""
    rows = db.session.query(YINRegistration.yin_code).filter(
        YINRegistration.yin_code.like('YIN%')
    ).all()
    used = set()
    for (code,) in rows:
        if code and len(code) > 3 and code[3:].isdigit():
            used.add(int(code[3:]))
    next_n = (max(used) + 1) if used else 1
    return f'YIN{next_n:04d}'


@app.route('/yin-register', methods=['GET', 'POST'])
def yin_register_page():
    programs = YINProgram.query.filter_by(is_active=True).order_by(YINProgram.created_at.desc()).all()

    if request.method == 'GET':
        return render_template('hr_yin_register.html', programs=programs)

    # ── POST: process registration ──
    prog_id_str  = request.form.get('prog_select', '').strip()
    full_name    = request.form.get('full_name', '').strip()
    phone        = request.form.get('phone', '').strip()
    email        = request.form.get('email', '').strip().lower()
    institution  = request.form.get('institution', '').strip()
    inst_type    = request.form.get('institution_type', '').strip()
    how_heard    = request.form.get('how_heard', '').strip()
    is_existing  = request.form.get('is_existing_member') == 'yes'
    existing_code = request.form.get('existing_yin_code', '').strip().upper()
    confirmed    = request.form.get('confirmed') == 'on'

    # ── Validate required fields ──
    if not full_name or not email or not phone:
        return render_template('hr_yin_register.html', programs=programs,
                               error='Full name, email, and phone number are required.')

    if not _validate_email(email):
        return render_template('hr_yin_register.html', programs=programs,
                               error='Please enter a valid email address (e.g. name@example.com).')

    phone_ok, phone_norm = _validate_phone(phone)
    if not phone_ok:
        return render_template('hr_yin_register.html', programs=programs,
                               error='Please enter a valid Ghana phone number — 10 digits starting with 0 (e.g. 0244123456).')

    # ── Duplicate check ──
    existing_reg = _yin_duplicate_check(email, phone_norm)
    if existing_reg:
        return render_template('hr_yin_register.html', programs=programs,
                               error=f'You are already registered. Your YIN code is {existing_reg.yin_code}. '
                                     f'Check your records or contact admin if you think this is an error.')

    # Resolve program; fall back to first active, or create a General one
    prog = None
    if prog_id_str and prog_id_str.isdigit():
        prog = YINProgram.query.get(int(prog_id_str))
    if prog is None:
        prog = YINProgram.query.filter_by(is_active=True).first()
    if prog is None:
        prog = YINProgram(name='General Membership', description='General YIN membership registration.', is_active=True)
        db.session.add(prog)
        db.session.flush()

    prog_name = prog.name

    # Generate or use existing YIN code
    yin_code = existing_code if is_existing else _generate_yin_code()

    reg = YINRegistration(
        program_id=prog.id,
        program_name=prog_name,
        full_name=full_name, phone=phone_norm, email=email.lower().strip(),
        institution=institution, institution_type=inst_type,
        how_heard=how_heard, is_existing_member=is_existing,
        yin_code=yin_code, confirmed=confirmed,
    )
    db.session.add(reg)
    db.session.commit()

    return render_template('hr_yin_register.html', programs=programs,
                           success=True, new_code=yin_code,
                           is_existing=is_existing, registered_prog=prog_name)


@app.route('/yin-programs')
def yin_programs():
    programs = YINProgram.query.filter_by(is_active=True).order_by(YINProgram.created_at.desc()).all()
    return render_template('hr_yin_programs.html', programs=programs)


@app.route('/yin-programs/<int:prog_id>/register', methods=['POST'])
def yin_register(prog_id):
    prog = YINProgram.query.get_or_404(prog_id)
    full_name        = request.form.get('full_name', '').strip()
    phone            = request.form.get('phone', '').strip()
    email            = request.form.get('email', '').strip().lower()
    institution      = request.form.get('institution', '').strip()
    institution_type = request.form.get('institution_type', '').strip()
    how_heard        = request.form.get('how_heard', '').strip()
    is_existing      = request.form.get('is_existing_member') == 'yes'
    existing_code    = request.form.get('existing_yin_code', '').strip().upper()
    confirmed        = request.form.get('confirmed') == 'on'

    _active_programs = YINProgram.query.filter_by(is_active=True).all()

    if not full_name or not email or not phone:
        return render_template('hr_yin_programs.html',
                               programs=_active_programs,
                               error='Full name, email, and phone number are required.',
                               open_prog_id=prog_id)

    if not _validate_email(email):
        return render_template('hr_yin_programs.html',
                               programs=_active_programs,
                               error='Please enter a valid email address (e.g. name@example.com).',
                               open_prog_id=prog_id)

    phone_ok, phone_norm = _validate_phone(phone)
    if not phone_ok:
        return render_template('hr_yin_programs.html',
                               programs=_active_programs,
                               error='Please enter a valid Ghana phone number (e.g. 0244123456 or 0541234567).',
                               open_prog_id=prog_id)

    existing_reg = _yin_duplicate_check(email, phone_norm)
    if existing_reg:
        return render_template('hr_yin_programs.html',
                               programs=_active_programs,
                               error=f'You are already registered (YIN Code: {existing_reg.yin_code}). '
                                     f'Contact us if you need assistance.',
                               open_prog_id=prog_id)

    yin_code = existing_code if is_existing else _generate_yin_code()

    reg = YINRegistration(
        program_id=prog.id, program_name=prog.name,
        full_name=full_name, phone=phone_norm, email=email,
        institution=institution, institution_type=institution_type,
        how_heard=how_heard, is_existing_member=is_existing,
        yin_code=yin_code, confirmed=confirmed,
    )
    db.session.add(reg)
    db.session.commit()
    return render_template('hr_yin_programs.html',
                           programs=_active_programs,
                           success=True, new_code=yin_code, is_existing=is_existing,
                           registered_prog=prog.name)


@app.route('/admin/yin-programs', methods=['GET', 'POST'])
def admin_yin_programs():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        desc = request.form.get('description', '').strip()
        if name:
            db.session.add(YINProgram(name=name, description=desc))
            db.session.commit()
    programs = YINProgram.query.order_by(YINProgram.created_at.desc()).all()
    reg_counts = {
        prog.id: YINRegistration.query.filter_by(program_id=prog.id).count()
        for prog in programs
    }
    dedup_deleted = request.args.get('dedup_deleted', type=int)
    phones_fixed  = request.args.get('phones_fixed', type=int)
    codes_fixed   = request.args.get('codes_fixed', type=int)
    return render_template('admin_yin_programs.html', programs=programs, reg_counts=reg_counts,
                           dedup_deleted=dedup_deleted, phones_fixed=phones_fixed,
                           codes_fixed=codes_fixed)


@app.route('/admin/yin-programs/<int:prog_id>/registrations-csv')
def admin_yin_program_csv(prog_id):
    if not session.get('admin_logged_in') and not session.get('super_admin_logged_in'):
        return redirect(url_for('admin_login'))
    prog = YINProgram.query.get_or_404(prog_id)
    rows = YINRegistration.query.filter_by(program_id=prog_id).order_by(YINRegistration.created_at.desc()).all()
    import csv, io
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(['ID','Program','Full Name','Phone','Email','Institution','Type','How Heard','Existing Member','YIN Code','Confirmed','Date'])
    for r in rows:
        w.writerow([r.id, r.program_name, r.full_name, r.phone, r.email,
                    r.institution, r.institution_type, r.how_heard,
                    'Yes' if r.is_existing_member else 'No',
                    r.yin_code, 'Yes' if r.confirmed else 'No',
                    r.created_at.strftime('%Y-%m-%d %H:%M')])
    out.seek(0)
    from flask import Response
    safe_name = prog.name.replace(' ', '_').replace('/', '-')[:40]
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment;filename=YIN_{safe_name}_registrations.csv'})


@app.route('/admin/yin-registrations/deduplicate', methods=['POST'])
def admin_yin_deduplicate():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    all_regs = YINRegistration.query.order_by(YINRegistration.id.asc()).all()
    seen_emails = {}
    seen_phones = {}
    to_delete = set()
    for reg in all_regs:
        email_key = (reg.email or '').lower().strip()
        phone_key = _normalize_phone(reg.phone) if reg.phone else ''
        duplicate = False
        if email_key and email_key in seen_emails:
            duplicate = True
        if phone_key and phone_key in seen_phones:
            duplicate = True
        if duplicate:
            to_delete.add(reg.id)
        else:
            if email_key:
                seen_emails[email_key] = reg.id
            if phone_key:
                seen_phones[phone_key] = reg.id
    deleted = 0
    for reg_id in to_delete:
        reg = YINRegistration.query.get(reg_id)
        if reg:
            db.session.delete(reg)
            deleted += 1
    db.session.commit()
    return redirect(url_for('admin_yin_programs', dedup_deleted=deleted))


@app.route('/admin/yin-registrations/normalize-phones', methods=['POST'])
def admin_yin_normalize_phones():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    all_regs = YINRegistration.query.all()
    fixed = 0
    for reg in all_regs:
        changed = False
        if reg.phone:
            normed = _normalize_phone(reg.phone)
            if normed != reg.phone:
                reg.phone = normed
                changed = True
        if reg.email:
            cleaned = reg.email.lower().strip()
            if cleaned != reg.email:
                reg.email = cleaned
                changed = True
        if changed:
            fixed += 1
    db.session.commit()
    return redirect(url_for('admin_yin_programs', phones_fixed=fixed))


@app.route('/admin/yin-registrations/fix-codes', methods=['POST'])
def admin_yin_fix_codes():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    all_regs = YINRegistration.query.order_by(YINRegistration.id.asc()).all()

    # Build set of already-used codes (first occurrence wins, duplicates flagged)
    seen = {}      # code -> first reg.id that owns it
    to_fix = []    # regs that need a new code (duplicate or missing)
    for reg in all_regs:
        code = (reg.yin_code or '').strip()
        if not code:
            to_fix.append(reg)
        elif code in seen:
            to_fix.append(reg)   # keep oldest owner, reassign this one
        else:
            seen[code] = reg.id

    # Collect all currently-used numeric values
    used_nums = set()
    for code in seen:
        if len(code) > 3 and code[3:].isdigit():
            used_nums.add(int(code[3:]))

    fixed = 0
    for reg in to_fix:
        n = (max(used_nums) + 1) if used_nums else 1
        used_nums.add(n)
        reg.yin_code = f'YIN{n:04d}'
        fixed += 1

    db.session.commit()
    return redirect(url_for('admin_yin_programs', codes_fixed=fixed))


@app.route('/admin/yin-programs/<int:prog_id>/toggle', methods=['POST'])
def admin_yin_toggle(prog_id):
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))
    prog = YINProgram.query.get_or_404(prog_id)
    prog.is_active = not prog.is_active
    db.session.commit()
    return redirect(url_for('admin_yin_programs'))


@app.route('/super-admin/yin-registrations-csv')
def super_admin_yin_csv():
    if not session.get('admin_logged_in') and not session.get('super_admin_logged_in'):
        return redirect(url_for('admin_login'))
    rows = YINRegistration.query.order_by(YINRegistration.created_at.desc()).all()
    import csv, io
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(['ID','Program','Full Name','Phone','Email','Institution','Type','How Heard','Existing Member','YIN Code','Confirmed','Date'])
    for r in rows:
        w.writerow([r.id, r.program_name, r.full_name, r.phone, r.email,
                    r.institution, r.institution_type, r.how_heard,
                    'Yes' if r.is_existing_member else 'No',
                    r.yin_code, 'Yes' if r.confirmed else 'No',
                    r.created_at.strftime('%Y-%m-%d %H:%M')])
    out.seek(0)
    from flask import Response
    return Response(out.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment;filename=yin_registrations.csv'})


@app.route('/api/mentorship/apply', methods=['POST'])
def api_mentorship_apply():
    data = request.get_json() or {}
    if not data.get('full_name') or not data.get('email'):
        return _j({'ok': False, 'error': 'Full name and email are required.'}, 400)
    if not data.get('interest_area'):
        return _j({'ok': False, 'error': 'Please select an area of interest.'}, 400)
    app_rec = MentorshipApplication(
        full_name    = data.get('full_name', '').strip(),
        email        = data.get('email', '').strip().lower(),
        phone        = data.get('phone', ''),
        institution  = data.get('institution', ''),
        program      = data.get('program', ''),
        year_of_study= data.get('year_of_study', ''),
        interest_area= data.get('interest_area', ''),
        availability = data.get('availability', ''),
        why_mentorship= data.get('why_mentorship', ''),
        linkedin     = data.get('linkedin', ''),
        status       = 'pending',
    )
    db.session.add(app_rec)
    db.session.commit()
    return _j({'ok': True, 'message': 'Application submitted. We will be in touch within 5 business days.'})


@app.route('/api/survey/submit', methods=['POST'])
def api_survey_submit():
    data = request.get_json() or {}
    rec = CVSurveyResponse(
        full_name           = data.get('full_name', ''),
        email               = data.get('email', '').strip().lower(),
        yin_member          = data.get('yin_member', 'no').capitalize(),
        stock_pitch         = data.get('stock_pitch', 'no').capitalize(),
        want_internship     = data.get('want_internship', 'no').capitalize(),
        want_national_service= data.get('want_national_service', 'no').capitalize(),
    )
    db.session.add(rec)
    db.session.commit()
    return _j({'ok': True, 'message': 'Survey responses recorded. Thank you!'})


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

# --- DATABASE INITIALIZATION ---
if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise

# --- APPLICATION RUNNER ---
if __name__ == '__main__':
    import socket
    
    # Get the actual IP address dynamically
    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip
    
    your_ip = get_ip()
    
    print("\n" + "="*60)
    print("💰 INVESTMENT CALCULATOR APPLICATION")
    print("="*60)
    print("Starting server...")
    print(f"Local access: http://127.0.0.1:5000")
    print(f"Network access: http://{your_ip}:5000")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # CRITICAL: Use '0.0.0.0' to allow network access
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("\n💡 Troubleshooting steps:")
        print("1. Check if port 5000 is in use: netstat -ano | findstr :5000")
        print("2. Try a different port: Change port=5000 to port=5001")
        print("3. Make sure all dependencies are installed: pip install -r requirements.txt")