# Investment Calculator Flask Application
# File: app.py
# Description: A Flask web application for financial calculations including DCF, FCFE, FCFF, credit risk,
#              leverage ratios, and more, with SQLAlchemy for database integration and WTForms for input validation.

# --- STANDARD LIBRARY IMPORTS ---
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# --- THIRD-PARTY IMPORTS ---
from flask import Flask, jsonify, render_template, request, send_from_directory, session, redirect, url_for, make_response, flash, abort
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

app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'e1efa2b32b1bac66588d074bac02a168212082d8befd0b6466f5ee37a8c2836a'),
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50 MB limit for book uploads
    SESSION_TYPE='filesystem',
    SESSION_FILE_THRESHOLD=500,
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=86400,
    WTF_CSRF_TIME_LIMIT=7200,
    SQLALCHEMY_DATABASE_URI='sqlite:///site.db',
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
    preferred_date  = db.Column(db.String(100), default='')
    notes           = db.Column(db.Text, default='')
    status          = db.Column(db.String(50), default='Pending')
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


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
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    users = SiteUser.query.order_by(SiteUser.created_at.desc()).all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/users/export')
def admin_users_export():
    if not session.get('admin'):
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
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    books = Book.query.order_by(Book.created_at.desc()).all()
    return render_template('admin_books.html', books=books)


@app.route('/admin/books/new', methods=['GET', 'POST'])
def admin_book_new():
    if not session.get('admin'):
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
    if not session.get('admin'):
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
    if not session.get('admin'):
        return redirect(url_for('admin_login'))
    book = Book.query.get_or_404(book_id)
    db.session.delete(book)
    db.session.commit()
    return redirect(url_for('admin_books'))


@app.route('/admin/book-requests')
def admin_book_requests():
    if not session.get('admin'):
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
    if not session.get('admin'):
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

@app.route('/portfolio-risk')
@app.route('/portfolio_risks')
def portfolio_risks():
    return render_template('portfolio_risks.html', result=None)

@app.route('/portfolio-return')
@app.route('/portfolio_return')
def portfolio_return():
    return render_template('portfolio_return.html')

@app.route('/volatility')
def volatility():
    return render_template('volatility.html')

@app.route('/risk-calculator')
@app.route('/risk_calculator')
def risk_calculator():
    return render_template('risk_calculator.html')

@app.route('/risk-assessment')
@app.route('/risk_assessment')
def risk_assessment():
    return render_template('risk_assessment.html')

@app.route('/portfolio-diversification')
@app.route('/portfolio_diversification')
def portfolio_diversification():
    return render_template('portfolio_diversification.html')

@app.route('/expected-return')
@app.route('/expected_return')
def expected_return():
    return render_template('expected_return.html')

@app.route('/duration')
def duration():
    return render_template('duration.html')

@app.route('/bonds')
def bonds():
    return render_template('bonds.html')

@app.route('/cds')
def cds():
    return render_template('cds.html')

@app.route('/dvm')
def dvm():
    return render_template('dvm.html')

@app.route('/intrinsic-value')
@app.route('/intrinsic_value')
def intrinsic_value():
    return render_template('intrinsic_value.html', form=request.form or {}, result=None, error=None)

@app.route('/valuation-methods')
@app.route('/valuation_methods')
def valuation_methods():
    return render_template('valuation_methods.html')

@app.route('/multi-method-valuation')
@app.route('/multi_method_valuation')
def multi_method_valuation():
    return render_template('multi_method_valuation.html')

@app.route('/specialized-industry-multiples')
@app.route('/Specialized_Industry_Multiples')
def specialized_industry_multiples():
    return render_template('Specialized_Industry_Multiples.html')

@app.route('/valuation-performance-multiples')
@app.route('/Valuation_Performance_Multiples')
def valuation_performance_multiples():
    return render_template('Valuation_Performance_Multiples.html')

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
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'investiq2026admin')


def admin_required(f):
    """Decorator: redirect to /admin/login if not authenticated."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session.permanent = True
            return redirect(url_for('admin_dashboard'))
        error = 'Incorrect password. Please try again.'
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


@app.route('/admin')
@admin_required
def admin_dashboard():
    article_count = Article.query.count()
    video_count   = Video.query.count()
    published     = Article.query.filter_by(is_published=True).count()
    recent        = Article.query.order_by(Article.created_at.desc()).limit(5).all()
    job_count     = JobListing.query.filter_by(is_active=True).count()
    app_count     = JobApplication.query.count()
    return render_template('admin_dashboard.html', article_count=article_count,
                           video_count=video_count, published=published, recent=recent,
                           job_count=job_count, app_count=app_count)


# ---- ARTICLE CRUD ----

@app.route('/admin/articles')
@admin_required
def admin_articles():
    posts = Article.query.order_by(Article.created_at.desc()).all()
    return render_template('admin_articles.html', posts=posts)


@app.route('/admin/articles/new', methods=['GET', 'POST'])
@admin_required
def admin_article_new():
    if request.method == 'POST':
        title   = request.form.get('title', '').strip()
        summary = request.form.get('summary', '').strip()
        body    = request.form.get('body', '').strip()
        category = request.form.get('category', 'General').strip()
        thumbnail_url = request.form.get('thumbnail_url', '').strip()
        is_published  = request.form.get('is_published') == '1'

        if not title or not body:
            return render_template('admin_article_form.html', error='Title and body are required.',
                                   action='New', article=None)

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

    return render_template('admin_article_form.html', action='New', article=None, error=None)


@app.route('/admin/articles/<int:article_id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_article_edit(article_id):
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
    return render_template('admin_article_form.html', action='Edit', article=article, error=None)


@app.route('/admin/articles/<int:article_id>/delete', methods=['POST'])
@admin_required
def admin_article_delete(article_id):
    article = Article.query.get_or_404(article_id)
    db.session.delete(article)
    db.session.commit()
    logger.info(f'Admin deleted article {article_id}')
    return redirect(url_for('admin_articles'))


# ---- VIDEO CRUD ----

@app.route('/admin/videos')
@admin_required
def admin_videos():
    vids = Video.query.order_by(Video.created_at.desc()).all()
    return render_template('admin_videos.html', videos=vids)


@app.route('/admin/videos/new', methods=['GET', 'POST'])
@admin_required
def admin_video_new():
    if request.method == 'POST':
        title        = request.form.get('title', '').strip()
        youtube_url  = request.form.get('youtube_url', '').strip()
        description  = request.form.get('description', '').strip()
        is_featured  = request.form.get('is_featured') == '1'
        is_published = request.form.get('is_published') == '1'

        if not title or not youtube_url:
            return render_template('admin_video_form.html', action='New', video=None,
                                   error='Title and YouTube URL are required.')

        # If featuring this video, un-feature others
        if is_featured:
            Video.query.update({'is_featured': False})

        video = Video(title=title, youtube_url=youtube_url, description=description,
                      is_featured=is_featured, is_published=is_published)
        db.session.add(video)
        db.session.commit()
        logger.info(f'Admin added video: {title}')
        return redirect(url_for('admin_videos'))

    return render_template('admin_video_form.html', action='New', video=None, error=None)


@app.route('/admin/videos/<int:video_id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_video_edit(video_id):
    video = Video.query.get_or_404(video_id)
    if request.method == 'POST':
        video.title        = request.form.get('title', '').strip()
        video.youtube_url  = request.form.get('youtube_url', '').strip()
        video.description  = request.form.get('description', '').strip()
        video.is_featured  = request.form.get('is_featured') == '1'
        video.is_published = request.form.get('is_published') == '1'

        if video.is_featured:
            Video.query.filter(Video.id != video_id).update({'is_featured': False})

        db.session.commit()
        logger.info(f'Admin updated video {video_id}')
        return redirect(url_for('admin_videos'))
    return render_template('admin_video_form.html', action='Edit', video=video, error=None)


@app.route('/admin/videos/<int:video_id>/delete', methods=['POST'])
@admin_required
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
    'AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','JPM','GS','BAC',
    'BRK-B','V','MA','UNH','XOM','JNJ','PG','HD','NFLX','DIS',
    'BABA','TSM','NVO','ASML','SAP','SIE.DE','BHP','RIO','TTE.PA','SHEL',
]

_stock_cache = {'global': [], 'gse': [], 'ts': 0}

@app.route('/api/stocks/ticker')
def api_stocks_ticker():
    """Return rolling ticker data for global + GSE stocks (cached 5 min)."""
    import time, yfinance as yf
    now = time.time()
    if now - _stock_cache['ts'] < 300:
        return jsonify(_stock_cache)

    # GSE stocks
    gse_data = []
    try:
        r = http_requests.get('https://dev.kwayisi.org/apis/gse/live', timeout=8)
        if r.ok:
            for s in r.json():
                gse_data.append({
                    'symbol': s.get('name', s.get('code','')),
                    'price': s.get('price', 0),
                    'change': s.get('change', 0),
                    'change_pct': s.get('change_percent', s.get('pct', 0)),
                    'market': 'GSE',
                })
    except Exception:
        pass

    # Global stocks via yfinance
    global_data = []
    try:
        tickers = yf.Tickers(' '.join(GLOBAL_TICKERS))
        for sym in GLOBAL_TICKERS:
            try:
                info = tickers.tickers[sym].fast_info
                price = round(float(info.last_price), 2)
                prev = round(float(info.previous_close), 2)
                chg = round(price - prev, 2)
                pct = round((chg / prev * 100), 2) if prev else 0
                global_data.append({'symbol': sym, 'price': price, 'change': chg, 'change_pct': pct, 'market': 'GLOBAL'})
            except Exception:
                pass
    except Exception:
        pass

    _stock_cache.update({'global': global_data, 'gse': gse_data, 'ts': now})
    return jsonify({'global': global_data, 'gse': gse_data})


@app.route('/api/stocks/lookup')
def api_stock_lookup():
    """Return key fundamentals for a given ticker symbol."""
    symbol = request.args.get('symbol', '').upper().strip()
    market = request.args.get('market', 'global').lower()
    if not symbol:
        return jsonify({'error': 'No symbol'}), 400

    if market == 'gse':
        try:
            r = http_requests.get('https://dev.kwayisi.org/apis/gse/live', timeout=8)
            if r.ok:
                for s in r.json():
                    if s.get('name','').upper() == symbol or s.get('code','').upper() == symbol:
                        return jsonify({
                            'symbol': symbol,
                            'name': s.get('name', symbol),
                            'current_price': s.get('price', 0),
                            'shares_outstanding': s.get('shares', ''),
                            'traded_volume': s.get('volume', ''),
                            'market_cap': s.get('market_cap', ''),
                            'pe_ratio': s.get('pe', ''),
                            'eps': s.get('eps', ''),
                            'dividend_yield': s.get('dividend_yield', ''),
                            'sector': s.get('sector', ''),
                            'currency': 'GHS',
                            'exchange': 'GSE',
                        })
        except Exception:
            pass
        return jsonify({'error': 'GSE stock not found'}), 404
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
    cv_data = None
    error = None
    if request.method == 'POST':
        try:
            cv_data = {
                'full_name': request.form.get('full_name', '').strip(),
                'email': request.form.get('email', '').strip(),
                'phone': request.form.get('phone', '').strip(),
                'location': request.form.get('location', '').strip(),
                'linkedin': request.form.get('linkedin', '').strip(),
                'website': request.form.get('website', '').strip(),
                'summary': request.form.get('summary', '').strip(),
                'skills': request.form.get('skills', '').strip(),
                'projects': request.form.get('projects', '').strip(),
                'awards': request.form.get('awards', '').strip(),
                'memberships': request.form.get('memberships', '').strip(),
                'template': request.form.get('template', 'professional'),
                'work_experiences': [],
                'educations': [],
                'certifications': [],
                'languages': [],
            }
            # Parse dynamic work experience entries
            i = 0
            while True:
                title = request.form.get(f'work_experiences-{i}-job_title', '')
                if not title and i > 0:
                    break
                if title:
                    cv_data['work_experiences'].append({
                        'job_title': title,
                        'company': request.form.get(f'work_experiences-{i}-company', ''),
                        'location': request.form.get(f'work_experiences-{i}-location', ''),
                        'start_date': request.form.get(f'work_experiences-{i}-start_date', ''),
                        'end_date': request.form.get(f'work_experiences-{i}-end_date', ''),
                        'responsibilities': request.form.get(f'work_experiences-{i}-responsibilities', ''),
                    })
                i += 1
                if i > 20:
                    break
            # Parse education entries
            i = 0
            while True:
                degree = request.form.get(f'educations-{i}-degree', '')
                if not degree and i > 0:
                    break
                if degree:
                    cv_data['educations'].append({
                        'degree': degree,
                        'institution': request.form.get(f'educations-{i}-institution', ''),
                        'end_date': request.form.get(f'educations-{i}-end_date', ''),
                        'honors': request.form.get(f'educations-{i}-honors', ''),
                    })
                i += 1
                if i > 10:
                    break
            # Parse certifications
            i = 0
            while True:
                name = request.form.get(f'certifications-{i}-name', '')
                if not name and i > 0:
                    break
                if name:
                    cv_data['certifications'].append({
                        'name': name,
                        'organization': request.form.get(f'certifications-{i}-organization', ''),
                        'year': request.form.get(f'certifications-{i}-year', ''),
                    })
                i += 1
                if i > 10:
                    break
            # Parse languages
            i = 0
            while True:
                lang = request.form.get(f'languages-{i}-language', '')
                if not lang and i > 0:
                    break
                if lang:
                    cv_data['languages'].append({
                        'language': lang,
                        'proficiency': request.form.get(f'languages-{i}-proficiency', ''),
                    })
                i += 1
                if i > 10:
                    break
            if not cv_data['full_name'] or not cv_data['email']:
                error = 'Full name and email are required.'
                cv_data = None
        except Exception as e:
            error = 'Error processing CV. Please check all fields.'
            logger.error(f'CV build error: {e}')
    return render_template('hr_cv_builder.html', cv_data=cv_data, error=error)


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
    success = False
    booking_type = request.form.get('booking_type', 'individual')
    if request.method == 'POST':
        try:
            booking = TrainingBooking(
                booking_type=booking_type,
                full_name=request.form.get('full_name', '').strip(),
                email=request.form.get('email', '').strip(),
                phone=request.form.get('phone', '').strip(),
                organization=request.form.get('organization', '').strip(),
                participants=int(request.form.get('participants', 1) or 1),
                category=request.form.get('category', '').strip(),
                preferred_date=request.form.get('preferred_date', '').strip(),
                notes=request.form.get('notes', '').strip(),
            )
            if not booking.full_name or not booking.email:
                error = 'Full name and email are required.'
            else:
                db.session.add(booking)
                db.session.commit()
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
<tr><td><b>Notes:</b></td><td>{booking.notes or "None"}</td></tr>
</table>'''
                )
                success = True
        except Exception as e:
            error = 'Booking failed. Please try again.'
            logger.error(f'Training booking error: {e}')
    return render_template('hr_training.html', error=error, success=success, booking_type=booking_type)


@app.route('/referral', methods=['GET', 'POST'])
def referral_page():
    success = False
    if request.method == 'POST':
        success = True
    return render_template('hr_referral.html', success=success)


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