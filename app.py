# Standard Library Imports
# -----------------------
# Imports for file handling, system operations, and data structures
import io
import logging
import os
import re
import subprocess
import tempfile
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime

# Third-Party Imports
# -------------------
# Flask and Extensions
from flask import (
    Flask, flash, redirect, render_template, request,
    send_from_directory, send_file, session, url_for
)
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField

# Data & Numerical Processing
import numpy as np
import numpy_financial as npf  # For IRR calculations

# Form Handling
from werkzeug.utils import secure_filename
from wtforms import (
    DateTimeField, FieldList, FileField, FormField,
    IntegerField, SelectField, StringField, SubmitField, TextAreaField
)
from wtforms.validators import DataRequired, Email, Optional

# Environment Configuration
from dotenv import load_dotenv

# ENVIRONMENT SETUP BLOCK
# -----------------------
# Loads environment variables from a .env file
load_dotenv()

# APP INITIALIZATION BLOCK
# ------------------------
# Initializes the Flask application
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# CONFIGURATION BLOCK
# -------------------
# Sets up Flask app configurations for database, mail, and uploads
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///cleanvisionhr.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/author_photos'
app.config['MAIL_SERVER'] = 'smtpout.secureserver.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'info@cleanvisionhr.com'
app.config['MAIL_PASSWORD'] = 'IFokbu@m@1'
app.config['MAIL_DEFAULT_SENDER'] = 'info@cleanvisionhr.com'

# EXTENSIONS INITIALIZATION BLOCK
# -------------------------------
# Initializes Flask extensions (SQLAlchemy, Migrate, Mail)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)

# MODELS BLOCK
# ------------
# Defines database models for the application
class ContactMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)

class BlogPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    author = db.Column(db.String(100), nullable=False, default="Admin")
    author_photo = db.Column(db.String(100), nullable=True)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    slug = db.Column(db.String(200), nullable=False, unique=True)

# FORMS BLOCK
# -----------
# Defines form classes using Flask-WTF
class ContactForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    message = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Send')

class BlogForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    author = StringField('Author', validators=[DataRequired()])
    author_photo = FileField('Author Photo', validators=[FileAllowed(['jpg', 'png', 'jpeg'])])
    content = TextAreaField('Content', validators=[DataRequired()])
    submit = SubmitField('Post')

# HELPER FUNCTIONS BLOCK
# ----------------------
# Defines utility functions and namedtuples used in the application
DCFResult = namedtuple('DCFResult', ['total_pv', 'pv_cash_flows', 'terminal_value', 'pv_terminal', 'total_dcf'])
DVMResult = namedtuple('DVMResult', ['intrinsic_value', 'formula', 'pv_dividends', 'terminal_value', 'pv_terminal'])

def generate_slug(title):
    """Generate a URL-friendly slug from the title."""
    slug = re.sub(r'[^\w\s-]', '', title.lower()).strip()
    slug = re.sub(r'\s+', '-', slug)
    base_slug = slug
    counter = 1
    while BlogPost.query.filter_by(slug=slug).first():
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug

def calculate_twr(returns):
    twr = 1
    for r in returns:
        twr *= (1 + r)
    return twr - 1

def calculate_mwr(cash_flows):
    return npf.irr(cash_flows)

def calculate_modified_dietz(mv0, mv1, cash_flow, weight):
    return (mv1 - mv0 - cash_flow) / (mv0 + (cash_flow * weight))

def calculate_simple_dietz(mv0, mv1, cash_flow):
    return (mv1 - mv0 - cash_flow) / (mv0 + cash_flow / 2)

def calculate_irr(cash_flows):
    return npf.irr(cash_flows)

def calculate_hpr(p0, p1, dividend):
    return (p1 - p0 + dividend) / p0

def calculate_annualized_return(r, n):
    return (1 + r) ** (1 / n) - 1

def calculate_geometric_mean_return(returns):
    return np.prod([1 + r for r in returns]) ** (1 / len(returns)) - 1

def calculate_arithmetic_mean_return(returns):
    return sum(returns) / len(returns)

def calculate_real_return(nominal_return, inflation_rate):
    return (1 + nominal_return) / (1 + inflation_rate) - 1

def calculate_time_weighted_inflation(monthly_inflations):
    tw_inflation = np.prod([1 + (i / 100) for i in monthly_inflations]) - 1
    return tw_inflation

def calculate_cca(pe_ratio, earnings):
    if pe_ratio <= 0 or earnings < 0:
        raise ValueError("P/E ratio must be positive and earnings non-negative")
    return pe_ratio * earnings

def calculate_nav(assets, liabilities):
    if assets < 0 or liabilities < 0:
        raise ValueError("Assets and liabilities cannot be negative")
    return assets - liabilities

def calculate_market_cap(share_price, shares_outstanding):
    if share_price < 0 or shares_outstanding < 0:
        raise ValueError("Share price and shares outstanding cannot be negative")
    return share_price * shares_outstanding

def calculate_ev(market_cap, debt, cash):
    if market_cap < 0 or debt < 0 or cash < 0:
        raise ValueError("Market cap, debt, and cash cannot be negative")
    return market_cap + debt - cash

def calculate_replacement_cost(tangible_assets, intangible_assets, adjustment_factor):
    if tangible_assets < 0 or intangible_assets < 0 or not 0 <= adjustment_factor <= 1:
        raise ValueError("Invalid inputs for replacement cost")
    return tangible_assets + intangible_assets * adjustment_factor

def calculate_risk_adjusted_return(returns, risk_free_rate, beta, market_return):
    if risk_free_rate < 0 or market_return < 0 or beta < 0:
        raise ValueError("Invalid inputs for risk-adjusted return")
    expected_return = risk_free_rate / 100 + beta * (market_return / 100 - risk_free_rate / 100)
    return (returns / 100) - expected_return

def parse_comma_separated(text):
    try:
        return [float(x.strip()) for x in text.split(',')]
    except ValueError:
        raise ValueError("Invalid numeric format")

def parse_covariance_matrix(text, num_assets):
    rows = text.split(';')
    if len(rows) != num_assets:
        raise ValueError(f"Covariance matrix must have {num_assets} rows")
    matrix = [parse_comma_separated(row) for row in rows]
    if any(len(row) != num_assets for row in matrix):
        raise ValueError(f"Each row must have {num_assets} elements")
    matrix = np.array(matrix)
    if not np.allclose(matrix, matrix.T) or np.any(np.linalg.eigvals(matrix) < -1e-10):
        raise ValueError("Covariance matrix must be symmetric and positive semi-definite")
    return matrix

def calculate_expected_return(weights, returns):
    if len(weights) != len(returns) or not 0.99 <= sum(weights) <= 1.01 or any(w < 0 for w in weights):
        raise ValueError("Invalid weights")
    return np.sum(np.array(weights) * np.array(returns))

def calculate_portfolio_metrics(num_assets, returns, weights, volatilities):
    if num_assets != len(returns) or num_assets != len(weights) or num_assets != len(volatilities):
        raise ValueError("Inconsistent input lengths")
    if not 0.99 <= sum(weights) <= 1.01 or any(w < 0 for w in weights) or any(v < 0 for v in volatilities):
        raise ValueError("Invalid weights or volatilities")
    expected_return = sum(r * w for r, w in zip(returns, weights)) / 100
    portfolio_volatility = np.sqrt(sum((w * v / 100) ** 2 for w, v in zip(weights, volatilities)))
    return expected_return, portfolio_volatility

def calculate_forex_profit(investment, initial_rate, final_rate):
    if any(x <= 0 for x in [investment, initial_rate, final_rate]):
        raise ValueError("All inputs must be positive")
    foreign_currency = investment * initial_rate
    final_value = foreign_currency / final_rate
    return final_value - investment

def calculate_cagr(start_value, end_value, years):
    """
    Calculate Compound Annual Growth Rate (CAGR) as a percentage.
    Handles edge cases and caps the growth rate.
    """
    if not all(isinstance(x, (int, float)) for x in [start_value, end_value]) or years <= 0:
        return 0
    try:
        if start_value == 0:
            return 0
        cagr = ((end_value / start_value) ** (1 / years) - 1) if start_value != 0 else 0
        capped_cagr = min(max(cagr, -0.05), 0.05)  # Cap at ±5%
        return capped_cagr * 100  # Return as percentage
    except (ZeroDivisionError, ValueError):
        return 0
        
def calculate_esg_metrics(esg_amount, total_portfolio, num_esg_assets, esg_scores, esg_weights):
    if esg_amount > total_portfolio or esg_amount < 0 or total_portfolio <= 0:
        raise ValueError("Invalid investment or portfolio values")
    if num_esg_assets != len(esg_scores) or num_esg_assets != len(esg_weights):
        raise ValueError("Inconsistent ESG inputs")
    if not 0.99 <= sum(esg_weights) <= 1.01 or any(w < 0 for w in esg_weights) or any(s < 0 or s > 100 for s in esg_scores):
        raise ValueError("Invalid ESG weights or scores")
    esg_proportion = esg_amount / total_portfolio
    weighted_esg_score = sum(s * w for s, w in zip(esg_scores, esg_weights))
    return esg_proportion, weighted_esg_score

def calculate_hedge_fund_returns(strategy, investment, leverage, target_return, volatility):
    if any(x <= 0 for x in [investment, leverage, volatility]) or target_return < 0:
        raise ValueError("Invalid inputs")
    strategy_multipliers = {'long-short': 1.0, 'arbitrage': 0.8, 'global-macro': 1.2}
    multiplier = strategy_multipliers.get(strategy, 1.0)
    leveraged_return = target_return / 100 * leverage * multiplier
    leveraged_volatility = volatility / 100 * leverage * multiplier
    expected_value = investment * (1 + leveraged_return)
    return expected_value, leveraged_return, leveraged_volatility

def calculate_dcf(fcfs, risk_free_rate, market_return, beta, debt, equity, tax_rate, growth_rate, use_exit_multiple=False, exit_ebitda_multiple=None, ebitda_last_year=None):
    assert len(fcfs) == 5, "Provide exactly 5 years of FCF"
    total_value = debt + equity
    if total_value <= 0:
        raise ValueError("Total value must be positive")
    cost_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    cost_debt = 0.05
    wacc = (equity / total_value) * cost_equity + (debt / total_value) * cost_debt * (1 - tax_rate)
    pv_fcfs = sum(fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs))
    last_fcf = fcfs[-1]
    if use_exit_multiple:
        if exit_ebitda_multiple is None or ebitda_last_year is None:
            raise ValueError("Exit multiple and EBITDA required")
        terminal_value = ebitda_last_year * exit_ebitda_multiple
    else:
        if wacc <= growth_rate:
            raise ValueError("WACC must exceed growth rate")
        fcf_next = last_fcf * (1 + growth_rate)
        terminal_value = fcf_next / (wacc - growth_rate)
    pv_terminal = terminal_value / (1 + wacc) ** 5
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = max(enterprise_value - debt, 0)
    return enterprise_value, equity_value

def calculate_vc_method(exit_value, target_roi, investment_amount, exit_horizon, dilution_factor=1.0):
    if any(x <= 0 for x in [exit_value, target_roi, investment_amount, exit_horizon]) or not 0 < dilution_factor <= 1:
        raise ValueError("Invalid inputs")
    adjusted_exit_value = exit_value * dilution_factor
    post_money_valuation = adjusted_exit_value / target_roi
    pre_money_valuation = post_money_valuation - investment_amount
    return pre_money_valuation, post_money_valuation

def calculate_arr_multiple(arr, arr_multiple, control_premium=0.0, illiquidity_discount=0.0):
    if arr <= 0 or arr_multiple <= 0 or control_premium < 0 or illiquidity_discount < 0:
        raise ValueError("Invalid inputs")
    base_valuation = arr * arr_multiple
    return base_valuation * (1 + control_premium) * (1 - illiquidity_discount)

def calculate_target_price(fcf, explicit_growth, n, g, r, debt, cash, shares):
    """
    Calculate target price for a given projection period using discounted cash flows.
    """
    if not all(isinstance(x, (int, float)) for x in [fcf, explicit_growth, n, g, r, debt, cash, shares]):
        raise ValueError("All inputs must be numeric")
    if g >= r:
        raise ValueError("Perpetual growth rate must be less than discount rate")
    if shares <= 0:
        raise ValueError("Shares outstanding must be positive")
    if n <= 0:
        raise ValueError("Projection period must be positive")
    projected_fcf = fcf * (1 + explicit_growth) ** n
    terminal_value = (projected_fcf * (1 + g)) / (r - g)
    pv_terminal = terminal_value / (1 + r) ** n
    target_price = max((pv_terminal - debt + cash) / shares, 0)
    return target_price

def calculate_portfolio_volatility(weights, cov_matrix):
    weights_array = np.array(weights)
    portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
    return np.sqrt(portfolio_variance)

def calculate_gordon_growth(d1, r, g):
    if r <= g:
        raise ValueError("Discount rate must exceed growth rate")
    intrinsic_value = d1 / (r - g)
    formula = f"{d1:.2f} / ({r*100:.2f}% - {g*100:.2f}%)"
    return {'intrinsic_value': intrinsic_value, 'formula': formula}

def calculate_multi_stage(dividends, r, terminal_growth):
    pv_dividends = [d / ((1 + r) ** (i + 1)) for i, d in enumerate(dividends)]
    total_pv = sum(pv_dividends)
    terminal_value = (dividends[-1] * (1 + terminal_growth)) / (r - terminal_growth)
    pv_terminal = terminal_value / ((1 + r) ** len(dividends))
    return {
        'intrinsic_value': total_pv + pv_terminal,
        'pv_dividends': pv_dividends,
        'terminal_value': terminal_value,
        'pv_terminal': pv_terminal
    }

def calculate_no_growth(d, r):
    intrinsic_value = d / r
    formula = f"{d:.2f} / {r*100:.2f}%"
    return {'intrinsic_value': intrinsic_value, 'formula': formula}

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# CUSTOM JINJA2 FILTERS BLOCK
# ---------------------------
# Defines custom filters for use in Jinja2 templates
@app.template_filter('first_two_sentences')
def first_two_sentences(content):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text()
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = ' '.join(sentences[:2]).strip()
    return result + ' Read more...' if result else 'Read more...'

@app.template_filter('round')
def round_filter(value, decimals=2):
    return round(value, decimals)

@app.template_filter('commafy')
def commafy(value):
    return "{:,.2f}".format(value)

# ROUTES BLOCK
# ------------
# Defines all route handlers for the Flask application
@app.route('/ads.txt')
def ads_txt():
    return send_from_directory('static', 'ads.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/expected-return', methods=['GET', 'POST'])
def expected_return():
    if request.method == 'POST':
        try:
            num_assets = int(request.form['num_assets'])
            weights = [float(request.form[f'weight_{i}']) for i in range(1, num_assets + 1)]
            returns = [float(request.form[f'return_{i}']) for i in range(1, num_assets + 1)]
            expected_return = calculate_expected_return(weights, returns)
            result = f"<p>Portfolio Expected Return: {expected_return:.2%}</p>"
            return render_template('expected_return.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('expected_return.html', error=str(e), form_data=request.form)
    return render_template('expected_return.html', form_data={})

@app.route('/volatility', methods=['GET', 'POST'])
def volatility():
    if request.method == 'POST':
        try:
            num_assets = int(request.form['num_assets'])
            weights = [float(request.form[f'weight_{i}']) for i in range(1, num_assets + 1)]
            cov_matrix = np.array([[float(request.form[f'cov_{i}_{j}']) for j in range(1, num_assets + 1)] for i in range(1, num_assets + 1)])
            portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
            result = f"<p>Portfolio Volatility: {portfolio_volatility:.2%}</p>"
            return render_template('volatility.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('volatility.html', error=str(e), form_data=request.form)
    return render_template('volatility.html', form_data={})

@app.route('/calculate-fcf', methods=['GET', 'POST'])
def calculate_fcf():
    fcfs = None
    ocfs = None
    capex = None
    error = None
    if request.method == 'POST':
        try:
            ocfs = [float(request.form[f'ocf_{i}']) for i in range(1, 6)]
            capex = [float(request.form[f'capex_{i}']) for i in range(1, 6)]
            if any(ocf < 0 or cap < 0 for ocf, cap in zip(ocfs, capex)):
                error = "OCF and CAPEX must be non-negative"
            else:
                fcfs = [ocf - cap for ocf, cap in zip(ocfs, capex)]
        except ValueError:
            error = "Invalid numbers entered"
    return render_template('calculate_fcf.html', fcfs=fcfs, ocfs=ocfs, capex=capex, error=error, currency_symbol="$")

@app.route('/portfolio-diversification', methods=['GET', 'POST'])
def portfolio_diversification():
    if request.method == 'POST':
        try:
            num_assets = int(request.form['num_assets'])
            returns = [float(request.form[f'return_{i}']) for i in range(1, num_assets + 1)]
            weights = [float(request.form[f'weight_{i}']) for i in range(1, num_assets + 1)]
            volatilities = [float(request.form[f'volatility_{i}']) for i in range(1, num_assets + 1)]
            expected_return, portfolio_volatility = calculate_portfolio_metrics(num_assets, returns, weights, volatilities)
            result = f"<p>Portfolio Expected Return: {expected_return:.2%}</p><p>Portfolio Volatility: {portfolio_volatility:.2%}</p>"
            return render_template('portfolio_diversification.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('portfolio_diversification.html', result=str(e), form_data=request.form)
    return render_template('portfolio_diversification.html', form_data={})

@app.route('/dcf', methods=['GET', 'POST'])
def dcf_calculator():
    error = None
    results = None
    if request.method == 'POST':
        try:
            years = int(request.form.get('years', 0))
            if not 1 <= years <= 10:
                raise ValueError("Years must be between 1 and 10")
            discount_rate = float(request.form['discount_rate']) / 100
            terminal_growth = float(request.form['terminal_growth']) / 100
            if discount_rate <= terminal_growth:
                raise ValueError("Discount rate must exceed terminal growth")
            cash_flows = [float(request.form[f'cash_flow_{i}']) for i in range(1, years + 1)]
            pv_cash_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, 1)]
            total_pv = sum(pv_cash_flows)
            terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** years
            total_dcf = total_pv + pv_terminal
            results = DCFResult(total_pv, pv_cash_flows, terminal_value, pv_terminal, total_dcf)
        except ValueError as e:
            error = str(e)
    return render_template('dcf.html', error=error, results=results)

@app.route('/dvm', methods=['GET', 'POST'])
def dvm_calculator():
    error = None
    results = None
    model_type = request.form.get('model_type', 'gordon_growth') if request.method == 'POST' else 'gordon_growth'
    if request.method == 'POST':
        try:
            r = float(request.form['r']) / 100
            if model_type == 'gordon_growth':
                d1 = float(request.form['d1'])
                g = float(request.form['g']) / 100
                result = calculate_gordon_growth(d1, r, g)
                results = DVMResult(result['intrinsic_value'], result['formula'], None, None, None)
            elif model_type == 'multi_stage':
                periods = int(request.form['periods'])
                terminal_growth = float(request.form['terminal_growth']) / 100
                dividends = [float(request.form[f'dividend_{i+1}']) for i in range(periods)]
                result = calculate_multi_stage(dividends, r, terminal_growth)
                results = DVMResult(result['intrinsic_value'], None, result['pv_dividends'], result['terminal_value'], result['pv_terminal'])
            elif model_type == 'no_growth':
                d = float(request.form['d'])
                result = calculate_no_growth(d, r)
                results = DVMResult(result['intrinsic_value'], result['formula'], None, None, None)
        except ValueError as e:
            error = str(e)
    return render_template('dvm.html', error=error, results=results, model_type=model_type)

@app.route('/forex', methods=['GET', 'POST'])
def forex_calculator():
    if request.method == 'POST':
        try:
            investment = float(request.form['investment'])
            initial_rate = float(request.form['initial_rate'])
            final_rate = float(request.form['final_rate'])
            profit = calculate_forex_profit(investment, initial_rate, final_rate)
            result = f"<p>Forex Profit/Loss: ${profit:,.2f}</p>"
            return render_template('forex.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('forex.html', result=str(e), form_data=request.form)
    return render_template('forex.html', form_data={})

@app.route('/valuation_methods', methods=['GET', 'POST'])
def valuation_methods():
    selected_method = request.form.get('method', 'CCA')
    if request.method == 'POST':
        try:
            method = request.form['method']
            result = {'method': method.replace('_', ' ').title()}
            if method == 'CCA':
                result['value'] = f"GHS {calculate_cca(float(request.form['pe_ratio']), float(request.form['earnings'])):,.2f}"
            elif method == 'NAV':
                result['value'] = f"GHS {calculate_nav(float(request.form['assets']), float(request.form['liabilities'])):,.2f}"
            elif method == 'Market Capitalization':
                result['value'] = f"GHS {calculate_market_cap(float(request.form['share_price']), float(request.form['shares_outstanding'])):,.2f}"
            elif method == 'EV':
                result['value'] = f"GHS {calculate_ev(float(request.form['market_cap']), float(request.form['debt']), float(request.form['cash'])):,.2f}"
            elif method == 'Replacement Cost':
                result['value'] = f"GHS {calculate_replacement_cost(float(request.form['tangible_assets']), float(request.form['intangible_assets']), float(request.form['adjustment_factor'])):,.2f}"
            elif method == 'Risk-Adjusted Return':
                result['value'] = f"{calculate_risk_adjusted_return(float(request.form['returns']), float(request.form['risk_free_rate']), float(request.form['beta']), float(request.form['market_return'])):.2%}"
            return render_template('valuation_methods.html', result=result, selected_method=method)
        except ValueError as e:
            return render_template('valuation_methods.html', error=str(e), selected_method=method)
    return render_template('valuation_methods.html', selected_method=selected_method)

@app.route('/esg-investments', methods=['GET', 'POST'])
def esg_investments():
    if request.method == 'POST':
        try:
            esg_amount = float(request.form['esg_amount'])
            total_portfolio = float(request.form['total_portfolio'])
            num_esg_assets = int(request.form['num_esg_assets'])
            esg_scores = [float(request.form[f'esg_score_{i}']) for i in range(1, num_esg_assets + 1)]
            esg_weights = [float(request.form[f'esg_weight_{i}']) for i in range(1, num_esg_assets + 1)]
            esg_proportion, weighted_esg_score = calculate_esg_metrics(esg_amount, total_portfolio, num_esg_assets, esg_scores, esg_weights)
            result = f"<p>ESG Proportion: {esg_proportion:.2%}</p><p>Weighted ESG Score: {weighted_esg_score:.2f}/100</p>"
            return render_template('esg.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('esg.html', result=str(e), form_data=request.form)
    return render_template('esg.html', form_data={})

@app.route('/hedge-funds', methods=['GET', 'POST'])
def hedge_funds():
    if request.method == 'POST':
        try:
            strategy = request.form['strategy']
            investment = float(request.form['investment'])
            leverage = float(request.form['leverage'])
            target_return = float(request.form['target_return'])
            volatility = float(request.form['volatility'])
            expected_value, leveraged_return, leveraged_volatility = calculate_hedge_fund_returns(strategy, investment, leverage, target_return, volatility)
            result = f"<p>Expected Value: ${expected_value:,.2f}</p><p>Leveraged Return: {leveraged_return:.2%}</p><p>Leveraged Volatility: {leveraged_volatility:.2%}</p>"
            return render_template('hedge_funds.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('hedge_funds.html', result=str(e), form_data=request.form)
    return render_template('hedge_funds.html', form_data={})

@app.route('/calculate-beta', methods=['GET', 'POST'])
def calculate_beta():
    beta = None
    error = None
    if request.method == 'POST':
        try:
            stock_returns = parse_comma_separated(request.form['stock_returns'])
            market_returns = parse_comma_separated(request.form['market_returns'])
            if len(stock_returns) != len(market_returns) or len(stock_returns) < 2:
                raise ValueError("Invalid return data")
            cov_matrix = np.cov(stock_returns, market_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else None
            if beta is None:
                raise ValueError("Market variance is zero")
            beta = round(beta, 4)
        except ValueError as e:
            error = str(e)
    return render_template('calculate_beta.html', beta=beta, error=error)

@app.route('/monthly-contribution', methods=['GET', 'POST'])
def monthly_contribution():
    result = None
    if request.method == 'POST':
        try:
            target = float(request.form['target_amount'])
            principal = float(request.form['starting_principal'])
            period = float(request.form['period'])
            rate = float(request.form['annual_return']) / 100
            if target <= 0 or principal < 0 or period <= 0:
                raise ValueError("Invalid inputs")
            months = period * 12
            monthly_rate = (1 + rate) ** (1 / 12) - 1 if rate != 0 else 0
            future_principal = principal * (1 + monthly_rate) ** months if rate != 0 else principal
            monthly_contribution = (target - future_principal) / (((1 + monthly_rate) ** months - 1) / monthly_rate) if rate != 0 else (target - principal) / months
            result = "{:,.2f}".format(monthly_contribution)
        except ValueError as e:
            return render_template('monthly_contribution.html', error=str(e))
    return render_template('monthly_contribution.html', result=result)

@app.route('/end-balance', methods=['GET', 'POST'])
def end_balance():
    result = None
    if request.method == 'POST':
        try:
            monthly = float(request.form['monthly_contribution'])
            principal = float(request.form['starting_principal'])
            period = float(request.form['period'])
            rate = float(request.form['annual_return']) / 100
            if monthly < 0 or principal < 0 or period <= 0:
                raise ValueError("Invalid inputs")
            months = period * 12
            monthly_rate = (1 + rate) ** (1 / 12) - 1 if rate != 0 else 0
            future_principal = principal * (1 + monthly_rate) ** months if rate != 0 else principal
            future_contributions = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate) if rate != 0 else monthly * months
            end_balance = future_principal + future_contributions
            result = "{:,.2f}".format(end_balance)
        except ValueError as e:
            return render_template('end_balance.html', error=str(e))
    return render_template('end_balance.html', result=result)

@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    result = None
    if request.method == 'POST':
        try:
            num_shares = float(request.form['num_shares'])
            purchase_price = float(request.form['purchase_price_per_share'])
            purchase_commission = float(request.form['purchase_commission']) / 100
            selling_price = float(request.form['selling_price_per_share'])
            sale_commission = float(request.form['sale_commission']) / 100
            dividends = float(request.form['dividends'])
            if any(x < 0 for x in [num_shares, purchase_price, purchase_commission, selling_price, sale_commission, dividends]):
                raise ValueError("Non-negative numbers required")
            total_purchase_cost = num_shares * purchase_price * (1 + purchase_commission)
            net_sale_proceeds = num_shares * selling_price * (1 - sale_commission)
            total_return = ((net_sale_proceeds - total_purchase_cost + dividends) / total_purchase_cost) * 100
            result = {'total_return': round(total_return, 2)}
        except ValueError as e:
            return render_template('stocks.html', error=str(e))
    return render_template('stocks.html', result=result)

@app.route('/mna', methods=['GET', 'POST'])
def mna_calculator():
    if request.method == 'POST':
        try:
            acquirer_eps = float(request.form['acquirer_eps'])
            acquirer_shares = float(request.form['acquirer_shares'])
            target_eps = float(request.form['target_eps'])
            target_shares = float(request.form['target_shares'])
            new_shares_issued = float(request.form['new_shares_issued'])
            synergy_value = float(request.form['synergy_value'])
            acquirer_earnings = acquirer_eps * acquirer_shares
            target_earnings = target_eps * target_shares
            combined_earnings = acquirer_earnings + target_earnings + synergy_value
            total_shares = acquirer_shares + new_shares_issued
            combined_eps = combined_earnings / total_shares
            eps_change = combined_eps - acquirer_eps
            status = "Accretive" if eps_change > 0 else "Dilutive" if eps_change < 0 else "Neutral"
            result = f"<p>Combined EPS: ${combined_eps:.2f}</p><p>EPS Change: ${eps_change:.2f} ({status})</p>"
            return render_template('mna.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('mna.html', result=str(e), form_data=request.form)
    return render_template('mna.html', form_data={})

@app.route('/pe-vc', methods=['GET', 'POST'])
def pe_vc_valuation():
    if request.method == 'POST':
        try:
            valuation_method = request.form['valuation_method']
            if valuation_method == "dcf":
                fcfs = [float(request.form[f'fcf_{i}']) for i in range(1, 6)]
                risk_free_rate = float(request.form['risk_free_rate']) / 100
                market_return = float(request.form['market_return']) / 100
                beta = float(request.form['beta'])
                debt = float(request.form['debt'])
                equity = float(request.form['equity'])
                tax_rate = float(request.form['tax_rate']) / 100
                growth_rate = float(request.form['growth_rate']) / 100
                use_exit_multiple = request.form.get('use_exit_multiple') == 'on'
                exit_ebitda_multiple = float(request.form['exit_ebitda_multiple']) if use_exit_multiple else None
                ebitda_last_year = float(request.form['ebitda_last_year']) if use_exit_multiple else None
                enterprise_value, equity_value = calculate_dcf(fcfs, risk_free_rate, market_return, beta, debt, equity, tax_rate, growth_rate, use_exit_multiple, exit_ebitda_multiple, ebitda_last_year)
                result = f"<p>Enterprise Value: ${enterprise_value:,.2f}</p><p>Equity Value: ${equity_value:,.2f}</p>"
            elif valuation_method == "vc":
                exit_value = float(request.form['exit_value'])
                target_roi = float(request.form['target_roi'])
                investment_amount = float(request.form['investment_amount'])
                exit_horizon = float(request.form['exit_horizon'])
                dilution_factor = float(request.form['dilution_factor'])
                pre_money, post_money = calculate_vc_method(exit_value, target_roi, investment_amount, exit_horizon, dilution_factor)
                result = f"<p>Pre-Money Valuation: ${pre_money:,.2f}</p><p>Post-Money Valuation: ${post_money:,.2f}</p>"
            elif valuation_method == "arr":
                arr = float(request.form['arr'])
                arr_multiple = float(request.form['arr_multiple'])
                control_premium = float(request.form['control_premium']) / 100
                illiquidity_discount = float(request.form['illiquidity_discount']) / 100
                valuation = calculate_arr_multiple(arr, arr_multiple, control_premium, illiquidity_discount)
                result = f"<p>Valuation: ${valuation:,.2f}</p>"
            return render_template('pe_vc.html', result=result, form_data=request.form)
        except ValueError as e:
            return render_template('pe_vc.html', result=str(e), form_data=request.form)
    return render_template('pe_vc.html', form_data={})

@app.route('/bonds', methods=['GET', 'POST'])
def bonds():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            tenor = float(request.form['tenor'])
            rate = float(request.form['rate']) / 100
            total_coupons = float(request.form['total_coupons'])
            if principal <= 0 or tenor <= 0:
                raise ValueError("Principal and tenor must be positive")
            maturity_amount = principal + total_coupons
            bond_yield = (total_coupons + (maturity_amount - principal)) / (principal * (tenor / 365)) * 100
            result = {'maturity_amount': "{:,.2f}".format(maturity_amount), 'bond_yield': round(bond_yield, 2)}
        except ValueError as e:
            return render_template('bonds.html', error=str(e))
    return render_template('bonds.html', result=result)

@app.route('/tbills', methods=['GET', 'POST'])
def tbills():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100
            tenor = float(request.form['tenor'])
            if principal <= 0 or tenor <= 0:
                raise ValueError("Principal and tenor must be positive")
            interest = (principal * tenor * rate) / 364
            maturity_value = principal + interest
            result = {'maturity_value': "{:,.2f}".format(maturity_value)}
        except ValueError as e:
            return render_template('tbills.html', error=str(e))
    return render_template('tbills.html', result=result)

@app.route('/mutual-funds', methods=['GET', 'POST'])
def mutual_funds():
    result = None
    if request.method == 'POST':
        try:
            nav_start = float(request.form['nav_start'])
            nav_end = float(request.form['nav_end'])
            dividends = float(request.form['dividends'])
            if nav_start <= 0:
                raise ValueError("NAV at start must be positive")
            total_return = (nav_end - nav_start + dividends) / nav_start * 100
            result = {'total_return': round(total_return, 2)}
        except ValueError as e:
            return render_template('mutual_funds.html', error=str(e))
    return render_template('mutual_funds.html', result=result)

@app.route('/duration', methods=['GET', 'POST'])
def duration():
    if request.method == 'POST':
        try:
            num_periods = int(request.form['num_periods'])
            cash_flows = [float(request.form[f'cf_{i}']) for i in range(1, num_periods + 1)]
            yield_rate = float(request.form['yield']) / 100
            compounding = int(request.form['compounding'])
            initial_price = float(request.form['initial_price'])
            price_drop = float(request.form['price_drop'])
            price_rise = float(request.form['price_rise'])
            if num_periods < 1 or num_periods > 10 or initial_price <= 0 or compounding < 1 or any(cf <= 0 for cf in cash_flows):
                raise ValueError("Invalid inputs")
            pv_sum = 0
            weighted_pv_sum = 0
            yield_per_period = yield_rate / compounding
            for t in range(1, num_periods + 1):
                pv = cash_flows[t-1] / (1 + yield_per_period) ** t
                pv_sum += pv
                weighted_pv_sum += (t / compounding) * pv
            macaulay_duration = weighted_pv_sum / pv_sum
            modified_duration = macaulay_duration / (1 + yield_rate / compounding)
            effective_duration = (price_drop - price_rise) / (2 * 0.01 * initial_price)
            result = {
                'macaulay_duration': round(macaulay_duration, 2),
                'modified_duration': round(modified_duration, 2),
                'effective_duration': round(effective_duration, 2)
            }
            return render_template('duration.html', result=result)
        except ValueError as e:
            return render_template('duration.html', error=str(e))
    return render_template('duration.html')

@app.route('/portfolio_return', methods=['GET', 'POST'])
def portfolio_return():
    if request.method == 'POST':
        try:
            method = request.form['method']
            data = parse_comma_separated(request.form['data'])
            average_inflation = float(request.form['average_inflation']) / 100
            monthly_inflation = parse_comma_separated(request.form['monthly_inflation']) if request.form['monthly_inflation'].strip() else []
            if method == 'twr':
                nominal_return = calculate_twr(data)
            elif method == 'mwr':
                nominal_return = calculate_mwr(data)
            elif method == 'modified_dietz':
                nominal_return = calculate_modified_dietz(*data)
            elif method == 'simple_dietz':
                nominal_return = calculate_simple_dietz(*data)
            elif method == 'irr':
                nominal_return = calculate_irr(data)
            elif method == 'hpr':
                nominal_return = calculate_hpr(*data)
            elif method == 'annualized':
                nominal_return = calculate_annualized_return(*data)
            elif method == 'geometric_mean':
                nominal_return = calculate_geometric_mean_return(data)
            elif method == 'arithmetic_mean':
                nominal_return = calculate_arithmetic_mean_return(data)
            real_return_avg = calculate_real_return(nominal_return, average_inflation)
            real_return_tw = calculate_real_return(nominal_return, calculate_time_weighted_inflation(monthly_inflation)) if monthly_inflation else None
            result = {
                'method': method.replace('_', ' ').title(),
                'nominal_return': f"{nominal_return:.2%}",
                'real_return_avg': f"{real_return_avg:.2%}",
                'real_return_tw': f"{real_return_tw:.2%}" if real_return_tw is not None else "Not Provided"
            }
            return render_template('portfolio_return.html', result=result)
        except ValueError as e:
            return render_template('portfolio_return.html', error=str(e))
    return render_template('portfolio_return.html')

@app.route('/etfs', methods=['GET', 'POST'])
def etfs():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            dividends = float(request.form['dividends'])
            if purchase_price <= 0:
                raise ValueError("Purchase price must be positive")
            total_return = (selling_price - purchase_price + dividends) / purchase_price * 100
            result = {'total_return': round(total_return, 2)}
        except ValueError as e:
            return render_template('etfs.html', error=str(e))
    return render_template('etfs.html', result=result)

@app.route('/cds', methods=['GET', 'POST'])
def cds():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100
            compounding_periods = float(request.form['compounding_periods'])
            years = float(request.form['years'])
            if principal <= 0 or compounding_periods <= 0 or years <= 0:
                raise ValueError("Invalid inputs")
            fv = principal * (1 + rate / compounding_periods) ** (years * compounding_periods)
            result = {'future_value': "{:,.2f}".format(fv)}
        except ValueError as e:
            return render_template('cds.html', error=str(e))
    return render_template('cds.html', result=result)

@app.route('/money-market', methods=['GET', 'POST'])
def money_market():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100
            days_held = float(request.form['days_held'])
            if principal <= 0 or days_held <= 0:
                raise ValueError("Invalid inputs")
            interest_earned = principal * rate * (days_held / 365)
            result = {'interest_earned': "{:,.2f}".format(interest_earned)}
        except ValueError as e:
            return render_template('money_market.html', error=str(e))
    return render_template('money_market.html', result=result)

@app.route('/options', methods=['GET', 'POST'])
def options():
    result = None
    if request.method == 'POST':
        try:
            option_type = request.form['option_type']
            stock_price = float(request.form['stock_price'])
            strike_price = float(request.form['strike_price'])
            premium = float(request.form['premium'])
            if option_type == 'call':
                profit = max(stock_price - strike_price - premium, 0)
            elif option_type == 'put':
                profit = max(strike_price - stock_price - premium, 0)
            else:
                raise ValueError("Invalid option type")
            result = {'profit': "{:,.2f}".format(profit)}
        except ValueError as e:
            return render_template('options.html', error=str(e))
    return render_template('options.html', result=result)

@app.route('/futures', methods=['GET', 'POST'])
def futures():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            contract_size = float(request.form['contract_size'])
            if contract_size <= 0:
                raise ValueError("Contract size must be positive")
            profit = (selling_price - purchase_price) * contract_size
            result = {'profit': "{:,.2f}".format(profit)}
        except ValueError as e:
            return render_template('futures.html', error=str(e))
    return render_template('futures.html', result=result)

@app.route('/cryptocurrency', methods=['GET', 'POST'])
def cryptocurrency():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            if purchase_price <= 0:
                raise ValueError("Purchase price must be positive")
            total_return = (selling_price - purchase_price) / purchase_price * 100
            result = {'total_return': round(total_return, 2)}
        except ValueError as e:
            return render_template('cryptocurrency.html', error=str(e))
    return render_template('cryptocurrency.html', result=result)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms')
def terms_conditions():
    return render_template('terms_conditions.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        contact_message = ContactMessage(name=form.name.data, email=form.email.data, message=form.message.data)
        db.session.add(contact_message)
        db.session.commit()
        company_msg = Message(subject='New Contact Message', recipients=['info@cleanvisionhr.com'])
        company_msg.body = f"Name: {contact_message.name}\nEmail: {contact_message.email}\nMessage: {contact_message.message}"
        mail.send(company_msg)
        html_body = render_template('contact_confirmation.html', name=contact_message.name)
        auto_response_msg = Message(subject="Thanks for Reaching Out!", recipients=[contact_message.email], html=html_body)
        mail.send(auto_response_msg)
        flash('Message sent successfully!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html', form=form)

@app.route('/early_exit', methods=['GET', 'POST'])
def early_exit():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            holding_period = float(request.form['holding_period'])
            selling_price = float(request.form['selling_price'])
            total_coupons = float(request.form.get('total_coupons', 0))
            if principal <= 0 or holding_period <= 0:
                raise ValueError("Invalid inputs")
            holding_period_return = (total_coupons + (selling_price - principal)) / (principal * (holding_period / 365)) * 100
            result = {'holding_period_return': round(holding_period_return, 2)}
        except ValueError as e:
            return render_template('early_exit.html', error=str(e))
    return render_template('early_exit.html', result=result)

@app.route('/tbills-rediscount', methods=['GET', 'POST'])
def tbills_rediscount():
    result = None
    if request.method == 'POST':
        try:
            settlement_amount = float(request.form['settlement_amount'])
            rate = float(request.form['rate']) / 100
            days_to_maturity = float(request.form['days_to_maturity'])
            initial_fv = float(request.form['initial_fv'])
            if settlement_amount <= 0 or days_to_maturity <= 0 or initial_fv <= 0:
                raise ValueError("Invalid inputs")
            settlement_fv = settlement_amount * (1 + rate) ** (days_to_maturity / 364)
            face_value_after_rediscount = initial_fv - settlement_fv
            result = {
                'settlement_fv': "{:,.2f}".format(settlement_fv),
                'face_value_after_rediscount': "{:,.2f}".format(face_value_after_rediscount)
            }
        except ValueError as e:
            return render_template('tbills_rediscount.html', error=str(e))
    return render_template('tbills_rediscount.html', result=result)

# ROUTES BLOCK
# ------------
# ROUTES BLOCK
# ------------
@app.route('/intrinsic-value', methods=['GET', 'POST'])
def intrinsic_value():
    """
    Route to calculate intrinsic value using DCF.
    Supports 3 to 5 years of FCF inputs.
    """
    if request.method == 'POST':
        try:
            # Helper function to safely convert form input to float
            def safe_float(value, default=None):
                try:
                    return float(value) if value and value.strip() else default
                except (ValueError, TypeError):
                    return default

            # Collect number of FCF years
            num_fcf_years = safe_float(request.form.get('num_fcf_years'), 5)
            num_fcf_years = int(num_fcf_years) if num_fcf_years in [3, 4, 5] else 5

            # Collect FCF inputs dynamically
            fcf = []
            for i in range(1, num_fcf_years + 1):
                fcf_value = safe_float(request.form.get(f'fcf_{i}'))
                if fcf_value is None:
                    raise ValueError(f"Free Cash Flow for year {i} is required")
                fcf.append(fcf_value)

            # Collect other inputs and convert percentages to decimals
            risk_free_rate = safe_float(request.form.get('risk_free_rate')) / 100
            market_return = safe_float(request.form.get('market_return')) / 100
            beta = safe_float(request.form.get('beta'))
            outstanding_shares = safe_float(request.form.get('outstanding_shares'))
            total_debt = safe_float(request.form.get('total_debt'), 0)
            cash_and_equivalents = safe_float(request.form.get('cash_and_equivalents'), 0)
            perpetual_growth_rate = safe_float(request.form.get('perpetual_growth_rate'))

            # Validate required inputs
            required_fields = {
                'Risk-free rate': risk_free_rate,
                'Market return': market_return,
                'Beta': beta,
                'Outstanding shares': outstanding_shares
            }
            for name, value in required_fields.items():
                if value is None:
                    raise ValueError(f"{name} is required")
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")

            # Calculate discount rate using CAPM
            discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)

            # Determine perpetual growth rate
            if perpetual_growth_rate is None:
                # Calculate CAGR-based growth rate (capped)
                cagr_years = num_fcf_years - 1 if num_fcf_years > 1 else 1
                perpetual_growth_rate = calculate_cagr(fcf[0], fcf[-1], cagr_years) / 100
            else:
                # Convert from percentage to decimal and validate
                perpetual_growth_rate = perpetual_growth_rate / 100
                if not -0.05 <= perpetual_growth_rate <= 0.05:
                    raise ValueError("Perpetual growth rate must be between -5% and 5%")

            # Validate discount rate vs. growth rate
            if discount_rate <= perpetual_growth_rate:
                raise ValueError(
                    f"Discount rate ({discount_rate:.2%}) must exceed growth rate ({perpetual_growth_rate:.2%}). "
                    f"Adjust risk-free rate, market return, beta, or perpetual growth rate."
                )

            # Intrinsic Value Calculation
            last_fcf = fcf[-1]
            enterprise_value = (last_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
            equity_value = enterprise_value - total_debt + cash_and_equivalents
            intrinsic_value = max(equity_value / outstanding_shares, 0)

            # Debug information
            debug = {
                'fcf': fcf,
                'discount_rate': discount_rate,
                'perpetual_growth_rate': perpetual_growth_rate,
                'enterprise_value': enterprise_value,
                'equity_value': equity_value
            }

            return render_template(
                'intrinsic_value.html',
                result=f"{intrinsic_value:.2f}",
                debug=debug,
                form_data=request.form,
                num_fcf_years=num_fcf_years
            )
        except ValueError as e:
            return render_template(
                'intrinsic_value.html',
                error=str(e),
                form_data=request.form,
                num_fcf_years=num_fcf_years
            )
        except Exception as e:
            return render_template(
                'intrinsic_value.html',
                error=f"Unexpected error: {str(e)}",
                form_data=request.form,
                num_fcf_years=num_fcf_years
            )
    return render_template('intrinsic_value.html', form_data={}, num_fcf_years=5)

@app.route('/target-price', methods=['GET', 'POST'])
def target_price():
    if request.method == 'POST':
        try:
            current_eps = float(request.form['current_eps'])
            growth_rate = float(request.form['growth_rate']) / 100
            pe_ratio = float(request.form['pe_ratio'])

            # Calculate for 1 year
            future_eps_1 = current_eps * (1 + growth_rate)
            target_price_1 = future_eps_1 * pe_ratio

            # Calculate for 2 years
            future_eps_2 = current_eps * (1 + growth_rate) ** 2
            target_price_2 = future_eps_2 * pe_ratio

            return render_template('target_price.html', 
                                   target_price_1=f"{target_price_1:.2f}",
                                   target_price_2=f"{target_price_2:.2f}")
        except ValueError:
            return render_template('target_price.html', error="Invalid input. Please enter numeric values.")
    return render_template('target_price.html')

@app.route('/blog')
def blog():
    posts = BlogPost.query.order_by(BlogPost.date_posted.desc()).all()
    return render_template('blog.html', posts=posts)

@app.route('/blog/<int:post_id>')
def blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    return render_template('blog_post.html', post=post)

@app.route('/admin/blog', methods=['GET', 'POST'])
def admin_blog():
    form = BlogForm()
    if form.validate_on_submit():
        try:
            file = form.author_photo.data
            filename = None
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif file:
                flash('Invalid file type', 'danger')
                return render_template('admin_blog.html', form=form, posts=BlogPost.query.order_by(BlogPost.date_posted.desc()).all())
            new_post = BlogPost(
                title=form.title.data,
                slug=generate_slug(form.title.data),
                content=form.content.data,
                author=form.author.data,
                author_photo=filename
            )
            db.session.add(new_post)
            db.session.commit()
            flash('Blog post created!', 'success')
            return redirect(url_for('admin_blog'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
    posts = BlogPost.query.order_by(BlogPost.date_posted.desc()).all()
    return render_template('admin_blog.html', form=form, posts=posts)

@app.route('/admin/blog/edit/<int:post_id>', methods=['GET', 'POST'])
def edit_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    form = BlogForm()
    if form.validate_on_submit():
        try:
            file = form.author_photo.data
            filename = post.author_photo
            if file and allowed_file(file.filename):
                if post.author_photo:
                    try:
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], post.author_photo))
                    except FileNotFoundError:
                        pass
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif file:
                flash('Invalid file type', 'danger')
                return render_template('edit_blog_post.html', form=form, post=post)
            post.title = form.title.data
            post.content = form.content.data
            post.author = form.author.data
            post.author_photo = filename
            db.session.commit()
            flash('Post updated!', 'success')
            return redirect(url_for('admin_blog'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error: {str(e)}', 'danger')
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
        form.author.data = post.author
    return render_template('edit_blog_post.html', form=form, post=post)

@app.route('/admin/blog/delete/<int:post_id>', methods=['POST'])
def delete_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    try:
        if post.author_photo:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], post.author_photo))
            except FileNotFoundError:
                pass
        db.session.delete(post)
        db.session.commit()
        flash('Post deleted!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error: {str(e)}', 'danger')
    return redirect(url_for('admin_blog'))

# APPLICATION RUNNER BLOCK
# ------------------------
# Runs the app with Waitress locally or Gunicorn in production
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates database tables on startup
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
else:
    if __name__ == 'app':
        from gunicorn.app.base import BaseApplication
        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.application = app
                self.options = options or {}
                super().__init__()
            def load_config(self):
                config = {key: value for key, value in self.options.items() if key in self.cfg.settings and value is not None}
                for key, value in config.items():
                    self.cfg.set(key.lower(), value)
            def load(self):
                return self.application
        options = {'bind': '0.0.0.0:5000', 'workers': 4}
        StandaloneApplication(app, options).run()