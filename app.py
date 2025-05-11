# Imports Block
from flask import (
    Flask, render_template, request, send_from_directory, flash, redirect,
    url_for, session, send_file
)
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import (
    StringField, TextAreaField, SubmitField, IntegerField, SelectField,
    FormField, FieldList, FileField, DateTimeField
)
from wtforms.validators import DataRequired, Email, Optional
import numpy as np
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from collections import namedtuple
import subprocess
import tempfile
import io
import logging
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy_financial as npf  # For IRR calculations

# ENVIRONMENT SETUP BLOCK
# Loads environment variables from a .env file for secure configuration
load_dotenv()  # Load variables from .env

# APP INITIALIZATION BLOCK
# Creates the Flask app instance and sets up basic configuration
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Fallback for local development
app.config['UPLOAD_FOLDER'] = 'static/author_photos'

# CONFIGURATION BLOCK
# Configures the Flask app for database (SQLAlchemy) and email (Flask-Mail) integrations
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cleanvisionhr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtpout.secureserver.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'info@cleanvisionhr.com'
app.config['MAIL_PASSWORD'] = 'IFokbu@m@1'
app.config['MAIL_DEFAULT_SENDER'] = 'info@cleanvisionhr.com'

# EXTENSIONS INITIALIZATION BLOCK
# Initializes Flask-SQLAlchemy and Flask-Mail extensions
db = SQLAlchemy(app)
mail = Mail(app)

# MODELS BLOCK
# Defines database models for ContactMessage and BlogPost using Flask-SQLAlchemy
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
    author_photo = db.Column(db.String(100), nullable=True)  # Store photo filename
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Create database tables
with app.app_context():
    db.create_all()
    
    # FORMS BLOCK
# Defines WTForms classes for ContactForm and BlogForm
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
# Utility functions and namedtuples used throughout the application
DCFResult = namedtuple('DCFResult', ['total_pv', 'pv_cash_flows', 'terminal_value', 'pv_terminal', 'total_dcf'])
DVMResult = namedtuple('DVMResult', ['intrinsic_value', 'formula', 'pv_dividends', 'terminal_value', 'pv_terminal'])

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
    if pe_ratio <= 0:
        raise ValueError("P/E ratio must be positive")
    if earnings < 0:
        raise ValueError("Earnings cannot be negative")
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
    if tangible_assets < 0 or intangible_assets < 0:
        raise ValueError("Tangible and intangible assets cannot be negative")
    if not 0 <= adjustment_factor <= 1:
        raise ValueError("Adjustment factor must be between 0 and 1")
    return tangible_assets + intangible_assets * adjustment_factor

def calculate_risk_adjusted_return(returns, risk_free_rate, beta, market_return):
    if risk_free_rate < 0 or market_return < 0:
        raise ValueError("Risk-free rate and market return cannot be negative")
    if beta < 0:
        raise ValueError("Beta cannot be negative")
    expected_return = risk_free_rate / 100 + beta * (market_return / 100 - risk_free_rate / 100)
    return (returns / 100) - expected_return  # Jensen's Alpha

def parse_comma_separated(text):
    """Parse a comma-separated string into a list of floats."""
    try:
        return [float(x.strip()) for x in text.split(',')]
    except ValueError:
        raise ValueError("Invalid numeric format. Use comma-separated numbers (e.g., 0.4, 0.6).")

def parse_covariance_matrix(text, num_assets):
    """Parse a semicolon-separated covariance matrix into a numpy array."""
    try:
        rows = text.split(';')
        if len(rows) != num_assets:
            raise ValueError(f"Covariance matrix must have {num_assets} rows.")
        matrix = []
        for row in rows:
            elements = parse_comma_separated(row)
            if len(elements) != num_assets:
                raise ValueError(f"Each row must have {num_assets} elements.")
            matrix.append(elements)
        matrix = np.array(matrix)
        # Check symmetry
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Covariance matrix must be symmetric.")
        # Check positive semi-definite (all eigenvalues non-negative)
        if np.any(np.linalg.eigvals(matrix) < -1e-10):
            raise ValueError("Covariance matrix must be positive semi-definite.")
        return matrix
    except ValueError as e:
        raise ValueError(f"Invalid covariance matrix: {str(e)}")

def calculate_expected_return(weights, returns):
    """Calculate portfolio expected return."""
    if len(weights) != len(returns):
        raise ValueError("Number of weights must match number of returns.")
    if len(weights) < 1 or len(weights) > 10:
        raise ValueError("Number of assets must be between 1 and 10.")
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError("Weights must sum to 1.")
    if any(w < 0 for w in weights):
        raise ValueError("Weights must be non-negative.")
    return np.sum(np.array(weights) * np.array(returns))

def calculate_portfolio_metrics(num_assets, returns, weights, volatilities):
    if num_assets != len(returns) or num_assets != len(weights) or num_assets != len(volatilities):
        raise ValueError("Number of assets must match returns, weights, and volatilities")
    if num_assets < 1 or num_assets > 10:
        raise ValueError("Number of assets must be between 1 and 10")
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError("Portfolio weights must sum to 1")
    if any(w < 0 for w in weights) or any(v < 0 for v in volatilities):
        raise ValueError("Weights and volatilities must be non-negative")
    
    expected_return = sum(r * w for r, w in zip(returns, weights)) / 100
    # Simplified portfolio volatility (assumes no correlation)
    portfolio_volatility = np.sqrt(sum((w * v / 100) ** 2 for w, v in zip(weights, volatilities)))
    return expected_return, portfolio_volatility

def calculate_forex_profit(investment, initial_rate, final_rate):
    if any(x <= 0 for x in [investment, initial_rate, final_rate]):
        raise ValueError("All inputs must be positive")
    base_currency = investment
    foreign_currency = base_currency * initial_rate
    final_value = foreign_currency / final_rate
    profit = final_value - base_currency
    return profit

def calculate_cagr(start_value, end_value, years):
    """Calculate Compound Annual Growth Rate (CAGR) even with negative values"""
    if start_value == 0:
        return 0  # Avoid division by zero
    try:
        # Use absolute value for start_value to handle negative FCF
        cagr = (end_value / abs(start_value)) ** (1 / years) - 1
    except ZeroDivisionError:
        return 0
    return cagr * 100  # Return as percentage

def calculate_esg_metrics(esg_amount, total_portfolio, num_esg_assets, esg_scores, esg_weights):
    if esg_amount > total_portfolio or esg_amount < 0 or total_portfolio <= 0:
        raise ValueError("Invalid investment or portfolio values")
    if num_esg_assets != len(esg_scores) or num_esg_assets != len(esg_weights):
        raise ValueError("Number of ESG assets must match scores and weights")
    if num_esg_assets < 1 or num_esg_assets > 5:
        raise ValueError("Number of ESG assets must be between 1 and 5")
    if abs(sum(esg_weights) - 1.0) > 0.01:
        raise ValueError("ESG weights must sum to 1")
    if any(w < 0 for w in esg_weights) or any(s < 0 or s > 100 for s in esg_scores):
        raise ValueError("Invalid ESG scores or weights")
    
    esg_proportion = esg_amount / total_portfolio
    weighted_esg_score = sum(s * w for s, w in zip(esg_scores, esg_weights))
    return esg_proportion, weighted_esg_score

def calculate_hedge_fund_returns(strategy, investment, leverage, target_return, volatility):
    if any(x <= 0 for x in [investment, leverage, volatility]) or target_return < 0:
        raise ValueError("Invalid inputs")
    
    # Strategy-specific adjustments (simplified)
    strategy_multipliers = {
        'long-short': 1.0,
        'arbitrage': 0.8,  # Lower risk/return
        'global-macro': 1.2  # Higher risk/return
    }
    multiplier = strategy_multipliers.get(strategy, 1.0)
    leveraged_return = target_return / 100 * leverage * multiplier
    leveraged_volatility = volatility / 100 * leverage * multiplier
    expected_value = investment * (1 + leveraged_return)
    return expected_value, leveraged_return, leveraged_volatility

def calculate_dcf(fcfs, risk_free_rate, market_return, beta, debt, equity, tax_rate, growth_rate, use_exit_multiple=False, exit_ebitda_multiple=None, ebitda_last_year=None):
    assert len(fcfs) == 5, "Provide exactly 5 years of FCF"
    total_value = debt + equity
    if total_value <= 0:
        raise ValueError("Total value (debt + equity) must be positive")
    cost_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    cost_debt = 0.05
    wacc = (equity / total_value) * cost_equity + (debt / total_value) * cost_debt * (1 - tax_rate)
    pv_fcfs = sum(fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(fcfs))
    last_fcf = fcfs[-1]
    if use_exit_multiple:
        if exit_ebitda_multiple is None or ebitda_last_year is None:
            raise ValueError("Exit EBITDA multiple and last year's EBITDA required")
        terminal_value = ebitda_last_year * exit_ebitda_multiple
    else:
        if wacc <= growth_rate:
            raise ValueError(f"WACC ({wacc:.2%}) must be greater than growth rate ({growth_rate:.2%})")
        fcf_next = last_fcf * (1 + growth_rate)
        terminal_value = fcf_next / (wacc - growth_rate)
    pv_terminal = terminal_value / (1 + wacc) ** 5
    enterprise_value = pv_fcfs + pv_terminal
    equity_value = max(enterprise_value - debt, 0)
    return enterprise_value, equity_value

def calculate_vc_method(exit_value, target_roi, investment_amount, exit_horizon, dilution_factor=1.0):
    if any(x <= 0 for x in [exit_value, target_roi, investment_amount, exit_horizon]):
        raise ValueError("All inputs must be positive")
    if dilution_factor <= 0 or dilution_factor > 1:
        raise ValueError("Dilution factor must be between 0 and 1")
    adjusted_exit_value = exit_value * dilution_factor
    post_money_valuation = adjusted_exit_value / target_roi
    pre_money_valuation = post_money_valuation - investment_amount
    return pre_money_valuation, post_money_valuation

def calculate_arr_multiple(arr, arr_multiple, control_premium=0.0, illiquidity_discount=0.0):
    if arr <= 0 or arr_multiple <= 0:
        raise ValueError("ARR and ARR multiple must be positive")
    if control_premium < 0 or illiquidity_discount < 0:
        raise ValueError("Control premium and illiquidity discount cannot be negative")
    base_valuation = arr * arr_multiple
    adjusted_valuation = base_valuation * (1 + control_premium) * (1 - illiquidity_discount)
    return adjusted_valuation

def calculate_intrinsic_value_full(
    fcfs: list[float],
    risk_free_rate: float,
    market_return: float,
    beta: float,
    outstanding_shares: float,
    total_debt: float,
    cash_and_equivalents: float,
    growth_rate: float = None,
    auto_growth_rate: bool = True
) -> tuple[float, float, float]:
    """
    Calculates the intrinsic value per share, growth rate, and discount rate using perpetual growth DCF from historical FCF.
    
    Parameters:
    - fcfs: Historical Free Cash Flows (years 1-5)
    - risk_free_rate: e.g., 0.03 for 3%
    - market_return: e.g., 0.08 for 8%
    - beta: e.g., 1.2
    - outstanding_shares: Number of shares
    - total_debt: Total debt of the company
    - cash_and_equivalents: Cash and equivalents of the company
    - growth_rate: Optional manual growth rate (if not auto-computed)
    - auto_growth_rate: Whether to auto-compute growth rate from historical FCF (default: True)

    Returns:
    - intrinsic_value_per_share: Intrinsic value per share
    - g: Growth rate used in the calculation
    - discount_rate: Discount rate calculated via CAPM
    """
    
    assert len(fcfs) == 5, "Provide exactly 5 years of historical FCF"

    # Determine growth rate based on auto_growth_rate
    if auto_growth_rate:
        if fcfs[0] == 0:
            g = 0.0
        else:
            g = (fcfs[-1] / fcfs[0]) ** (1 / (len(fcfs) - 1)) - 1
    else:
        g = growth_rate

    # Use the last FCF
    last_fcf = fcfs[-1]

    # Calculate discount rate using CAPM
    discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)

    # Check if discount_rate > g with detailed error message
    if discount_rate <= g:
        raise ValueError(
            f"Discount rate ({discount_rate:.2%}) must be greater than growth rate ({g:.2%}) for the perpetual growth model."
        )

    # Calculate next year's FCF
    fcf_next = last_fcf * (1 + g)

    # Calculate enterprise value using perpetual growth formula
    enterprise_value = fcf_next / (discount_rate - g)

    # Calculate equity value
    equity_value = enterprise_value - total_debt + cash_and_equivalents

    # Calculate intrinsic value per share
    intrinsic_value_per_share = equity_value / outstanding_shares if equity_value > 0 else 0

    return intrinsic_value_per_share, g, discount_rate

def calculate_target_price(fcf, explicit_growth, n, g, r, debt, cash, shares):
    """Calculate target price for a given projection period."""
    if g >= r:
        raise ValueError("Perpetual growth rate must be less than discount rate.")
    # Project FCF for n years
    projected_fcf = [fcf * (1 + explicit_growth)**i for i in range(1, n+1)]
    # Terminal value at year n
    terminal_value = (projected_fcf[-1] * (1 + g)) / (r - g)
    # Target price (adjusted for debt/cash)
    target_price = (terminal_value - debt + cash) / shares
    return target_price

# ROUTES BLOCK
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
            form_data = {
                'num_assets': request.form['num_assets']
            }
            num_assets = int(form_data['num_assets'])
            weights = []
            returns = []
            for i in range(1, num_assets + 1):
                form_data[f'weight_{i}'] = request.form[f'weight_{i}']
                form_data[f'return_{i}'] = request.form[f'return_{i}']
                weights.append(float(form_data[f'weight_{i}']))
                returns.append(float(form_data[f'return_{i}']))
            
            expected_return = calculate_expected_return(weights, returns)
            result = f"<p>Portfolio Expected Return: {expected_return:.2%}</p>"
            return render_template('expected_return.html', result=result, form_data=form_data)
        except ValueError as e:
            error = f"Error: {str(e)}"
            return render_template('expected_return.html', error=error, form_data=form_data)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            return render_template('expected_return.html', error=error, form_data=form_data)
    return render_template('expected_return.html', form_data={})

def calculate_portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility (standard deviation) using weights and covariance matrix."""
    weights_array = np.array(weights)
    portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
    return np.sqrt(portfolio_variance)

@app.route('/volatility', methods=['GET', 'POST'])
def volatility():
    if request.method == 'POST':
        try:
            form_data = {
                'num_assets': request.form['num_assets']
            }
            num_assets = int(form_data['num_assets'])
            weights = []
            for i in range(1, num_assets + 1):
                form_data[f'weight_{i}'] = request.form[f'weight_{i}']
                weights.append(float(form_data[f'weight_{i}']))
            
            # Build covariance matrix from individual inputs
            cov_matrix = []
            for i in range(1, num_assets + 1):
                row = []
                for j in range(1, num_assets + 1):
                    form_data[f'cov_{i}_{j}'] = request.form[f'cov_{i}_{j}']
                    row.append(float(form_data[f'cov_{i}_{j}']))
                cov_matrix.append(row)
            cov_matrix = np.array(cov_matrix)
            
            # Validate covariance matrix
            if not np.allclose(cov_matrix, cov_matrix.T):
                raise ValueError("Covariance matrix must be symmetric.")
            if np.any(np.linalg.eigvals(cov_matrix) < -1e-10):
                raise ValueError("Covariance matrix must be positive semi-definite.")
            
            portfolio_volatility = calculate_portfolio_volatility(weights, cov_matrix)
            result = f"<p>Portfolio Volatility: {portfolio_volatility:.2%}</p>"
            return render_template('volatility.html', result=result, form_data=form_data)
        except ValueError as e:
            error = f"Error: {str(e)}"
            return render_template('volatility.html', error=error, form_data=form_data)
        except Exception as e:
            error = f"Unexpected error: {str(e)}"
            return render_template('volatility.html', error=error, form_data=form_data)
    return render_template('volatility.html', form_data={})

@app.route('/calculate-fcf', methods=['GET', 'POST'])
def calculate_fcf():
    fcfs = None
    ocfs = None
    capex = None
    error = None
    currency_symbol = "$"  # Configurable currency symbol

    if request.method == 'POST':
        try:
            ocfs = [float(request.form[f'ocf_{i}']) for i in range(1, 6)]
            capex = [float(request.form[f'capex_{i}']) for i in range(1, 6)]

            if any(ocf < 0 or cap < 0 for ocf, cap in zip(ocfs, capex)):
                error = "Please enter valid non-negative numbers for OCF and CAPEX."
            else:
                fcfs = [ocf - cap for ocf, cap in zip(ocfs, capex)]
        except ValueError:
            error = "Please enter valid numbers for all fields."

    return render_template('calculate_fcf.html', fcfs=fcfs, ocfs=ocfs, capex=capex, error=error, currency_symbol=currency_symbol)

@app.route('/portfolio-diversification', methods=['GET', 'POST'])
def portfolio_diversification():
    if request.method == 'POST':
        try:
            form_data = {
                'num_assets': request.form['num_assets']
            }
            num_assets = int(request.form['num_assets'])
            returns = []
            weights = []
            volatilities = []
            for i in range(1, num_assets + 1):
                form_data[f'return_{i}'] = request.form[f'return_{i}']
                form_data[f'weight_{i}'] = request.form[f'weight_{i}']
                form_data[f'volatility_{i}'] = request.form[f'volatility_{i}']
                returns.append(float(form_data[f'return_{i}']))
                weights.append(float(form_data[f'weight_{i}']))
                volatilities.append(float(form_data[f'volatility_{i}']))

            expected_return, portfolio_volatility = calculate_portfolio_metrics(num_assets, returns, weights, volatilities)
            result = f"""
                <p>Portfolio Expected Return: {expected_return:.2%}</p>
                <p>Portfolio Volatility (Risk): {portfolio_volatility:.2%}</p>
            """
            return render_template('portfolio_diversification.html', result=result, form_data=form_data)
        except ValueError as e:
            result = f"Error: {str(e)}"
            return render_template('portfolio_diversification.html', result=result, form_data=request.form)
    return render_template('portfolio_diversification.html', form_data={})

# DCF Calculator Route
@app.route('/dcf', methods=['GET', 'POST'])
def dcf_calculator():
    error = None
    results = None
    if request.method == 'POST':
        try:
            # Validate and parse inputs
            years = int(request.form.get('years', 0))
            if years < 1 or years > 10:
                raise ValueError("Forecast period must be between 1 and 10 years.")
            
            discount_rate = float(request.form.get('discount_rate', '')) / 100
            terminal_growth = float(request.form.get('terminal_growth', '')) / 100
            if discount_rate <= 0:
                raise ValueError("Discount rate must be positive.")
            if discount_rate <= terminal_growth:
                raise ValueError("Discount rate must exceed terminal growth rate.")

            # Collect cash flows
            cash_flows = []
            for i in range(1, years + 1):
                cf = request.form.get(f'cash_flow_{i}', '')
                if not cf:
                    raise ValueError(f"Cash flow for Year {i} is missing.")
                cash_flows.append(float(cf))

            # Perform DCF calculations
            pv_cash_flows = [cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows, 1)]
            total_pv = sum(pv_cash_flows)
            last_cash_flow = cash_flows[-1]
            terminal_value = (last_cash_flow * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** years
            total_dcf = total_pv + pv_terminal

            results = DCFResult(total_pv, pv_cash_flows, terminal_value, pv_terminal, total_dcf)
        except ValueError as e:
            error = str(e) if str(e) else "Invalid input detected."
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"
    return render_template('dcf.html', error=error, results=results)

# DVM Calculator Route
@app.route('/dvm', methods=['GET', 'POST'])
def dvm_calculator():
    error = None
    results = None
    model_type = request.form.get('model_type', 'gordon_growth') if request.method == 'POST' else 'gordon_growth'

    if request.method == 'POST':
        try:
            r = float(request.form.get('r', '')) / 100
            if r <= 0:
                raise ValueError("Discount rate must be positive.")

            if model_type == 'gordon_growth':
                d1 = float(request.form.get('d1', ''))
                g = float(request.form.get('g', '')) / 100
                if r <= g:
                    raise ValueError("Discount rate must exceed growth rate.")
                result = calculate_gordon_growth(d1, r, g)
                results = DVMResult(
                    intrinsic_value=result['intrinsic_value'],
                    formula=result['formula']
                )

            elif model_type == 'multi_stage':
                periods = int(request.form.get('periods', 0))
                if periods < 1 or periods > 5:
                    raise ValueError("Growth periods must be between 1 and 5.")
                terminal_growth = float(request.form.get('terminal_growth', '')) / 100
                if r <= terminal_growth:
                    raise ValueError("Discount rate must exceed terminal growth rate.")
                
                dividends = []
                for i in range(periods):
                    d = request.form.get(f'dividend_{i+1}', '')
                    if not d:
                        raise ValueError(f"Dividend for Year {i+1} is missing.")
                    dividends.append(float(d))
                
                result = calculate_multi_stage(dividends, r, terminal_growth)
                results = DVMResult(
                    intrinsic_value=result['intrinsic_value'],
                    pv_dividends=result['pv_dividends'],
                    terminal_value=result['terminal_value'],
                    pv_terminal=result['pv_terminal']
                )

            elif model_type == 'no_growth':
                d = float(request.form.get('d', ''))
                result = calculate_no_growth(d, r)
                results = DVMResult(
                    intrinsic_value=result['intrinsic_value'],
                    formula=result['formula']
                )

            else:
                raise ValueError("Invalid model type selected.")

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = f"An unexpected error occurred: {str(e)}"

    return render_template('dvm.html', error=error, results=results, model_type=model_type)

# Add these calculation functions
def calculate_gordon_growth(d1, r, g):
    intrinsic_value = d1 / (r - g)
    formula = f"{d1:.2f} / ({r*100:.2f}% - {g*100:.2f}%)"
    return {'intrinsic_value': intrinsic_value, 'formula': formula}

def calculate_multi_stage(dividends, r, terminal_growth):
    pv_dividends = []
    total_pv = 0
    for i, d in enumerate(dividends):
        pv = d / ((1 + r) ** (i + 1))
        pv_dividends.append(pv)
        total_pv += pv
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

@app.route('/forex', methods=['GET', 'POST'])
def forex_calculator():
    if request.method == 'POST':
        try:
            form_data = {
                'investment': request.form['investment'],
                'initial_rate': request.form['initial_rate'],
                'final_rate': request.form['final_rate']
            }
            investment = float(form_data['investment'])
            initial_rate = float(form_data['initial_rate'])
            final_rate = float(form_data['final_rate'])

            profit = calculate_forex_profit(investment, initial_rate, final_rate)
            result = f"""
                <p>Forex Profit/Loss: ${profit:,.2f}</p>
                <p>Initial Investment: ${investment:,.2f}</p>
                <p>Initial Exchange Rate: {initial_rate:.4f}</p>
                <p>Final Exchange Rate: {final_rate:.4f}</p>
            """
            return render_template('forex.html', result=result, form_data=form_data)
        except ValueError as e:
            result = f"Error: {str(e)}"
            return render_template('forex.html', result=result, form_data=request.form)
    return render_template('forex.html', form_data={})

# Valuation Route
@app.route('/valuation_methods', methods=['GET', 'POST'])
def valuation_methods():
    selected_method = request.form.get('method', 'CCA')  # Default to CCA
    if request.method == 'POST':
        try:
            method = request.form['method']
            result = {'method': method.replace('_', ' ').title()}

            # Helper function to safely convert form input to float
            def safe_float(value):
                return float(value) if value.strip() else 0.0

            # Handle each valuation method
            if method == 'CCA':
                pe_ratio = safe_float(request.form.get('pe_ratio', '0'))
                earnings = safe_float(request.form.get('earnings', '0'))
                result['value'] = f"GHS {calculate_cca(pe_ratio, earnings):,.2f}"

            elif method == 'NAV':
                assets = safe_float(request.form.get('assets', '0'))
                liabilities = safe_float(request.form.get('liabilities', '0'))
                result['value'] = f"GHS {calculate_nav(assets, liabilities):,.2f}"

            elif method == 'Market Capitalization':
                share_price = safe_float(request.form.get('share_price', '0'))
                shares_outstanding = safe_float(request.form.get('shares_outstanding', '0'))
                result['value'] = f"GHS {calculate_market_cap(share_price, shares_outstanding):,.2f}"

            elif method == 'EV':
                market_cap = safe_float(request.form.get('market_cap', '0'))
                debt = safe_float(request.form.get('debt', '0'))
                cash = safe_float(request.form.get('cash', '0'))
                result['value'] = f"GHS {calculate_ev(market_cap, debt, cash):,.2f}"

            elif method == 'Replacement Cost':
                tangible_assets = safe_float(request.form.get('tangible_assets', '0'))
                intangible_assets = safe_float(request.form.get('intangible_assets', '0'))
                adjustment_factor = safe_float(request.form.get('adjustment_factor', '1'))
                result['value'] = f"GHS {calculate_replacement_cost(tangible_assets, intangible_assets, adjustment_factor):,.2f}"

            elif method == 'Risk-Adjusted Return':
                returns = safe_float(request.form.get('returns', '0'))
                risk_free_rate = safe_float(request.form.get('risk_free_rate', '0'))
                beta = safe_float(request.form.get('beta', '0'))
                market_return = safe_float(request.form.get('market_return', '0'))
                result['value'] = f"{calculate_risk_adjusted_return(returns, risk_free_rate, beta, market_return):.2%}"

            return render_template('valuation_methods.html', result=result, selected_method=method)

        except ValueError as ve:
            return render_template('valuation_methods.html', error=str(ve), selected_method=method)
        except ZeroDivisionError:
            return render_template('valuation_methods.html', error="Division by zero error. Please check inputs.", selected_method=method)
        except Exception as e:
            return render_template('valuation_methods.html', error=f"An unexpected error occurred: {str(e)}", selected_method=method)

    return render_template('valuation_methods.html', selected_method=selected_method)

@app.route('/esg-investments', methods=['GET', 'POST'])
def esg_investments():
    if request.method == 'POST':
        try:
            form_data = {
                'esg_amount': request.form['esg_amount'],
                'total_portfolio': request.form['total_portfolio'],
                'num_esg_assets': request.form['num_esg_assets']
            }
            esg_amount = float(form_data['esg_amount'])
            total_portfolio = float(form_data['total_portfolio'])
            num_esg_assets = int(form_data['num_esg_assets'])
            esg_scores = []
            esg_weights = []
            for i in range(1, num_esg_assets + 1):
                form_data[f'esg_score_{i}'] = request.form[f'esg_score_{i}']
                form_data[f'esg_weight_{i}'] = request.form[f'esg_weight_{i}']
                esg_scores.append(float(form_data[f'esg_score_{i}']))
                esg_weights.append(float(form_data[f'esg_weight_{i}']))

            esg_proportion, weighted_esg_score = calculate_esg_metrics(
                esg_amount, total_portfolio, num_esg_assets, esg_scores, esg_weights
            )
            result = f"""
                <p>ESG Investment Proportion: {esg_proportion:.2%}</p>
                <p>Weighted ESG Score: {weighted_esg_score:.2f}/100</p>
            """
            return render_template('esg.html', result=result, form_data=form_data)
        except ValueError as e:
            result = f"Error: {str(e)}"
            return render_template('esg.html', result=result, form_data=request.form)
    return render_template('esg.html', form_data={})

@app.route('/hedge-funds', methods=['GET', 'POST'])
def hedge_funds():
    if request.method == 'POST':
        try:
            form_data = {
                'strategy': request.form['strategy'],
                'investment': request.form['investment'],
                'leverage': request.form['leverage'],
                'target_return': request.form['target_return'],
                'volatility': request.form['volatility']
            }
            strategy = form_data['strategy']
            investment = float(form_data['investment'])
            leverage = float(form_data['leverage'])
            target_return = float(form_data['target_return'])
            volatility = float(form_data['volatility'])

            expected_value, leveraged_return, leveraged_volatility = calculate_hedge_fund_returns(
                strategy, investment, leverage, target_return, volatility
            )
            result = f"""
                <p>Expected Portfolio Value: ${expected_value:,.2f}</p>
                <p>Leveraged Return: {leveraged_return:.2%}</p>
                <p>Leveraged Volatility (Risk): {leveraged_volatility:.2%}</p>
                <p>Strategy: {strategy.replace('-', ' ').title()}</p>
            """
            return render_template('hedge_funds.html', result=result, form_data=form_data)
        except ValueError as e:
            result = f"Error: {str(e)}"
            return render_template('hedge_funds.html', result=result, form_data=request.form)
    return render_template('hedge_funds.html', form_data={})

@app.route('/calculate-beta', methods=['GET', 'POST'])
def calculate_beta():
    beta = None
    error = None

    if request.method == 'POST':
        try:
            stock_returns_str = request.form['stock_returns']
            market_returns_str = request.form['market_returns']

            # Parse inputs into lists of floats
            stock_returns = [float(x.strip()) for x in stock_returns_str.split(',')]
            market_returns = [float(x.strip()) for x in market_returns_str.split(',')]

            if len(stock_returns) != len(market_returns):
                error = "Stock and market returns must have the same number of periods."
            elif len(stock_returns) < 2:
                error = "At least two periods of returns are required to calculate Beta."
            else:
                cov_matrix = np.cov(stock_returns, market_returns)
                cov_stock_market = cov_matrix[0, 1]
                var_market = cov_matrix[1, 1]
                beta = cov_stock_market / var_market if var_market != 0 else None

                if beta is None:
                    error = "Cannot calculate Beta: Market returns variance is zero."
                else:
                    beta = round(beta, 4)
        except ValueError:
            error = "Please enter valid comma-separated numbers for returns."

    return render_template('calculate_beta.html', beta=beta, error=error)

@app.route('/monthly-contribution', methods=['GET', 'POST'])
def monthly_contribution():
    result = None
    if request.method == 'POST':
        try:
            target = float(request.form['target_amount'])
            principal = float(request.form['starting_principal'])
            period = float(request.form['period'])
            rate = float(request.form['annual_return']) / 100  # Convert percentage to decimal

            if target <= 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('monthly_contribution.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                monthly_contribution = (target - principal) / months
            else:
                monthly_rate = (1 + rate) ** (1 / 12) - 1
                future_principal = principal * (1 + monthly_rate) ** months
                monthly_contribution = (target - future_principal) / (((1 + monthly_rate) ** months - 1) / monthly_rate)

            result = "{:,.2f}".format(monthly_contribution)
        except ValueError:
            return render_template('monthly_contribution.html', error="Please enter valid numbers.")

    return render_template('monthly_contribution.html', result=result)

@app.route('/end-balance', methods=['GET', 'POST'])
def end_balance():
    result = None
    if request.method == 'POST':
        try:
            monthly = float(request.form['monthly_contribution'])
            principal = float(request.form['starting_principal'])
            period = float(request.form['period'])
            rate = float(request.form['annual_return']) / 100  # Convert percentage to decimal

            if monthly < 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('end_balance.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                end_balance = principal + monthly * months
            else:
                monthly_rate = (1 + rate) ** (1 / 12) - 1
                future_principal = principal * (1 + monthly_rate) ** months
                future_contributions = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)
                end_balance = future_principal + future_contributions

            result = "{:,.2f}".format(end_balance)
        except ValueError:
            return render_template('end_balance.html', error="Please enter valid numbers.")

    return render_template('end_balance.html', result=result)

@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    result = None
    if request.method == 'POST':
        try:
            num_shares = float(request.form['num_shares'])
            purchase_price_per_share = float(request.form['purchase_price_per_share'])
            purchase_commission = float(request.form['purchase_commission']) / 100
            selling_price_per_share = float(request.form['selling_price_per_share'])
            sale_commission = float(request.form['sale_commission']) / 100
            dividends = float(request.form['dividends'])

            if any(x < 0 for x in [num_shares, purchase_price_per_share, purchase_commission, selling_price_per_share, sale_commission, dividends]):
                return render_template('stocks.html', error="Please enter valid non-negative numbers.")

            purchase_consideration = num_shares * purchase_price_per_share
            purchase_commission_amount = purchase_consideration * purchase_commission
            total_purchase_cost = purchase_consideration + purchase_commission_amount

            selling_consideration = num_shares * selling_price_per_share
            sale_commission_amount = selling_consideration * sale_commission
            net_sale_proceeds = selling_consideration - sale_commission_amount

            capital_gain = ((net_sale_proceeds - total_purchase_cost) / total_purchase_cost) * 100
            dividend_yield = (dividends / total_purchase_cost) * 100
            total_return = ((net_sale_proceeds - total_purchase_cost + dividends) / total_purchase_cost) * 100

            result = {
                'total_return': round(total_return, 2),
                'capital_gain': round(capital_gain, 2),
                'dividend_yield': round(dividend_yield, 2)
            }
        except ValueError:
            return render_template('stocks.html', error="Please enter valid numbers.")

    return render_template('stocks.html', result=result)

@app.route('/mna', methods=['GET', 'POST'])
def mna_calculator():
    if request.method == 'POST':
        try:
            # Capture form data
            form_data = {
                'acquirer_eps': request.form['acquirer_eps'],
                'acquirer_shares': request.form['acquirer_shares'],
                'target_eps': request.form['target_eps'],
                'target_shares': request.form['target_shares'],
                'new_shares_issued': request.form['new_shares_issued'],
                'synergy_value': request.form['synergy_value']
            }
            # Convert to floats for calculation
            acquirer_eps = float(form_data['acquirer_eps'])
            acquirer_shares = float(form_data['acquirer_shares'])
            target_eps = float(form_data['target_eps'])
            target_shares = float(form_data['target_shares'])
            new_shares_issued = float(form_data['new_shares_issued'])
            synergy_value = float(form_data['synergy_value'])

            # Perform M&A calculations
            acquirer_earnings = acquirer_eps * acquirer_shares
            target_earnings = target_eps * target_shares
            combined_earnings = acquirer_earnings + target_earnings + synergy_value
            total_shares = acquirer_shares + new_shares_issued
            combined_eps = combined_earnings / total_shares
            eps_change = combined_eps - acquirer_eps
            status = "Accretive" if eps_change > 0 else "Dilutive" if eps_change < 0 else "Neutral"

            # Format results
            result = f"""
                <p>Pre-Deal Acquirer EPS: ${acquirer_eps:.2f}</p>
                <p>Post-Deal Combined EPS: ${combined_eps:.2f}</p>
                <p>EPS Change: ${eps_change:.2f} ({status})</p>
                <p>Total Combined Earnings (incl. Synergy): ${combined_earnings:,.2f}</p>
                <p>Total Shares Outstanding: {total_shares:,.0f}</p>
            """
            return render_template('mna.html', result=result, form_data=form_data)
        except ValueError:
            return render_template('mna.html', result="Error: Please enter valid numeric values.", form_data=request.form)
    return render_template('mna.html', form_data={})

@app.route('/pe-vc', methods=['GET', 'POST'])
def pe_vc_valuation():
    if request.method == 'POST':
        try:
            form_data = {
                'valuation_method': request.form['valuation_method'],
                'fcf_1': request.form.get('fcf_1', ''),
                'fcf_2': request.form.get('fcf_2', ''),
                'fcf_3': request.form.get('fcf_3', ''),
                'fcf_4': request.form.get('fcf_4', ''),
                'fcf_5': request.form.get('fcf_5', ''),
                'risk_free_rate': request.form.get('risk_free_rate', ''),
                'market_return': request.form.get('market_return', ''),
                'beta': request.form.get('beta', ''),
                'debt': request.form.get('debt', ''),
                'equity': request.form.get('equity', ''),
                'tax_rate': request.form.get('tax_rate', ''),
                'growth_rate': request.form.get('growth_rate', ''),
                'use_exit_multiple': request.form.get('use_exit_multiple', 'off'),
                'exit_ebitda_multiple': request.form.get('exit_ebitda_multiple', ''),
                'ebitda_last_year': request.form.get('ebitda_last_year', ''),
                'exit_value': request.form.get('exit_value', ''),
                'target_roi': request.form.get('target_roi', ''),
                'investment_amount': request.form.get('investment_amount', ''),
                'exit_horizon': request.form.get('exit_horizon', ''),
                'dilution_factor': request.form.get('dilution_factor', '1.0'),
                'arr': request.form.get('arr', ''),
                'arr_multiple': request.form.get('arr_multiple', ''),
                'control_premium': request.form.get('control_premium', '0.0'),
                'illiquidity_discount': request.form.get('illiquidity_discount', '0.0')
            }
            valuation_method = form_data['valuation_method']
            result = ""

            if valuation_method == "dcf":
                fcfs = [float(form_data[f'fcf_{i}']) for i in range(1, 6)]
                risk_free_rate = float(form_data['risk_free_rate']) / 100
                market_return = float(form_data['market_return']) / 100
                beta = float(form_data['beta'])
                debt = float(form_data['debt'])
                equity = float(form_data['equity'])
                tax_rate = float(form_data['tax_rate']) / 100
                growth_rate = float(form_data['growth_rate']) / 100
                use_exit_multiple = form_data['use_exit_multiple'] == 'on'
                exit_ebitda_multiple = float(form_data['exit_ebitda_multiple']) if form_data['exit_ebitda_multiple'] else None
                ebitda_last_year = float(form_data['ebitda_last_year']) if form_data['ebitda_last_year'] else None

                enterprise_value, equity_value = calculate_dcf(
                    fcfs, risk_free_rate, market_return, beta, debt, equity, tax_rate,
                    growth_rate, use_exit_multiple, exit_ebitda_multiple, ebitda_last_year
                )
                result = f"""
                    <p>DCF Valuation:</p>
                    <p>Enterprise Value: ${enterprise_value:,.2f}</p>
                    <p>Equity Value: ${equity_value:,.2f}</p>
                """

            elif valuation_method == "vc":
                exit_value = float(form_data['exit_value'])
                target_roi = float(form_data['target_roi'])
                investment_amount = float(form_data['investment_amount'])
                exit_horizon = float(form_data['exit_horizon'])
                dilution_factor = float(form_data['dilution_factor'])

                pre_money_valuation, post_money_valuation = calculate_vc_method(
                    exit_value, target_roi, investment_amount, exit_horizon, dilution_factor
                )
                result = f"""
                    <p>VC Method Valuation:</p>
                    <p>Pre-Money Valuation: ${pre_money_valuation:,.2f}</p>
                    <p>Post-Money Valuation: ${post_money_valuation:,.2f}</p>
                """

            elif valuation_method == "arr":
                arr = float(form_data['arr'])
                arr_multiple = float(form_data['arr_multiple'])
                control_premium = float(form_data['control_premium']) / 100
                illiquidity_discount = float(form_data['illiquidity_discount']) / 100

                valuation = calculate_arr_multiple(arr, arr_multiple, control_premium, illiquidity_discount)
                result = f"""
                    <p>ARR Multiple Valuation:</p>
                    <p>Valuation: ${valuation:,.2f}</p>
                """

            return render_template('pe_vc.html', result=result, form_data=form_data)
        except ValueError as e:
            result = f"Error: {str(e)}"
            return render_template('pe_vc.html', result=result, form_data=form_data)
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

            if any(x < 0 for x in [principal, tenor, rate, total_coupons]) or principal == 0:
                return render_template('bonds.html', error="Please enter valid positive numbers.")

            maturity_amount = principal + total_coupons
            bond_yield = (total_coupons + (maturity_amount - principal)) / (principal * (tenor / 365)) * 100

            result = {
                'maturity_amount': "{:,.2f}".format(maturity_amount),
                'bond_yield': round(bond_yield, 2)
            }
        except ValueError:
            return render_template('bonds.html', error="Please enter valid numbers.")
    return render_template('bonds.html', result=result)

@app.route('/tbills', methods=['GET', 'POST'])
def tbills():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100
            tenor = float(request.form['tenor'])

            if principal <= 0 or rate < 0 or tenor <= 0:
                return render_template('tbills.html', error="Please enter valid positive numbers.")

            interest = (principal * tenor * rate) / 364
            maturity_value = principal + interest

            result = {
                'maturity_value': "{:,.2f}".format(maturity_value)
            }
        except ValueError:
            return render_template('tbills.html', error="Please enter valid numbers.")

    return render_template('tbills.html', result=result)

@app.route('/mutual-funds', methods=['GET', 'POST'])
def mutual_funds():
    result = None
    if request.method == 'POST':
        try:
            nav_start = float(request.form['nav_start'])
            nav_end = float(request.form['nav_end'])
            dividends = float(request.form['dividends'])

            if nav_start <= 0 or nav_end < 0 or dividends < 0:
                return render_template('mutual_funds.html', error="Please enter valid numbers. NAV at Start must be positive.")

            total_return = (nav_end - nav_start + dividends) / nav_start * 100
            result = {
                'total_return': round(total_return, 2)
            }
        except ValueError:
            return render_template('mutual_funds.html', error="Please enter valid numbers.")

    return render_template('mutual_funds.html', result=result)

@app.route('/duration', methods=['GET', 'POST'])
def duration():
    if request.method == 'POST':
        try:
            # Collect inputs
            num_periods = int(request.form['num_periods'])
            cash_flows = []
            for i in range(1, num_periods + 1):
                cf = request.form.get(f'cf_{i}', '')
                if not cf:
                    return render_template('duration.html', error=f"Cash flow for period {i} is required.")
                cash_flows.append(float(cf))

            yield_rate = float(request.form['yield']) / 100  # Convert % to decimal
            compounding = int(request.form['compounding'])
            initial_price = float(request.form['initial_price'])
            price_drop = float(request.form['price_drop'])
            price_rise = float(request.form['price_rise'])
            yield_change = 0.01  # Fixed 1% for Effective Duration

            # Validate inputs
            if num_periods < 1 or num_periods > 10:
                return render_template('duration.html', error="Number of periods must be between 1 and 10.")
            if yield_rate < 0 or initial_price <= 0 or compounding < 1:
                return render_template('duration.html', error="Invalid input: Yield, initial price, and compounding must be positive.")
            if any(cf <= 0 for cf in cash_flows):
                return render_template('duration.html', error="All cash flows must be positive.")

            # Macaulay Duration (in years)
            pv_sum = 0
            weighted_pv_sum = 0
            yield_per_period = yield_rate / compounding
            for t in range(1, num_periods + 1):
                pv = cash_flows[t-1] / (1 + yield_per_period) ** t
                pv_sum += pv
                weighted_pv_sum += (t / compounding) * pv  # Time in years

            if pv_sum == 0:
                return render_template('duration.html', error="Invalid cash flows: Sum of present values is zero.")

            macaulay_duration = weighted_pv_sum / pv_sum  # Now in years

            # Modified Duration
            modified_duration = macaulay_duration / (1 + yield_rate / compounding)

            # Effective Duration
            effective_duration = (price_drop - price_rise) / (2 * yield_change * initial_price)

            # Prepare results with rounding
            result = {
                'macaulay_duration': round(macaulay_duration, 2),
                'modified_duration': round(modified_duration, 2),
                'effective_duration': round(effective_duration, 2)
            }

            return render_template('duration.html', result=result)

        except (ValueError, ZeroDivisionError) as e:
            return render_template('duration.html', error="Invalid input: Please ensure all fields are valid numbers.")
        except Exception as e:
            return render_template('duration.html', error="An unexpected error occurred. Please try again.")

    return render_template('duration.html')

@app.route('/portfolio_return', methods=['GET', 'POST'])
def portfolio_return():
    if request.method == 'POST':
        try:
            # Collect inputs
            method = request.form['method']
            data_input = request.form['data'].strip()
            if not data_input:
                raise ValueError("Data field cannot be empty.")
            data = [float(x) for x in data_input.split(',') if x.strip()]
            average_inflation = float(request.form['average_inflation']) / 100
            monthly_inflation = request.form.get('monthly_inflation', '').strip()

            # Validate inputs based on method
            if method == 'twr' and len(data) < 2:
                raise ValueError("Time-Weighted Return requires at least two periodic returns.")
            elif method == 'mwr' and (len(data) < 2 or data[0] >= 0):
                raise ValueError("Money-Weighted Return requires cash flows starting with a negative initial investment.")
            elif method == 'modified_dietz' and len(data) != 4:
                raise ValueError("Modified Dietz requires 4 inputs: initial value, final value, cash flow, weight.")
            elif method == 'simple_dietz' and len(data) != 3:
                raise ValueError("Simple Dietz requires 3 inputs: initial value, final value, cash flow.")
            elif method == 'irr' and (len(data) < 2 or data[0] >= 0):
                raise ValueError("Internal Rate of Return requires cash flows starting with a negative initial investment.")
            elif method == 'hpr' and len(data) != 3:
                raise ValueError("Holding Period Return requires 3 inputs: initial price, final price, dividend.")
            elif method == 'annualized' and len(data) != 2:
                raise ValueError("Annualized Return requires 2 inputs: total return, number of years.")
            elif method in ['geometric_mean', 'arithmetic_mean'] and len(data) < 1:
                raise ValueError(f"{method.replace('_', ' ').title()} requires at least one periodic return.")

            # Compute nominal return
            if method == 'twr':
                nominal_return = calculate_twr(data)
            elif method == 'mwr':
                nominal_return = calculate_mwr(data)
            elif method == 'modified_dietz':
                mv0, mv1, cash_flow, weight = data
                nominal_return = calculate_modified_dietz(mv0, mv1, cash_flow, weight)
            elif method == 'simple_dietz':
                mv0, mv1, cash_flow = data
                nominal_return = calculate_simple_dietz(mv0, mv1, cash_flow)
            elif method == 'irr':
                nominal_return = calculate_irr(data)
            elif method == 'hpr':
                p0, p1, dividend = data
                nominal_return = calculate_hpr(p0, p1, dividend)
            elif method == 'annualized':
                r, n = data
                nominal_return = calculate_annualized_return(r, n)
            elif method == 'geometric_mean':
                nominal_return = calculate_geometric_mean_return(data)
            elif method == 'arithmetic_mean':
                nominal_return = calculate_arithmetic_mean_return(data)

            # Calculate real return using average inflation
            real_return_avg = calculate_real_return(nominal_return, average_inflation)

            # Calculate real return using time-weighted inflation (if provided)
            real_return_tw = None
            if monthly_inflation:
                monthly_inflations = [float(x) for x in monthly_inflation.split(',') if x.strip()]
                if not monthly_inflations:
                    raise ValueError("Monthly inflation rates must be valid numbers.")
                tw_inflation = calculate_time_weighted_inflation(monthly_inflations)
                real_return_tw = calculate_real_return(nominal_return, tw_inflation)

            # Prepare results
            result = {
                'method': method.replace('_', ' ').title(),
                'nominal_return': f"{nominal_return:.2%}" if nominal_return is not None else "N/A",
                'real_return_avg': f"{real_return_avg:.2%}" if real_return_avg is not None else "N/A",
                'real_return_tw': f"{real_return_tw:.2%}" if real_return_tw is not None else "Not Provided"
            }

            return render_template('portfolio_return.html', result=result)

        except ValueError as ve:
            return render_template('portfolio_return.html', error=str(ve))
        except RuntimeError as re:
            return render_template('portfolio_return.html', error="IRR calculation failed to converge. Please check cash flows.")
        except Exception as e:
            return render_template('portfolio_return.html', error=f"An unexpected error occurred: {e}")

    return render_template('portfolio_return.html')

@app.route('/etfs', methods=['GET', 'POST'])
def etfs():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            dividends = float(request.form['dividends'])

            if purchase_price <= 0 or selling_price < 0 or dividends < 0:
                return render_template('etfs.html', error="Please enter valid numbers. Purchase Price must be positive.")

            total_return = (selling_price - purchase_price + dividends) / purchase_price * 100
            result = {
                'total_return': round(total_return, 2)
            }
        except ValueError:
            return render_template('etfs.html', error="Please enter valid numbers.")

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

            if principal <= 0 or rate < 0 or compounding_periods <= 0 or years <= 0:
                return render_template('cds.html', error="Please enter valid positive numbers.")

            fv = principal * (1 + rate / compounding_periods) ** (years * compounding_periods)
            result = {
                'future_value': "{:,.2f}".format(fv)
            }
        except ValueError:
            return render_template('cds.html', error="Please enter valid numbers.")

    return render_template('cds.html', result=result)

@app.route('/money-market', methods=['GET', 'POST'])
def money_market():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            rate = float(request.form['rate']) / 100
            days_held = float(request.form['days_held'])

            if principal <= 0 or rate < 0 or days_held <= 0:
                return render_template('money_market.html', error="Please enter valid positive numbers.")

            interest_earned = principal * rate * (days_held / 365)
            result = {
                'interest_earned': "{:,.2f}".format(interest_earned)
            }
        except ValueError:
            return render_template('money_market.html', error="Please enter valid numbers.")

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

            if stock_price < 0 or strike_price < 0 or premium < 0:
                return render_template('options.html', error="Please enter valid non-negative numbers.")

            if option_type == 'call':
                profit = max(stock_price - strike_price - premium, 0)
            elif option_type == 'put':
                profit = max(strike_price - stock_price - premium, 0)
            else:
                return render_template('options.html', error="Invalid option type.")

            result = {
                'profit': "{:,.2f}".format(profit)
            }
        except ValueError:
            return render_template('options.html', error="Please enter valid numbers.")

    return render_template('options.html', result=result)

@app.route('/futures', methods=['GET', 'POST'])
def futures():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            contract_size = float(request.form['contract_size'])

            if purchase_price < 0 or selling_price < 0 or contract_size <= 0:
                return render_template('futures.html', error="Please enter valid numbers. Contract Size must be positive.")

            profit = (selling_price - purchase_price) * contract_size
            result = {
                'profit': "{:,.2f}".format(profit)
            }
        except ValueError:
            return render_template('futures.html', error="Please enter valid numbers.")

    return render_template('futures.html', result=result)

@app.route('/cryptocurrency', methods=['GET', 'POST'])
def cryptocurrency():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])

            if purchase_price <= 0 or selling_price < 0:
                return render_template('cryptocurrency.html', error="Please enter valid numbers. Purchase Price must be positive.")

            total_return = (selling_price - purchase_price) / purchase_price * 100
            result = {
                'total_return': round(total_return, 2)
            }
        except ValueError:
            return render_template('cryptocurrency.html', error="Please enter valid numbers.")

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
        user_email = contact_message.email
        name = contact_message.name
        html_body = render_template('contact_confirmation.html', name=name)
        auto_response_msg = Message(subject="Thanks for Reaching Out! We’ll Get Back to You Soon", sender=("Admin", "info@cleanvisionhr.com"), recipients=[user_email], html=html_body)
        mail.send(auto_response_msg)
        flash('Your message has been sent successfully!', 'success')
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

            if any(x < 0 for x in [principal, holding_period, selling_price, total_coupons]) or principal == 0 or holding_period == 0:
                return render_template('early_exit.html', error="Please enter valid positive numbers.")

            holding_period_return = (total_coupons + (selling_price - principal)) / (principal * (holding_period / 365)) * 100

            result = {
                'holding_period_return': round(holding_period_return, 2)
            }
        except ValueError:
            return render_template('early_exit.html', error="Please enter valid numbers.")
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

            if settlement_amount <= 0 or rate < 0 or days_to_maturity <= 0 or initial_fv <= 0:
                return render_template('tbills_rediscount.html', error="Please enter valid positive numbers.")

            settlement_fv = settlement_amount * (1 + rate) ** (days_to_maturity / 364)
            face_value_after_rediscount = initial_fv - settlement_fv

            result = {
                'settlement_fv': "{:,.2f}".format(settlement_fv),
                'face_value_after_rediscount': "{:,.2f}".format(face_value_after_rediscount)
            }
        except ValueError:
            return render_template('tbills_rediscount.html', error="Please enter valid numbers.")

    return render_template('tbills_rediscount.html', result=result)

# Intrinsic value calculation route
@app.route('/intrinsic-value', methods=['GET', 'POST'])
def intrinsic_value():
    if request.method == 'POST':
        try:
            # Allow negative FCF values
            fcf = [float(request.form[f'fcf_{i}']) for i in range(1, 6)]
            risk_free_rate = float(request.form['risk_free_rate']) / 100
            market_return = float(request.form['market_return']) / 100
            beta = float(request.form['beta'])
            outstanding_shares = float(request.form['outstanding_shares'])
            total_debt = float(request.form['total_debt'])
            cash_and_equivalents = float(request.form['cash_and_equivalents'])
            projection_period = int(request.form['projection_period'])

            # Calculate discount rate using CAPM
            discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)

            # Calculate growth rate from historical FCF (CAGR)
            perpetual_growth_rate = calculate_cagr(fcf[0], fcf[-1], 4) / 100

            # Critical validation: discount rate must exceed growth rate
            if discount_rate <= perpetual_growth_rate:
                return render_template('intrinsic_value.html', 
                                     error="Discount rate must exceed growth rate for valid calculations.")

            # Intrinsic Value Calculation
            last_fcf = fcf[-1]
            enterprise_value = (last_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
            equity_value = enterprise_value - total_debt + cash_and_equivalents
            intrinsic_value = equity_value / outstanding_shares

            # Target Price Calculation
            projected_fcf = last_fcf * (1 + perpetual_growth_rate) ** projection_period
            terminal_value = (projected_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
            target_equity_value = terminal_value - total_debt + cash_and_equivalents
            target_price = target_equity_value / outstanding_shares

            return render_template(
                'intrinsic_value.html',
                result=f"{intrinsic_value:.2f}",
                target_price=f"{target_price:.2f}",
                projection_period=projection_period
            )
        except ValueError:
            return render_template('intrinsic_value.html', error="Please enter valid numeric values.")
    return render_template('intrinsic_value.html')

@app.route('/blog')
def blog():
    posts = BlogPost.query.order_by(BlogPost.date_posted.desc()).all()
    return render_template('blog.html', posts=posts)

@app.route('/blog/<int:post_id>')
def blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    return render_template('blog_post.html', post=post)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            elif file and not allowed_file(file.filename):
                flash('Invalid file type. Allowed types: PNG, JPG, JPEG, GIF.', 'danger')
                return render_template('admin_blog.html', form=form, posts=BlogPost.query.order_by(BlogPost.date_posted.desc()).all())

            new_post = BlogPost(
                title=form.title.data,
                content=form.content.data,
                author=form.author.data,
                author_photo=filename
            )
            db.session.add(new_post)
            db.session.commit()
            flash('Blog post created successfully!', 'success')
            return redirect(url_for('admin_blog'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error creating post: {str(e)}', 'danger')
    
    posts = BlogPost.query.order_by(BlogPost.date_posted.desc()).all()
    return render_template('admin_blog.html', form=form, posts=posts)

@app.route('/admin/blog/edit/<int:post_id>', methods=['GET', 'POST'])
def edit_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    form = BlogForm()
    
    if form.validate_on_submit():
        try:
            file = form.author_photo.data
            filename = post.author_photo  # Keep existing photo if no new upload
            if file and allowed_file(file.filename):
                # Remove old photo if it exists
                if post.author_photo:
                    try:
                        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], post.author_photo))
                    except FileNotFoundError:
                        pass
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            elif file and not allowed_file(file.filename):
                flash('Invalid file type. Allowed types: PNG, JPG, JPEG, GIF.', 'danger')
                return render_template('edit_blog_post.html', form=form, post=post)

            post.title = form.title.data
            post.content = form.content.data
            post.author = form.author.data
            post.author_photo = filename
            db.session.commit()
            flash('Blog post updated successfully!', 'success')
            return redirect(url_for('admin_blog'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating post: {str(e)}', 'danger')
    
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
        form.author.data = post.author
    
    return render_template('edit_blog_post.html', form=form, post=post)

@app.route('/admin/blog/delete/<int:post_id>', methods=['POST'])
def delete_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    try:
        # Remove associated photo file
        if post.author_photo:
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], post.author_photo))
            except FileNotFoundError:
                pass
        
        db.session.delete(post)
        db.session.commit()
        flash('Blog post deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting post: {str(e)}', 'danger')
    
    return redirect(url_for('admin_blog'))

# APPLICATION RUNNER BLOCK
# Code to run the Flask app with Waitress (local) or Gunicorn (production)

if __name__ == '__main__':
    # For local development, use Waitress
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
else:
    # For production on Render, use Gunicorn
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

        options = {
            'bind': '0.0.0.0:5000',
            'workers': 4,  # Adjust based on your needs
        }
        StandaloneApplication(app, options).run()


# TEMPLATE FILTERS BLOCK
# Custom Jinja2 filters for use in templates

@app.template_filter('commafy')
def commafy(value):
    return "{:,.2f}".format(value)