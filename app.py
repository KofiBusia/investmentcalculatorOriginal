# Investment Calculator Flask Application
# File: app.py
# Description: A Flask web application for financial calculations including DCF, valuation methods,
#              leverage ratios, and more, with SQLAlchemy for database integration and WTForms for input validation.

# --- STANDARD LIBRARY IMPORTS ---
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# --- THIRD-PARTY IMPORTS ---
from flask import Flask, jsonify, render_template, request, send_from_directory, session, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Length, NumberRange
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_session import Session
from dotenv import load_dotenv
import statistics

# --- ENVIRONMENT CONFIGURATION ---
load_dotenv()
logger = logging.getLogger(__name__)

# --- FLASK APPLICATION INITIALIZATION ---
app = Flask(__name__)

# --- CONFIGURATION SETTINGS ---
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'e1efa2b32b1bac66588d074bac02a168212082d8befd0b6466f5ee37a8c2836a'),
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5 MB limit
    SESSION_TYPE='filesystem',
    SESSION_FILE_THRESHOLD=500,
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=86400,
    WTF_CSRF_TIME_LIMIT=7200,
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'sqlite:///site.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SESSION_FILE_DIR=os.path.join(os.path.dirname(__file__), 'instance', 'sessions')
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

# --- SQLALCHEMY INITIALIZATION ---
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --- DATABASE MODELS ---
class ValuationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    period_name = db.Column(db.String(10), nullable=False)
    currency = db.Column(db.String(3), nullable=False)
    weighted_average = db.Column(db.Float, nullable=False)
    pb_value = db.Column(db.Float, nullable=False)
    ptbv_value = db.Column(db.Float, nullable=False)
    pe_value = db.Column(db.Float, nullable=False)
    ddm_value = db.Column(db.Float, nullable=False)

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

class FCFEForm(FlaskForm):
    net_income = FloatField('Net Income', validators=[DataRequired(), NumberRange(min=-1000000, max=1000000)])
    capex = FloatField('Capital Expenditures', validators=[DataRequired(), NumberRange(min=0, max=1000000)])
    depreciation = FloatField('Depreciation', validators=[DataRequired(), NumberRange(min=0, max=1000000)])
    change_in_working_capital = FloatField('Change in Working Capital', validators=[DataRequired(), NumberRange(min=-1000000, max=1000000)])
    debt_issued = FloatField('Debt Issued', validators=[DataRequired(), NumberRange(min=0, max=1000000)])
    debt_repaid = FloatField('Debt Repaid', validators=[DataRequired(), NumberRange(min=0, max=1000000)])

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

def calculate_fcfe(net_income, capex, depreciation, change_in_working_capital, debt_issued, debt_repaid):
    """Calculate Free Cash Flow to Equity (FCFE)."""
    return net_income + depreciation - capex - change_in_working_capital + debt_issued - debt_repaid

# --- ROUTES ---
@app.route('/')
def index():
    """Render the homepage."""
    logger.debug("Rendering index page")
    return render_template('index.html')

@app.route('/help')
def help():
    """Render the help page with calculator information."""
    try:
        with open('calculators.json') as f:
            calculators = json.load(f)
    except FileNotFoundError:
        calculators = []
        logger.warning("calculators.json not found, returning empty list")
    return render_template('help.html', calculators=calculators)

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

@app.route('/valuation_methods', methods=['GET', 'POST'])
def valuation_methods():
    """Handle valuation methods form and calculations."""
    selected_method = request.form.get('method', 'CCA') if request.method == 'POST' else 'CCA'
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    error = None
    result = None
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
            logger.info(f"Valuation method {method} calculated successfully")
        except ValueError as e:
            error = str(e)
            logger.error(f"Valuation methods error: {e}")
    return render_template('valuation_methods.html', result=result, selected_method=selected_method, error=error, form_data=form_data)

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

@app.route('/valuation-performance', methods=['GET', 'POST'])
def valuation_performance():
    """Handle valuation performance multiples calculations."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            data = request.form
            selected_formula = data.get('formula')
            results = []
            input_data = []
            years_filled = 0

            for i in range(1, 6):
                try:
                    market_cap = float(data.get(f'market_cap_{i}', 0)) if data.get(f'market_cap_{i}') else None
                    total_debt = float(data.get(f'total_debt_{i}', 0)) if data.get(f'total_debt_{i}') else None
                    preferred_stock = float(data.get(f'preferred_stock_{i}', 0)) if data.get(f'preferred_stock_{i}') else None
                    minority_interest = float(data.get(f'minority_interest_{i}', 0)) if data.get(f'minority_interest_{i}') else None
                    cash = float(data.get(f'cash_{i}', 0)) if data.get(f'cash_{i}') else None
                    non_operating_assets = float(data.get(f'non_operating_assets_{i}', 0)) if data.get(f'non_operating_assets_{i}') else None
                    ebitda = float(data.get(f'ebitda_{i}', 0)) if data.get(f'ebitda_{i}') else None
                    ebit = float(data.get(f'ebit_{i}', 0)) if data.get(f'ebit_{i}') else None
                    revenue = float(data.get(f'revenue_{i}', 0)) if data.get(f'revenue_{i}') else None
                    net_income = float(data.get(f'net_income_{i}', 0)) if data.get(f'net_income_{i}') else None
                    equity = float(data.get(f'equity_{i}', 0)) if data.get(f'equity_{i}') else None
                    total_assets = float(data.get(f'total_assets_{i}', 0)) if data.get(f'total_assets_{i}') else None
                    avg_total_assets = float(data.get(f'avg_total_assets_{i}', 0)) if data.get(f'avg_total_assets_{i}') else None
                    avg_equity = float(data.get(f'avg_equity_{i}', 0)) if data.get(f'avg_equity_{i}') else None
                    share_price = float(data.get(f'share_price_{i}', 0)) if data.get(f'share_price_{i}') else None
                    eps = float(data.get(f'eps_{i}', 0)) if data.get(f'eps_{i}') else None
                    bvps = float(data.get(f'bvps_{i}', 0)) if data.get(f'bvps_{i}') else None
                    eps_growth = float(data.get(f'eps_growth_{i}', 0)) if data.get(f'eps_growth_{i}') else None
                    tax_rate = float(data.get(f'tax_rate_{i}', 0)) if data.get(f'tax_rate_{i}') else None

                    is_year_filled = all(v is not None for v in [market_cap, total_debt, preferred_stock, minority_interest, cash, non_operating_assets, ebitda, ebit, revenue, net_income, equity, total_assets, avg_total_assets, avg_equity, share_price, eps, bvps, eps_growth, tax_rate])
                    is_year_empty = all(v is None for v in [market_cap, total_debt, preferred_stock, minority_interest, cash, non_operating_assets, ebitda, ebit, revenue, net_income, equity, total_assets, avg_total_assets, avg_equity, share_price, eps, bvps, eps_growth, tax_rate])

                    if i == 1 and not is_year_filled:
                        return render_template('valuation_performance_multiples.html', error='Please provide all required inputs for Year 1.', form_data=form_data)
                    if not is_year_empty and not is_year_filled:
                        return render_template('valuation_performance_multiples.html', error=f'Please provide all required inputs for Year {i} or leave it empty.', form_data=form_data)
                    if is_year_filled:
                        ev = market_cap + total_debt + preferred_stock + minority_interest - cash - non_operating_assets
                        result = None
                        if selected_formula == 'ev':
                            result = ev
                        elif selected_formula == 'ev_ebitda':
                            if ebitda == 0:
                                return render_template('valuation_performance_multiples.html', error=f'EBITDA cannot be zero for Year {i} in EV/EBITDA calculation.', form_data=form_data)
                            result = ev / ebitda
                        elif selected_formula == 'ev_ebit':
                            if ebit == 0:
                                return render_template('valuation_performance_multiples.html', error=f'EBIT cannot be zero for Year {i} in EV/EBIT calculation.', form_data=form_data)
                            result = ev / ebit
                        elif selected_formula == 'ev_sales':
                            if revenue == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Revenue cannot be zero for Year {i} in EV/Sales calculation.', form_data=form_data)
                            result = ev / revenue
                        elif selected_formula == 'pe':
                            if eps == 0:
                                return render_template('valuation_performance_multiples.html', error=f'EPS cannot be zero for Year {i} in P/E calculation.', form_data=form_data)
                            result = share_price / eps
                        elif selected_formula == 'pb':
                            if bvps == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Book Value Per Share cannot be zero for Year {i} in P/B calculation.', form_data=form_data)
                            result = share_price / bvps
                        elif selected_formula == 'peg':
                            if eps == 0 or eps_growth == 0:
                                return render_template('valuation_performance_multiples.html', error=f'EPS or EPS Growth Rate cannot be zero for Year {i} in PEG calculation.', form_data=form_data)
                            result = (share_price / eps) / (eps_growth / 100)
                        elif selected_formula == 'ebitda_margin':
                            if revenue == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Revenue cannot be zero for Year {i} in EBITDA Margin calculation.', form_data=form_data)
                            result = (ebitda / revenue) * 100
                        elif selected_formula == 'ebit_margin':
                            if revenue == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Revenue cannot be zero for Year {i} in EBIT Margin calculation.', form_data=form_data)
                            result = (ebit / revenue) * 100
                        elif selected_formula == 'net_margin':
                            if revenue == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Revenue cannot be zero for Year {i} in Net Margin calculation.', form_data=form_data)
                            result = (net_income / revenue) * 100
                        elif selected_formula == 'roe':
                            if equity == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Equity cannot be zero for Year {i} in ROE calculation.', form_data=form_data)
                            result = (net_income / equity) * 100
                        elif selected_formula == 'roa':
                            if total_assets == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Total Assets cannot be zero for Year {i} in ROA calculation.', form_data=form_data)
                            result = (net_income / total_assets) * 100
                        elif selected_formula == 'roic':
                            nopat = ebit * (1 - tax_rate / 100)
                            invested_capital = total_debt + equity - cash - non_operating_assets
                            if invested_capital == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Invested Capital cannot be zero for Year {i} in ROIC calculation.', form_data=form_data)
                            result = (nopat / invested_capital) * 100
                        elif selected_formula == 'roa_fin':
                            if avg_total_assets == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Average Total Assets cannot be zero for Year {i} in ROA (Financials) calculation.', form_data=form_data)
                            result = (net_income / avg_total_assets) * 100
                        elif selected_formula == 'roe_fin':
                            if avg_equity == 0:
                                return render_template('valuation_performance_multiples.html', error=f'Average Equity cannot be zero for Year {i} in ROE (Financials) calculation.', form_data=form_data)
                            result = (net_income / avg_equity) * 100

                        input_data.append({
                            'year': i,
                            'market_cap': market_cap,
                            'total_debt': total_debt,
                            'preferred_stock': preferred_stock,
                            'minority_interest': minority_interest,
                            'cash': cash,
                            'non_operating_assets': non_operating_assets,
                            'ebitda': ebitda,
                            'ebit': ebit,
                            'revenue': revenue,
                            'net_income': net_income,
                            'equity': equity,
                            'total_assets': total_assets,
                            'avg_total_assets': avg_total_assets,
                            'avg_equity': avg_equity,
                            'share_price': share_price,
                            'eps': eps,
                            'bvps': bvps,
                            'eps_growth': eps_growth,
                            'tax_rate': tax_rate,
                            'result': result
                        })
                        results.append(result)
                        years_filled += 1
                except ValueError:
                    return render_template('valuation_performance_multiples.html', error=f'Invalid input for Year {i}. Please ensure all inputs are valid numbers.', form_data=form_data)

            if years_filled == 0:
                return render_template('valuation_performance_multiples.html', error='Please provide at least one year of data.', form_data=form_data)

            average_result = sum(results) / years_filled
            unit = '%' if selected_formula in ['ebitda_margin', 'ebit_margin', 'net_margin', 'roe', 'roa', 'roic', 'roa_fin', 'roe_fin'] else 'x'
            logger.info(f"Valuation performance calculated for {years_filled} years with formula {selected_formula}")
            return render_template('valuation_performance_multiples.html', results=input_data, average_result=average_result, unit=unit, form_data=form_data, selected_formula=selected_formula)
        except Exception as e:
            logger.error(f"Valuation performance error: {e}")
            return render_template('valuation_performance_multiples.html', error=f"An error occurred: {str(e)}", form_data=form_data)
    return render_template('valuation_performance_multiples.html', form_data=form_data)

@app.route('/specialized-industry', methods=['GET', 'POST'])
def specialized_industry():
    """Handle specialized industry multiples calculations."""
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST':
        try:
            data = request.form
            selected_formula = data.get('formula')
            results = []
            input_data = []
            periods_filled = 0

            if selected_formula == 'ltm_ebitda':
                for i in range(1, 5):
                    try:
                        ebitda = float(data.get(f'ebitda_q{i}', 0)) if data.get(f'ebitda_q{i}') else None
                        is_quarter_filled = ebitda is not None
                        is_quarter_empty = ebitda is None

                        if i == 1 and not is_quarter_filled:
                            return render_template('specialized_industry_multiples.html', error=f'Please provide all required inputs for Quarter {i}.', form_data=form_data)
                        if not is_quarter_empty and not is_quarter_filled:
                            return render_template('specialized_industry_multiples.html', error=f'Please provide all required inputs for Quarter {i} or leave it empty.', form_data=form_data)
                        if is_quarter_filled:
                            results.append(ebitda)
                            input_data.append({'quarter': i, 'ebitda': ebitda, 'result': ebitda})
                            periods_filled += 1
                    except ValueError:
                        return render_template('specialized_industry_multiples.html', error=f'Invalid input for Quarter {i}. Please ensure all inputs are valid numbers.', form_data=form_data)

                if periods_filled == 0:
                    return render_template('specialized_industry_multiples.html', error='Please provide at least one quarter of data.', form_data=form_data)

                average_result = sum(results)
                unit = 'GHS'
                logger.info(f"Specialized industry LTM EBITDA calculated for {periods_filled} quarters")
                return render_template('specialized_industry_multiples.html', results=input_data, average_result=average_result, unit=unit, form_data=form_data, selected_formula=selected_formula)

            elif selected_formula == 'ntm_ebitda':
                try:
                    current_fy = float(data.get('current_fy_ebitda', 0)) if data.get('current_fy_ebitda') else None
                    next_fy = float(data.get('next_fy_ebitda', 0)) if data.get('next_fy_ebitda') else None
                    months_remaining = float(data.get('months_remaining', 0)) if data.get('months_remaining') else None
                    months_passed = float(data.get('months_passed', 0)) if data.get('months_passed') else None

                    if any(v is None for v in [current_fy, next_fy, months_remaining, months_passed]):
                        return render_template('specialized_industry_multiples.html', error='Please provide all required inputs for NTM EBITDA.', form_data=form_data)
                    if months_remaining + months_passed != 12:
                        return render_template('specialized_industry_multiples.html', error='Months Remaining and Months Passed must sum to 12.', form_data=form_data)

                    result = (current_fy * months_remaining / 12) + (next_fy * months_passed / 12)
                    input_data.append({
                        'current_fy': current_fy,
                        'next_fy': next_fy,
                        'months_remaining': months_remaining,
                        'months_passed': months_passed,
                        'result': result
                    })
                    results.append(result)
                    periods_filled = 1
                    average_result = result
                    unit = 'GHS'
                    logger.info("Specialized industry NTM EBITDA calculated")
                    return render_template('specialized_industry_multiples.html', results=input_data, average_result=average_result, unit=unit, form_data=form_data, selected_formula=selected_formula)
                except ValueError:
                    return render_template('specialized_industry_multiples.html', error='Invalid input for NTM EBITDA. Please ensure all inputs are valid numbers.', form_data=form_data)

            else:
                for i in range(1, 6):
                    try:
                        market_cap = float(data.get(f'market_cap_{i}', 0)) if data.get(f'market_cap_{i}') else None
                        total_debt = float(data.get(f'total_debt_{i}', 0)) if data.get(f'total_debt_{i}') else None
                        preferred_stock = float(data.get(f'preferred_stock_{i}', 0)) if data.get(f'preferred_stock_{i}') else None
                        minority_interest = float(data.get(f'minority_interest_{i}', 0)) if data.get(f'minority_interest_{i}') else None
                        cash = float(data.get(f'cash_{i}', 0)) if data.get(f'cash_{i}') else None
                        non_operating_assets = float(data.get(f'non_operating_assets_{i}', 0)) if data.get(f'non_operating_assets_{i}') else None
                        ebitda = float(data.get(f'ebitda_{i}', 0)) if data.get(f'ebitda_{i}') else None
                        ebit = float(data.get(f'ebit_{i}', 0)) if data.get(f'ebit_{i}') else None
                        revenue = float(data.get(f'revenue_{i}', 0)) if data.get(f'revenue_{i}') else None
                        prior_revenue = float(data.get(f'prior_revenue_{i}', 0)) if data.get(f'prior_revenue_{i}') else None
                        eps = float(data.get(f'eps_{i}', 0)) if data.get(f'eps_{i}') else None
                        prior_eps = float(data.get(f'prior_eps_{i}', 0)) if data.get(f'prior_eps_{i}') else None
                        tax_rate = float(data.get(f'tax_rate_{i}', 0)) if data.get(f'tax_rate_{i}') else None
                        rent_expense = float(data.get(f'rent_expense_{i}', 0)) if data.get(f'rent_expense_{i}') else None
                        subscribers = float(data.get(f'subscribers_{i}', 0)) if data.get(f'subscribers_{i}') else None
                        boe = float(data.get(f'boe_{i}', 0)) if data.get(f'boe_{i}') else None
                        square_footage = float(data.get(f'square_footage_{i}', 0)) if data.get(f'square_footage_{i}') else None
                        mau = float(data.get(f'mau_{i}', 0)) if data.get(f'mau_{i}') else None
                        ffo_per_share = float(data.get(f'ffo_per_share_{i}', 0)) if data.get(f'ffo_per_share_{i}') else None
                        tangible_bvps = float(data.get(f'tangible_bvps_{i}', 0)) if data.get(f'tangible_bvps_{i}') else None
                        share_price = float(data.get(f'share_price_{i}', 0)) if data.get(f'share_price_{i}') else None

                        is_year_filled = all(v is not None for v in [market_cap, total_debt, preferred_stock, minority_interest, cash, non_operating_assets, ebitda, ebit, revenue, prior_revenue, eps, prior_eps, tax_rate, rent_expense, subscribers, boe, square_footage, mau, ffo_per_share, tangible_bvps, share_price])
                        is_year_empty = all(v is None for v in [market_cap, total_debt, preferred_stock, minority_interest, cash, non_operating_assets, ebitda, ebit, revenue, prior_revenue, eps, prior_eps, tax_rate, rent_expense, subscribers, boe, square_footage, mau, ffo_per_share, tangible_bvps, share_price])

                        if i == 1 and not is_year_filled:
                            return render_template('specialized_industry_multiples.html', error=f'Please provide all required inputs for Year {i}.', form_data=form_data)
                        if not is_year_empty and not is_year_filled:
                            return render_template('specialized_industry_multiples.html', error=f'Please provide all required inputs for Year {i} or leave it empty.', form_data=form_data)
                        if is_year_filled:
                            ev = market_cap + total_debt + preferred_stock + minority_interest - cash - non_operating_assets
                            result = None
                            if selected_formula == 'net_debt':
                                result = total_debt - cash
                            elif selected_formula == 'net_debt_ebitda':
                                if ebitda == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'EBITDA cannot be zero for Year {i} in Net Debt/EBITDA calculation.', form_data=form_data)
                                result = (total_debt - cash) / ebitda
                            elif selected_formula == 'revenue_growth':
                                if prior_revenue == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'Prior Revenue cannot be zero for Year {i} in Revenue Growth calculation.', form_data=form_data)
                                result = ((revenue - prior_revenue) / prior_revenue) * 100
                            elif selected_formula == 'eps_growth':
                                if prior_eps == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'Prior EPS cannot be zero for Year {i} in EPS Growth calculation.', form_data=form_data)
                                result = ((eps - prior_eps) / prior_eps) * 100
                            elif selected_formula == 'unlevered_pe':
                                ebiat = ebit * (1 - tax_rate / 100)
                                if ebiat == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'EBIAT cannot be zero for Year {i} in Unlevered P/E calculation.', form_data=form_data)
                                result = ev / ebiat
                            elif selected_formula == 'tev_ebitdar':
                                ebitdar = ebitda + rent_expense
                                tev = ev
                                if ebitdar == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'EBITDAR cannot be zero for Year {i} in TEV/EBITDAR calculation.', form_data=form_data)
                                result = tev / ebitdar
                            elif selected_formula == 'ev_subscribers':
                                if subscribers == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'Subscribers cannot be zero for Year {i} in EV/Subscribers calculation.', form_data=form_data)
                                result = ev / subscribers
                            elif selected_formula == 'ev_boe':
                                if boe == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'BOE cannot be zero for Year {i} in EV/BOE calculation.', form_data=form_data)
                                result = ev / boe
                            elif selected_formula == 'ev_square_foot':
                                if square_footage == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'Square Footage cannot be zero for Year {i} in EV/Square Foot calculation.', form_data=form_data)
                                result = ev / square_footage
                            elif selected_formula == 'ev_mau':
                                if mau == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'MAU cannot be zero for Year {i} in EV/MAU calculation.', form_data=form_data)
                                result = ev / mau
                            elif selected_formula == 'p_ffo':
                                if ffo_per_share == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'FFO Per Share cannot be zero for Year {i} in P/FFO calculation.', form_data=form_data)
                                result = share_price / ffo_per_share
                            elif selected_formula == 'p_tbv':
                                if tangible_bvps == 0:
                                    return render_template('specialized_industry_multiples.html', error=f'Tangible Book Value Per Share cannot be zero for Year {i} in P/TBV calculation.', form_data=form_data)
                                result = share_price / tangible_bvps

                            input_data.append({
                                'year': i,
                                'market_cap': market_cap,
                                'total_debt': total_debt,
                                'preferred_stock': preferred_stock,
                                'minority_interest': minority_interest,
                                'cash': cash,
                                'non_operating_assets': non_operating_assets,
                                'ebitda': ebitda,
                                'ebit': ebit,
                                'revenue': revenue,
                                'prior_revenue': prior_revenue,
                                'eps': eps,
                                'prior_eps': prior_eps,
                                'tax_rate': tax_rate,
                                'rent_expense': rent_expense,
                                'subscribers': subscribers,
                                'boe': boe,
                                'square_footage': square_footage,
                                'mau': mau,
                                'ffo_per_share': ffo_per_share,
                                'tangible_bvps': tangible_bvps,
                                'share_price': share_price,
                                'result': result
                            })
                            results.append(result)
                            periods_filled += 1
                    except ValueError:
                        return render_template('specialized_industry_multiples.html', error=f'Invalid input for Year {i}. Please ensure all inputs are valid numbers.', form_data=form_data)

                if periods_filled == 0:
                    return render_template('specialized_industry_multiples.html', error='Please provide at least one year of data.', form_data=form_data)

                average_result = sum(results) / periods_filled
                unit = '%' if selected_formula in ['revenue_growth', 'eps_growth'] else 'GHS' if selected_formula == 'net_debt' else 'x'
                logger.info(f"Specialized industry calculated for {periods_filled} periods with formula {selected_formula}")
                return render_template('specialized_industry_multiples.html', results=input_data, average_result=average_result, unit=unit, form_data=form_data, selected_formula=selected_formula)
        except Exception as e:
            logger.error(f"Specialized industry error: {e}")
            return render_template('specialized_industry_multiples.html', error=f"An error occurred: {str(e)}", form_data=form_data)
    return render_template('specialized_industry_multiples.html', form_data=form_data)

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

                    valuation = ValuationResult(
                        period_name=period,
                        currency=currency,
                        weighted_average=weighted_average,
                        pb_value=pb_value,
                        ptbv_value=ptbv_value,
                        pe_value=pe_value,
                        ddm_value=ddm_value
                    )
                    db.session.add(valuation)
                except ValueError as e:
                    logger.error(f"Calculation error for period {period}: {e}")
                    periods.append({'period_name': period, 'error': str(e)})
                    continue
            db.session.commit()
            logger.info("Multiples master valuation calculation and storage successful")
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

@app.route('/calculate-fcfe', methods=['GET', 'POST'])
def calculate_fcfe_route():
    """Handle FCFE calculation form and results."""
    form = FCFEForm()
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    start_year = int(form_data.get('start_year', datetime.now().year))  # Default to current year
    if request.method == 'POST' and form.validate_on_submit():
        try:
            net_income = form.net_income.data
            capex = form.capex.data
            depreciation = form.depreciation.data
            change_in_working_capital = form.change_in_working_capital.data
            debt_issued = form.debt_issued.data
            debt_repaid = form.debt_repaid.data

            fcfe = calculate_fcfe(net_income, capex, depreciation, change_in_working_capital, debt_issued, debt_repaid)
            logger.info("FCFE calculation successful")
            return render_template('fcfe.html', result={'fcfe': round(fcfe, 2)}, form=form, form_data=form_data, start_year=start_year)
        except ValueError as e:
            logger.error(f"FCFE calculation error: {e}")
            return render_template('fcfe.html', error=str(e), form=form, form_data=form_data, start_year=start_year)
        except Exception as e:
            logger.error(f"Unexpected error in FCFE calculation: {e}")
            return render_template('fcfe.html', error="An unexpected error occurred. Please try again.", form=form, form_data=form_data, start_year=start_year)
    return render_template('fcfe.html', form=form, form_data=form_data, start_year=start_year)

@app.route('/multi-method-valuation', methods=['GET', 'POST'])
def multi_method_valuation():
    """Handle multi-method valuation form and calculations."""
    form = PeriodForm()  # Reuse PeriodForm for validation
    form_data = request.form.to_dict() if request.method == 'POST' else {}
    if request.method == 'POST' and form.validate_on_submit():
        try:
            currency = form.currency.data
            weight_scenario = form.weight_scenario.data
            current_price = form.current_price.data
            years_high = int(form.pe_years.data)  # Reuse pe_years as years_high
            growth_high = form.eps_growth.data  # Reuse eps_growth as growth_high
            growth_terminal = form.dividend_growth.data  # Reuse dividend_growth as terminal growth
            discount_rate = form.required_return.data
            ddm_base_dividend = form.dividend_per_share.data
            ddm_sensitivity_dividend = ddm_base_dividend * 0.9  # Sensitivity at 90% of base
            dcf_fcfe = form.eps.data  # Reuse eps as proxy for fcfe
            pe_eps = form.eps.data
            pe_growth = form.eps_growth.data
            pe_multiple = form.pe_multiple.data
            pe_years = form.pe_years.data

            # Enhanced validation
            if any(x <= 0 for x in [current_price, ddm_base_dividend, dcf_fcfe, pe_eps, pe_multiple, pe_years, years_high]):
                raise ValueError("All monetary values and years must be positive and non-zero")
            if any(x < 0 or x > 100 for x in [growth_high, growth_terminal, pe_growth]) or discount_rate <= 0 or discount_rate > 50:
                raise ValueError("Rates must be between 0 and 100%, discount rate between 0.01 and 50")
            if growth_terminal >= growth_high:
                raise ValueError("Terminal growth rate must be less than high-growth rate")
            if discount_rate <= growth_terminal:
                raise ValueError("Discount rate must be greater than terminal growth rate")
            if years_high > 20 or pe_years > 5:
                raise ValueError("High-growth years must be <= 20, P/E years <= 5")
            if weight_scenario not in ['conservative', 'balanced', 'growth']:
                raise ValueError("Invalid weighting scenario")
            if currency not in ['USD', 'GHS', 'EUR', 'GBP', 'JPY']:
                raise ValueError("Invalid currency selected")

            ddm_base = calculate_two_stage_ddm(ddm_base_dividend, growth_high, years_high, growth_terminal, discount_rate)
            ddm_sensitivity = calculate_two_stage_ddm(ddm_sensitivity_dividend, growth_high, years_high, growth_terminal, discount_rate)
            dcf_value = calculate_two_stage_dcf(dcf_fcfe, growth_high, years_high, growth_terminal, discount_rate)
            pe_target = calculate_pe_target(pe_eps, pe_growth, pe_years, pe_multiple)

            weights = {
                'conservative': [30, 20, 30, 20],
                'balanced': [20, 20, 40, 20],
                'growth': [20, 20, 30, 30]
            }[weight_scenario]
            weight_priority = {
                'conservative': 'DDM Base and DCF',
                'balanced': 'DCF',
                'growth': 'DCF and P/E'
            }[weight_scenario]
            weight_rationale = {
                'conservative': 'emphasis on dividend stability and cash flow reliability',
                'balanced': 'cash flow focus',
                'growth': 'growth potential and market alignment'
            }[weight_scenario]
            weight_max_index = {'conservative': 0, 'balanced': 2, 'growth': 2}[weight_scenario]

            values = [ddm_base, ddm_sensitivity, dcf_value, pe_target]
            weighted_average = sum(v * w / 100 for v, w in zip(values, weights))

            sensitivity = {
                'discount_rate_low': discount_rate * 0.9,
                'discount_rate_high': discount_rate * 1.1,
                'growth_high_low': growth_high * 0.9,
                'growth_high_high': growth_high * 1.1,
                'value_low': sum([
                    calculate_two_stage_ddm(ddm_base_dividend, growth_high * 0.9, years_high, growth_terminal, discount_rate * 0.9) * (weights[0] / 100),
                    calculate_two_stage_ddm(ddm_sensitivity_dividend, growth_high * 0.9, years_high, growth_terminal, discount_rate * 0.9) * (weights[1] / 100),
                    calculate_two_stage_dcf(dcf_fcfe, growth_high * 0.9, years_high, growth_terminal, discount_rate * 0.9) * (weights[2] / 100),
                    calculate_pe_target(pe_eps, pe_growth * 0.9, pe_years, pe_multiple) * (weights[3] / 100)
                ]),
                'value_high': sum([
                    calculate_two_stage_ddm(ddm_base_dividend, growth_high * 1.1, years_high, growth_terminal, discount_rate * 1.1) * (weights[0] / 100),
                    calculate_two_stage_ddm(ddm_sensitivity_dividend, growth_high * 1.1, years_high, growth_terminal, discount_rate * 1.1) * (weights[1] / 100),
                    calculate_two_stage_dcf(dcf_fcfe, growth_high * 1.1, years_high, growth_terminal, discount_rate * 1.1) * (weights[2] / 100),
                    calculate_pe_target(pe_eps, pe_growth * 1.1, pe_years, pe_multiple) * (weights[3] / 100)
                ])
            }

            over_under_valuation = (current_price / weighted_average - 1) * 100 if weighted_average > 0 else float('inf')

            result = {
                'currency': currency,
                'ddm_base': round(ddm_base, 2),
                'ddm_sensitivity': round(ddm_sensitivity, 2),
                'dcf': round(dcf_value, 2),
                'pe_target': round(pe_target, 2),
                'weighted_average': round(weighted_average, 2),
                'over_under_valuation': round(over_under_valuation, 2),
                'pe_years': pe_years,
                'current_price': round(current_price, 2),
                'weights': weights,
                'weight_priority': weight_priority,
                'weight_rationale': weight_rationale,
                'weight_max_index': weight_max_index,
                'discount_rate': round(discount_rate, 2),
                'growth_high': round(growth_high, 2),
                'sensitivity': {
                    'discount_rate_low': round(sensitivity['discount_rate_low'], 2),
                    'discount_rate_high': round(sensitivity['discount_rate_high'], 2),
                    'growth_high_low': round(sensitivity['growth_high_low'], 2),
                    'growth_high_high': round(sensitivity['growth_high_high'], 2),
                    'value_low': round(sensitivity['value_low'], 2),
                    'value_high': round(sensitivity['value_high'], 2)
                }
            }
            logger.info("Multi-method valuation calculation successful")
            return render_template('multi_method_valuation.html', result=result, form=form, form_data=form_data)
        except ValueError as e:
            logger.error(f"Multi-method valuation error: {e}")
            return render_template('multi_method_valuation.html', error=str(e), form=form, form_data=form_data)
        except Exception as e:
            logger.error(f"Unexpected error in multi-method valuation: {e}")
            return render_template('multi_method_valuation.html', error="An unexpected error occurred. Please try again.", form=form, form_data=form_data)
    return render_template('multi_method_valuation.html', form=form, form_data=form_data)

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
    if os.getenv('FLASK_ENV') != 'production':
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)