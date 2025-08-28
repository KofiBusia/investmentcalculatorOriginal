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
    current_year = 2025  # Based on current date August 28, 2025
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
            fv_fcfe = [last_fcFE * (1 + perpetual_growth_rate) ** t for t in range(1, num_years + 2)]  # FV for years start_year + num_years to start_year + 2*num_years (e.g., 2026 to 2030 for 5 years starting 2025)

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
    current_year = 2025  # Based on current date August 28, 2025
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