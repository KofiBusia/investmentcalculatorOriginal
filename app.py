from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for, session, send_file
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, IntegerField, SelectField, FloatField
from wtforms.validators import DataRequired, Email
import numpy as np
import numpy_financial as npf
import os
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass
from collections import namedtuple
from datetime import datetime
from werkzeug.utils import secure_filename

load_dotenv()

dapp = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cleanvisionhr.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Flask-Mail configuration for GoDaddy
app.config['MAIL_SERVER'] = 'smtpout.secureserver.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'info@cleanvisionhr.com'
app.config['MAIL_PASSWORD'] = 'IFokbu@m@1'
app.config['MAIL_DEFAULT_SENDER'] = 'info@cleanvisionhr.com'
mail = Mail(app)

app.config['MAIL_SERVER'] = 'smtpout.secureserver.net'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'info@cleanvisionhr.com'
app.config['MAIL_PASSWORD'] = 'IFokbu@m@1'
app.config['MAIL_DEFAULT_SENDER'] = 'info@cleanvisionhr.com'
mail = Mail(app)

# Models
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
 date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Forms
class ContactForm(FlaskForm):
 name = StringField('Name', validators=[DataRequired()])
 email = StringField('Email', validators=[DataRequired(), Email()])
 message = TextAreaField('Message', validators=[DataRequired()])
 submit = SubmitField('Send')

class BlogForm(FlaskForm):
 title = StringField('Title', validators=[DataRequired()])
 content = TextAreaField('Content', validators=[DataRequired()])
 author = StringField('Author', validators=[DataRequired()])
 submit = SubmitField('Post')

# Result Structures
DCFResult = namedtuple('DCFResult', ['total_pv', 'pv_cash_flows', 'terminal_value', 'pv_terminal', 'total_dcf'])
DVMResult = namedtuple('DVMResult', ['intrinsic_value', 'formula', 'pv_dividends', 'terminal_value', 'pv_terminal'])

# Template Filter
@app.template_filter('commafy')
def commafy(value):
 return "{:,.2f}".format(value)

# Helper Functions
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
 return (returns / 100) - expected_return

def parse_comma_separated(text):
 try:
     
  return [float(x.strip()) for x in text.split(',')]
 except ValueError:
  raise ValueError("Invalid numeric format. Use comma-separated numbers (e.g., 0.4, 0.6).")

def parse_comma_separated(row):
    # Assuming this helper function exists to parse row elements
    return list(map(float, row.split(',')))

def parse_covariance_matrix(text, num_assets):
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
        if not np.allclose(matrix, matrix.T):
            raise ValueError("Covariance matrix must be symmetric.")
        if np.any(np.linalg.eigvals(matrix) < -1e-10):
            raise ValueError("Covariance matrix must be positive semi-definite.")
        return matrix
    except ValueError as e:
        raise ValueError(f"Invalid covariance matrix: {str(e)}")

def calculate_expected_return(weights, returns):
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
    if start_value == 0:
        return 0
    try:
        cagr = (end_value / abs(start_value)) ** (1 / years) - 1
    except ZeroDivisionError:
        return 0
    return cagr * 100

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
        raise ValueError("Total value (debt + equity) must be positive")
    
    cost_equity = risk_free_rate + beta * (market_return - risk_free_rate)
    cost_debt = 0.05  # This should probably be a parameter
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

def calculate_intrinsic_value_full(fcfs, risk_free_rate, market_return, beta, outstanding_shares, total_debt, cash_and_equivalents, growth_rate=None, auto_growth_rate=True):
    assert len(fcfs) == 5, "Provide exactly 5 years of historical FCF"
    
    if auto_growth_rate:
        g = calculate_cagr(fcfs[0], fcfs[-1], 4) / 100 if fcfs[0] != 0 else 0.0
    else:
        g = growth_rate
    
    last_fcf = fcfs[-1]
    discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)
    
    if discount_rate <= g:
        raise ValueError(f"Discount rate ({discount_rate:.2%}) must be > growth rate ({g:.2%}).")
    
    fcf_next = last_fcf * (1 + g)
    enterprise_value = fcf_next / (discount_rate - g)
    equity_value = max(enterprise_value - total_debt + cash_and_equivalents, 0)
    intrinsic_value_per_share = equity_value / outstanding_shares
    return intrinsic_value_per_share, g, discount_rate

def calculate_target_price(fcf, explicit_growth, n, g, r, debt, cash, shares):
    if g >= r:
        raise ValueError("Perpetual growth rate must be less than discount rate.")
    
    projected_fcf = [fcf * (1 + explicit_growth)**i for i in range(1, n+1)]
    terminal_value = (projected_fcf[-1] * (1 + g)) / (r - g)
    target_price = (terminal_value - debt + cash) / shares
    return target_price

# Helper functions (should be defined before routes that use them)
def calculate_portfolio_volatility(weights, cov_matrix):
    weights_array = np.array(weights)
    portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
    return np.sqrt(portfolio_variance)

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

# Routes
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
            form_data = {'num_assets': request.form['num_assets']}
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
    
    return render_template('expected_return.html', form_data={})

@app.route('/volatility', methods=['GET', 'POST'])
def volatility():
    if request.method == 'POST':
        try:
            form_data = {'num_assets': request.form['num_assets']}
            num_assets = int(form_data['num_assets'])
            weights = []
            
            for i in range(1, num_assets + 1):
                form_data[f'weight_{i}'] = request.form[f'weight_{i}']
                weights.append(float(form_data[f'weight_{i}']))
            
            cov_matrix = []
            for i in range(1, num_assets + 1):
                row = []
                for j in range(1, num_assets + 1):
                    form_data[f'cov_{i}_{j}'] = request.form[f'cov_{i}_{j}']
                    row.append(float(form_data[f'cov_{i}_{j}']))
                cov_matrix.append(row)
            
            cov_matrix = np.array(cov_matrix)
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
    
    return render_template('volatility.html', form_data={})

# ... (remaining routes follow the same pattern with proper indentation)

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
            result = {'profit': "{:,.2f}".format(profit)}
        
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
            result = {'total_return': round(total_return, 2)}
        
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
        contact_message = ContactMessage(
            name=form.name.data,
            email=form.email.data,
            message=form.message.data
        )
        db.session.add(contact_message)
        db.session.commit()

        # Send email to company
        company_msg = Message(subject='New Contact Message', recipients=['info@cleanvisionhr.com'])
        company_msg.body = f"Name: {contact_message.name}\nEmail: {contact_message.email}\nMessage: {contact_message.message}"
        mail.send(company_msg)

        # Send auto-response
        html_body = render_template('contact_confirmation.html', name=contact_message.name)
        auto_response_msg = Message(
            subject="Thanks for Reaching Out! We'll Get Back to You Soon",
            sender=("Admin", "info@cleanvisionhr.com"),
            recipients=[contact_message.email],
            html=html_body
        )
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
            result = {'holding_period_return': round(holding_period_return, 2)}
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


@app.route('/intrinsic-value', methods=['GET', 'POST'])
def intrinsic_value():
    if request.method == 'POST':
        try:
            fcf = [float(request.form[f'fcf_{i}']) for i in range(1, 6)]
            risk_free_rate = float(request.form['risk_free_rate']) / 100
            market_return = float(request.form['market_return']) / 100
            beta = float(request.form['beta'])
            outstanding_shares = float(request.form['outstanding_shares'])
            total_debt = float(request.form['total_debt'])
            cash_and_equivalents = float(request.form['cash_and_equivalents'])
            projection_period = int(request.form['projection_period'])

            if outstanding_shares <= 0:
                raise ValueError("Outstanding shares must be positive.")

            discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)
            perpetual_growth_rate = calculate_cagr(fcf[0], fcf[-1], 4) / 100

            if discount_rate <= perpetual_growth_rate:
                raise ValueError("Discount rate must exceed growth rate.")

            last_fcf = fcf[-1]
            enterprise_value = (last_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
            equity_value = max(enterprise_value - total_debt + cash_and_equivalents, 0)
            intrinsic_value = equity_value / outstanding_shares

            projected_fcf = last_fcf * (1 + perpetual_growth_rate) ** projection_period
            terminal_value = (projected_fcf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)
            target_equity_value = max(terminal_value - total_debt + cash_and_equivalents, 0)
            target_price = target_equity_value / outstanding_shares

            return render_template(
                'intrinsic_value.html',
                result=f"{intrinsic_value:.2f}",
                target_price=f"{target_price:.2f}",
                projection_period=projection_period
            )
        except ValueError as e:
            return render_template('intrinsic_value.html', error=str(e))
    return render_template('intrinsic_value.html')


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
        new_post = BlogPost(
            title=form.title.data,
            content=form.content.data,
            author=form.author.data
        )
        db.session.add(new_post)
        db.session.commit()
        flash('Blog post created successfully!', 'success')
        return redirect(url_for('admin_blog'))
    posts = BlogPost.query.order_by(BlogPost.date_posted.desc()).all()
    return render_template('admin_blog.html', form=form, posts=posts)


@app.route('/admin/blog/edit/<int:post_id>', methods=['GET', 'POST'])
def edit_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    form = BlogForm()
    if form.validate_on_submit():
        post.title = form.title.data
        post.content = form.content.data
        post.author = form.author.data
        db.session.commit()
        flash('Blog post updated successfully!', 'success')
        return redirect(url_for('admin_blog'))
    elif request.method == 'GET':
        form.title.data = post.title
        form.content.data = post.content
        form.author.data = post.author
    return render_template('edit_blog_post.html', form=form, post=post)


@app.route('/admin/blog/delete/<int:post_id>', methods=['POST'])
def delete_blog_post(post_id):
    post = BlogPost.query.get_or_404(post_id)
    db.session.delete(post)
    db.session.commit()
    flash('Blog post deleted successfully!', 'success')
    return redirect(url_for('admin_blog'))

if __name__ == '__main__':
    # For local development, use Waitress
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
else:
    # For production on Render, use Gunicorn
    if __name__ == 'app':
        from gunicorn.app.base import BaseApplication

        class StandaloneApplication(BaseApplication):
            def __init__(self, intrinsic_value, formula=None, pv_dividends=None, terminal_value=None, pv_terminal=None):
                self.intrinsic_value = intrinsic_value
                self.formula = formula
                self.pv_dividends = pv_dividends
                self.terminal_value = terminal_value
                self.pv_terminal = pv_terminal
                self.application = app
                super().__init__(options)

            def load_config(self):
                pass

            def load(self):
                return self.application

        options = {
            'bind': '0.0.0.0:5000',
            'workers': 4,  # Adjust based on your needs
        }
        StandaloneApplication(app, options).run()