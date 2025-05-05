from flask import Flask, render_template, request, send_from_directory, flash, redirect, url_for
from flask_mail import Mail, Message
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()  # ← This loads variables from .env

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Set via environment; default for local dev only

# Email configuration using environment variables
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')  # e.g., 'smtp.gmail.com'
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))  # Default to 587 if not set
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'  # Convert string to boolean
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # e.g., 'your-email@gmail.com'
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # Email password or app-specific password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')  # e.g., 'your-email@gmail.com'

mail = Mail(app)

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
) -> float:
    """
    Calculates the intrinsic value per share using perpetual growth DCF from historical FCF.
    
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
    - Intrinsic value per share.
    """
    assert len(fcfs) == 5, "Provide exactly 5 years of historical FCF"

    # Determine growth rate based on auto_growth_rate
    if auto_growth_rate:
        if fcfs[0] == 0:
            g = 0.0
        else:
            g = (fcfs[-1] / fcfs[0]) ** (1 / (len(fcfs) - 1)) - 1
        growth_source = "auto-computed"
    else:
        g = growth_rate
        growth_source = "manually input"

    # Use the last FCF
    last_fcf = fcfs[-1]

    # Calculate discount rate using CAPM
    discount_rate = risk_free_rate + beta * (market_return - risk_free_rate)

    # Check if discount_rate > g with detailed error message
    if discount_rate <= g:
        raise ValueError(
            f"Discount rate ({discount_rate:.2%}) must be greater than growth rate ({g:.2%}) for the perpetual growth model. "
            f"The growth rate was {growth_source}. Consider lowering the growth rate or increasing the discount rate "
            "by adjusting the risk-free rate, market return, or beta."
        )

    # Calculate next year's FCF
    fcf_next = last_fcf * (1 + g)

    # Calculate enterprise value using perpetual growth formula
    enterprise_value = fcf_next / (discount_rate - g)

    # Calculate equity value
    equity_value = enterprise_value - total_debt + cash_and_equivalents

    # Calculate intrinsic value per share
    intrinsic_value_per_share = equity_value / outstanding_shares if equity_value > 0 else 0

    return intrinsic_value_per_share

@app.route('/ads.txt')
def ads_txt():
    return send_from_directory('static', 'ads.txt')

@app.route('/')
def index():
    return render_template('index.html')

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
            # Capture form data as strings to retain in the form
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

            # Validation
            if acquirer_shares <= 0 or target_shares <= 0 or new_shares_issued < 0:
                result = "Error: Shares outstanding and new shares issued must be positive."
                return render_template('mna.html', result=result, form_data=form_data)
            total_shares = acquirer_shares + new_shares_issued
            if total_shares <= 0:
                result = "Error: Total shares outstanding must be positive."
                return render_template('mna.html', result=result, form_data=form_data)

            # Perform M&A calculations
            acquirer_earnings = acquirer_eps * acquirer_shares
            target_earnings = target_eps * target_shares
            combined_earnings = acquirer_earnings + target_earnings + synergy_value
            combined_eps = combined_earnings / total_shares
            eps_change = combined_eps - acquirer_eps
            status = "Accretive" if eps_change > 0 else "Dilutive" if eps_change < 0 else "Neutral"

            # Format results
            result = f"""
                <p>Pre-Deal Acquirer EPS: ${acquirer_eps:,.2f}</p>
                <p>Post-Deal Combined EPS: ${combined_eps:,.2f}</p>
                <p>EPS Change: ${eps_change:,.2f} ({status})</p>
                <p>Total Combined Earnings (incl. Synergy): ${combined_earnings:,.2f}</p>
                <p>Total Shares Outstanding: {total_shares:,.0f}</p>
            """
            return render_template('mna.html', result=result, form_data=form_data)
        except ValueError:
            result = "Error: Please enter valid numeric values."
            return render_template('mna.html', result=result, form_data=request.form)
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
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        try:
            # Send email to info@cleanvisionhr.com
            msg_to_admin = Message(
                subject=f"New Contact Form Submission from {name}",
                recipients=['info@cleanvisionhr.com'],
                body=f"Name: {name}\nEmail: {email}\nMessage: {message}"
            )
            mail.send(msg_to_admin)
            
            # Send auto-response to the user
            msg_to_user = Message(
                subject="Thank you for contacting us!",
                recipients=[email],
                body=f"Dear {name},\n\nThank you for reaching out to us. We have received your message and will get back to you shortly.\n\nBest regards,\nInvestment Calculator Team"
            )
            mail.send(msg_to_user)
            
            flash("Your message has been sent successfully!", "success")
        except Exception as e:
            app.logger.error(f"Error sending email: {str(e)}")
            flash("An error occurred while sending your message. Please try again later.", "danger")
        
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

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

@app.route('/intrinsic-value', methods=['GET', 'POST'])
def intrinsic_value():
    result = None
    error = None
    if request.method == 'POST':
        try:
            # Extract 5 years of FCF
            fcfs = [float(request.form[f'fcf_{i}']) for i in range(1, 6)]
            risk_free_rate = float(request.form['risk_free_rate']) / 100
            market_return = float(request.form['market_return']) / 100
            beta = float(request.form['beta'])
            outstanding_shares = float(request.form['outstanding_shares'])
            total_debt = float(request.form['total_debt'])
            cash_and_equivalents = float(request.form['cash_and_equivalents'])
            auto_growth_rate = request.form.get('auto_growth_rate') == 'on'

            # Handle growth rate
            if auto_growth_rate:
                growth_rate = None  # Will be auto-computed
            else:
                growth_rate = float(request.form['manual_growth_rate']) / 100

            # Validation
            if outstanding_shares <= 0:
                error = "Outstanding shares must be positive."
            else:
                intrinsic_value = calculate_intrinsic_value_full(
                    fcfs, risk_free_rate, market_return, beta,
                    outstanding_shares, total_debt, cash_and_equivalents,
                    growth_rate=growth_rate, auto_growth_rate=auto_growth_rate
                )
                result = "{:,.2f}".format(intrinsic_value)

        except ValueError as e:
            error = str(e)
        except Exception as e:
            error = "An error occurred: " + str(e)

    return render_template('intrinsic_value.html', result=result, error=error)

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