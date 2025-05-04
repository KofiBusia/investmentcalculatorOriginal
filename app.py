from flask import Flask, render_template, request, send_from_directory
from flask_mail import Mail, Message
import numpy as np  # Added for Beta calculation

app = Flask(__name__)

# Email configuration (update with your email provider's settings)
app.config['MAIL_SERVER'] = 'smtp.yourmailprovider.com'  # e.g., 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@example.com'  # Your email
app.config['MAIL_PASSWORD'] = 'your-email-password'  # Your email password
app.config['MAIL_DEFAULT_SENDER'] = 'your-email@example.com'

mail = Mail(app)

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
        
        return render_template('contact.html', success="Your message has been sent successfully!")
    
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