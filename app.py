from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

@app.route('/ads.txt')
def ads_txt():
    return send_from_directory('static', 'ads.txt')

# Rest of your code remains unchanged

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monthly-contribution', methods=['GET', 'POST'])
def monthly_contribution():
    result = None
    if request.method == 'POST':
        try:
            target = float(request.form['target_amount'])
            principal = float(request.form['starting_principal'])
            period = float(request.form['period'])
            rate = float(request.form['annual_return']) / 100  # Convert percentage to decimal (e.g., 20% → 0.20)

            if target <= 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('monthly_contribution.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                monthly_contribution = (target - principal) / months
            else:
                # Calculate the compounded monthly return: (1 + annual_rate)^(1/12) - 1
                monthly_rate = (1 + rate) ** (1 / 12) - 1
                # Future value of the starting principal with compound interest
                future_principal = principal * (1 + monthly_rate) ** months
                # Solve for monthly contribution using the annuity future value formula: P = FV / [((1 + r)^n - 1) / r]
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
            rate = float(request.form['annual_return']) / 100  # Convert percentage to decimal (e.g., 20% → 0.20)

            if monthly < 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('end_balance.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                end_balance = principal + monthly * months
            else:
                # Calculate the compounded monthly return: (1 + annual_rate)^(1/12) - 1
                monthly_rate = (1 + rate) ** (1 / 12) - 1
                # Future value of the starting principal with compound interest
                future_principal = principal * (1 + monthly_rate) ** months
                # Future value of monthly contributions using the annuity formula: P * [((1 + r)^n - 1) / r]
                future_contributions = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)
                # Total end balance
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
            purchase_commission = float(request.form['purchase_commission']) / 100  # Convert percentage to decimal
            selling_price_per_share = float(request.form['selling_price_per_share'])
            sale_commission = float(request.form['sale_commission']) / 100  # Convert percentage to decimal
            dividends = float(request.form['dividends'])

            if any(x < 0 for x in [num_shares, purchase_price_per_share, purchase_commission, selling_price_per_share, sale_commission, dividends]):
                return render_template('stocks.html', error="Please enter valid non-negative numbers.")

            # Calculate total purchase price (consideration) and total cost including commission
            purchase_consideration = num_shares * purchase_price_per_share
            purchase_commission_amount = purchase_consideration * purchase_commission
            total_purchase_cost = purchase_consideration + purchase_commission_amount

            # Calculate total selling price (consideration) and net proceeds after commission
            selling_consideration = num_shares * selling_price_per_share
            sale_commission_amount = selling_consideration * sale_commission
            net_sale_proceeds = selling_consideration - sale_commission_amount

            # Calculate metrics
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
            rate = float(request.form['rate']) / 100  # Convert percentage to decimal (e.g., 21% → 0.21)
            tenor = float(request.form['tenor'])

            if principal <= 0 or rate < 0 or tenor <= 0:
                return render_template('tbills.html', error="Please enter valid positive numbers.")

            # Calculate Maturity Value using the formula: ((Principal * Tenor * Rate) / 364) + Principal
            interest = (principal * tenor * rate) / 364
            maturity_value = principal + interest

            result = {
                'maturity_value': "{:,.2f}".format(maturity_value)  # Format with commas and 2 decimal places
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

# Routes for additional pages
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/privacy-policy')
def privacy_policy():
    return render_template('privacy_policy.html')

@app.route('/terms')
def terms_conditions():
    return render_template('terms_conditions.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/early_exit', methods=['GET', 'POST'])
def early_exit():
    result = None
    if request.method == 'POST':
        try:
            principal = float(request.form['principal'])
            holding_period = float(request.form['holding_period'])
            selling_price = float(request.form['selling_price'])
            total_coupons = float(request.form.get('total_coupons', 0))  # Optional field

            if any(x < 0 for x in [principal, holding_period, selling_price, total_coupons]) or principal == 0 or holding_period == 0:
                return render_template('early_exit.html', error="Please enter valid positive numbers.")

            holding_period_return = (total_coupons + (selling_price - principal)) / (principal * (holding_period / 365)) * 100

            result = {
                'holding_period_return': round(holding_period_return, 2)
            }
        except ValueError:
            return render_template('early_exit.html', error="Please enter valid numbers.")
    return render_template('early_exit.html', result=result)

# New route for Treasury Bills Rediscount
@app.route('/tbills-rediscount', methods=['GET', 'POST'])
def tbills_rediscount():
    result = None
    if request.method == 'POST':
        try:
            settlement_amount = float(request.form['settlement_amount'])
            rate = float(request.form['rate']) / 100  # Convert percentage to decimal (e.g., 30% → 0.3)
            days_to_maturity = float(request.form['days_to_maturity'])
            initial_fv = float(request.form['initial_fv'])

            if settlement_amount <= 0 or rate < 0 or days_to_maturity <= 0 or initial_fv <= 0:
                return render_template('tbills_rediscount.html', error="Please enter valid positive numbers.")

            # Calculate Settlement Face Value (FV) using the formula: FV = Settlement Amount * (1 + r)^(x/364)
            settlement_fv = settlement_amount * (1 + rate) ** (days_to_maturity / 364)
            
            # Calculate Face Value After Rediscount: Initial FV - Partial FV
            face_value_after_rediscount = initial_fv - settlement_fv

            result = {
                'settlement_fv': "{:,.2f}".format(settlement_fv),  # Format with commas and 2 decimal places
                'face_value_after_rediscount': "{:,.2f}".format(face_value_after_rediscount)
            }
        except ValueError:
            return render_template('tbills_rediscount.html', error="Please enter valid numbers.")

    return render_template('tbills_rediscount.html', result=result)

if __name__ == '__main__':
    # For local development, use waitress
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