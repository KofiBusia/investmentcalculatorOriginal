from flask import Flask, render_template, request

app = Flask(__name__)

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
            rate = float(request.form['annual_return']) / 100

            if target <= 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('monthly_contribution.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                monthly_contribution = (target - principal) / months
            else:
                monthly_rate = rate / 12
                monthly_contribution = (target - (principal * (1 + monthly_rate) ** months)) * monthly_rate / ((1 + monthly_rate) ** months - 1)
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
            rate = float(request.form['annual_return']) / 100

            if monthly < 0 or principal < 0 or period <= 0 or rate < 0:
                return render_template('end_balance.html', error="Please enter valid positive numbers.")

            months = period * 12
            if rate == 0:
                end_balance = principal + monthly * months
            else:
                monthly_rate = rate / 12
                end_balance = (principal * (1 + monthly_rate) ** months) + (monthly * ((1 + monthly_rate) ** months - 1) / monthly_rate)
            result = "{:,.2f}".format(end_balance)
        except ValueError:
            return render_template('end_balance.html', error="Please enter valid numbers.")

    return render_template('end_balance.html', result=result)

@app.route('/stocks', methods=['GET', 'POST'])
def stocks():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            num_shares = float(request.form['num_shares'])
            dividends = float(request.form['dividends'])
            initial_investment = float(request.form['initial_investment'])

            if any(x < 0 for x in [purchase_price, selling_price, num_shares, dividends, initial_investment]):
                return render_template('stocks.html', error="Please enter valid non-negative numbers.")

            capital_gain = (selling_price - purchase_price) * num_shares
            total_return = (dividends + capital_gain) / initial_investment * 100
            result = {
                'capital_gain': "{:,.2f}".format(capital_gain),
                'total_return': round(total_return, 2)
            }
        except ValueError:
            return render_template('stocks.html', error="Please enter valid numbers.")

    return render_template('stocks.html', result=result)

@app.route('/bonds', methods=['GET', 'POST'])
def bonds():
    result = None
    if request.method == 'POST':
        try:
            purchase_price = float(request.form['purchase_price'])
            selling_price = float(request.form['selling_price'])
            coupon_payments = float(request.form['coupon_payments'])

            if any(x < 0 for x in [purchase_price, selling_price, coupon_payments]):
                return render_template('bonds.html', error="Please enter valid non-negative numbers.")

            price_change = selling_price - purchase_price
            total_return = (coupon_payments + price_change) / purchase_price * 100
            result = {
                'total_return': round(total_return, 2)
            }
        except ValueError:
            return render_template('bonds.html', error="Please enter valid numbers.")

    return render_template('bonds.html', result=result)

@app.route('/tbills', methods=['GET', 'POST'])
def tbills():
    result = None
    if request.method == 'POST':
        try:
            face_value = float(request.form['face_value'])
            purchase_price = float(request.form['purchase_price'])
            days_to_maturity = float(request.form['days_to_maturity'])

            if face_value <= 0 or purchase_price <= 0 or days_to_maturity <= 0:
                return render_template('tbills.html', error="Please enter valid positive numbers.")

            discount_yield = ((face_value - purchase_price) / purchase_price) * (364 / days_to_maturity) * 100
            result = {
                'discount_yield': round(discount_yield, 2)
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