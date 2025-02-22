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
            monthly_rate = rate / 12
            monthly_contribution = (target - (principal * (1 + monthly_rate) ** months)) * monthly_rate / ((1 + monthly_rate) ** months - 1)
            result = round(monthly_contribution, 2)
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
            monthly_rate = rate / 12
            end_balance = (principal * (1 + monthly_rate) ** months) + (monthly * ((1 + monthly_rate) ** months - 1) / monthly_rate)
            result = round(end_balance, 2)
        except ValueError:
            return render_template('end_balance.html', error="Please enter valid numbers.")

    return render_template('end_balance.html', result=result)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
    