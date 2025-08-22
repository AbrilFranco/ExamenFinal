import io, base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from model import train_and_evaluate, get_total_data

app = Flask(__name__)
plt.switch_backend('Agg')

# --- RUTAS ---
@app.route('/')
def home():
    total = get_total_data()
    return render_template('index.html', total_data=total)

@app.route('/train', methods=['POST'])
def train():
    try:
        sample_size = int(request.form['training_samples'])
    except (ValueError, KeyError):
        return redirect(url_for('home'))

    results = train_and_evaluate(sample_size)

    return render_template(
        'results.html',
        accuracy=results["accuracy"],
        f1_score=results["f1_score"],
        plot_cm=base64.b64encode(open(results["confusion_matrix_url"], "rb").read()).decode(),
        plot_roc=base64.b64encode(open(results["roc_curve_url"], "rb").read()).decode(),
        plot_pr=base64.b64encode(open(results["pr_curve_url"], "rb").read()).decode(),
        sample_size=results["data_used"],
        f1_condition_met=(0.98 <= float(results["f1_score"].strip('%')) / 100 < 1.0)
    )

# ⚠️ Solo para pruebas locales
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
