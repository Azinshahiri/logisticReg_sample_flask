from flask import Flask, render_template
from pill_logistic import run_logistic_model
from pill_grid import run_logistic_with_gridsearch

app = Flask(__name__)

@app.route('/')
def index():
    basic_results = run_logistic_model()
    grid_results = run_logistic_with_gridsearch()

    return render_template(
        'index.html',
        basic=basic_results,
        grid=grid_results
    )

if __name__ == '__main__':
    app.run(debug=True)


