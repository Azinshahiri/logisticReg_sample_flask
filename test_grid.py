from pill_grid import run_logistic_with_gridsearch

def test_grid_metrics_and_params():
    result = run_logistic_with_gridsearch()
    assert 'C' in result['best_params']
    assert 'penalty' in result['best_params']
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['f1'] <= 1
