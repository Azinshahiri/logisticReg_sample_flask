from pill_logistic import run_logistic_model

def test_basic_logistic_metrics():
    result = run_logistic_model()
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['precision'] <= 1
    assert 0 <= result['recall'] <= 1
    assert 0 <= result['f1'] <= 1
