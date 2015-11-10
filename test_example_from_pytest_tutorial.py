## from py.test tutorial
## http://pytest.org/latest/getting-started.html#getstarted 

# content of test_sample.py
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5
