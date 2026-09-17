# MNIST Digit Classifier

`DigitClassifier` wraps three unrelated algorithms — CNN, Random Forest, and
a random baseline — behind one `predict(image) -> int` call. Each model
actually expects a different input shape internally (28x28x1 / flat 784 /
10x10 crop); `DigitClassifier` handles the conversion so callers don't have
to care which one they picked.

    from digit_classifier import DigitClassifier

    clf = DigitClassifier("cnn")   # or "rf", "rand"
    clf.predict(image)             # 28x28x1 in, int 0-9 out

`train()` isn't implemented — per the task, it just raises
`NotImplementedError`. Interface still declares it, since that's part of
what a classifier is supposed to do; this exercise just doesn't need it.

## Setup

    cd mnist
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

## Running it

    python demo.py              # quick look at all three algorithms
    python -m pytest tests/ -v  # test suite

If you've got a ROS environment sourced in your shell, it can leak a
`PYTHONPATH` that breaks pytest's plugin loading in a fresh venv — if you
hit an unrelated `ModuleNotFoundError: No module named 'yaml'`, run
`unset PYTHONPATH` first.
