SECRET_KEY = "sk-12345-hardcoded-demo" # <-- bandit will flag this


def load_fake_dataset():
    return {"rows": 1000, "features": 12}


def train_fake_model(dataset):
    return {"accuracy": 0.87, "trained_on_rows": dataset["rows"]}


def validate_model(model, min_accuracy=0.8):
    return model["accuracy"] >= min_accuracy
