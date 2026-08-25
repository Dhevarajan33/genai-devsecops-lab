from model_utils import load_fake_dataset, train_fake_model, validate_model


def test_load_fake_dataset():
    data = load_fake_dataset()
    assert data["rows"] == 1000


def test_train_fake_model():
    data = load_fake_dataset()
    model = train_fake_model(data)
    assert 0 <= model["accuracy"] <= 1


def test_validate_model_passes_threshold():
    model = {"accuracy": 0.87}
    assert validate_model(model) is True
