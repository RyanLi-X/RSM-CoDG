import torch


def _labels_to_vector(labels, device):
    return labels.to(device=device, dtype=torch.long).view(-1)


@torch.no_grad()
def _test_classifier(dataLoader, model, cuda, batch_size, name):
    del batch_size
    print(f"Testing {name}")
    count = 0
    data_set_all = 0

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    for test_input, label in dataLoader:
        test_input = test_input.to(device)
        label = _labels_to_vector(label, device)
        output = model(test_input)
        if isinstance(output, tuple):
            output = output[1] if len(output) > 1 else output[0]
        _, pred = torch.max(output, dim=1)
        count += pred.eq(label).sum().item()
        data_set_all += label.numel()

    return float(count) / data_set_all if data_set_all else 0.0


def test_rsm_codg(dataLoader, RSMCoDGTestModel, cuda, batch_size):
    return _test_classifier(dataLoader, RSMCoDGTestModel, cuda, batch_size, "RSM-CoDG")


def evaluate_model_detailed(dataLoader, model, cuda, class_names=None):
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix

    print("Running detailed model evaluation...")

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for test_input, label in dataLoader:
            test_input = test_input.to(device)
            label = _labels_to_vector(label, device)
            logits = model(test_input)
            if isinstance(logits, tuple):
                logits = logits[1] if len(logits) > 1 else logits[0]
            probabilities = torch.softmax(logits, dim=1)
            _, pred = torch.max(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)

            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())

    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))

    print(f"Overall accuracy: {accuracy:.6f}")
    print(f"Average prediction confidence: {np.mean(all_confidences):.6f}")

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]

    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))

    print("\nConfusion matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    return {
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
        "confidences": all_confidences,
        "confusion_matrix": cm,
    }
