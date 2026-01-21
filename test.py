import torch
from torch.autograd import Variable

def test_rsm_codg(dataLoader, model, cuda, batch_size):
    """Test RSM-CoDG model"""
    print("Testing RSM-CoDG")
    index = 0
    count = 0
    data_set_all = 0
    
    if cuda:
        model = model.cuda()
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            
            # Model prediction
            x_shared_pred = model(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
            
            index += batch_size
    
    acc = float(count) / data_set_all
    return acc

def test_esagn(dataLoader, model, cuda, batch_size):
    """Test ESAGN model"""
    print("Testing ESAGN")
    index = 0
    count = 0
    data_set_all = 0
    
    if cuda:
        model = model.cuda()
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            
            x_shared_pred = model(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
            
            index += batch_size
    
    acc = float(count) / data_set_all
    return acc

def test_dgsgdt(dataLoader, model, cuda, batch_size):
    """Test DG-SGDT model"""
    print("Testing DG-SGDT")
    index = 0
    count = 0
    data_set_all = 0
    
    if cuda:
        model = model.cuda()
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            
            test_input, label = Variable(test_input), Variable(label)
            data_set_all += len(label)
            
            x_shared_pred = model(test_input)
            _, pred = torch.max(x_shared_pred, dim=1)
            count += pred.eq(label.squeeze().data.view_as(pred)).sum()
            
            index += batch_size
    
    acc = float(count) / data_set_all
    return acc

def evaluate_model_detailed(dataLoader, model, cuda, class_names=None):
    """Detailed evaluation with extra metrics"""
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Running detailed evaluation...")
    
    if cuda:
        model = model.cuda()
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (test_input, label) in enumerate(dataLoader):
            if cuda:
                test_input, label = test_input.cuda(), label.cuda()
            
            test_input, label = Variable(test_input), Variable(label)
            
            # Model prediction
            logits = model(test_input)
            probabilities = torch.softmax(logits, dim=1)
            
            _, pred = torch.max(logits, dim=1)
            max_probs, _ = torch.max(probabilities, dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(label.squeeze().cpu().numpy())
            all_confidences.extend(max_probs.cpu().numpy())
    
    # Metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    print(f"Overall accuracy: {accuracy:.6f}")
    print(f"Mean confidence: {np.mean(all_confidences):.6f}")
    
    # Class names
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]
    
    # Classification report
    print("\nClassification report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names, digits=4))
    
    # Confusion matrix
    print("\nConfusion matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'confusion_matrix': cm
    }