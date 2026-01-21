import os
import time
from model import *
import numpy as np
from test import *
from collections import defaultdict
import copy

# Check CUDA status
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name()}")

def train_rsm_codg(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    """Train the RSM-CoDG model"""
    # Data loaders
    source_loader = data_loader_dict['source_loader']
    
    # Create RSM-CoDG model
    model = RSMCoDGModel(
        cuda=cuda, 
        number_of_category=args.cls_classes,
        dropout_rate=args.dropout_rate
    )
    
    if cuda:
        model = model.cuda()

    # Build source iterators
    source_iters = []
    for i in range(len(source_loader)):
        source_iters.append(iter(source_loader[i]))

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    # Early stopping state
    best_test_acc = 0
    patience = args.patience
    wait = 0
    acc_final = 0

    print("Start training RSM-CoDG ...")
    print(f"DG loss weights: {args.dg_loss_weights}")
    print(f"DG warmup epochs: {args.dg_warmup_epochs}")
    
    for epoch in range(args.epoch_training):
        print(f"Epoch: {epoch + 1}/{args.epoch_training}")
        start_time = time.time()
        model.train()
        
        # DG loss weight schedule
        if epoch < args.dg_warmup_epochs:
            dg_weight_multiplier = (epoch + 1) / args.dg_warmup_epochs
        else:
            dg_weight_multiplier = args.dg_max_weight
        
        # Running stats
        total_loss = 0
        total_cls_loss = 0
        total_dg_losses = {
            'attention_sparsity': 0,
            'feature_orthogonal': 0,
            'attention_contrastive': 0,
            'feature_mmd': 0,
            'feature_consistency': 0
        }
        total_correct = 0
        total_samples = 0
        
        for i in range(1, iteration + 1):
            epoch_loss = 0
            epoch_cls_loss = 0
            epoch_dg_losses = {
                'attention_sparsity': 0,
                'feature_orthogonal': 0,
                'attention_contrastive': 0,
                'feature_mmd': 0,
                'feature_consistency': 0
            }
            epoch_correct = 0
            epoch_samples = 0
            
            # Iterate each source domain
            for j in range(len(source_iters)):
                try:
                    batch_data, batch_labels = next(source_iters[j])
                except StopIteration:
                    source_iters[j] = iter(source_loader[j])
                    batch_data, batch_labels = next(source_iters[j])

                if cuda:
                    batch_data = batch_data.cuda()
                    batch_labels = batch_labels.cuda()

                # Build subject ids
                subject_ids = torch.full((batch_data.size(0),), j, dtype=torch.long)
                if cuda:
                    subject_ids = subject_ids.cuda()

                optimizer.zero_grad()

                # Forward pass
                x_pred, x_logits, dg_losses = model(
                    batch_data, 
                    subject_ids=subject_ids, 
                    labels=batch_labels.squeeze(), 
                    apply_noise=True
                )
                
                # Classification loss
                cls_loss = F.nll_loss(x_pred, batch_labels.squeeze())
                
                # Total loss
                total_batch_loss = cls_loss
                
                # Add DG losses
                if dg_losses:
                    current_dg_loss = 0
                    for loss_name, loss_value in dg_losses.items():
                        if loss_value is not None and loss_name in args.dg_loss_weights:
                            weighted_loss = args.dg_loss_weights[loss_name] * loss_value * dg_weight_multiplier
                            current_dg_loss += weighted_loss
                            epoch_dg_losses[loss_name] += weighted_loss.item()
                    
                    total_batch_loss += current_dg_loss
                
                # Backprop
                total_batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                # Accuracy
                _, pred = torch.max(x_pred, dim=1)
                correct = pred.eq(batch_labels.squeeze().data.view_as(pred)).sum().item()
                
                epoch_loss += total_batch_loss.item()
                epoch_cls_loss += cls_loss.item()
                epoch_correct += correct
                epoch_samples += len(batch_labels)

            total_loss += epoch_loss
            total_cls_loss += epoch_cls_loss
            for key in total_dg_losses:
                total_dg_losses[key] += epoch_dg_losses[key]
            total_correct += epoch_correct
            total_samples += epoch_samples

        # LR schedule
        scheduler.step()

        # Epoch averages
        avg_loss = total_loss / iteration
        avg_cls_loss = total_cls_loss / iteration
        avg_dg_losses = {key: val / iteration for key, val in total_dg_losses.items()}
        train_acc = float(total_correct) / total_samples
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Train time: {epoch_time:.2f}s")
        print(f"Train samples: {total_samples}")
        print(f"Total loss: {avg_loss:.6f}")
        print(f"Cls loss: {avg_cls_loss:.6f}")
        print(f"DG weight scale: {dg_weight_multiplier:.3f}")
        print("DG losses:")
        for loss_name, loss_value in avg_dg_losses.items():
            if loss_value > 0:
                print(f"  - {loss_name}: {loss_value:.6f}")
        print(f"Train acc: {train_acc:.6f}")

        # Log train metrics
        writer.add_scalars(f'subject: {one_subject + 1} train RSM-CoDG/loss',
                          {
                              'total_loss': avg_loss,
                              'cls_loss': avg_cls_loss,
                          }, epoch + 1)
        
        # Log each DG loss
        for loss_name, loss_value in avg_dg_losses.items():
            if loss_value > 0:
                writer.add_scalar(f'subject: {one_subject + 1} train RSM-CoDG/dg_{loss_name}', 
                                 loss_value, epoch + 1)
        
            writer.add_scalar(f'subject: {one_subject + 1} train RSM-CoDG/train_accuracy', 
                         train_acc, epoch + 1)
        
            writer.add_scalar(f'subject: {one_subject + 1} train RSM-CoDG/dg_weight_multiplier', 
                         dg_weight_multiplier, epoch + 1)
        
            writer.add_scalar(f'subject: {one_subject + 1} train RSM-CoDG/learning_rate', 
                         optimizer.param_groups[0]['lr'], epoch + 1)

        # Evaluate model
        testModel = RSMCoDGTestModel(model)
        acc_rsm_codg = test_rsm_codg(data_loader_dict["test_loader"], testModel, cuda, args.batch_size)
        print(f"Test acc: {acc_rsm_codg:.6f}")
        
        writer.add_scalars(f'subject: {one_subject + 1} test RSM-CoDG/test_acc',
                          {'test_acc': acc_rsm_codg}, epoch + 1)

        # Early stopping
        if acc_rsm_codg > best_test_acc:
            best_test_acc = acc_rsm_codg
            acc_final = acc_rsm_codg
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_model_state = copy.deepcopy(testModel.state_dict())
            wait = 0
            print(f"*** New best test acc: {best_test_acc:.6f} ***")
        else:
            wait += 1
            print(f"No improvement for {wait} epochs, best: {best_test_acc:.6f}")
            if wait >= patience:
                print(f"Early stop at epoch {epoch + 1}")
                break

        print("-" * 80)

    # Save best model
    modelDir = f"model/{args.way}/{args.index}/"
    try:
        os.makedirs(modelDir, exist_ok=True)
    except:
        pass

    # Persist trained weights
    torch.save(best_model_state, f"{modelDir}{one_subject}_rsm_codg_model.pth")
    torch.save(best_test_model_state, f"{modelDir}{one_subject}_rsm_codg_test_model.pth")
    
    # Persist config
    model_config = {
        'args': vars(args),
        'final_accuracy': acc_final,
        'best_epoch': epoch + 1 - wait,
        'dg_loss_weights': args.dg_loss_weights,
        'dg_weight_multiplier_final': dg_weight_multiplier
    }
    
    import json
    with open(f"{modelDir}{one_subject}_rsm_codg_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\nSubject {one_subject + 1} done")
    print(f"Best test acc: {acc_final:.6f}")
    print(f"Model saved to: {modelDir}")
    print(f"Config saved to: {modelDir}{one_subject}_rsm_codg_config.json")

    return acc_final