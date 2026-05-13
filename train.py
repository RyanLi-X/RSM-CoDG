import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from model import RSMCoDGModel, RSMCoDGTestModel
from test import test_rsm_codg


def _labels_to_vector(labels, device):
    return labels.to(device=device, dtype=torch.long).view(-1)


def _build_dg_weight_multiplier(epoch, args):
    warmup_epochs = max(int(getattr(args, "dg_warmup_epochs", 0)), 0)
    max_weight = float(getattr(args, "dg_max_weight", 1.0))
    if warmup_epochs <= 0:
        return max_weight
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    return max_weight


def _weighted_dg_loss(dg_losses, loss_weights, multiplier):
    total = None
    logged = {}

    for loss_name, loss_value in dg_losses.items():
        if loss_value is None or loss_name not in loss_weights:
            continue
        weighted = float(loss_weights[loss_name]) * loss_value * multiplier
        total = weighted if total is None else total + weighted
        logged[loss_name] = weighted.detach().item()

    return total, logged


def train_rsm_codg(data_loader_dict, optimizer_config, cuda, args, iteration, writer, one_subject):
    source_loaders = data_loader_dict["source_loader"]
    test_loader = data_loader_dict["test_loader"]

    if not source_loaders:
        raise ValueError("source_loader is empty; cannot train RSM-CoDG.")

    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    model = RSMCoDGModel(
        cuda=(device.type == "cuda"),
        number_of_category=args.cls_classes,
        dropout_rate=args.dropout_rate,
        time_steps=args.time_steps,
    ).to(device)

    source_iters = [iter(loader) for loader in source_loaders]
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    loss_weights = getattr(args, "dg_loss_weights", {})
    patience = int(getattr(args, "patience", 50))
    num_iterations = int(iteration) if iteration is not None else min(len(loader) for loader in source_loaders)

    best_test_acc = float("-inf")
    best_model_state = None
    best_test_model_state = None
    best_epoch = 0
    wait = 0

    for epoch in range(args.epoch_training):
        print(f"Epoch: {epoch + 1}/{args.epoch_training}")
        start_time = time.time()
        model.train()

        dg_weight_multiplier = _build_dg_weight_multiplier(epoch, args)
        total_loss = 0.0
        total_cls_loss = 0.0
        total_dg_losses = {
            "attention_sparsity": 0.0,
            "feature_orthogonal": 0.0,
            "attention_contrastive": 0.0,
            "feature_mmd": 0.0,
            "feature_consistency": 0.0,
        }
        for loss_name in loss_weights:
            total_dg_losses.setdefault(loss_name, 0.0)
        total_correct = 0
        total_samples = 0
        total_batches = 0

        for _ in range(num_iterations):
            for source_idx, loader in enumerate(source_loaders):
                try:
                    batch_data, batch_labels = next(source_iters[source_idx])
                except StopIteration:
                    source_iters[source_idx] = iter(loader)
                    batch_data, batch_labels = next(source_iters[source_idx])

                batch_data = batch_data.to(device)
                batch_labels = _labels_to_vector(batch_labels, device)
                subject_ids = torch.full(
                    (batch_data.size(0),),
                    source_idx,
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)
                x_pred, _, dg_losses, cls_loss = model(
                    batch_data,
                    subject_ids=subject_ids,
                    labels=batch_labels,
                    apply_noise=True,
                )

                if cls_loss is None:
                    cls_loss = F.nll_loss(x_pred, batch_labels)

                dg_loss, logged_dg = _weighted_dg_loss(dg_losses, loss_weights, dg_weight_multiplier)
                total_batch_loss = cls_loss if dg_loss is None else cls_loss + dg_loss

                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                pred = x_pred.argmax(dim=1)
                total_correct += pred.eq(batch_labels).sum().item()
                total_samples += batch_labels.numel()
                total_loss += total_batch_loss.detach().item()
                total_cls_loss += cls_loss.detach().item()
                for loss_name, loss_value in logged_dg.items():
                    total_dg_losses[loss_name] += loss_value
                total_batches += 1

        scheduler.step()

        avg_loss = total_loss / max(num_iterations, 1)
        avg_cls_loss = total_cls_loss / max(num_iterations, 1)
        avg_dg_losses = {
            name: value / max(num_iterations, 1)
            for name, value in total_dg_losses.items()
        }
        train_acc = float(total_correct) / total_samples if total_samples else 0.0
        epoch_time = time.time() - start_time

        print(f"Train ACC: {train_acc:.6f} | Loss: {avg_loss:.6f} | Time: {epoch_time:.1f}s")

        writer.add_scalars(
            f"subject: {one_subject + 1} train RSM-CoDG/loss",
            {"total_loss": avg_loss, "cls_loss": avg_cls_loss},
            epoch + 1,
        )
        for loss_name, loss_value in avg_dg_losses.items():
            if loss_value <= 0:
                continue
            writer.add_scalar(
                f"subject: {one_subject + 1} train RSM-CoDG/dg_{loss_name}",
                loss_value,
                epoch + 1,
            )
        writer.add_scalar(
            f"subject: {one_subject + 1} train RSM-CoDG/train_accuracy",
            train_acc,
            epoch + 1,
        )
        writer.add_scalar(
            f"subject: {one_subject + 1} train RSM-CoDG/dg_weight_multiplier",
            dg_weight_multiplier,
            epoch + 1,
        )
        writer.add_scalar(
            f"subject: {one_subject + 1} train RSM-CoDG/learning_rate",
            optimizer.param_groups[0]["lr"],
            epoch + 1,
        )

        test_model = RSMCoDGTestModel(model).to(device)
        acc_rsm_codg = test_rsm_codg(test_loader, test_model, cuda=(device.type == "cuda"), batch_size=args.batch_size)
        print(f"Val ACC: {acc_rsm_codg:.6f}")

        writer.add_scalars(
            f"subject: {one_subject + 1} test RSM-CoDG/test_acc",
            {"test_acc": acc_rsm_codg},
            epoch + 1,
        )

        if acc_rsm_codg > best_test_acc:
            best_test_acc = acc_rsm_codg
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_test_model_state = copy.deepcopy(test_model.state_dict())
            wait = 0
            print(f"*** New best: {best_test_acc:.6f} ***")
        else:
            wait += 1
            print(f"No improvement for {wait} epoch(s), current best: {best_test_acc:.6f}")
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

        print("-" * 80)

    model_dir = f"model/{args.way}/{args.index}/"
    os.makedirs(model_dir, exist_ok=True)

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())
        best_test_model_state = copy.deepcopy(RSMCoDGTestModel(model).state_dict())
        best_test_acc = 0.0

    model_path = f"{model_dir}{one_subject}_rsm_codg_model.pth"
    test_model_path = f"{model_dir}{one_subject}_rsm_codg_test_model.pth"
    config_path = f"{model_dir}{one_subject}_rsm_codg_config.json"

    torch.save(best_model_state, model_path)
    torch.save(best_test_model_state, test_model_path)

    model_config = {
        "args": vars(args),
        "final_accuracy": best_test_acc,
        "best_epoch": best_epoch,
        "dg_loss_weights": loss_weights,
        "dg_weight_multiplier_final": dg_weight_multiplier,
    }
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"\nSubject {one_subject + 1} training complete.")
    print(f"Best test accuracy: {best_test_acc:.6f}")
    print(f"Model saved to: {model_path}")
    print(f"Test model saved to: {test_model_path}")
    print(f"Config saved to: {config_path}")

    return best_test_acc


trainUnified = train_rsm_codg
