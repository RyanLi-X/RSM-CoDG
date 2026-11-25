import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from model_region import RSMCoDG
from test import evaluate


def next_source_batch(train_iters, train_loaders):
    xs, ys, ids = [], [], []
    for j, loader in enumerate(train_loaders):
        try:
            x, y = next(train_iters[j])
        except StopIteration:
            train_iters[j] = iter(loader)
            x, y = next(train_iters[j])
        xs.append(x)
        ys.append(y)
        ids.append(torch.full((len(x),), j, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys).view(-1), torch.cat(ids)


def train_one_fold(train_loaders, val_loader, test_loader, args, cuda, writer, target, train_ids, val_ids):
    model = RSMCoDG(num_source_subjects=len(train_loaders), num_classes=args.num_classes,
                    time_steps=args.time_steps, dropout_rate=args.dropout_rate,
                    use_calibration=not args.no_calibration)
    if cuda:
        model = model.cuda()
    train_iters = [iter(loader) for loader in train_loaders]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)

    best_val_acc, best_epoch, wait = -1.0, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    tag = f'target {target}'

    for epoch in range(1, args.epoch_training + 1):
        start = time.time()
        model.train()
        dg_scale = args.dg_max_weight * min(1.0, epoch / args.dg_warmup_epochs)  # linear warm-up
        totals = {'total': 0.0, 'cls': 0.0, **{name: 0.0 for name in args.dg_loss_weights}}
        correct = seen = 0

        for _ in range(args.iterations):
            x, y, subject_ids = next_source_batch(train_iters, train_loaders)
            if cuda:
                x, y, subject_ids = x.cuda(), y.cuda(), subject_ids.cuda()

            optimizer.zero_grad()
            logits, dg_losses = model(x, subject_ids=subject_ids, labels=y, apply_noise=True)
            cls_loss = F.cross_entropy(logits, y)
            weighted = {name: args.dg_loss_weights[name] * value * dg_scale
                        for name, value in dg_losses.items() if name in args.dg_loss_weights}
            loss = cls_loss + sum(weighted.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            totals['total'] += loss.item()
            totals['cls'] += cls_loss.item()
            for name, value in weighted.items():
                totals[name] += value.item()
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += y.size(0)
        scheduler.step()

        avg = {name: value / args.iterations for name, value in totals.items()}
        train_acc = correct / seen
        val_acc = evaluate(model, val_loader, cuda)
        writer.add_scalars(f'{tag}/loss', {'total': avg['total'], 'cls': avg['cls']}, epoch)
        for name in args.dg_loss_weights:
            writer.add_scalar(f'{tag}/dg_{name}', avg[name], epoch)
        writer.add_scalars(f'{tag}/accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar(f'{tag}/dg_weight', dg_scale, epoch)
        writer.add_scalar(f'{tag}/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch:3d} | {time.time() - start:5.1f}s | loss {avg['total']:.4f} | "
              f"cls {avg['cls']:.4f} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, wait = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, cuda)
    writer.add_scalar('test_acc', test_acc, target)

    model_dir = os.path.join('model', args.way, args.index)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_state, os.path.join(model_dir, f'target{target}_model.pth'))
    with open(os.path.join(model_dir, f'target{target}_result.json'), 'w') as f:
        json.dump({'args': vars(args), 'target': target, 'train_subjects': train_ids, 'val_subjects': val_ids,
                   'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'test_acc': test_acc}, f, indent=2)
    print(f"Target {target}: best val_acc {best_val_acc:.4f} at epoch {best_epoch} | test_acc {test_acc:.4f}")
    return test_acc, best_val_acc, best_epoch
