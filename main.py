from preprocess import getDataLoaders
import math
from torch.utils.tensorboard import SummaryWriter
import argparse
from train import *
import random
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main(data_loader_dict, args, optim_config, cuda, writer, one_subject, seed=3):
    set_seed(seed)

    iteration = 7  # seed3 only
    acc = train_rsm_codg(data_loader_dict, optim_config, cuda, args, iteration, writer, one_subject)
    return acc

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description='RSM-CoDG: Region-aware Spatiotemporal Modeling with Collaborative DG')
    
    # Experiment setup
    parser.add_argument("--way", type=str, default='RSM-CoDG/seed3', help="experiment tag")
    parser.add_argument("--index", type=str, default='RSMCoDG_1024', help="tensorboard index")
    
    # Dataset setup
    parser.add_argument("--dataset_name", type=str, nargs='?', default='seed3', choices=['seed3'],
                       help="dataset name (seed3 only)")
    parser.add_argument("--session", type=str, nargs='?', default='1', help="selected session")
    parser.add_argument("--subjects", type=int, choices=[15], default=15, help="the number of all subject")
    parser.add_argument("--dim", type=int, default=310, help="dim of input")
    
    # Model setup
    parser.add_argument("--input_dim", type=int, default=310, help="input dim is the same with sample's last dim")
    parser.add_argument("--hid_dim", type=int, default=64, help="hidden dim for RGRM and MSTT")
    parser.add_argument("--epoch_training", type=int, default=200, help="epoch of training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="dropout rate")
    
    # DG loss weights
    parser.add_argument("--weight_feature_orthogonal", type=float, default=0.2, 
                       help="weight for feature orthogonal loss")
    parser.add_argument("--weight_attention_contrastive", type=float, default=0.2, 
                       help="weight for attention contrastive loss")
    parser.add_argument("--weight_feature_mmd", type=float, default=0.2, 
                       help="weight for feature MMD loss")
    
    # DG schedule
    parser.add_argument("--dg_warmup_epochs", type=int, default=50, 
                       help="number of epochs to warm up domain generalization losses")
    parser.add_argument("--dg_max_weight", type=float, default=1.0, 
                       help="maximum weight for domain generalization losses")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=50, help="patience for early stopping")

    args = parser.parse_args()
    args.source_subjects = args.subjects - 1
    args.seed3_path = ""

    if cuda:
        args.num_workers_train = 16
        args.num_workers_test = 8
    else:
        args.num_workers_train = 16
        args.num_workers_test = 8

    # Initial console summary before dataset-specific overrides
    print("=== RSM-CoDG: Region-aware spatiotemporal modeling + collaborative DG ===")
    print(f"Dataset (pre-adjust): {args.dataset_name}")
    args.path = args.seed3_path
    args.cls_classes = 3
    args.time_steps = 30
    args.batch_size = 512

    optim_config = {"lr": args.lr, "weight_decay": args.weight_decay}

    # Build DG weight dict
    args.dg_loss_weights = {
        'feature_orthogonal': args.weight_feature_orthogonal,
        'attention_contrastive': args.weight_attention_contrastive,
        'feature_mmd': args.weight_feature_mmd
    }

    # Leave-one-subject-out
    acc_list = []
    writer = SummaryWriter("data/session" + args.session + "/" + args.way + "/" + args.index)

    print("=== RSM-CoDG: Region-aware spatiotemporal modeling + collaborative DG ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Classes: {args.cls_classes}")
    print(f"Time steps: {args.time_steps}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dropout: {args.dropout_rate}")
    print("DG loss weights:")
    for loss_name, weight in args.dg_loss_weights.items():
        print(f"  - {loss_name}: {weight}")
    print(f"DG warmup epochs: {args.dg_warmup_epochs}")
    print(f"DG max weight: {args.dg_max_weight}")
    print(f"Start {args.subjects}-fold leave-one-subject-out...\n")
    
    for one_subject in range(0, args.subjects):
        print(f"===== Train Subject {one_subject + 1}/{args.subjects} =====")
        
        # 1. Data prep
        source_loaders, test_loader = getDataLoaders(one_subject, args)
        
        if source_loaders is None or test_loader is None:
            print(f"Subject {one_subject + 1} data failed to load, skip...")
            continue
            
        data_loader_dict = {"source_loader": source_loaders, "test_loader": test_loader}
        
        # 2. Train
        acc = main(data_loader_dict, args, optim_config, cuda, writer, one_subject)
        writer.add_scalars('single experiment acc: ', {'test acc': acc}, one_subject + 1)
        writer.flush()
        acc_list.append(acc)
        
        print(f"Subject {one_subject + 1} done, acc: {acc:.6f}\n")

    # Summary
    if len(acc_list) > 0:
        avg_acc = np.average(acc_list)
        std_acc = np.std(acc_list)
        
        print("=== Final results ===")
        print(f"Average acc: {avg_acc:.6f}")
        print(f"Std: {std_acc:.6f}")
        print(f"Per-subject acc: {[f'{acc:.6f}' for acc in acc_list]}")
        
        writer.add_text('final acc avg', str(avg_acc))
        writer.add_text('final acc std', str(std_acc))
        acc_list_str = [str(x) for x in acc_list]
        writer.add_text('final each acc', ",".join(acc_list_str))
        writer.add_scalars('final experiment acc scala: /avg', {'test acc': avg_acc})
        writer.add_scalars('final experiment acc scala: /std', {'test acc': std_acc})
        
        print("\nDG effects:")
        print("- Multi-scale attention: spatial + temporal")
        print("- Unified DG constraints: 5 complementary losses")
        print(f"- Adaptive scheduling: first {args.dg_warmup_epochs} epochs warmup, max weight {args.dg_max_weight}")
        print(f"Results saved to: data/session{args.session}/{args.way}/{args.index}/")
    else:
            print("No subject finished training!")
        
    writer.close()