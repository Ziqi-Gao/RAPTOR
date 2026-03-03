import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_controllers import NeuralController
from utils import load_model

import numpy as np
import pickle
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, f1_score
import torch

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']

def get_halubench_data(tokenizer, source_ds='pubmedQA'):

    # Load HaluBench dataset
    ds = load_dataset("PatronusAI/HaluBench")['test']
    # Filter for PubMedQA data
    ds = ds.filter(lambda x: x['source_ds'] == source_ds)

    template = "Consider the factuality of the ANSWER to the following QUESTION based on the CONTEXT provided.\n\n"
    template += "CONTEXT: {context}\n\n"
    template += "QUESTION: {question}\n\n"
    template += "ANSWER: {answer}"

    inputs = []
    labels = []
    for item in ds:
        formatted_str = template.format(context=item['passage'], question=item['question'], answer=item['answer'])
        chat = [
            {
                "role": "user", 
                "content": formatted_str
            },
        ]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))
        labels.append(int(item['label']=='FAIL'))

    print("="*100)
    print("inputs", inputs[0])
    print("="*100)
    print("labels", labels[0])
    print("="*100)
    # Split dataset into train and test
    return np.array(inputs), np.array(labels)

def get_cross_val_splits(n_total, n_folds=5, source_ds='', target_ds='', seed=None):
    """
    Generate k-fold cross validation splits with train/validation splits for each fold.
    n_total: total number of samples
    n_folds: number of folds for cross validation
    source_ds: source dataset name for file path
    seed: random seed for reproducibility
    """
    seed_suffix = f"_seed_{seed}" if seed is not None else ""
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halubench_results/{source_ds}_{target_ds}'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{source_ds}_cv_splits_nfolds_{n_folds}_ntotal_{n_total}{seed_suffix}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)
    fold_size = n_total // n_folds
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle indices once
    shuffled_indices = np.random.permutation(indices)
    
    # Create folds
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n_total
        test_indices = shuffled_indices[start_idx:end_idx]
        val_indices = np.array([i for i in shuffled_indices if i not in test_indices])
        
        splits.append({
            'val_indices': val_indices,
            'test_indices': test_indices,
            'fold': fold
        })

    with open(out_name, 'wb') as f:
        pickle.dump(splits, f)

    return splits

def split_states_on_idx(inputs, split):
    """
    Split hidden states into train/val/test based on indices.
    """
    val_inputs, test_inputs = {}, {}
    for layer_idx, layer_states in inputs.items():
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return val_inputs, test_inputs

def compute_overall_metrics(predictions, labels, threshold=0.5):
    """
    Compute overall metrics from all folds.
    predictions: torch.Tensor with shape (n_samples,)
    labels: torch.Tensor with shape (n_samples,)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    # Compute overall metrics
    overall_auc = roc_auc_score(labels, predictions)
    overall_f1 = f1_score(labels, predictions > threshold)
    
    metrics = {
        'auc': overall_auc,
        'f1': overall_f1,
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--source_ds', type=str, default='RAGTruth')
    parser.add_argument('--target_ds', type=str, default='pubmedQA')
    parser.add_argument('--n_folds', type=int, default=10)
    parser.add_argument('--n_seeds', type=int, default=3)
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    source_ds = args.source_ds
    target_ds = args.target_ds
    n_folds = args.n_folds
    n_seeds = args.n_seeds

    unsupervised = control_method=='pca'
    if control_method not in ['rfm']:
        n_components=1
        
    use_logistic=(control_method=='logistic')
    
    original_control_method = str(control_method)
    if control_method=='rfm':
        use_rfm=True
    else:
        use_rfm=False

    language_model, tokenizer = load_model(model=model_name)
    source_inputs, source_labels = get_halubench_data(tokenizer, source_ds)
    target_inputs, target_labels = get_halubench_data(tokenizer, target_ds)

    print("Number of target inputs", len(target_inputs))
    print("Number of target labels", len(target_labels))

    print("Getting hidden states")
    hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'{target_ds}_{model_name}_unsupervised_{unsupervised}.pth')
    if os.path.exists(hidden_states_path):
        with open(hidden_states_path, 'rb') as f:
            hidden_states = pickle.load(f) 
    else:
        from direction_utils import get_hidden_states
        controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            rfm_iters=5,
            batch_size=1
        )
    
        hidden_states = get_hidden_states(target_inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(hidden_states_path, 'wb') as f:
            pickle.dump(hidden_states, f)

    # Initialize list to store all predictions across seeds
    all_seeds_best_layer_predictions = []
    all_seeds_aggregated_predictions = []
    
    for seed in range(n_seeds):
        print(f"\nProcessing seed {seed}/{n_seeds}...")
        
        # Split data into n_folds folds with specific seed
        splits = get_cross_val_splits(
            n_total=len(target_inputs),
            n_folds=n_folds,
            source_ds=source_ds,
            target_ds=target_ds,
            seed=seed
        )
        
        # Initialize list to store all predictions for this seed
        seed_best_layer_predictions = []
        seed_aggregated_predictions = []
        seed_idx = []
        
        for fold in tqdm(range(n_folds)):
            split = splits[fold]
            
            # Split the data into train/val/test
            val_hidden_states_on_fold, test_hidden_states_on_fold = split_states_on_idx(hidden_states, split)
            val_labels_on_fold = target_labels[split['val_indices']]
            test_labels_on_fold = target_labels[split['test_indices']]

            # Average predictions over all default folds
            default_n_folds = 10
            test_best_layer_predictions = []
            test_aggregated_predictions = []
            
            for default_fold in range(default_n_folds):
                controller = NeuralController(
                    language_model,
                    tokenizer,
                    control_method=control_method,
                    rfm_iters=5,
                    batch_size=1
                )

                controller.load(concept=f'{source_ds}_fold_{default_fold}_out_of_{default_n_folds}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

                # Train on training set, validate on validation set, final evaluation on test set
                val_metrics, test_metrics, _, test_predictions = controller.evaluate_directions(
                    val_hidden_states_on_fold, val_labels_on_fold,
                    test_hidden_states_on_fold, test_labels_on_fold,
                    n_components=n_components,
                    use_logistic=use_logistic,
                    use_rfm=use_rfm,
                    unsupervised=unsupervised,
                )
                
                test_best_layer_predictions.append(test_predictions['best_layer'])
                test_aggregated_predictions.append(test_predictions['aggregation'])
            
            # Average predictions over all default folds
            test_best_layer_predictions = torch.stack(test_best_layer_predictions).mean(dim=0)
            test_aggregated_predictions = torch.stack(test_aggregated_predictions).mean(dim=0)
            
            seed_best_layer_predictions.append(test_best_layer_predictions)
            seed_aggregated_predictions.append(test_aggregated_predictions)
            seed_idx.append(torch.from_numpy(split['test_indices']))
            
            # Save individual fold metrics
            results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halubench_results/{source_ds}_{target_ds}'
            os.makedirs(results_dir, exist_ok=True)
            
            out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_fold_{fold}_val_metrics.pkl'
            with open(out_name, 'wb') as f:
                pickle.dump(val_metrics, f)

            out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_fold_{fold}_test_metrics.pkl'
            with open(out_name, 'wb') as f:
                pickle.dump(test_metrics, f)

        # Combine all predictions for this seed
        seed_best_layer_predictions = torch.cat(seed_best_layer_predictions, dim=0)
        seed_aggregated_predictions = torch.cat(seed_aggregated_predictions, dim=0)
        seed_idx = torch.cat(seed_idx, dim=0)
        
        # Sort predictions according to index order
        sorted_order = torch.argsort(seed_idx)
        seed_best_layer_predictions = seed_best_layer_predictions[sorted_order]
        seed_aggregated_predictions = seed_aggregated_predictions[sorted_order]
        seed_idx = seed_idx[sorted_order]

        assert torch.allclose(seed_idx, torch.arange(len(seed_idx)))
        
        # Compute and save seed-specific overall metrics
        seed_best_layer_metrics = compute_overall_metrics(seed_best_layer_predictions, target_labels)
        seed_aggregated_metrics = compute_overall_metrics(seed_aggregated_predictions, target_labels)
        
        print(f"\nSeed {seed} Metrics:")
        print(f"Best Layer AUC: {seed_best_layer_metrics['auc']:.3f}")
        print(f"Aggregated AUC: {seed_aggregated_metrics['auc']:.3f}")
        print(f"Best Layer F1: {seed_best_layer_metrics['f1']:.3f}")
        print(f"Aggregated F1: {seed_aggregated_metrics['f1']:.3f}")
        
        # Save seed-specific metrics and predictions
        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_best_layer_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(seed_best_layer_metrics, f)

        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_aggregated_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(seed_aggregated_metrics, f)
            
        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_all_predictions.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(seed_best_layer_predictions, f)

        out_name = f'{results_dir}/{model_name}_{original_control_method}_seed_{seed}_aggregated_predictions.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(seed_aggregated_predictions, f)
            
        # Store predictions for averaging across seeds
        all_seeds_best_layer_predictions.append(seed_best_layer_predictions)
        all_seeds_aggregated_predictions.append(seed_aggregated_predictions)
    
    # Average predictions across all seeds
    all_seeds_best_layer_predictions = torch.stack(all_seeds_best_layer_predictions).mean(dim=0)
    all_seeds_aggregated_predictions = torch.stack(all_seeds_aggregated_predictions).mean(dim=0)
    
    # Compute overall metrics averaged across seeds
    overall_best_layer_metrics = compute_overall_metrics(all_seeds_best_layer_predictions, target_labels)
    overall_aggregated_metrics = compute_overall_metrics(all_seeds_aggregated_predictions, target_labels)
    
    print("\nOverall Metrics (Averaged Across Seeds):")
    print(f"Best Layer AUC: {overall_best_layer_metrics['auc']:.3f}")
    print(f"Aggregated AUC: {overall_aggregated_metrics['auc']:.3f}")
    print(f"Best Layer F1: {overall_best_layer_metrics['f1']:.3f}")
    print(f"Aggregated F1: {overall_aggregated_metrics['f1']:.3f}")

    # Save overall metrics and predictions (averaged across seeds)
    out_name = f'{results_dir}/{model_name}_{original_control_method}_overall_best_layer_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(overall_best_layer_metrics, f)

    out_name = f'{results_dir}/{model_name}_{original_control_method}_overall_aggregated_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(overall_aggregated_metrics, f)


if __name__ == '__main__':              
    main()