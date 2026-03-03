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


def get_halubench_data(source_ds='pubmedQA', prompt_version='v1'):
    # Load HaluBench dataset
    ds = load_dataset("PatronusAI/HaluBench")['test']
    # Filter for PubMedQA data
    ds = ds.filter(lambda x: x['source_ds'] == source_ds)

    if prompt_version == 'v1':
        template = "Is the ANSWER to the following QUESTION correct strictly based on the CONTEXT provided. State 'Yes' if correct, or 'No' if incorrect.\n\n"
        template += "CONTEXT: {context}\n\n"
        template += "QUESTION: {question}\n\n"
        template += "ANSWER: {answer}\n\n"

    inputs = []
    labels = []
    for item in ds:
        formatted_str = template.format(context=item['passage'], question=item['question'], answer=item['answer'])
        inputs.append(formatted_str)
        labels.append(int(item['label']=='FAIL'))

    return np.array(inputs), np.array(labels)


def get_cross_val_splits(n_total, n_folds=5, source_ds=''):
    """
    Generate k-fold cross validation splits with train/validation splits for each fold.
    n_total: total number of samples
    n_folds: number of folds for cross validation
    source_ds: source dataset name for file path
    """
    results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halubench_results/{source_ds}'
    os.makedirs(results_dir, exist_ok=True)
    out_name = f'{results_dir}/{source_ds}_cv_splits_nfolds_{n_folds}_ntotal_{n_total}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)
    fold_size = n_total // n_folds
    
    # Shuffle indices once
    shuffled_indices = np.random.permutation(indices)
    
    # Create folds
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n_total
        test_indices = shuffled_indices[start_idx:end_idx]
        remaining_indices = np.array([i for i in shuffled_indices if i not in test_indices])
        
        # Split remaining data into train/val (70/30)
        n_train = int(len(remaining_indices) * 0.7)
        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:]
        
        splits.append({
            'train_indices': train_indices,
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
    train_inputs, val_inputs, test_inputs = {}, {}, {}
    for layer_idx, layer_states in inputs.items():
        train_inputs[layer_idx] = layer_states[split['train_indices']]
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return train_inputs, val_inputs, test_inputs

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
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--source_ds', type=str, default='RAGTruth')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    source_ds = args.source_ds
    n_folds = args.n_folds
    prompt_version = args.prompt_version
    tuning_metric = args.tuning_metric

    if control_method not in ['rfm']:
        n_components=1
        tuning_metric = 'auc'

    language_model, tokenizer = load_model(model=model_name)
    unformatted_inputs, labels = get_halubench_data(source_ds, prompt_version)

    inputs = []
    for prompt in unformatted_inputs:
        chat = [
            {
                "role": "user", 
                "content": prompt
            },
        ]
        inputs.append(tokenizer.apply_chat_template(chat, tokenize=False))

    print("Number of inputs", len(inputs))
    print("Number of labels", len(labels))

    print("Getting hidden states")
    hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'{source_ds}_{model_name}_prompt_{prompt_version}.pth')
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
            batch_size=1,
        )
    
        hidden_states = get_hidden_states(inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(hidden_states_path, 'wb') as f:
            pickle.dump(hidden_states, f)

    # Split data into n_folds folds
    splits = get_cross_val_splits(
        n_total=len(inputs),
        n_folds=n_folds,
        source_ds=source_ds
    )
    
    # Initialize list to store all predictions
    all_best_layer_predictions = []
    all_aggregated_predictions = []
    all_idx = []
    for fold in tqdm(range(n_folds)):
        split = splits[fold]
        
        # Split the data into train/val/test
        train_hidden_states_on_fold, val_hidden_states_on_fold, test_hidden_states_on_fold = split_states_on_idx(hidden_states, split)
        train_labels_on_fold = labels[split['train_indices']]
        val_labels_on_fold = labels[split['val_indices']]
        test_labels_on_fold = labels[split['test_indices']]

        controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            rfm_iters=5,
            batch_size=1,
            n_components=n_components
        )

        try:
            print(f"Loading directions for fold {fold}")
            controller.load(concept=f'{source_ds}_fold_{fold}_out_of_{n_folds}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            print(f"Computing directions for fold {fold}")
            controller.compute_directions(train_hidden_states_on_fold, train_labels_on_fold, val_hidden_states_on_fold, val_labels_on_fold, tuning_metric=tuning_metric)
            controller.save(concept=f'{source_ds}_fold_{fold}_out_of_{n_folds}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_top_k_{n_components}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        # Train on training set, validate on validation set, final evaluation on test set
        val_metrics, test_metrics, _, test_predictions = controller.evaluate_directions(
            train_hidden_states_on_fold, train_labels_on_fold,
            val_hidden_states_on_fold, val_labels_on_fold,
            test_hidden_states_on_fold, test_labels_on_fold,
            n_components=n_components,
            agg_model=control_method,
        )
        
        all_best_layer_predictions.append(test_predictions['best_layer'])
        all_aggregated_predictions.append(test_predictions['aggregation'])
        all_idx.append(torch.from_numpy(split['test_indices']))
        
        # Save individual fold metrics
        results_dir = f'{NEURAL_CONTROLLERS_DIR}/results/halubench_results/{source_ds}'
        os.makedirs(results_dir, exist_ok=True)
        
        out_name = f'{results_dir}/{model_name}_{control_method}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_fold_{fold}_val_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(val_metrics, f)

        out_name = f'{results_dir}/{model_name}_{control_method}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_fold_{fold}_test_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(test_metrics, f)

    # Combine all predictions
    all_best_layer_predictions = torch.cat(all_best_layer_predictions, dim=0)
    all_aggregated_predictions = torch.cat(all_aggregated_predictions, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    
    # Sort predictions according to index order
    sorted_order = torch.argsort(all_idx)
    all_best_layer_predictions = all_best_layer_predictions[sorted_order]
    all_aggregated_predictions = all_aggregated_predictions[sorted_order]
    all_idx = all_idx[sorted_order]

    # Compute and save overall metrics
    best_layer_metrics = compute_overall_metrics(all_best_layer_predictions, labels)
    aggregated_metrics = compute_overall_metrics(all_aggregated_predictions, labels)
    print("\nOverall Metrics:")
    print(f"Best Layer AUC: {best_layer_metrics['auc']:.3f}")
    print(f"Aggregated AUC: {aggregated_metrics['auc']:.3f}")
    print(f"Best Layer F1: {best_layer_metrics['f1']:.3f}")
    print(f"Aggregated F1: {aggregated_metrics['f1']:.3f}")

    # Save overall metrics and predictions
    out_name = f'{results_dir}/{source_ds}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(best_layer_metrics, f)

    out_name = f'{results_dir}/{source_ds}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(aggregated_metrics, f)

    # Save predictions
    predictions_file = f'{results_dir}/{source_ds}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-predictions.pkl'
    with open(predictions_file, 'wb') as f:
        pickle.dump({
            'aggregation': all_aggregated_predictions,
            'best_layer': all_best_layer_predictions,
        }, f)

if __name__ == '__main__':              
    main()