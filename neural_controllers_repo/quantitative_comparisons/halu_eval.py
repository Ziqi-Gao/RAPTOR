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
import random
import json
import torch
from sklearn.metrics import roc_auc_score, f1_score

random.seed(0)

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results/halu_eval_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def read_hallucination_prompts_from_lines(lines):
    dicts = []
    for line in lines:
        x = json.loads(line)
        dicts.append(x)
    return dicts

def get_halu_eval_data(hal_type, prompt_version='v1'):
    if prompt_version == 'v1':
        qa_template = "Is the ANSWER to the following QUESTION factual? State yes or no.\n\n"
        qa_template += 'QUESTION: {question}\n\nANSWER: {answer}'

        general_template = "Is the RESPONSE to the following QUERY factual? State yes or no.\n\n"
        general_template += 'QUERY: {query}\n\nRESPONSE: {response}'

    if hal_type=='qa':
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/qa_data.txt'
        with open(data_path, 'r') as f:
            lines = f.readlines()
            raw_prompts = read_hallucination_prompts_from_lines(lines)

        # Generate training data
        inputs = []
        labels = []
        for prompt in raw_prompts:
            x_true = qa_template.format(question=prompt['question'], answer=prompt['right_answer'])
            x_false = qa_template.format(question=prompt['question'], answer=prompt['hallucinated_answer'])
            inputs.append(x_true)
            inputs.append(x_false)
            labels += [0,1]
            
    elif hal_type=='general':
        # Get general data for evaluation
        data_path = f'{NEURAL_CONTROLLERS_DIR}/data/hallucinations/halu_eval/general_data.txt'
        with open(data_path, 'r') as f:
            lines = f.readlines()
            eval_prompts = read_hallucination_prompts_from_lines(lines)
            
        inputs = []
        labels = []
        for prompt in eval_prompts:
            x = general_template.format(query=prompt['user_query'], response=prompt['chatgpt_response'])
            inputs.append(x)
            labels.append(int(prompt['hallucination'].lower().strip() == 'yes'))

    return inputs, np.array(labels)

def get_cross_val_splits(n_total, n_folds=5, hal_type=''):
    """
    Generate k-fold cross validation splits with train/validation splits for each fold.
    n_total: total number of samples
    n_folds: number of folds for cross validation
    hal_type: hallucination type for file path
    """
    out_name = f'{RESULTS_DIR}/{hal_type}_cv_splits_nfolds_{n_folds}_ntotal_{n_total}.pkl'
    try:
        with open(out_name, 'rb') as f:
            splits = pickle.load(f)
            return splits
    except:
        pass

    splits = []
    indices = np.arange(n_total)
    fold_size = n_total // n_folds
    shuffled_indices = np.random.permutation(indices)
    for fold in range(n_folds):
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < n_folds - 1 else n_total
        test_indices = shuffled_indices[start_idx:end_idx]
        remaining_indices = np.array([i for i in shuffled_indices if i not in test_indices])
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
    train_inputs, val_inputs, test_inputs = {}, {}, {}
    for layer_idx, layer_states in inputs.items():
        train_inputs[layer_idx] = layer_states[split['train_indices']]
        val_inputs[layer_idx] = layer_states[split['val_indices']]
        test_inputs[layer_idx] = layer_states[split['test_indices']]
    return train_inputs, val_inputs, test_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_method', type=str, default='rfm')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--n_components', type=int, default=3)
    parser.add_argument('--hal_type', type=str, default='general')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    control_method = args.control_method
    model_name = args.model_name
    n_components = args.n_components
    hal_type = args.hal_type
    prompt_version = args.prompt_version
    n_folds = args.n_folds
    tuning_metric = args.tuning_metric

    if control_method not in ['rfm']:
        n_components=1
        tuning_metric = 'auc'
    
    print("Num components:", n_components)
    
    language_model, tokenizer = load_model(model=model_name)
    
    controller = NeuralController(
            language_model,
            tokenizer,
            control_method=control_method,
            rfm_iters=5,
            batch_size=1,
            n_components=n_components
    )
    unformatted_inputs, labels = get_halu_eval_data(hal_type, prompt_version)
    inputs = []
    for unformatted_input in unformatted_inputs:
        chat = [{
            'role': 'user',
            'content': unformatted_input
        }]
        x = tokenizer.apply_chat_template(chat, tokenize=False)
        inputs.append(x)

    print("="*100)
    print(inputs[0])
    print("="*100)
    print(labels[0])
    print("="*100)
    
    hidden_states_path = os.path.join(f'{NEURAL_CONTROLLERS_DIR}', f'hidden_states', 
                                      f'halu_eval_{hal_type}_{model_name}_prompt_{prompt_version}.pth')
    if os.path.exists(hidden_states_path):
        with open(hidden_states_path, 'rb') as f:
            hidden_states = pickle.load(f)
    else:
        from direction_utils import get_hidden_states
        hidden_states = get_hidden_states(inputs, language_model, tokenizer, 
                                      controller.hidden_layers, 
                                      controller.hyperparams['forward_batch_size'])
    
        with open(hidden_states_path, 'wb') as f:
            pickle.dump(hidden_states, f)

    # Cross-validation splits
    splits = get_cross_val_splits(n_total=len(inputs), n_folds=n_folds, hal_type=hal_type)
    
    all_best_layer_predictions = []
    all_aggregated_predictions = []
    all_idx = []
    for fold in tqdm(range(n_folds)):
        split = splits[fold]
        print(f"Fold {fold+1} of {n_folds}")
        train_hidden_states_on_fold, val_hidden_states_on_fold, test_hidden_states_on_fold = split_states_on_idx(hidden_states, split)
        train_labels_on_fold = labels[split['train_indices']]
        val_labels_on_fold = labels[split['val_indices']]
        test_labels_on_fold = labels[split['test_indices']]

        try:
            print(f"Loading directions")
            controller.load(concept=f'halu_eval_{hal_type}_{model_name}_prompt_{prompt_version}_fold_{fold}_out_of_{n_folds}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')
        except:
            print(f"Loading failed, computing directions")
            controller.compute_directions(train_hidden_states_on_fold, train_labels_on_fold, 
                                          val_hidden_states_on_fold, val_labels_on_fold,
                                          tuning_metric=tuning_metric)
            controller.save(concept=f'halu_eval_{hal_type}_{model_name}_prompt_{prompt_version}_fold_{fold}_out_of_{n_folds}', model_name=model_name, path=f'{NEURAL_CONTROLLERS_DIR}/directions/')

        print("Evaluating directions")
        _, _, _, test_predictions = controller.evaluate_directions(
            train_hidden_states_on_fold, train_labels_on_fold,
            val_hidden_states_on_fold, val_labels_on_fold,
            test_hidden_states_on_fold, test_labels_on_fold,
            n_components=n_components,
        )
        all_best_layer_predictions.append(torch.tensor(test_predictions['best_layer']))
        all_aggregated_predictions.append(torch.tensor(test_predictions['aggregation']))
        all_idx.append(torch.from_numpy(split['test_indices']))
        print("Done evaluating directions")

    # Aggregate predictions across folds
    all_best_layer_predictions = torch.cat(all_best_layer_predictions, dim=0)
    all_aggregated_predictions = torch.cat(all_aggregated_predictions, dim=0)
    all_idx = torch.cat(all_idx, dim=0)
    sorted_order = torch.argsort(all_idx)

    all_best_layer_predictions = all_best_layer_predictions[sorted_order]
    all_aggregated_predictions = all_aggregated_predictions[sorted_order]
    all_idx = all_idx[sorted_order]

    def compute_overall_metrics(predictions, labels, threshold=0.5):
        if isinstance(labels, torch.Tensor):
            labels_ = labels.cpu().numpy()
        else:
            labels_ = labels
        if isinstance(predictions, torch.Tensor):
            predictions_ = predictions.cpu().numpy()
        else:
            predictions_ = predictions
        overall_auc = roc_auc_score(labels_, predictions_)
        overall_f1 = f1_score(labels_, predictions_ > threshold)
        metrics = {
            'auc': overall_auc,
            'f1': overall_f1,
        }
        return metrics

    best_layer_metrics = compute_overall_metrics(all_best_layer_predictions, labels)
    aggregated_metrics = compute_overall_metrics(all_aggregated_predictions, labels)
    print("\nOverall Metrics:")
    print(f"Best Layer AUC: {best_layer_metrics['auc']:.3f}")
    print(f"Aggregated AUC: {aggregated_metrics['auc']:.3f}")
    print(f"Best Layer F1: {best_layer_metrics['f1']:.3f}")
    print(f"Aggregated F1: {aggregated_metrics['f1']:.3f}")

    # Save overall metrics and predictions
    out_name = f'{RESULTS_DIR}/halu_eval_{hal_type}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-best_layer_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(best_layer_metrics, f)

    out_name = f'{RESULTS_DIR}/halu_eval_{hal_type}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-aggregated_metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(aggregated_metrics, f)

    predictions_file = f'{RESULTS_DIR}/halu_eval_{hal_type}-{model_name}-{control_method}-prompt_{prompt_version}-tuning_metric_{tuning_metric}-top_k_{n_components}-predictions.pkl'
    with open(predictions_file, 'wb') as f:
        pickle.dump({
            'aggregation': all_aggregated_predictions,
            'best_layer': all_best_layer_predictions,
        }, f)

if __name__ == '__main__':              
    main()