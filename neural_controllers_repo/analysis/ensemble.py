import os
import pickle
import torch

import sys
NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
sys.path.append(NEURAL_CONTROLLERS_DIR)

from direction_utils import compute_prediction_metrics
from quantitative_comparisons.halubench import get_halubench_data
from quantitative_comparisons.fava import get_fava_annotated_data
from quantitative_comparisons.multiclass_halu_eval_wild import get_multiclass_halu_eval_wild_data

results_dir = f'{NEURAL_CONTROLLERS_DIR}/results'

def load_and_bag_predictions_pairs(results_path, ensemble_pairs, model_name, prompt_version='v1', tuning_metric='top_agop_vectors_ols_auc'):
    """
    Load and bag predictions for a provided dataset name and a list of (prediction_type, control_method) pairs.
    Args:
        dataset_name (str): Name of the dataset (e.g., 'RAGTruth', 'pubmedQA', etc.)
        ensemble_pairs (list of (str, str)): List of (prediction_type, control_method) pairs, e.g. [('aggregation', 'linear'), ('best_layer', 'rfm')]
        model_name (str): Model name string (e.g., 'llama_3.3_70b_4bit_it')
        prompt_version (str): Prompt version (default 'v1')
    Returns:
        torch.Tensor: Bagged predictions across the specified pairs
    """
    preds = []
    for prediction_type, control_method in ensemble_pairs:
        pred_path = os.path.join(
            results_path, f"{model_name}_{control_method}_prompt_{prompt_version}_tuning_metric_{tuning_metric}_predictions.pkl"
        )
        with open(pred_path, 'rb') as f:
            pred_dict = pickle.load(f)
            pred = pred_dict[prediction_type]
            if isinstance(pred, list):
                pred = torch.tensor(pred)
            elif isinstance(pred, torch.Tensor):
                pass
            else:
                pred = torch.tensor(pred)
            preds.append(pred)
    if not preds:
        raise FileNotFoundError("No predictions found for the given configuration.")
    bagged_preds = torch.stack(preds).mean(dim=0)
    return bagged_preds

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bag predictions for specified (prediction_type, control_method) pairs.")
    parser.add_argument('--dataset', type=str, default='RAGTruth')
    parser.add_argument('--model_name', type=str, default='llama_3.3_70b_4bit_it')
    parser.add_argument('--prompt_version', type=str, default='v1')
    parser.add_argument('--tuning_metric', type=str, default='top_agop_vectors_ols_auc')
    args = parser.parse_args()

    ensemble_elements = [
        # 'aggregation,linear',
        # 'aggregation,rfm',
        # 'aggregation,logistic',
        'best_layer,linear',
        # 'best_layer,rfm',
        # 'best_layer,logistic',
    ]

    # Parse ensemble_pairs into list of (prediction_type, control_method)
    ensemble_pairs = []
    for pair in ensemble_elements:
        parts = pair.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid ensemble pair: {pair}. Should be in the form prediction_type,control_method")
        ensemble_pairs.append((parts[0], parts[1]))

    if args.dataset.lower() == 'fava':
        results_path = f'{results_dir}/fava_annotated_results'
    elif args.dataset.lower() == 'halu_eval_wild':
        results_path = f'{results_dir}/halu_eval_wild_results'
    else:
        results_path = f'{results_dir}/halubench_results/{args.dataset}'

    bagged_preds = load_and_bag_predictions_pairs(
        results_path=results_path,
        ensemble_pairs=ensemble_pairs,
        model_name=args.model_name,
        prompt_version=args.prompt_version,
        tuning_metric=args.tuning_metric
    )

    # Load ground truth labels
    if args.dataset.lower() == 'fava':
        _, labels = get_fava_annotated_data(args.prompt_version)
    elif args.dataset.lower() == 'halu_eval_wild':
        _, labels = get_multiclass_halu_eval_wild_data(args.prompt_version)
    else:
        _, labels = get_halubench_data(args.dataset, args.prompt_version)
    labels = torch.tensor(labels)
    # Compute and print metrics
    metrics = compute_prediction_metrics(bagged_preds, labels)
    print("\nMetrics on bagged predictions:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")






