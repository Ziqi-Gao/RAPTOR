import os
import pickle
import pandas as pd
import re
from glob import glob

import os

NEURAL_CONTROLLERS_DIR = os.environ['NEURAL_CONTROLLERS_DIR']
RESULTS_DIR = os.path.join(NEURAL_CONTROLLERS_DIR, 'results')

LABELS = {
    'None' : 'ToxicChat-T5-Large',
    'gpt-4o' : 'GPT-4o',
    'llama_3.1_70b_4bit_it' : 'Llama-3.1-70b-4-bit',
    'llama_3.3_70b_4bit_it' : 'Llama-3.3-70b-4-bit',
    'llama_3_8b_it' : 'Llama-3.1-8b',
    'logistic' : 'Logistic',
    'linear' : 'Lin. Reg.',
    'rfm' : 'RFM',
}

DATASETS = {
    'toxic_chat' : 'ToxicChat',
    'fava' : 'FAVA',
    'halu_eval_general' : 'HaluEval (General)',
    'halu_eval_wild' : 'HaluEval (Wild)',
    'pubmedQA' : 'PubMedQA',
    'RAGTruth' : 'RAGTruth',
}

JUDGE_METHODS = set(['gpt-4o', 'llama_3_8b_it', 'None', 'llama_3.1_70b_4bit_it', 'llama_3.3_70b_4bit_it'])

# Recursively find all *_metrics.pkl files in results
metrics_files = [y for x in os.walk(RESULTS_DIR) for y in glob(os.path.join(x[0], '*metrics.pkl'))]
print(f"Found {len(metrics_files)} metrics files in {RESULTS_DIR}")
rows = []

# Regex to parse filenames for new formats
probe_re = re.compile(r'(?P<dataset>[^-]+)-(?P<model>[^-]+)-(?P<method>[^-]+)-prompt_(?P<prompt_version>v\d+)-tuning_metric_(?P<tuning_metric>[^-]+)-top_k_(?P<n_components>\d+)-(?P<agg_type>aggregated|best_layer)_metrics\.pkl$')
judge_re = re.compile(r'^(?P<dataset>.+?)-(?P<judge_type>.+?)-(?P<judge_model>.+?)-prompt_(?P<prompt_version>v\d+)-metrics\.pkl$')

for file in metrics_files:
    try:
        with open(file, 'rb') as f:
            metrics = pickle.load(f)
    except Exception as e:
        print(f"Could not load {file}: {e}")
        continue
    fname = os.path.basename(file)
    # Try to match probe or judge pattern
    m_probe = probe_re.match(fname)
    m_judge = judge_re.match(fname)
    if m_probe:
        row = {
            'file': file,
            'dataset': m_probe.group('dataset'),
            'model': m_probe.group('model'),
            'method': m_probe.group('method'),
            'prompt_version': m_probe.group('prompt_version'),
            'tuning_metric': m_probe.group('tuning_metric'),
            'n_components': m_probe.group('n_components'),
            'aggregation': m_probe.group('agg_type'),
        }
    elif m_judge:
        print(f"Found judge file: {fname}")
        row = {
            'file': file,
            'dataset': m_judge.group('dataset'),
            'judge_type': m_judge.group('judge_type'),
            'judge_model': m_judge.group('judge_model'),
            'prompt_version': m_judge.group('prompt_version'),
            'aggregation': '',
        }
    else:
        continue
    # Flatten metrics dict for DataFrame
    for k, v in metrics.items():
        row[k] = v
    rows.append(row)

# Create DataFrame
if rows:
    df = pd.DataFrame(rows)

    # Reorder columns
    cols = [
        'dataset', 'model', 'method', 'judge_type', 'judge_model',
        'prompt_version', 'tuning_metric', 'n_components', 'aggregation', 'auc', 'file'
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    
    out_csv = os.path.join(RESULTS_DIR, 'all_results_table.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved table to {out_csv}")

    # Prepare for LaTeX output: only dataset, method, auc
    df_latex = df.copy()
    def format_method(row):
        if pd.notnull(row.get('judge_model', None)) and row.get('judge_model', '') != '':
            judge_label = LABELS.get(row['judge_model'], row['judge_model'])
            return judge_label
        method_label = LABELS.get(row.get('method', ''), row.get('method', ''))
        model = row.get('model', '')
        model_label = LABELS.get(model, model) if model else ''
        agg = row.get('aggregation', '')
        if model_label and agg:
            return f"{method_label} ({model_label}, {agg})"
        elif model_label:
            return f"{method_label} ({model_label})"
        elif agg:
            return f"{method_label} ({agg})"
        return method_label
    df_latex['method'] = df_latex.apply(format_method, axis=1)
    
    latex_cols = [c for c in ['dataset', 'method', 'auc'] if c in df_latex.columns]
    df_latex = df_latex[latex_cols]

    # Pivot so each dataset is a column, each row is a method, auc is the value
    df_pivot = df_latex.pivot_table(index='method', columns='dataset', values='auc')

    # Truncate to three decimals for LaTeX output
    def truncate(x):
        if pd.isna(x):
            return '-'
        if isinstance(x, float):
            return f"{int(x * 1000) / 1000:.3f}"
        return x
    df_pivot_fmt = df_pivot.applymap(truncate)

    # Add \textbf{} to max value in each column
    for col in df_pivot.columns:
        col_vals = df_pivot[col]
        valid_mask = col_vals.apply(lambda x: isinstance(x, float) and not pd.isna(x))
        if valid_mask.any():
            max_val = col_vals[valid_mask].max()
            max_str = f"{int(max_val * 1000) / 1000:.3f}"
            df_pivot_fmt[col] = df_pivot_fmt[col].apply(lambda x: f"\\textbf{{{x}}}" if x == max_str else x)

    # Map dataset names to labels
    df_pivot_fmt.columns = [DATASETS.get(ds, ds) for ds in df_pivot.columns]
    new_column_order = ['FAVA', 'HaluEval (General)', 'HaluEval (Wild)', 'PubMedQA', 'RAGTruth', 'ToxicChat']

    # Reorder the columns
    df_pivot_fmt = df_pivot_fmt[new_column_order]

    index_list = list(df_pivot_fmt.index)
    def sort_key(item):
        # Priority for model version
        if "3.3" in item:
            version_priority = 0  # 3.3 comes first
        elif "3.1-70b" in item:
            version_priority = 1  # 3.1-70b comes second
        elif "3.1-8b" in item:
            version_priority = 2  # 3.1-8b comes last
        else:
            version_priority = 3  # Any other models
        
        # Secondary sort by aggregation method
        if "aggregated" in item:
            agg_priority = 0  # aggregated comes before best_layer
        else:
            agg_priority = 1
        
        return (version_priority, agg_priority)

    rfm_methods = [x for x in index_list if x.startswith('RFM')]
    rfm_methods = sorted(rfm_methods, key=sort_key)

    linear_methods = [x for x in index_list if x.startswith('Lin. Reg.')]
    linear_methods = sorted(linear_methods, key=sort_key)

    logistic_methods = [x for x in index_list if x.startswith('Logistic')]
    logistic_methods = sorted(logistic_methods, key=sort_key)

    judge_models = [x for x in index_list if x not in rfm_methods and x not in linear_methods and x not in logistic_methods]
    judge_models = sorted(judge_models, key=sort_key)
    df_pivot_fmt = df_pivot_fmt.loc[rfm_methods + linear_methods + logistic_methods + judge_models]

    print("\nLaTeX table (methods as rows, datasets as columns, auc as value):")
    print(df_pivot_fmt.to_latex(index=True, na_rep='-'))

    # Create a table showing max between best_layer and aggregated for RFM
    rfm_rows = [x for x in df_pivot.index if x.startswith('RFM')]
    models = set()
    for row in rfm_rows:
        if '(' in row:
            model = row.split('(')[1].split(',')[0].strip()
            models.add(model)
    
    max_rfm_results = {}
    for model in models:
        model_rows = [x for x in rfm_rows if model in x]
        if len(model_rows) > 0:
            max_vals = df_pivot.loc[model_rows].max()
            max_rfm_results[f"RFM ({model})"] = max_vals
    
    df_max_rfm = pd.DataFrame(max_rfm_results).T
    
    # Format the max RFM table
    def truncate_and_bold_max(x):
        if pd.isna(x):
            return '-'
        if isinstance(x, float):
            return f"{int(x * 1000) / 1000:.3f}"
        return x
    
    df_max_rfm_fmt = df_max_rfm.applymap(truncate_and_bold_max)
    
    # Bold the maximum value in each column
    for col in df_max_rfm.columns:
        col_vals = df_max_rfm[col]
        valid_mask = col_vals.apply(lambda x: isinstance(x, float) and not pd.isna(x))
        if valid_mask.any():
            max_val = col_vals[valid_mask].max()
            max_str = f"{int(max_val * 1000) / 1000:.3f}"
            df_max_rfm_fmt[col] = df_max_rfm_fmt[col].apply(lambda x: f"\\textbf{{{x}}}" if x == max_str else x)
    
    # Map dataset names and reorder columns
    df_max_rfm_fmt.columns = [DATASETS.get(ds, ds) for ds in df_max_rfm.columns]
    df_max_rfm_fmt = df_max_rfm_fmt[new_column_order]
    df_max_rfm_fmt = df_max_rfm_fmt.loc[['RFM (Llama-3.3-70b-4-bit)', 'RFM (Llama-3.1-70b-4-bit)', 'RFM (Llama-3.1-8b)']]
    
    print("\nLaTeX table (Max RFM results across aggregation methods):")
    print(df_max_rfm_fmt.to_latex(index=True, na_rep='-'))

    # Calculate averages across datasets for each method
    print("\nAverage performance across datasets:")
    
    # For RFM methods, we already have the max between aggregated/best_layer in df_max_rfm
    rfm_averages = df_max_rfm.mean(axis=1)
    
    # For other methods (Linear, Logistic), calculate max between aggregated/best_layer
    other_methods = {}
    for method in ['Lin. Reg.', 'Logistic']:
        for model in models:
            method_rows = [x for x in df_pivot.index if x.startswith(method) and model in x]
            if len(method_rows) > 0:
                max_vals = df_pivot.loc[method_rows].max()
                avg_performance = max_vals.mean()
                other_methods[f"{method} ({model})"] = avg_performance
    
    # For judge models (which don't have aggregated/best_layer variants)
    judge_averages = {}
    for judge in JUDGE_METHODS:
        if judge in LABELS:
            judge_row = LABELS[judge]
            if judge_row in df_pivot.index:
                avg = df_pivot.loc[judge_row].mean()
                judge_averages[judge_row] = avg

    # Combine all averages and sort them
    all_averages = {**dict(zip(rfm_averages.index, rfm_averages)), **other_methods, **judge_averages}
    sorted_averages = dict(sorted(all_averages.items(), key=lambda x: x[1], reverse=True))
    
    print("\nAverages across all datasets (sorted by performance):")
    for method, avg in sorted_averages.items():
        print(f"{method}: {avg:.3f}")

else:
    print("No metrics files found.")
