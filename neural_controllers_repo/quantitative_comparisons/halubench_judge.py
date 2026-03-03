import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import load_model

import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score
import torch
from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from halubench import get_halubench_data

NEURAL_CONTROLLERS_DIR = os.environ.get('NEURAL_CONTROLLERS_DIR', str(Path(__file__).parent.parent))
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results'

def compute_metrics(predictions, labels, threshold=0.5):
    """
    Compute metrics for predictions.
    predictions: torch.Tensor with shape (n_samples,)
    labels: torch.Tensor with shape (n_samples,)
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    metrics = {
        'auc': roc_auc_score(labels, predictions),
        'f1': f1_score(labels, predictions > threshold)
    }
    
    return metrics

def save_predictions(predictions, probabilities, judge_type, judge_model, source_ds, prompt_version):
    """Save all predictions to a file."""
    os.makedirs(f'{RESULTS_DIR}/halubench_results/{source_ds}', exist_ok=True)
    out_name = f'{RESULTS_DIR}/halubench_results/{source_ds}/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model, source_ds, prompt_version):
    """Load predictions from file if they exist."""
    out_name = f'{RESULTS_DIR}/halubench_results/{source_ds}/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None

class HaluBenchJudge(ABC):
    def __init__(self, judge_prompt=None, judge_model=None):
        self.judge_prompt = judge_prompt if judge_prompt else '{statement}'
        self.judge_model = judge_model
        
    @abstractmethod
    def get_judgement(self, prompt):
        """Get a single judgement with probability for a prompt."""
        pass

    def get_all_predictions(self, inputs):
        """Get predictions for all inputs at once."""
        predictions = []
        probabilities = []  # Store probabilities for AUC calculation
        
        for input_text in tqdm(inputs):
            prediction, probability = self.get_judgement(input_text)
            predictions.append(prediction)
            probabilities.append(probability)
        
        return torch.tensor(predictions), torch.tensor(probabilities)
            
    def evaluate_inputs(self, source_ds, prompt_version):
        self.source_ds = source_ds
        
        # First try to load precomputed predictions
        judge_type = self.__class__.__name__.replace('HaluBenchJudge', '').lower()
        
        all_predictions, all_probabilities = load_predictions(judge_type, self.judge_model, source_ds, prompt_version)
        
        if all_predictions is None:
            # Load the dataset
            inputs, labels = get_halubench_data(source_ds, prompt_version=prompt_version)

            print("="*100)
            print(inputs[0])
            print("="*100)
            print("Label:",labels[0])
            print("="*100)
            
            # Get all judgements at once for efficiency
            all_predictions, all_probabilities = self.get_all_predictions(inputs)
            save_predictions(all_predictions, all_probabilities, judge_type, self.judge_model, source_ds, prompt_version)
            
            # Calculate metrics
            metrics = compute_metrics(all_probabilities, labels)
        else:
            # Load the dataset just to get labels
            _, labels = get_halubench_data(source_ds)
            
            # Calculate metrics using loaded predictions
            metrics = compute_metrics(all_probabilities, labels)
        
        # Save metrics separately
        results_dir = f'{RESULTS_DIR}/halubench_results/{source_ds}'
        os.makedirs(results_dir, exist_ok=True)
        
        out_name = f'{results_dir}/{judge_type}_{self.judge_model}_metrics.pkl'
        with open(out_name, 'wb') as f:
            pickle.dump(metrics, f)
            
        return metrics, all_predictions

class LlamaHaluBenchJudge(HaluBenchJudge):
    def __init__(self, judge_model='llama_3.3_70b_4bit_it'):
        super().__init__(judge_model=judge_model)
        self.model, self.tokenizer = load_model(model=judge_model)
        
    def get_judgement(self, prompt):
        """Get a single judgement with probability for a prompt."""
        chat = [
            {'role': 'system', 'content': 'You are a helpful assistant who follows instructions exactly.'},
            {'role': 'user', 'content': prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 1,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,
            }
            
            outputs = self.model.generate(**inputs, **gen_kwargs)
            token_scores = outputs.scores[0][0]
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            total_prob = yes_prob + no_prob
            no_prob_norm = no_prob / total_prob if total_prob > 0 else 0.5
            
            return (no_prob > yes_prob, no_prob_norm)

class OpenAIHaluBenchJudge(HaluBenchJudge):
    def __init__(self, judge_model=None):
        super().__init__(judge_model=judge_model)
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        _, self.tokenizer = load_model('llama_3.3_70b_4bit_it')  # Just for tokenizing inputs
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def get_judgement(self, prompt):
        full_prompt = self.judge_prompt.format(statement=prompt)
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": full_prompt}
            ],
            logprobs=True,
            top_logprobs=20,
            max_tokens=5,
            temperature=0
        )
        
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        yes_prob = None
        no_prob = None
        
        for logprob in logprobs:
            token = logprob.token
            if token == 'Yes':
                yes_prob = logprob.logprob
            elif token == 'No':
                no_prob = logprob.logprob
        
        # If we didn't find exact "Yes"/"No", look for close matches
        if yes_prob is None or no_prob is None:
            for logprob in logprobs:
                token = logprob.token
                if yes_prob is None and ('yes' in token.lower() or 'y' == token.lower()):
                    yes_prob = logprob.logprob
                elif no_prob is None and ('no' in token.lower() or 'n' == token.lower()):
                    no_prob = logprob.logprob
        
        if yes_prob is not None and no_prob is not None:
            # Convert from log probabilities to probabilities
            yes_prob_value = torch.exp(torch.tensor(yes_prob)).item()
            no_prob_value = torch.exp(torch.tensor(no_prob)).item()
            # Normalize probabilities
            total = yes_prob_value + no_prob_value
            no_prob_norm = no_prob_value / total if total > 0 else 0.5
            return (no_prob > yes_prob, no_prob_norm)
        elif yes_prob is not None:
            return (0, 0.0)
        elif no_prob is not None:
            return (1, 1.0)
        else:
            # Fallback to content
            is_no = 'n' in response.choices[0].message.content.lower()
            return (is_no, is_no)

class AnthropicHaluBenchJudge(HaluBenchJudge):
    def __init__(self, judge_model=None):
        super().__init__(judge_model=judge_model)
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        _, self.tokenizer = load_model('llama_3.3_70b_4bit_it')  # Just for tokenizing inputs
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def get_judgement(self, prompt):
        full_prompt = self.judge_prompt.format(statement=prompt)
        response = self.client.messages.create(
            model=self.judge_model,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=5,
            temperature=0,
            logprobs=True,
            top_logprobs=20
        )
        
        # Extract logprobs from the response
        logprobs = response.content[0].logprobs[0].tokens[0].top_logprobs
        yes_prob = None
        no_prob = None
        
        # Find logprobs for Yes/No tokens
        for token_logprob in logprobs:
            token = token_logprob.token
            if token == 'Yes':
                yes_prob = token_logprob.logprob
            elif token == 'No':
                no_prob = token_logprob.logprob
        
        # If we didn't find exact "Yes"/"No", look for close matches
        if yes_prob is None or no_prob is None:
            for token_logprob in logprobs:
                token = token_logprob.token
                if yes_prob is None and ('yes' in token.lower() or 'y' == token.lower()):
                    yes_prob = token_logprob.logprob
                elif no_prob is None and ('no' in token.lower() or 'n' == token.lower()):
                    no_prob = token_logprob.logprob
        
        if yes_prob is not None and no_prob is not None:
            # Convert from log probabilities to probabilities
            yes_prob_value = torch.exp(torch.tensor(yes_prob)).item()
            no_prob_value = torch.exp(torch.tensor(no_prob)).item()
            # Normalize probabilities
            total = yes_prob_value + no_prob_value
            no_prob_norm = no_prob_value / total if total > 0 else 0.5
            return (no_prob > yes_prob, no_prob_norm)
        elif yes_prob is not None:
            return (0, 0.0)  # Not a hallucination
        elif no_prob is not None:
            return (1, 1.0)  # Hallucination
        else:
            # Fallback to content
            content = response.content[0].text.lower()
            is_no = 'n' in content
            return (is_no, float(is_no))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_ds', type=str, default='FinanceBench')
    parser.add_argument('--judge_type', type=str, default='openai') #, choices=['openai', 'llama', 'gemma', 'anthropic']
    parser.add_argument('--judge_model', type=str, default='gpt-4o') #llama_3.3_70b_4bit_it #gpt-4o
    parser.add_argument('--prompt_version', type=str, default='v1')
    args = parser.parse_args()
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    judge_model = args.judge_model
    source_ds = args.source_ds
    judge_type = args.judge_type
    prompt_version = args.prompt_version

    if judge_type == 'llama':
        judge = LlamaHaluBenchJudge(judge_model=judge_model)
    elif judge_type == 'openai':
        judge = OpenAIHaluBenchJudge(judge_model=judge_model)
    elif judge_type == 'anthropic':
        judge = AnthropicHaluBenchJudge(judge_model=judge_model)
    else:
        raise ValueError(f"Invalid judge type: {judge_type}")
    
    metrics, predictions = judge.evaluate_inputs(source_ds, prompt_version)
    # Save metrics to file
    results_dir = f'{RESULTS_DIR}/halubench_results/{source_ds}'
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = f'{results_dir}/{source_ds}-{judge_type}-{judge_model}-prompt_{prompt_version}-metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    print("\nMetrics:")
    print(f"AUC: {metrics['auc']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")

if __name__ == '__main__':              
    main()