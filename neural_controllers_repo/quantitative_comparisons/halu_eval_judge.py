import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

NEURAL_CONTROLLERS_DIR = os.environ.get('NEURAL_CONTROLLERS_DIR', str(Path(__file__).parent.parent))
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results'

from utils import load_model

import torch
import pickle
from direction_utils import compute_prediction_metrics
from tqdm import tqdm
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from abc import ABC, abstractmethod
from utils import load_model
from sklearn.metrics import roc_auc_score
from halu_eval import get_halu_eval_data

class HallucinationJudge(ABC):
    def __init__(self):
        pass
    
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
    
    def evaluate_inputs(self, inputs, labels, prompt_version):
        # First try to load precomputed predictions
        judge_type = self.__class__.__name__.replace('Judge', '').lower()
        judge_model = getattr(self, 'judge_model', None)
        
        predictions, probabilities = load_predictions(judge_type, judge_model, prompt_version)
        
        if predictions is None:
            # Get all judgements at once for efficiency
            predictions, probabilities = self.get_all_predictions(inputs)
            save_predictions(predictions, probabilities, judge_type, judge_model, prompt_version)
        
        # Evaluate metrics for the whole dataset
        labels = torch.tensor(labels)
        metrics = compute_prediction_metrics(probabilities, labels)

        # Calculate AUC
        auc = roc_auc_score(labels.cpu().numpy(), probabilities.cpu().numpy())
        metrics['auc'] = auc
        return metrics

class OpenAIJudge(HallucinationJudge):
    def __init__(self, judge_model):
        super().__init__()
        self.judge_model = judge_model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(min=1, max=1024),
    )
    def get_judgement(self, prompt):
        full_prompt = f"{prompt}"
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

class LlamaJudge(HallucinationJudge):
    def __init__(self, judge_model=None):
        super().__init__()
        self.judge_model = judge_model
        self.model, self.tokenizer = load_model(self.judge_model)
        
    def get_judgement(self, prompt):
        full_prompt = f"{prompt}"
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user', 'content':full_prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters for probability calculation
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,   # No temperature scaling
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]  # [batch, sequence_position, vocab_size]
            
            # Apply softmax the same way the model would during sampling
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities so they sum to 1
            total_prob = yes_prob + no_prob
            no_prob_norm = no_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return (no_prob > yes_prob, no_prob_norm)

class GemmaJudge(HallucinationJudge):
    def __init__(self, judge_model=None):
        super().__init__()
        self.judge_model = judge_model
        self.model, self.tokenizer = load_model(self.judge_model)
        
    def get_judgement(self, prompt):
        full_prompt = f"{prompt}"
        chat = [
            {'role':'user', 'content':full_prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        # We'll use the model's generate method with return_dict_in_generate=True
        # to get the full scores (probabilities) for each possible next token
        with torch.no_grad():
            # Set up generation parameters for probability calculation
            gen_kwargs = {
                "max_new_tokens": 1,  # We only care about the first token
                "do_sample": False,   # No sampling, we just want the probabilities
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,   # No temperature scaling
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]  # [batch, sequence_position, vocab_size]
            
            # Apply softmax the same way the model would during sampling
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities so they sum to 1
            total_prob = yes_prob + no_prob
            no_prob_norm = no_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on these probabilities
            return (no_prob > yes_prob, no_prob_norm)

def save_predictions(predictions, probabilities, judge_type, judge_model, prompt_version):
    """Save all predictions to a file."""
    os.makedirs(f'{RESULTS_DIR}/halu_eval_results', exist_ok=True)
    out_name = f'{RESULTS_DIR}/halu_eval_results/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model, prompt_version):
    """Load predictions from file if they exist."""
    out_name = f'{RESULTS_DIR}/halu_eval_results/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hal_type', type=str, default='general')
    parser.add_argument('--judge_type', type=str, default='llama') #, choices=['openai', 'llama', 'gemma', 'anthropic']
    parser.add_argument('--judge_model', type=str, default='llama_3.3_70b_4bit_it') # choices=['gpt-4o', 'llama_3.3_70b_4bit_it', 'claude-3-opus-20240229']
    parser.add_argument('--prompt_version', type=str, default='v1')
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    inputs, labels = get_halu_eval_data(args.hal_type, args.prompt_version)

    print("="*100)
    print(inputs[0])
    print("="*100)
    print(labels[0])
    print("="*100)
    
    os.makedirs(f'{RESULTS_DIR}/halu_eval_results', exist_ok=True)

    if args.judge_type == 'openai':
        judge = OpenAIJudge(args.judge_model)
    elif args.judge_type == 'llama':
        judge = LlamaJudge(args.judge_model)
    elif args.judge_type == 'gemma':
        judge = GemmaJudge(args.judge_model)

    # Evaluate and get results for the dataset
    metrics = judge.evaluate_inputs(inputs, labels, args.prompt_version)
    
    # Print metrics for the dataset
    print("\nMetrics for the dataset:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Save results for the dataset
    test_out_name = f'{RESULTS_DIR}/halu_eval_results/halu_eval_{args.hal_type}-{args.judge_type}-{args.judge_model}-prompt_{args.prompt_version}-metrics.pkl'
    with open(test_out_name, 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == '__main__':
    main()