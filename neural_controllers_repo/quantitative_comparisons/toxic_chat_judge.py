import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

NEURAL_CONTROLLERS_DIR = os.environ.get('NEURAL_CONTROLLERS_DIR', '.')
RESULTS_DIR = f'{NEURAL_CONTROLLERS_DIR}/results'

from utils import load_model
import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset
import random
from openai import OpenAI
from abc import ABC, abstractmethod
from utils import load_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import direction_utils
from sklearn.metrics import roc_auc_score
from toxic_chat import get_prompt

random.seed(0)

def get_data():
    ds = load_dataset("lmsys/toxic-chat", "toxicchat1123")
    train_inputs = [x['user_input'] for x in ds['train']]
    test_inputs = [x['user_input'] for x in ds['test']]
        
    train_labels = [x['toxicity'] for x in ds['train']]
    test_labels = [x['toxicity'] for x in ds['test']]
    return train_inputs, train_labels, test_inputs, test_labels

class ToxicityJudge(ABC):
    def __init__(self, judge_prompt):
        self.judge_prompt = judge_prompt
    
    @abstractmethod
    def get_judgement(self, prompt):
        pass
    
    def get_all_predictions(self, inputs):
        """Get predictions for all inputs at once."""
        predictions = []
        probabilities = []  # Store probabilities for AUC calculation
        for input_text in tqdm(inputs):
            prompt = self.judge_prompt.format(query=input_text)
            print("prompt:", prompt)
            
            prediction, probability = self.get_judgement(prompt)
            predictions.append(prediction)
            probabilities.append(probability)
            print("prediction:", prediction, "probability:", probability)
        
        return torch.tensor(predictions), torch.tensor(probabilities)
    
    def evaluate_inputs(self, test_inputs, test_labels):
        predictions, probabilities = self.get_all_predictions(test_inputs)
        targets = torch.tensor(test_labels).reshape(-1, 1)
        probabilities = torch.tensor(probabilities).reshape(-1, 1)
        print("targets:", targets.shape, "probabilities:", probabilities.shape)
        metrics = direction_utils.compute_prediction_metrics(probabilities, targets)
        
        # Calculate AUC
        auc = roc_auc_score(targets.numpy(), probabilities.numpy())
        metrics['auc'] = auc
        
        return metrics, predictions, probabilities

class OpenAIJudge(ToxicityJudge):
    def __init__(self, judge_prompt, judge_model):
        super().__init__(judge_prompt)
        self.judge_model = judge_model
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    def get_judgement(self, prompt):
        response = self.client.chat.completions.create(
            model=self.judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant who follows instructions exactly."},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=20,
            max_tokens=5,
            temperature=0
        )
        
        content = response.choices[0].message.content.lower()
        prediction = 1 if 'yes' in content else 0
        
        # Try to extract probabilities from logprobs
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
            yes_prob_norm = yes_prob_value / total if total > 0 else 0.5
            return (yes_prob > no_prob, yes_prob_norm)
        elif yes_prob is not None:
            return (1, 1.0)
        elif no_prob is not None:
            return (0, 0.0)
        else:
            # Fallback to content
            return (prediction, float(prediction))

class GemmaJudge(ToxicityJudge):
    def __init__(self, judge_prompt, judge_model='gemma_2_9b_it'):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model(judge_model)
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'user',
             'content':prompt
            }
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        with torch.no_grad():
            # Set up generation parameters to get probabilities
            gen_kwargs = {
                "max_new_tokens": 1,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]
            
            # Apply softmax to get probabilities
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities
            total_prob = yes_prob + no_prob
            yes_prob_norm = yes_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on probabilities
            return (yes_prob > no_prob, yes_prob_norm)
    
    
class LlamaJudge(ToxicityJudge):
    def __init__(self, judge_prompt, judge_model='llama_3_8b_it'):
        super().__init__(judge_prompt)
        self.model, self.tokenizer = load_model(judge_model)
        
    def get_judgement(self, prompt):
        chat = [
            {'role':'system', 'content':'You are a helpful assistant who follows instructions exactly.'},
            {'role':'user','content':prompt}
        ]
        wrapped_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(wrapped_prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        
        # Get token IDs for "yes" and "no"
        yes_token_id = self.tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_token_id = self.tokenizer.encode('No', add_special_tokens=False)[0]
        
        with torch.no_grad():
            # Set up generation parameters to get probabilities
            gen_kwargs = {
                "max_new_tokens": 1,
                "do_sample": False,
                "return_dict_in_generate": True,
                "output_scores": True,
                "temperature": 0.0,
            }
            
            # Generate a token and get the scores
            outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Extract the scores (logits) for the first generated token
            token_scores = outputs.scores[0][0]
            
            # Apply softmax to get probabilities
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)
            
            # Get the probabilities for "yes" and "no" tokens
            yes_prob = token_probs[yes_token_id].item()
            no_prob = token_probs[no_token_id].item()
            
            # Normalize the probabilities
            total_prob = yes_prob + no_prob
            yes_prob_norm = yes_prob / total_prob if total_prob > 0 else 0.5
            
            # Make the decision based on probabilities
            return (yes_prob > no_prob, yes_prob_norm)

class T5ToxicJudge(ToxicityJudge):
    def __init__(self, judge_prompt):
        super().__init__(judge_prompt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("lmsys/toxicchat-t5-large-v1.0").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        
    def get_judgement(self, prompt, position=0):
        prefix = "ToxicChat: "
        inputs = self.tokenizer.encode(prefix + prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_new_tokens=5, 
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Print generated tokens
            generated_ids = outputs.sequences[0].tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print("Generated text:", generated_text)
            
            # Print individual tokens
            for i, token_id in enumerate(generated_ids):
                token = self.tokenizer.decode([token_id])
                print(f"Token {i}: {token} (ID: {token_id})")

            # exit()
            
            # Calculate probabilities for the second token
            token_scores = outputs.scores[position][0]  # Changed from [0][0] to [1][0] to get second token
            token_probs = torch.nn.functional.softmax(token_scores, dim=-1)

            # print("outputs.scores", len(outputs.scores))
            # print("outputs.scores[position]", len(outputs.scores[position]))
            # print("outputs.scores[position][0]", outputs.scores[position][0].shape)
            
            # Get tokens for positive and negative
            pos_tokens = [self.tokenizer.encode("positive", add_special_tokens=False)[0]]
            neg_tokens = [self.tokenizer.encode("negative", add_special_tokens=False)[0]]

            # print("pos_tokens:", pos_tokens)
            # print("neg_tokens:", neg_tokens)

            # print("token_probs:", token_probs.shape)
            # print("token_scores:", token_scores.shape)

            # Sum probabilities for related tokens
            pos_prob = sum(token_probs[t].item() for t in pos_tokens)
            neg_prob = sum(token_probs[t].item() for t in neg_tokens)
            
            print("pos_prob:", pos_prob)
            print("neg_prob:", neg_prob)

            # exit()
            # Normalize
            total = pos_prob + neg_prob
            pos_prob_norm = pos_prob / total if total > 0 else 0.5
            
            return (pos_prob > neg_prob, pos_prob_norm)

def save_predictions(predictions, probabilities, judge_type, judge_model, prompt_version):
    """Save all predictions to a file."""
    os.makedirs(f'{RESULTS_DIR}/toxic_chat_results', exist_ok=True)
    out_name = f'{RESULTS_DIR}/toxic_chat_results/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump({'predictions': predictions, 'probabilities': probabilities}, f)

def load_predictions(judge_type, judge_model, prompt_version):
    """Load predictions from file if they exist."""
    out_name = f'{RESULTS_DIR}/toxic_chat_results/{judge_type}_{judge_model}_prompt_{prompt_version}_all_predictions.pkl'
    if os.path.exists(out_name):
        with open(out_name, 'rb') as f:
            data = pickle.load(f)
            return data['predictions'], data['probabilities']
    return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--judge_type', type=str, default='toxicchat') # choices=['openai', 'llama', 'toxicchat', 'gemma'],
    parser.add_argument('--judge_model', type=str, default=None) # , choices=['gpt-4o', 'llama_3.3_70b_4bit_it']
    parser.add_argument('--prompt_version', type=str, default='v1')
    args = parser.parse_args()
    
    for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")
    
    _, _, inputs, labels = get_data()
    
    if args.judge_type == 'openai':
        judge_prompt = get_prompt(args.prompt_version)
        judge = OpenAIJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'llama':
        judge_prompt = get_prompt(args.prompt_version)
        judge = LlamaJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'gemma':
        judge_prompt = get_prompt(args.prompt_version)
        judge = GemmaJudge(judge_prompt, args.judge_model)
    elif args.judge_type == 'toxicchat': 
        judge_prompt='{query}'
        judge = T5ToxicJudge(judge_prompt)
    
    # Try to load predictions or generate new ones
    all_predictions, all_probabilities = load_predictions(args.judge_type, args.judge_model, args.prompt_version)
    if all_predictions is None:
        metrics, all_predictions, all_probabilities = judge.evaluate_inputs(inputs, labels)
        save_predictions(all_predictions, all_probabilities, args.judge_type, args.judge_model, args.prompt_version)
    else:
        # Calculate metrics from loaded predictions
        targets = torch.tensor(labels).reshape(-1, 1)
        probabilities = torch.tensor(all_probabilities).reshape(-1, 1)
        metrics = direction_utils.compute_prediction_metrics(probabilities, targets)
        auc = roc_auc_score(targets.numpy(), probabilities.numpy())
        metrics['auc'] = auc
    
    for k, v in metrics.items():
        print(f"{k:<20} : {v}")
    
    os.makedirs(f'{RESULTS_DIR}/toxic_chat_results', exist_ok=True)
    out_name = f'{RESULTS_DIR}/toxic_chat_results/toxic_chat-{args.judge_type}-{args.judge_model}-prompt_{args.prompt_version}-metrics.pkl'
    with open(out_name, 'wb') as f:
        pickle.dump(metrics, f)

if __name__ == '__main__':
    main()
    
    