import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import json
from collections import defaultdict
import random
import wandb
import sys
import numpy as np
from utils.data_utils import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train a prompt retriever model')
    parser.add_argument('--total_samples', type=int, default=10000,
                        help='Total number of samples for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='Number of training epochs')
    parser.add_argument('--model_name', type=str, 
                        default="princeton-nlp/sup-simcse-roberta-base",
                        help='Path to the pretrained model')
    parser.add_argument('--state_dir', type=str,
                        default="library_results/llama3/states",
                        help='Directory containing state files')
    parser.add_argument('--output_model', type=str,
                        default='prompt_classifier_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--project_name', type=str, default="retriever",
                        help='Project name for wandb logging')
    return parser.parse_args()

class PromptClassifier(nn.Module):
    def __init__(self, sentence_bert):
        super().__init__()
        self.sentence_bert = sentence_bert
        self.classifier = nn.Sequential(
            nn.Linear(sentence_bert.config.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2)
        )

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.sentence_bert(input_ids1, attention_mask=attention_mask1)
        outputs2 = self.sentence_bert(input_ids2, attention_mask=attention_mask2)
        pooled_output1 = outputs1.last_hidden_state[:, 0]
        pooled_output2 = outputs2.last_hidden_state[:, 0]
        concatenated = torch.cat((pooled_output1, pooled_output2), dim=1)
        logits = self.classifier(concatenated)
        return logits
    
class PromptRetriever:
    def __init__(self, model_path, device=None, batch_size=128, model_name=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.sentence_bert = AutoModel.from_pretrained(self.model_name)
        self.model = PromptClassifier(self.sentence_bert)
        self.model.load_state_dict(torch.load(model_path))
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.icl_pool_embedding = None

    def preprocess_icl_pool(self, icl_pool):
        icl_inputs = self._tokenize(icl_pool)
        with torch.no_grad():
            outputs1 = self.sentence_bert(icl_inputs['input_ids'].to(self.device),
                                          attention_mask=icl_inputs['attention_mask'].to(self.device))
        self.icl_pool_embedding = outputs1.last_hidden_state[:, 0]

    def retrieve(self, natural_prompt, icl_pool):
        if self.icl_pool_embedding is None:
            self.preprocess_icl_pool(icl_pool)

        scores = []
        natural_inputs = self._tokenize(natural_prompt)

        with torch.no_grad():
            outputs1 = self.sentence_bert(natural_inputs['input_ids'].to(self.device),
                                        attention_mask=natural_inputs['attention_mask'].to(self.device))
            natural_embedding = outputs1.last_hidden_state[:, 0]

            concatenated = torch.cat((natural_embedding.repeat(len(icl_pool), 1),
                                    self.icl_pool_embedding), dim=1)
            logits = self.model.classifier(concatenated)
        batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        
        for i, (prompt, score) in enumerate(zip(icl_pool, batch_scores)):
            scores.append((prompt, float(score),  i))

       
        return scores

    def _tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


class DataPreparation:
    @staticmethod
    def prepare_data(state_dir, total_samples=10000):
        file_lists = []
        icl_prompts = defaultdict(list)
        
        for filename in tqdm(os.listdir(state_dir)):
            if filename.endswith("icl_prompts.json"):
                file_lists.append(os.path.join(state_dir, filename))
        
        for file in file_lists:
            with open(file, "r") as f:
                tmp_prompts = json.load(f)
            task_name = file.split("_tv")[0].split("/")[-1]
            icl_prompts[task_name] += tmp_prompts
        
        with open("dataset_files/natural_prompts_manual.json", "r") as f:
            natural_prompts = json.load(f)
        

        tasks = ["superglue_rte", "superglue_wic",  "glue_qnli", "glue_sst2", "glue_mnli","arc_challenge", "bbh_boolean_expressions", "bbh_date_understanding", "bbh_reasoning_about_colored_objects", "bbh_temporal_sequences", "boolq", "commonsense_qa", "hellaswag", "openbookqa", "math_qa", "mmlu_pro_math","bbq_age", "crows_pairs", "ethics_justice", "ethics_commonsense"]
        completion_prompts = defaultdict(list)
        for task in tasks:
            data = load_dataset(task, "dataset_files")
            word_pairs = data['train'][np.random.choice(len(data['train']), 2, replace=False)]
            for p in natural_prompts[task][:10]:
                for i in range(2):
                    completion_prompts[task].append(p.format(input=word_pairs["input"][i])+"\nA:")
        
        task_pairs = defaultdict(list)
        for task in icl_prompts.keys():
            natural_task_prompts = completion_prompts[task]
            icl_task_prompts = icl_prompts[task]
            
            for natural_prompt in natural_task_prompts:
                for icl_prompt in icl_task_prompts:
                    task_pairs[task].append((natural_prompt, icl_prompt, 1))
                
                other_tasks = [t for t in icl_prompts.keys() if t != task]
                for t in other_tasks: 
                    random_icl_prompt = random.sample(icl_prompts[t], 10)
                    for r_icl_prompt in random_icl_prompt:
                        task_pairs[task].append((natural_prompt, r_icl_prompt, 0))
        
        balanced_task_pairs = defaultdict(lambda: defaultdict(list))
        for task, pairs in task_pairs.items():
            for pair in pairs:
                label = pair[2]
                balanced_task_pairs[task][label].append(pair)

        task_num = len(list(task_pairs.keys()))
        train_samples_per_task_per_label = int(total_samples / task_num / 2)
        test_samples_per_task_per_label = 5
        test_set, train_set, valid_set = [], [], []

        for task, label_pairs in balanced_task_pairs.items():
            for label, pairs in label_pairs.items():
                random.shuffle(pairs)
                test_set.extend(pairs[:test_samples_per_task_per_label])
                train_set.extend(pairs[test_samples_per_task_per_label:test_samples_per_task_per_label+train_samples_per_task_per_label])
                valid_set.extend(pairs[-5:])

        random.shuffle(train_set)
        random.shuffle(valid_set)
        random.shuffle(test_set)
        
        dataset = {
            "train": train_set,
            "test": test_set,
            "valid": valid_set
        }

        with open("dataset_files/train_data.json", "w") as f:
            json.dump(dataset, f, indent=4)
        return dataset
    
    def prepare_data_icl(state_dir, total_samples=1000):
        file_lists = []
        icl_prompts = defaultdict(list)
        
        for filename in tqdm(os.listdir(state_dir)):
            if filename.endswith("icl_prompts.json"):
                file_lists.append(os.path.join(state_dir, filename))
        
        for file in file_lists:
            with open(file, "r") as f:
                tmp_prompts = json.load(f)
            task_name = file.split("_tv")[0].split("/")[-1]
            icl_prompts[task_name] += tmp_prompts
        
        prompt = "Q: {input}\nA:"
        completion_prompts = defaultdict(list)
        for task_name in icl_prompts.keys():
            data = load_dataset(task_name, "dataset_files")
            word_pairs = data['train'][np.random.choice(len(data['train']), 2, replace=False)]
            
            for i in range(2):
                completion_prompts[task_name].append(prompt.format(input=word_pairs["input"][i]))
        
        task_pairs = defaultdict(list)
        for task in icl_prompts.keys():
            natural_task_prompts = completion_prompts[task]
            icl_task_prompts = icl_prompts[task]
            
            for natural_prompt in natural_task_prompts:
                for icl_prompt in icl_task_prompts:
                    task_pairs[task].append((natural_prompt, icl_prompt, 1))
                
                other_tasks = [t for t in icl_prompts.keys() if t != task]
                for t in other_tasks: 
                    random_icl_prompt = random.sample(icl_prompts[t], 10)
                    for r_icl_prompt in random_icl_prompt:
                        task_pairs[task].append((natural_prompt, r_icl_prompt, 0))
        
        balanced_task_pairs = defaultdict(lambda: defaultdict(list))
        for task, pairs in task_pairs.items():
            for pair in pairs:
                label = pair[2]
                balanced_task_pairs[task][label].append(pair)

        task_num = len(list(task_pairs.keys()))
        train_samples_per_task_per_label = int(total_samples / task_num / 2)
        test_samples_per_task_per_label = 5
        test_set, train_set, valid_set = [], [], []

        for task, label_pairs in balanced_task_pairs.items():
            for label, pairs in label_pairs.items():
                random.shuffle(pairs)
                test_set.extend(pairs[:test_samples_per_task_per_label])
                train_set.extend(pairs[test_samples_per_task_per_label:test_samples_per_task_per_label+train_samples_per_task_per_label])
                valid_set.extend(pairs[-5:])

        random.shuffle(train_set)
        random.shuffle(valid_set)
        random.shuffle(test_set)

        dataset = {
            "train": train_set,
            "test": test_set,
            "valid": valid_set
        }

        with open("dataset_files/train_data_icl.json", "w") as f:
            json.dump(dataset, f, indent=4)
        return dataset

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, learning_rate=2e-6):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, num_epochs=15, project_name="retriever"):
        wandb.init(project=project_name, config={
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epochs": num_epochs,
            "batch_size": self.train_dataloader.batch_size
        })

        wandb.watch(self.model)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}"):
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                outputs = self.model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")

            wandb.log({"train_loss": avg_train_loss, "epoch": epoch+1})

            val_accuracy, avg_val_loss = self.validate()
            print(f"Validation Accuracy: {val_accuracy}%")
            print(f"Validation Loss: {avg_val_loss}")

            wandb.log({
                "val_accuracy": val_accuracy,
                "val_loss": avg_val_loss,
                "epoch": epoch+1
            })

        wandb.finish()

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids1, attention_mask1, input_ids2, attention_mask2, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids1, attention_mask1, input_ids2, attention_mask2)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(self.val_dataloader)
        return val_accuracy, avg_val_loss

def prepare_dataloader(tokenizer, data_set, batch_size=32):
    natural_prompts, icl_prompts, labels = zip(*data_set)
    inputs1 = tokenizer(list(icl_prompts), padding=True, truncation=True, return_tensors="pt")
    inputs2 = tokenizer(list(natural_prompts), padding=True, truncation=True, return_tensors="pt")
    labels = torch.tensor(labels)
    dataset = TensorDataset(inputs1['input_ids'], inputs1['attention_mask'],
                            inputs2['input_ids'], inputs2['attention_mask'], labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    data = DataPreparation.prepare_data(
        state_dir=args.state_dir, 
        total_samples=args.total_samples
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sentence_bert = AutoModel.from_pretrained(args.model_name)

    # Prepare dataloaders
    train_dataloader = prepare_dataloader(tokenizer, data["train"], batch_size=args.batch_size)
    val_dataloader = prepare_dataloader(tokenizer, data["valid"], batch_size=args.batch_size)
    test_dataloader = prepare_dataloader(tokenizer, data["test"], batch_size=args.batch_size)

    # Initialize model
    model = PromptClassifier(sentence_bert)
    model.to(device)

    # Train model
    trainer = Trainer(model, train_dataloader, val_dataloader, device, 
                     learning_rate=args.learning_rate)
    trainer.train(num_epochs=args.num_epochs, project_name=args.project_name)

    # Save model
    torch.save(model.state_dict(), args.output_model)

    # Evaluate model
    retriever = PromptRetriever(args.output_model, device, model_name=args.model_name)
    accuracy = evaluate_classifier(retriever.model, test_dataloader, device)
    print(f"Final Test Accuracy: {100 * accuracy}%")

def evaluate_classifier(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids1, attention_mask1, input_ids2, attention_mask2, labels = [b.to(device) for b in batch]
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == "__main__":
    main()