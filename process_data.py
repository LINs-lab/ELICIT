import datasets
import random
from collections import defaultdict, Counter
from utils.model_utils import seed_everything
import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import wget
import re



def construct_deepmind():
    data = datasets.load_dataset("deepmind/aqua_rat", "raw")
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for item in data["options"]:
                choices_string += "\n" + item.replace(")", ". ", 1)
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string  , "output": data["correct"]})

        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/deepmind.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_antonym():
    with open("dataset_files/abstractive/antonym.json", "r") as f:
        train_data = json.load(f)
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["input"], "output": data["output"]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/antonym_processed.json", "w")as f:
        json.dump(final_data, f, indent=4)

   
def construct_glue(args):
    data = datasets.load_dataset('glue', 'mnli')
    train_data = list(data["train"])
    
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
    0: "True",
    1: "Neither",
    2: "False",
    }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "premise: "+  data["premise"] + "\nhypothesis: " + data["hypothesis"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/glue_mnli.json", "w")as f:
        json.dump(final_data, f, indent=4)



def select_fr_en_data(data, num_examples, exclude=None):
    
    input_space = list(data.keys())
    if exclude is None:
        input_space_new = input_space
    else: 
        input_space_new = [item for item in input_space if item not in 
                           exclude]
    
    selected_data = np.random.choice(input_space_new, num_examples).tolist()
    results = [{"input": d, "output": data[d]} for d in selected_data]

    return results
   

def construct_fr_en(args):
    data = datasets.load_dataset("json", data_files = "data/translation/fr_en.json")
    data = {f" {k}": f" {v}" for k, v in data["train"][0].items()}
    eval_data = select_fr_en_data(data, 1000)
    exclued_data = eval_data
    os.makedirs("our_data/translation/fr-en", exist_ok=True)
    for sample_num in [0, 1, 4, 16, 32, 64, 128, 256, 512, 1024]:
        demo_data = select_fr_en_data(data, sample_num, exclued_data)
        results = {
            "demon": demo_data,
            "test": eval_data
        }
        with open(f"our_data/translation/fr-en/sample{sample_num}_seed{args.seed}.json", "w")as f:
            json.dump(results, f, indent=4)



def construct_gsm8k(args):
    data = datasets.load_dataset('gsm8k', 'main')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test = list(data["test"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            parts = data["answer"].split("#### ")
            first_part = parts[0]
            answer_part = parts[-1].strip()
            first_part= first_part.replace("\n", " ").strip()
            if first_part[-1] != ".":
                first_part += "."
            answer = first_part + " The answer is " + answer_part + "."
            
            answer = re.sub(r"<<.*?>>", "", answer)
            processed_data.append({"input": data["question"] , "output": answer})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/gsm8k.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_boolq(args):
    data = datasets.load_dataset('boolq')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test = list(data["validation"])
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input":  "context: " + data["passage"] + "\nquestion: "+ data["question"], "output":data["answer"]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/boolq.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_crows_pairs(args):
    data = datasets.load_dataset('crows_pairs')
    train_data = list(data["test"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if np.random.random() > 0.5:
                processed_data.append({"input":  "sentence A: " + data["sent_more"] + "\nsentence B: " + data["sent_less"]  , "output": 'A'})
            else:
                processed_data.append({"input":  "sentence A: " + data["sent_less"] + "\nsentence B: " +  data["sent_more"]  , "output": 'B'})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/crows_pairs.json", "w")as f:
        json.dump(final_data, f, indent=4)
  
def construct_financial_phrasebank(args):
    data = datasets.load_dataset("financial_phrasebank", 'sentences_allagree')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
        0:"negative",
        1:"neutral",
        2:"positive",
    }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sentence"] , "output": labels[data["label"]]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/financial_phrasebank.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_openbookqa(args):
    data = datasets.load_dataset("openbookqa", "main")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            data["choices"]["label"] = ["A", "B", "C", "D"]
            for i in range(4):
                choices_string += "\n" + data["choices"]["label"][i] + ". " + data["choices"]["text"][i]
            processed_data.append({"input": data["question_stem"] + "\nOptions:" + choices_string  , "output": data["answerKey"]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/openbookqa.json", "w")as f:
        json.dump(final_data, f, indent=4)



def construct_superglue_copa(args):
    data = datasets.load_dataset("super_glue", "copa")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            premise = data["premise"].strip()
            if data["question"] == "cause":
                input_prefix = "Effect: "
                output_prefix = "Cause: "
            elif data["question"] == "effect":
                input_prefix = "Cause: "
                output_prefix = "Effect: "
            premise = input_prefix + premise
            answer_index = data["label"]
            choice1 = output_prefix + data["choice1"]
            choice2 = output_prefix + data["choice2"]
            choices_string = " (A) " + choice1 + " (B) " + choice2
            if answer_index == 0 :
                answer_string = "A"
            else:
                answer_string = "B"
            processed_data.append({"input": premise + choices_string  , "output": answer_string})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/superglue_copa.json", "w")as f:
        json.dump(final_data, f, indent=4)
   
    

def construct_piqa(args):
    data = datasets.load_dataset('piqa')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": ("goal: " + data["goal"] + " solution 1: " + data["sol1"] + " solution 2: " + data["sol2"]).replace("\t", "").replace("\r", "") , "output": str(data["label"]+1)})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/piqa.json", "w")as f:
        json.dump(final_data, f, indent=4)



def construct_winogrande(args):
    data = datasets.load_dataset("winogrande", "winogrande_xl")
    
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if int(data["answer"]) == 1:
                processed_data.append({"input": data["sentence"] + " (A) " + data["option1"] + " (B) " + data["option2"], "output": "A"})
            elif int(data["answer"]) == 2:
                processed_data.append({"input": data["sentence"] + " (A) " + data["option1"] + " (B) " + data["option2"], "output": "B"})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/piqa.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_ade_classification():
    data = datasets.load_dataset('ade_corpus_v2', 'Ade_corpus_v2_classification')
    train_data = list(data["train"])
    
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data  in data_list:
            processed_data.append({"input": data["text"], "output": "no" if data["label"] == 0 else "yes"})
        return processed_data

    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }

    with open(f"dataset_files/abstractive/ade_corpus_v2.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_amazon_polarity():
    data = datasets.load_dataset('amazon_polarity')
    train = list(data["train"])
    test = list(data["test"])
    test, valid = train_test_split(test, test_size=0.3
    , random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "title: "+  data["title"] + " content: " + data["content"], "output": "negative" if data["label"] == 0 else "positive"})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/amazon_polarity.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_anli():
    data = datasets.load_dataset('anli')
    train = list(data["train_r1"])
    test = list(data["dev_r1"])
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    labels = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "premise: "+  data["premise"].replace("\n", " ") + " hypothesis: " + data["hypothesis"].replace("\n", " "), "output": labels[data["label"]]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/anli.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_app_reviews():
    data = datasets.load_dataset('app_reviews')
    train_data = list(data["train"])
    
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "review: "+  data["review"], "output": str(data["star"])})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/app_reviews.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_aqua_rat():
    data = datasets.load_dataset("aqua_rat", "raw")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for option in data["options"]:
                choices_string += " (" + option
                    
            processed_data.append({"input": data["question"].replace("\n", " ") + choices_string, "output": data["correct"]})
        return processed_data
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/app_reviews.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_art():
    data = datasets.load_dataset('art')
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:  
            processed_data.append({"input": ("observation 1: "+ data["observation_1"] + " observation 2: " + data["observation_2"] + " hypothesis 1: "+ data["hypothesis_1"] + " hypothesis 2: "+ data["hypothesis_2"]).replace("\n", " ").replace("\t", " "), "output": str(data["label"])})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)
    }
    with open(f"dataset_files/abstractive/art.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_blimp():
    ALL_PARADIGMS = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
    ]
    def process_single(data_list):
        processed_data = []
        np.random.seed(42)
        for data in data_list:
            if np.random.random() > 0.5:
                processed_data.append({"input": "sentence 1: " + data["sentence_good"] + " sentence 2: " + data["sentence_bad"], "output":str(1)})
            else:
                processed_data.append({"input": "sentence 1: " + data["sentence_bad"] + " sentence 2: " + data["sentence_good"], "output":str(2)})
        return processed_data

    for subset in ALL_PARADIGMS:
        data = datasets.load_dataset('blimp', subset)
        train_data = list(data["train"])
    
        train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
        test, valid = train_test_split(valid, test_size=0.3, random_state=42)

        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)
            }
        with open(f"dataset_files/abstractive/blimp_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_climate_fever():
    data = datasets.load_dataset("climate_fever")
    train_data = list(data["test"])
    
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    
    labels = {
            0: "Supports",
            1: "Refutes",
            2: "Not enough info",
            3: "Disputed",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["claim"], "output": labels[data["claim_label"]]})
            
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)
            }
    with open(f"dataset_files/abstractive/climate_fever.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_codah():
    data = datasets.load_dataset("codah", "fold_0")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choice_string = ""
            for idx, candidate in enumerate(data["candidate_answers"]):
                choice_string += " (" + id2alphabet[idx] + ") " + candidate
            processed_data.append({"input": data["question_propmt"] + choice_string, "output": id2alphabet[data["correct_answer_idx"]]})
            
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)
            }
    with open(f"dataset_files/abstractive/codah.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_cos_e():
    data = datasets.load_dataset("cos_e", "v1.11")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choice_string = ""
            for idx, candidate in enumerate(data["choices"]):
                choice_string += " (" + id2alphabet[idx] + ") " + candidate
                if candidate == data["answer"]:
                    answer = id2alphabet[idx]
            processed_data.append({"input": data["question"].replace("\t", " ").replace("\n", " ") + choice_string, "output": answer})
            
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/cosine_e.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_cosmos_qa():
    data = datasets.load_dataset("cosmos_qa")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choice_string = ""
            for idx in range(4):
                choice_string += " (" + id2alphabet[idx] + ") " + data["answer" + str(idx)]
            processed_data.append({"input": data["question"] + " " + data["context"] + " " + choice_string, "output": id2alphabet[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/cosmos_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_dbpedia_14():
    labels = {
            0:"Company",
            1:"EducationalInstitution",
            2:"Artist",
            3:"Athlete",
            4:"OfficeHolder",
            5:"MeanOfTransportation",
            6:"Building",
            7:"NaturalPlace",
            8:"Village",
            9:"Animal",
            10:"Plant",
            11:"Album",
            12:"Film",
            13:"WrittenWork",
        }
    data = datasets.load_dataset('dbpedia_14')
    train = list(data["train"])
    test = list(data["test"])
    test, valid = train_test_split(test, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["content"].replace("\t", " ").strip(), "output":labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/dbpedia_14.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_definite_pronoun_resolution():
    data = datasets.load_dataset('definite_pronoun_resolution')
    train = list(data["train"])
    test = list(data["test"])
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sentence"] + " (A) " + data["candidates"][0] + " (B) " + data["candidates"][1], "output": "A" if data["label"] == 0 else "B"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/definite_pronoun_resolution.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_discovery():
    data =  datasets.load_dataset('discovery', 'discovery')
    train_data = list(data["test"])
    LABELS = [
    "[no-conn]",
    "absolutely,",
    "accordingly",
    "actually,",
    "additionally",
    "admittedly,",
    "afterward",
    "again,",
    "already,",
    "also,",
    "alternately,",
    "alternatively",
    "although,",
    "altogether,",
    "amazingly,",
    "and",
    "anyway,",
    "apparently,",
    "arguably,",
    "as_a_result,",
    "basically,",
    "because_of_that",
    "because_of_this",
    "besides,",
    "but",
    "by_comparison,",
    "by_contrast,",
    "by_doing_this,",
    "by_then",
    "certainly,",
    "clearly,",
    "coincidentally,",
    "collectively,",
    "consequently",
    "conversely",
    "curiously,",
    "currently,",
    "elsewhere,",
    "especially,",
    "essentially,",
    "eventually,",
    "evidently,",
    "finally,",
    "first,",
    "firstly,",
    "for_example",
    "for_instance",
    "fortunately,",
    "frankly,",
    "frequently,",
    "further,",
    "furthermore",
    "generally,",
    "gradually,",
    "happily,",
    "hence,",
    "here,",
    "historically,",
    "honestly,",
    "hopefully,",
    "however",
    "ideally,",
    "immediately,",
    "importantly,",
    "in_contrast,",
    "in_fact,",
    "in_other_words",
    "in_particular,",
    "in_short,",
    "in_sum,",
    "in_the_end,",
    "in_the_meantime,",
    "in_turn,",
    "incidentally,",
    "increasingly,",
    "indeed,",
    "inevitably,",
    "initially,",
    "instead,",
    "interestingly,",
    "ironically,",
    "lastly,",
    "lately,",
    "later,",
    "likewise,",
    "locally,",
    "luckily,",
    "maybe,",
    "meaning,",
    "meantime,",
    "meanwhile,",
    "moreover",
    "mostly,",
    "namely,",
    "nationally,",
    "naturally,",
    "nevertheless",
    "next,",
    "nonetheless",
    "normally,",
    "notably,",
    "now,",
    "obviously,",
    "occasionally,",
    "oddly,",
    "often,",
    "on_the_contrary,",
    "on_the_other_hand",
    "once,",
    "only,",
    "optionally,",
    "or,",
    "originally,",
    "otherwise,",
    "overall,",
    "particularly,",
    "perhaps,",
    "personally,",
    "plus,",
    "preferably,",
    "presently,",
    "presumably,",
    "previously,",
    "probably,",
    "rather,",
    "realistically,",
    "really,",
    "recently,",
    "regardless,",
    "remarkably,",
    "sadly,",
    "second,",
    "secondly,",
    "separately,",
    "seriously,",
    "significantly,",
    "similarly,",
    "simultaneously",
    "slowly,",
    "so,",
    "sometimes,",
    "soon,",
    "specifically,",
    "still,",
    "strangely,",
    "subsequently,",
    "suddenly,",
    "supposedly,",
    "surely,",
    "surprisingly,",
    "technically,",
    "thankfully,",
    "then,",
    "theoretically,",
    "thereafter,",
    "thereby,",
    "therefore",
    "third,",
    "thirdly,",
    "this,",
    "though,",
    "thus,",
    "together,",
    "traditionally,",
    "truly,",
    "truthfully,",
    "typically,",
    "ultimately,",
    "undoubtedly,",
    "unfortunately,",
    "unsurprisingly,",
    "usually,",
    "well,",
    "yet,",
]
    
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence 1: " + data["sentence1"] + " sentence 2: "+ data["sentence2"], "output": LABELS[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/discovery.json", "w")as f:
        json.dump(final_data, f, indent=4)
    
def construct_dream():
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    data = datasets.load_dataset("dream")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx in range(3):
                choices_string += " (" + id2alphabet[idx] + ") " + data["choice"][idx]
                if data["choice"][idx] == data["answer"]:
                    output = id2alphabet[idx]
            processed_data.append({"input": data["question"] + " " + " ".join(data["dialogue"]) + " " + choices_string, "output": output})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/dream.json", "w")as f:
        json.dump(final_data, f, indent=4)
   
def construct_emotion():
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    data = datasets.load_dataset('emotion')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"].strip(), "output": labels[data["label"]]})
        return processed_data

    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/emotion.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_ethos():
    data = datasets.load_dataset('ethos', 'multilabel')
    dimensions = ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]
    train = list(data["train"])
    train, valid = train_test_split(train, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    
    
    for dimension in dimensions:

        if dimension == "violence":
            labels = {
                0: "no",
                1: "yes",
            }
        elif dimension == "directed_vs_generalized":
            labels = {
                0:"generalied",
                1:"directed",
            }
        else:
            labels= {
                0:"false",
                1:"true",
            }
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                processed_data.append({"input": data["text"].strip(), "output": labels[data[dimension]]})
            return processed_data
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/ethos_{dimension}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_glue_cola():
    data = datasets.load_dataset('glue', 'cola')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sentence"].strip(), "output": "No" if data["label"] == 0 else "Yes"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_cola.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_glue_mrpc():
    data = datasets.load_dataset('glue', 'mrpc')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence 1: " + data["sentence1"] + " sentence 2: " + data["sentence2"], "output": "not_equivalent" if data["label"] == 0 else "equivalent"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_mrpc.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_glue_qnli():
    data = datasets.load_dataset('glue', 'qnli')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question: " + data["question"] + "\nresponse: " + data["sentence"], "output": "Yes" if data["label"] == 0 else "No"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_qnli.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_glue_qqp():
    data = datasets.load_dataset('glue', 'qqp')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question 1: " + data["question1"] + " question 2: " + data["question2"], "output": "not_duplicate" if data["label"] == 0 else "duplicate"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_qqp.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_glue_rte():
    data = datasets.load_dataset('glue', 'rte')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence 1: " + data["sentence1"] + " sentence 2: " + data["sentence2"], "output": "entailment" if data["label"] == 0 else "not_entailment"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_rte.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_glue_sst2():
    data = datasets.load_dataset('glue', 'sst2')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence: " + data["sentence"], "output": "negative" if data["label"] == 0 else "positive"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_sst2.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_glue_wnli():
    data = datasets.load_dataset('glue', 'wnli')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence 1: " + data["sentence1"] + " sentence 2: "+ data["sentence2"], "output": "not_entailment" if data["label"] == 0 else "entailment"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/glue_wnli.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_google_wellformed_query():
    data =  datasets.load_dataset('google_wellformed_query')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["rating"] < 0.4:
                processed_data.append({"input": data["content"].strip(), "output": "not well-formed"})
            elif data["rating"] > 0.6:
                processed_data.append({"input": data["content"].strip(), "output": "well-formed"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/google_wellformed_query.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_hate_speech_offensive():
    data = datasets.load_dataset('hate_speech_offensive')
    labels = {
            0:"hate speech",
            1:"offensive language",
            2:"neither",
        }
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["tweet"].replace("\n", " "), "output": labels[data["class"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/hate_speech_offensive.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_hate_speech18():
    data = datasets.load_dataset('hate_speech18')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
            0: "noHate",
            1: "hate",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["label"] > 1:
                continue
            processed_data.append({"input": data["text"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/hate_speech18.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_hatexplain():
    data = datasets.load_dataset('hatexplain')
    
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels = {
            0:"hatespeech",
            1:"normal",
            2:"offensive",
        }
    
    def get_majority(lst):
        c = Counter(lst)
        rank = c.most_common()
        if len(rank) == 1:
            return rank[0][0]
        elif rank[0][1] == rank[1][1]:
            return None
        else:
            return rank[0][0]
        
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            label = get_majority(data["annotators"]["label"])
            if label is not None:
                processed_data.append({"input":" ".join(data["post_tokens"]), "output": labels[label]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/hatexplain.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_health_fact():
    labels = {
            0:"false",
            1:"mixture",
            2:"true",
            3:"unproven",
    }
    data = datasets.load_dataset('health_fact')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["label"] < 0:
                continue
            processed_data.append({"input": data["claim"].replace("\n", " ").replace("\r"," ").replace("\t", " "), "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/health_fact.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_hellaswag():
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
    data = datasets.load_dataset("hellaswag")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i in range(len(data["endings"])):
                if i == int(data["label"]):
                    output = id2alphabet[i]
                choices_string += "\n" + id2alphabet[i] + ". " + data["endings"][i]
            processed_data.append({"input": data["ctx"] + "\nOptions:" + choices_string, "output": output})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/hellaswag.json", "w")as f:
        json.dump(final_data, f, indent=4)
    
def construct_imdb():
    data = datasets.load_dataset('imdb')
    
    train = list(data["train"])
    test = list(data["test"])
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    labels = {
            0: "negative",
            1: "positive",
        }

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"].replace("\n", "").replace("\t", ""), "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/imdb.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_lama():
    for subset in ["trex", "squad", "google_re", "conceptnet"]:
        data =  datasets.load_dataset("lama", subset)
        train_data = list(data["train"])
        train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
        test, valid = train_test_split(valid, test_size=0.3, random_state=42)
        
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                if subset in ["trex", "google_re"]:
                    input_text = data["template"].replace("[X]", data["sub_label"]).replace("[Y]", "[MASK]")
                else:
                    input_text = data["masked_sentence"]
                processed_data.append({"input": input_text, "output": data["obj_label"]})
            return processed_data
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/lama_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_liar():
    data = datasets.load_dataset('liar')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels = {
            0:"false",
            1:"half-true",
            2:"mostly-true",
            3:"true",
            4:"barely-true",
            5:"pants-fire",
        }

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            
            processed_data.append({"input": ("statement: "+ data["statement"] + " speaker: " + data["speaker"] + " context: " + data["context"]).replace("\n", " ").replace("\r", " ").replace("\t", " "), "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/liar.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_math_qa():
    data =  datasets.load_dataset('math_qa')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            options = data["options"].split(",")
            choices = "\nA. " + options[0][4:-1]
            choices += "\nB. " + options[1][5:-1]
            choices += "\nC. " + options[2][5:-1]
            choices += "\nD. " + options[3][5:-1]
            choices += "\nE. " + options[4][5:]
            processed_data.append({"input": data["Problem"] + "\nOptions:" + choices, "output": data["correct"].upper()})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/math_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_mc_taco():
    data = datasets.load_dataset('mc_taco')
    train_data = list(data["test"])
    train, test = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    valid = list(data["validation"])
    labels = {
            0: "no",
            1: "yes",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sentence"] + " " + data["question"] + " " + data["answer"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mc_taco.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_medical_questions_pairs():
    data = datasets.load_dataset('medical_questions_pairs')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
            0: "Similar",
            1: "Dissimilar",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question 1: " + data["question_1"] + " question 2: " + data["question_2"], "output": labels[data["label"]]})
            
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/medical_question_pairs.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_mocha():
    data = datasets.load_dataset('mocha')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["score"] % 1 !=0:
                continue
            
            processed_data.append({"input": "question: "+ data["question"] + " context: " + data["context"] + "reference: " + data["reference"] + " candidate" + data["candidate"], "output": data["score"]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mocha.json", "w")as f:
        json.dump(final_data, f, indent=4)
 
def construct_numer_sense():
    data = datasets.load_dataset('numer_sense')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sentence"], "output": data["target"]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/numer_sense.json", "w")as f:
        json.dump(final_data, f, indent=4)
 


def construct_onestop_english():
    data = datasets.load_dataset('onestop_english')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    labels = {
            0:"elementary",
            1:"intermediate",
            2:"advance",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            data["text"] = data["text"].replace("\n", " ")
            if data["text"].startswith("Intermediate  "):
                data["text"] = data["text"][14:]
            processed_data.append({"input": data["text"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/onestop_english.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_paws():
    data =  datasets.load_dataset('paws', 'labeled_final')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels = {
            0: "not_duplicate",
            1: "duplicate",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            
            processed_data.append({"input": "sentence 1: "+ data["sentence1"] + " sentence 2: " + data["sentence2"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/paws.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_poem_sentiment():
    data = datasets.load_dataset('poem_sentiment')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels= {
            0:"negative",
            1:"positive",
            2:"no_impact",
            #3:"mixed", # there is no `mixed` on the test set
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["label"] == 3:
                continue
            processed_data.append({"input": data["verse_text"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/poem_sentiment.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_proto_qa():
    if not os.path.exists("dataset_files/raw/proto_qa"):
        os.makedirs("dataset_files/raw/proto_qa")
    if not os.path.exists("dataset_files/raw/proto_qa/train.jsonl"):
        wget.download("https://raw.githubusercontent.com/iesl/protoqa-data/master/data/train/train.jsonl", "dataset_files/raw/proto_qa")
    with open("dataset_files/raw/proto_qa/train.jsonl") as fin:
        lines = fin.readlines()
        train_data = [json.loads(line) for line in lines]
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["question"]["normalized"], "output": "\t".join(data["answers"]["raw"].keys())})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/proto_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_qasc():
    data = datasets.load_dataset("qasc")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
 
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i in range(len(data["choices"]["label"])):
                if data["choices"]["label"][i] == data["answerKey"]:
                    answer_string = data["choices"]["label"][i]
 
                choices_string += " (" + data["choices"]["label"][i] + ") " + data["choices"]["text"][i]
            processed_data.append({"input": data["question"] + choices_string, "output": answer_string})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/qasc.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_quail():
    data = datasets.load_dataset("quail")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i, answer in enumerate(data["answers"]):
                choices_string += " (" + id2alphabet[i] + ") " + answer
                if i == data["correct_answer_id"]:
                    answer_string = id2alphabet[i]
            processed_data.append({"input": data["question"].replace("\n", " ") + choices_string + " " + data["context"].replace("\n", " "), "output": answer_string})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/quail.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_quarel():
    data = datasets.load_dataset("quarel")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["question"], "output": "A" if data["answer_index"] == 0 else "B"})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/quarel.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_quartz():
    data = datasets.load_dataset("quartz")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    for mode in ["with_knowledge", "no_knowledge"]:
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                choices_string = ""
                for i in range(len(data["choices"]["label"])):
                    choices_string += " (" + data["choices"]["label"][i] + ") " + data["choices"]["text"][i]
                if mode == "with_knowledge":
                    processed_data.append({"input": data["question"] + data["para"] + choices_string , "output": data["answerKey"]})
                elif mode == "no_knowledge":
                    processed_data.append({"input": data["question"] + choices_string , "output": data["answerKey"]})
            return processed_data
        final_data = {
                "train": process_single(train),
                "valid": process_single(valid),
                "test": process_single(test)}
        with open(f"dataset_files/abstractive/quartz_{mode}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_quoref():
    data = datasets.load_dataset("quoref")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": ("question: " + data["question"] + " context: " + data["context"]).replace("\n", " "), "output": "\t".join(data["answers"]["text"])})
        
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/quoref.json", "w")as f:
        json.dump(final_data, f, indent=4)

    

def construct_race():
    for subset in ["middle", "high"]:
        data = datasets.load_dataset("race", subset)
        train = list(data["train"])
        test = list(data["test"])
        valid = list(data["validation"])
        id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                choices_string = ""
                for i, answer in enumerate(data["options"]):
                    choices_string += " (" + id2alphabet[i] + ") " + answer.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    
                processed_data.append({"input": data["question"].replace("\n", " ").replace("\r", " ").replace("\t", " ") + choices_string + " " + data["article"].replace("\n", " ").replace("\r", " ").replace("\t", " "), "output": data["answer"]})
            return processed_data
        final_data = {
                "train": process_single(train),
                "valid": process_single(valid),
                "test": process_single(test)}
        with open(f"dataset_files/abstractive/race_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_ropes():
    data = datasets.load_dataset("ropes")

    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            
            processed_data.append({"input": ("question: " + data["question"] + " situation: " + data["situation"] + " background: " + data["background"]).replace("\n", " "), "output": "\t".join(data["answers"])})
            print(data["answers"])
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/ropes.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_rotten_tomatoes():
    data = datasets.load_dataset('rotten_tomatoes')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    
    labels =  {
            0: "negative",
            1: "positive",
        }

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"], "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/rotten_tomatoes.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_scicite():
    data = datasets.load_dataset("scicite")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels = {
            0:"method",
            1:"background",
            2:"result",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": (data["string"][data["citeStart"]: data["citeEnd"]] + " " +data["string"].replace("\n", " ") + " " + data["sectionName"]).replace("\n", " ").replace("\t", " "), "output": labels[data["label"]]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/scicite.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_sciq():
    data = datasets.load_dataset("sciq")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if len(data["support"].replace("\n", " ")) == 0:
                continue
            choices_string = ""
            all_answers = [data["distractor1"], data["distractor2"], data["distractor3"], data["correct_answer"]]
            random.shuffle(all_answers)
            for i, answer in enumerate(all_answers):
                choices_string += " (" + id2alphabet[i] + ") " + answer
                if answer == data["correct_answer"]:
                    answer_id = id2alphabet[i]
            processed_data.append({"input": data["question"].replace("\n", " ") + choices_string + " " + data["support"].replace("\n", " "), "output": answer_id})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/sciq.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_scitail():
    data = datasets.load_dataset('scitail', 'snli_format')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "setence 1: "+ data["sentence1"] + " sentence 2: " + data["sentence2"], "output": data["gold_label"]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/scitail.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_search_qa():
    data = datasets.load_dataset("search_qa", "train_test_val")
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question: " + data["question"] + "category: " + data["category"], "output": data["answer"]})
        return processed_data
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/search_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_sick():
    data = datasets.load_dataset('sick')
    train = list(data["train"])
    test = list(data["test"])
    valid = list(data["validation"])
    labels = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "sentence 1: " + data["sentence_A"] + " sentence 2: " + data["sentence_B"], "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/sick.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_sms_spam():
    data =  datasets.load_dataset('sms_spam')
    train_data = list(data["train"])
    train, valid = train_test_split(train_data, test_size=0.3, train_size=0.7, random_state=42)
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
            0:"ham",
            1:"spam"}
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["sms"].strip(), "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/sms_spam.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_social_i_qa():
    data = datasets.load_dataset("social_i_qa")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)

    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i, ans in enumerate([data["answerA"], data["answerB"], data["answerC"]]):
                if i == int(data["label"]) - 1:
                    answer_string = id2alphabet[i]
                choices_string += " (" + id2alphabet[i] + ") " + ans
            processed_data.append({"input": data["question"].replace("\n", " ") + choices_string + " " + data["context"].replace("\n", " "), "output": answer_string})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/sms_spam.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_squad():
    data = datasets.load_dataset("squad")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
  
    for mode in ["with_context", "no_context"]:
        def process_single(data_list):
            processed_data = []
            for data in data_list:
               if mode == "with_context":
                processed_data.append({"input": "question: " + data["question"] + " context: " + data["context"].replace("\t", " ").replace("\n", " "), "output": data["answers"]["text"][0]})
            else:
                processed_data.append({"input": "question: " + data["question"], "output": data["answers"]["text"][0]})
            return processed_data
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/squad_{mode}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_superglue_cb():
    data = datasets.load_dataset('super_glue', 'cb')
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    labels = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "premise: " + data["premise"] + " hypothesis: " + data["hypothesis"], "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_cb.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_superglue_multirc():
    data = datasets.load_dataset('super_glue', 'multirc')
    train = list(data["train"])[:1000]
    valid = list(data["validation"])[:1000]
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
    ID2ALPHABET = {i : chr(65+i) for i in range(26)}

  
    def process_single(data_list):
        processed_data = []
        paragraphs = {}
        for data in data_list:
            if data["idx"]["question"] not in paragraphs:
                paragraphs[data["idx"]["question"]] = [data["question"], data["paragraph"], [data["answer"]], [data["label"]]]
            else:
                paragraphs[data["idx"]["question"]][2].append(data["answer"])
                paragraphs[data["idx"]["question"]][3].append(data["label"])
            
            for paragraph in paragraphs.values():
                src = "question: {}".format(paragraph[0].replace("\n", " ").replace("\t", " ").replace("\r", " ")) + "\nOptions:"
                for idx, choice in enumerate(paragraph[2]):
                    src += "\n" + ID2ALPHABET[idx] + ". " + choice.replace("\n", " ").replace("\t", " ").replace("\t", " ")
                src += "\ncontext: {}".format(paragraph[1].replace("\n", " ").replace("\t", " ").replace("\r", " "))
                correct_answers = []
                for answer, label in zip(paragraph[2], paragraph[3]):
                    if label == 1:
                        correct_answers.append(answer)
                if len(correct_answers) == 0:
                    tar = "NO ANSWER!"
                else:
                    tar = "\t".join(correct_answers)
                processed_data.append({"input": src, "output": tar})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_multirc.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_superglue_record():
    data = datasets.load_dataset("super_glue", "record")
    train = list(data["train"])
    valid = list(data["validation"])
    test, valid = train_test_split(valid, test_size=0.3, random_state=42)
   
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question: " + data["query"] + " context: " + data["passage"].replace("\n", " "), "output": data["answers"][0]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_record.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_superglue_rte():
    data = datasets.load_dataset('super_glue', "rte")
    labels =  {
            0:"True",
            1:"False"
        }
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
   
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "premise: " + data["premise"].replace("\n", " ") + "\nhypothesis: " + data["hypothesis"].replace("\n", " "), "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_rte.json", "w")as f:
        json.dump(final_data, f, indent=4)
    
def construct_superglue_wic():
    data = datasets.load_dataset('super_glue', "wic")
    labels =  {
            0: "No",
            1: "Yes",
        }
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
   
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "word: " + data["word"] + "\nsentence 1: " + data["sentence1"] + "\nsentence 2: " + data["sentence2"],  "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_wic.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_superglue_wsc():
    data = datasets.load_dataset('super_glue', "wsc.fixed")
    labels =  {
            0: "false",
            1: "true",
        }
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
   
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "text: " + data["text"] + " span1_text" + data["span1_text"] + " span2_text: " + data["span2_text"],  "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/superglue_wsc.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_swag():
    data = datasets.load_dataset("swag", "regular")
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            candidates = [data["ending0"], data["ending1"], data["ending2"], data["ending3"]]
            choices_string = ""
            for i, ending in enumerate(candidates):
                if i == data["label"]:
                    output = id2alphabet[i]
                choices_string += " (" + id2alphabet[i] + ") " + ending
            processed_data.append({"input": data["startphrase"] + choices_string,  "output": output})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/swag.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_tab_fact():
    data = datasets.load_dataset('tab_fact', 'tab_fact')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    
    labels = {
            0: "refuted",
            1: "entailed",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "statement: "+ data["statement"] + " table_caption: " + data["table_caption"] + " table_text: " + data["table_text"].replace("\n", " "),  "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/tab_fact.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_trec_finegrained():
    data = datasets.load_dataset('trec')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    labels = {
            0:"manner",
            1:"cremat",
            2:"animal",
            3:"exp",
            4:"ind",
            5:"gr",
            6:"title",
            7:"def",
            8:"date",
            9:"reason",
            10:"event",
            11:"state",
            12:"desc",
            13:"count",
            14:"other",
            15:"letter",
            16:"religion",
            17:"food",
            18:"country",
            19:"color",
            20:"termeq",
            21:"city",
            22:"body",
            23:"dismed",
            24:"mount",
            25:"money",
            26:"product",
            27:"period",
            28:"substance",
            29:"sport",
            30:"plant",
            31:"techmeth",
            32:"volsize",
            33:"instru",
            34:"abb",
            35:"speed",
            36:"word",
            37:"lang",
            38:"perc",
            39:"code",
            40:"dist",
            41:"temp",
            42:"symbol",
            43:"ord",
            44:"veh",
            45:"weight",
            46:"currency",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["fine_label"] > 46:
                continue
            processed_data.append({"input": data["text"].replace("\t", "").replace("\n", "").replace("\r", ""),  "output": labels[data["fine_label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/trec_finegrained.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_trec():
    data = datasets.load_dataset('trec')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    
    labels = {
            0:"DESC",
            1:"ENTY",
            2:"ABBR",
            3:"HUM",
            4:"NUM",
            5:"LOC",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"],  "output": labels[data["coarse_label"]]})
        return processed_data
    
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/trec.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_tweet_eval():
    for subset in ['emoji', 'emotion', 'hate', 'irony', 'offensive', 'sentiment', 'stance_abortion', 'stance_atheism', 'stance_climate', 'stance_feminist', 'stance_hillary']:
        data = datasets.load_dataset('tweet_eval', subset)
        train = list(data["train"])
        train, test= train_test_split(train, test_size=0.3, random_state=42)
        test, valid = train_test_split(test, test_size=0.3, random_state=42)
        if subset == "emoji":
            labels = {
                0:"❤",
                1:"😍",
                2:"😂",
                3:"💕",
                4:"🔥",
                5:"😊",
                6:"😎",
                7:"✨",
                8:"💙",
                9:"😘",
                10:"📷",
                11:"🇺🇸",
                12:"☀",
                13:"💜",
                14:"😉",
                15:"💯",
                16:"😁",
                17:"🎄",
                18:"📸",
                19:"😜",
            }
        elif subset == "emotion":
            labels = {
                0:"anger",
                1:"joy",
                2:"optimism",
                3:"sadness",
            }
        elif subset == "hate":
            labels = {
                0:"non-hate",
                1:"hate",
            }
        elif subset == "irony":
            labels = {
                0:"non-irony",
                1:"hate",
            }
        elif subset == "offensive":
            labels = {
                0:"non-offensive",
                1:"hate",
            }
        elif subset == "sentiment":
            labels = {
                0:"negative",
                1:"neutral",
                2:"positive",
            }
        else:
            labels = {
                0:"none",
                1:"against",
                2:"favor",
            }
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                if len(data["text"].replace("\n", " ")):
                    processed_data.append({"input": data["text"].replace("\n", " "),  "output": labels[data["label"]]})
            return processed_data
    
        final_data = {
                "train": process_single(train),
                "valid": process_single(valid),
                "test": process_single(test)}
        with open(f"dataset_files/abstractive/tweet_eval_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_wiki_auto():
    data = datasets.load_dataset('wiki_auto', 'manual')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)

    labels = {
        0:"notAligned",
        1:"aligned",
    }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            if data["alignment_label"] > 1:
                continue
            processed_data.append({"input": "normal sentence: " + data["normal_sentence"] + " simple_sentence: " + data["simple_sentence"] ,  "output": labels[data["alignment_label"]]})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/wiki_auto.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_wiki_qa():
    data = datasets.load_dataset('wiki_qa')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    labels ={
        0: "false",
        1: "true",
    }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": "question: "+data["question"] + " answer: " + data["answer"] ,  "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/wiki_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_wiqa():
    data = datasets.load_dataset("wiqa")
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    id2alphabet = {0: "A", 1: "B", 2: "C", 3: "D"}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i, ans in enumerate(data["choices"]["text"]):
                if ans == id2alphabet[i]:
                    output = id2alphabet[i]
                choices_string += " (" + id2alphabet[i] + ") " + ans
            processed_data.append({"input": data["question_stem"].replace("\n"," ")+choices_string + " ".join(data["question_para_step"]) ,  "output": output})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/wiqa.json", "w")as f:
        json.dump(final_data, f, indent=4)
  

def construct_yahoo_answers_topics():
    data = datasets.load_dataset('yahoo_answers_topics')
    labels = {
            0:"Society & Culture",
            1:"Science & Mathematics",
            2:"Health",
            3:"Education & Reference",
            4:"Computers & Internet",
            5:"Sports",
            6:"Business & Finance",
            7:"Entertainment & Music",
            8:"Family & Relationships",
            9:"Politics & Government",
        }
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": ("question_title: " +data["question_title"] + " question_content: " + data["question_content"] + "best_answer: " + data["best_answer"]).replace("\n", "").replace("\t", "").replace("\r", "") ,  "output": labels[data["topic"]]})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/yahoo_answers_topics.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_yelp_polarity():
    data = datasets.load_dataset('yelp_polarity')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    labels = {
            0:"negative",
            1:"positive",
        }
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"].replace("\\n", " ") ,  "output": labels[data["label"]]})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/yelp_polarity.json", "w")as f:
        json.dump(final_data, f, indent=4)


def construct_yelp_review_full():
    data = datasets.load_dataset('yelp_review_full')
    train = list(data["train"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["text"].replace("\\n", " ") ,  "output": str(data["label"] + 1)})
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/yelp_review_full.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_arc_challenge():
    data = datasets.load_dataset("ai2_arc", "ARC-Challenge")
    train = list(data["train"])
    valid = list(data["validation"])
    test = list(data["test"])
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            
            choices_string = ""
            for label, string in zip(data['choices']["label"], data['choices']["text"]):
                choices_string += "\n" + label + ". " + string 
                # if label == "1": breakpoint()
            processed_data.append({"input": data["question"] + "\nOptions:" +  choices_string,  "output": data["answerKey"]})
            
        return processed_data
    
    final_data = {
        "train": process_single(train),
        "valid": process_single(valid),
        "test": process_single(test)}
    with open(f"dataset_files/abstractive/arc_challenge.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_bbh():
    for subset in ["reasoning_about_colored_objects", "object_counting", "boolean_expressions", "date_understanding", "temporal_sequences", "logical_deduction_five_objects", ]:
        data = datasets.load_dataset("lukaemon/bbh", subset)
        all_data = list(data["test"])
        random.shuffle(all_data)
       
        train = all_data[:128]
        test = all_data[128:208]
        valid = all_data[208:]

        def process_single(data_list):
            processed_data = []
            for data in data_list:
                if subset in ["reasoning_about_colored_objects", "date_understanding", "temporal_sequences", "logical_deduction_five_objects"]:
                    target = data["target"][1]
                    input = data["input"].replace("(", "").replace(")",".")
                else:
                    target = data["target"]
                    input = data["input"]
                processed_data.append({"input": input, "output": target})
            
            return processed_data
        
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/bbh_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_mmlu_pro():
    data = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    def filter_math(example):
        return example["category"] == "math"
   
    math_data = data.filter(filter_math)
    train_data = list(math_data["test"])
    train, test= train_test_split(train_data, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    ID2ALPHABET = {i : chr(65+i) for i in range(26)}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx, string in enumerate(data["options"]):
                if idx == data["answer_index"]:
                    answerKey = ID2ALPHABET[idx]
                choices_string +=  "\n" + ID2ALPHABET[idx] + ". " + string 
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string, "output": answerKey})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mmlu_pro_math.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_mmlu_pro_tweet():
    data = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
    def filter_math(example):
        return example["category"] == "math"
   
    math_data = data.filter(filter_math)
    train_data = list(math_data["test"])
    train, test= train_test_split(train_data, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx, string in enumerate(data["options"]):

                choices_string +=  "\n" + str(idx) + ". " + string 
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string, "output": data["answer_index"]})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mmlu_pro_math_tweet.json", "w")as f:
            json.dump(final_data, f, indent=4)


def construct_ethics():
    for subset in ["commonsense", "virtue", "justice"]:
        data = datasets.load_dataset("hendrycks/ethics", subset)
        labels = {
            0: "No",
            1: "Yes"
        }
        train = list(data["train"])
        valid = list(data["validation"])
        test = list(data["test"])
        def process_single(data_list):
            processed_data = []
            for data in data_list:
                if subset in ["justice", "virtue"]:
                    input = data["scenario"]
                else:
                    input = data["input"]
                processed_data.append({"input": input, "output": labels[data["label"]]})
            
            return processed_data
        
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/ethics_{subset}.json", "w")as f:
            json.dump(final_data, f, indent=4)

def construct_bbq():
    for subset in ["Age", "Religion"]:

        data = datasets.load_dataset("oskarvanderwal/bbq", subset)
        train_data = list(data["test"])
        train, test= train_test_split(train_data, test_size=0.3, random_state=42)
        test, valid = train_test_split(test, test_size=0.3, random_state=42)
        id2alphabet = {0: "A", 1: "B", 2: "C"}
        def process_single(data_list):
                processed_data = []
                for data in data_list:
                    choices_string = "\n" + id2alphabet[0] + ". " + data['ans0'] + "\n" + id2alphabet[1] + ". " + data['ans1'] + "\n" + id2alphabet[2] + ". " + data["ans2"]
                    processed_data.append({"input": "context: " + data["context"] + "\nquestion: " + data["question"] + "\nOptions:" + choices_string, "output": id2alphabet[data["label"]]})
                return processed_data
        
        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/bbq_{subset.lower()}.json", "w")as f:

            json.dump(final_data, f, indent=4)


def construct_commonsense_qa():
    data = datasets.load_dataset("commonsense_qa")
    train = list(data["train"])
    valid = list(data["validation"])
    train, test= train_test_split(train, test_size=0.3, random_state=42)
    
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for i in range(len(data["choices"]["label"])):
                choices_string += "\n" + data["choices"]["label"][i] + ". " + data["choices"]["text"][i]
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string, "output": data["answerKey"]})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/commonsense_qa.json", "w")as f:
        json.dump(final_data, f, indent=4)
 
def construct_mmlu():
    data = datasets.load_dataset("cais/mmlu", "high_school_psychology")
    train_data = list(data["test"])
    train, test= train_test_split(train_data, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    ID2ALPHABET = {i : chr(65+i) for i in range(26)}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx, string in enumerate(data["choices"]):
                if idx == data["answer"]:
                    answerKey = ID2ALPHABET[idx]
                choices_string +=  "\n" + ID2ALPHABET[idx] + ". " + string 
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string, "output": answerKey})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mmlu_high_school_psychology.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_mmlu_professional_law():
    data = datasets.load_dataset("cais/mmlu", "professional_law")
    train_data = list(data["test"])
    train, test= train_test_split(train_data, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    ID2ALPHABET = {i : chr(65+i) for i in range(26)}
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx, string in enumerate(data["choices"]):
                if idx == data["answer"]:
                    answerKey = ID2ALPHABET[idx]
                choices_string +=  "\n" + ID2ALPHABET[idx] + ". " + string 
            processed_data.append({"input": data["question"] + "\nOptions:" + choices_string, "output": answerKey})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/mmlu_professional_law.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_medqa():
    data = datasets.load_dataset("GBaker/MedQA-USMLE-4-options-hf")
    train_data = list(data["test"])
    train, test= train_test_split(train_data, test_size=0.3, random_state=42)
    test, valid = train_test_split(test, test_size=0.3, random_state=42)
    ID2ALPHABET = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    def process_single(data_list):
        processed_data = []
        for data in data_list:
            choices_string = ""
            for idx in range(4):
                if idx == data["label"]:
                    answerKey = ID2ALPHABET[idx]
                choices_string +=  "\n" + ID2ALPHABET[idx] + ". " + data[f"ending{idx}"]
            processed_data.append({"input": data["sent1"] + "\nOptions:" + choices_string, "output": answerKey})
            
        return processed_data
        
    final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
    with open(f"dataset_files/abstractive/medqa.json", "w")as f:
        json.dump(final_data, f, indent=4)

def construct_math():
    
    
    def read_data(dir_name):
        math_algebra = []

        for filename in os.listdir(dir_name):
        
            if(filename.endswith('.json')):
                d = json.load(open(dir_name + '/' + filename))
                math_algebra.append(d)
        
        complexity_dict = defaultdict(list)
        for d in math_algebra:
            complexity_dict[d['level']].append(d)
        return complexity_dict
    dir_name = 'dataset_files/raw/MATH/train/algebra'
    math_train = read_data(dir_name)
    math_test = read_data('dataset_files/raw/MATH/test/algebra')

    def process_single(data_list):
        processed_data = []
        for data in data_list:
            processed_data.append({"input": data["problem"], "output": data["solution"]})
        return processed_data
    


    for level in math_train.keys():
        train = math_train[level]
        train, valid= train_test_split(train, test_size=0.3, random_state=42)
        test = math_test[level]

        final_data = {
            "train": process_single(train),
            "valid": process_single(valid),
            "test": process_single(test)}
        with open(f"dataset_files/abstractive/math_level_{level[-1]}.json", "w")as f:
            json.dump(final_data, f, indent=4)
                           

def main(args):
    seed_everything(args.seed)
    if args.task == "math":
        construct_math()
    if args.task == "medqa":
        construct_medqa()
    if args.task == "mmlu":
        construct_mmlu()
    if args.task == "commonsense_qa":
        construct_commonsense_qa()
    if args.task == "mmlu_pro":
        construct_mmlu_pro()
    if args.task == "mmlu_pro_tweet":
        construct_mmlu_pro_tweet()
    if args.task == "glue_mnli":
        construct_glue(args)
    if args.task == "fr-en":
        construct_fr_en(args)
    if args.task == "gsm8k":
        construct_gsm8k(args)
    if args.task == "boolq":
        construct_boolq(args)
    if args.task == "bbq":
        construct_bbq()
    if args.task == "crows_pairs":
        construct_crows_pairs(args)
    if args.task == "financial_phrasebank":
        construct_financial_phrasebank(args)
    if args.task == "openbookqa":
        construct_openbookqa(args)
    if args.task == "superglue_copa":
        construct_superglue_copa(args)
    if args.task == "piqa":
        construct_piqa(args)
    if args.task == "arc_challenge":
        construct_arc_challenge()
    if args.task == "winogrande":
        construct_winogrande(args)
    if args.task == "ade_corpus_v2":
        construct_ade_classification()
    if args.task == "amazon_polarity":
        construct_amazon_polarity()
    if args.task == "anli":
        construct_anli()
    if args.task == "app_reviews":
        construct_app_reviews()
    if args.task == "aqua_rat":
        construct_aqua_rat()
    if args.task == "art":
        construct_art()
    if args.task == "blimp":
        construct_blimp()
    if args.task == "bbh":
        construct_bbh()
    if args.task == "ethics":
        construct_ethics()
    if args.task == "climate_fever":
        construct_climate_fever()
    if args.task == "codah":
        construct_codah()
    if args.task == "cos_e":
        construct_cos_e()
    if args.task == "cosmos_qa":
        construct_cosmos_qa()
    if args.task == "dbpedia_14":
        construct_dbpedia_14()
    if args.task == "definite_pronoun_resolution":
        construct_definite_pronoun_resolution()
    if args.task == "discorvery":
        construct_discovery()
    if args.task == "dream":
        construct_dream()
    if args.task == "emotion":
        construct_emotion()
    if args.task == "ethos":
        construct_ethos()
    if args.task == "glue_cola":
        construct_glue_cola()
    if args.task == "glue_mrpc":
        construct_glue_mrpc()
    if args.task == "glue_qnli":
        construct_glue_qnli()
    if args.task == "glue_qqp":
        construct_glue_qqp()
    if args.task == "glue_rte":
        construct_glue_rte()
    if args.task == "glue_sst2":
        construct_glue_sst2()
    if args.task == "glue_wnli":
        construct_glue_wnli()
    if args.task == "google_wellformed_query":
        construct_google_wellformed_query()
    if args.task == "hate_speech_offensive":
        construct_hate_speech_offensive()
    if args.task == "hate_speech18":
        construct_hate_speech18()
    if args.task == "hatexplain":
        construct_hatexplain()
    if args.task == "health_fact":
        construct_health_fact()
    if args.task == "hellaswag":
        construct_hellaswag()
    if args.task == "imdb":
        construct_imdb()
    if args.task == "lama":
        construct_lama()
    if args.task == "liar":
        construct_liar()
    if args.task == "math_qa":
        construct_math_qa()
    if args.task == "mc_taco":
        construct_mc_taco()
    if args.task == "medical_questions_pairs":
        construct_medical_questions_pairs()
    if args.task == "mocha":
        construct_mocha()
    if args.task == "numer_sense":
        construct_numer_sense()
    if args.task == "onestop_english":
        construct_onestop_english()
    if args.task == "paws":
        construct_paws()
    if args.task == "poem_sentiment":
        construct_poem_sentiment()
    if args.task == "proto_qa":
        construct_proto_qa()
    if args.task == "qasc":
        construct_qasc()
    if args.task == "quail":
        construct_quail()
    if args.task == "quarel":
        construct_quarel()
    if args.task == "quartz":
        construct_quartz()
    if args.task == "quoref":
        construct_quoref()
    if args.task == "race":
        construct_race()
    if args.task == "ropes":
        construct_ropes()
    if args.task == "rotten_tomatoes":
        construct_rotten_tomatoes()
    if args.task == "scicite":
        construct_scicite()
    if args.task == "sciq":
        construct_sciq()
    if args.task == "scitail":
        construct_scitail()
    if args.task == "search_qa":
        construct_search_qa()
    if args.task == "sick":
        construct_sick()
    if args.task == "sms_spam":
        construct_sms_spam()
    if args.task == "social_i_qa":
        construct_social_i_qa()
    if args.task == "squad":
        construct_squad()
    if args.task == "superglue_cb":
        construct_superglue_cb()
    if args.task == "superglue_multirc":
        construct_superglue_multirc()
    if args.task == "superglue_record":
        construct_superglue_record()
    if args.task == "superglue_rte":
        construct_superglue_rte()
    if args.task == "superglue_wic":
        construct_superglue_wic()
    if args.task == "superglue_wsc":
        construct_superglue_wsc()
    if args.task == "swag":
        construct_swag()
    if args.task == "tab_fact":
        construct_tab_fact()
    if args.task == "trec_finegrained":
        construct_trec_finegrained()
    if args.task == "trec":
        construct_trec()
    if args.task == "tweet_eval":
        construct_tweet_eval()
    if args.task == "wiki_auto":
        construct_wiki_auto()
    if args.task == "wiki_qa":
        construct_wiki_qa()
    if args.task == "wiqa":
        construct_wiqa()
    if args.task == "yahoo_answers_topics":
        construct_yahoo_answers_topics()
    if args.task == "yelp_polarity":
        construct_yelp_polarity()
    if args.task == "yelp_review_full":
        construct_yelp_review_full()
    if args.task == "antonym":
        construct_antonym()
    if args.task == "deepmind":
        construct_deepmind()
    if args.task == "mmlu_professional_law":
        construct_mmlu_professional_law()
    print(f"Succeed in processing {args.task}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="english-french")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)


