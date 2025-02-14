import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
import sys
import seaborn as sns
import string
import re
from collections import Counter
nshots=16
need_reeval = False
class ResultAnalyzer:
    def __init__(self, weight_fv=None, retrieve_method="retriever"):
        self.weight_fv = 2.0
        self.retrieve_method = retrieve_method
        self.datasets = {
            'nlu': ["superglue_rte", "superglue_wic",  "glue_qnli", "glue_sst2", "glue_mnli"], #,
            'reasoning': ["arc_challenge", "bbh_boolean_expressions", "bbh_date_understanding",
                          "bbh_reasoning_about_colored_objects", "bbh_temporal_sequences"],
            'knowledge': ["boolq", "commonsense_qa", "hellaswag", "openbookqa"],
            'math': ["math_qa", "mmlu_pro_math"],
            'safety': ["bbq_age", "crows_pairs", "ethics_justice", "ethics_commonsense"], #
            'unseen': ["glue_cola", "bbq_religion", "deepmind",
                       "mmlu_high_school_psychology", "bbh_logical_deduction_five_objects"]
        }

    @staticmethod
    def nested_dict():
        return defaultdict(lambda: defaultdict(list))

    @staticmethod
    def safe_mean(arr):
        """Calculate mean of numeric values, ignoring NaN"""
        return float(np.mean([x for x in arr if isinstance(x, (int, float)) and not np.isnan(x)]))

    def _get_filtered_files(self, results_dir, recall):
        """Get relevant files based on filtering criteria"""
        all_files = os.listdir(results_dir)
        file_list = [f for f in all_files if all(x in f for x in
                [self.retrieve_method, f"{self.weight_fv}fv", f"{recall}recall"])]
        return [f for f in file_list if "parsed" not in f]
    
    def get_answer_type(self, answer):
        if answer in string.ascii_uppercase:
            return 'capital'
        elif answer in ['True', 'False', 'Neither']:
            return 'true_false'
        elif answer in ['Yes', 'No']:
            return 'yes_no'
        elif answer in ['positive', 'negative']:
            return 'positive_negative'
        elif answer.isdigit():
            return 'number'
        return None
    
    def find_answer(self, text, answer):
        # arc chanllenge, bbh date understanding, bbh reasoning about colored, bbh temporal sequences,  bbq age, bbq religion, commonsenseqa, crows_pairs, deepmind, hellaswag, mathqa, mmlu high school psychology, MMLU pro math, openbook qa: ABC
        # bbh boolean, boolq, superglue rte: true/false
        # ethics commonsense, ethics justice, glue cola, glue qnli, superglue wic: Yes/No
        # glue mnli : True False Neither
        # glue sst2: positive / negative
        patterns = {
        'capital': r'([A-Z])(?:\.|:|<\|eot_id\|>)?\b',
        'true_false': r'\b(True|False|Neither)(?:\.|:|<\|eot_id\|>)?\b', 
        'number': r'(\d+)(?:\.|:|<\|eot_id\|>)?\b',
        'yes_no': r'\b(Yes|No)(?:\.|:|<\|eot_id\|>)?\b',
        'positive_negative': r'\b(positive|negative|Positive|Negative)(?:\.|:|<\|eot_id\|>)?\b'
        }
        # Strip the expected answer
        answer = answer.strip()
        pattern_type = self.get_answer_type(answer)
        if pattern_type == None: breakpoint()
        assert pattern_type is not None
        pattern = patterns[pattern_type]
        
        text = text.replace("X<|eot_id|>", "")
        if "answer" in text:
            p = text.split("answer")[-1]
            # breakpoint()
            
            for pattern in patterns.values():
                matches = re.findall(pattern, p)
                if len(matches) > 0:
                    pred_answer = matches[0].strip().replace("<|eot_id|>", "").strip(".").strip(":").lower()
                    if matches and  pred_answer== answer.lower():
                        return 1, pred_answer
                
        # Check each pattern
       
        matches = re.findall(pattern, text)
            
        if len(matches) > 0:
            pred_answer = matches[-1].strip().replace("<|eot_id|>", "").strip(".").strip(":").lower()
            if matches and  pred_answer== answer.lower():
                return 1, pred_answer
        try:
            return 0, pred_answer
        except:
            return 0, None
    
    def reeval(self, item_list, dataset = None):
        zs_acc_list = []
        intervene_acc_list = []
        pred_results = []
        for g in item_list["generation"]:
          
           zs_acc, pred_answer = self.find_answer(g["clean_output"], g["label"])
           
           zs_acc_list.append(zs_acc)
           intervene_acc, intervene_pred_answer = self.find_answer(g["intervene_output"], g["label"])
           intervene_acc_list.append(intervene_acc)
           item = {
               "clean_output": g["clean_output"],
               "clean_parsed_output": pred_answer,
                "zs_acc": zs_acc,
               "intervene_output": g["intervene_output"],
               "intervene_parsed_answer": intervene_pred_answer,
               "label": g["label"].strip(),
               "intervene_acc": intervene_acc
               
           }
           pred_results.append(item)

        return sum(zs_acc_list) / len(zs_acc_list), sum(intervene_acc_list)/len(intervene_acc_list), pred_results

    
    def _process_data_entry(self, data, all_results, category, dataset_name, results_dir):
        """Process a single data entry and update results dictionary"""
        if 'test_zero-shot_acc' not in data.keys():
            return
        metrics = all_results[category][dataset_name]

        # Process test accuracy
        if need_reeval:
            zs_acc, intervene_acc, pred_results = self.reeval(data, dataset_name)
        else:
            zs_acc, intervene_acc = float(data['test_zero-shot_acc']), (float(data['test_acc'][str(data["icl_best_layer"])])
                    if isinstance(data['test_acc'], dict)
                    else float(data['test_acc']))
        
        metrics["test_zero_acc"].append(zs_acc)

        metrics["test_intervene_acc"].append(intervene_acc)
        metrics["harm"].append(int(intervene_acc < zs_acc))
        metrics["retrieve_acc"].append(float(data['retrieve_acc']))

        # Process timing and length metrics
        metrics["0shot_time"].append(
            (float(data['clean_time'])/len(data["generation"])))
        metrics["intervene_time"].append(
            (float(data['intervene_time'])/len(data["generation"])))
        if "retrieve_time" in data.keys():
            metrics["retrieve_time"].append(
                (float(data["retrieve_time"])/len(data["generation"])))
        metrics["zs_length"].append(float(data['zs_lengths']))

        # Calculate chosen state numbers
        state_num = sum(len(item["chosen_states"])
                        for item in data["generation"])
        metrics["chosen_state_num"].append(
            float(state_num/len(data["generation"])))
        
        # Process BM25 results if available
    
        # print("valid" not in results_dir and "bm25_results" in data.keys() and data["bm25_results"] != {})
        if "valid" not in results_dir and "bm25_results" in data.keys() and data["bm25_results"] != {}:
            bm25_results = data["bm25_results"]
            if need_reeval:
                zs_acc, intervene_acc, bm25_pred_results = self.reeval(bm25_results)
            else:
                zs_acc, intervene_acc = float(bm25_results["acc"]), float(bm25_results["+tv_acc"])
            metrics["bm25_acc"].append(zs_acc)
            metrics["bm25_length"].append(float(bm25_results["length"]))
            metrics["bm25_time"].append(
                float(bm25_results["time"])/len(data["generation"]))
            metrics["bm25_tv"].append(intervene_acc)

        # Process nshots results
       
        if "nshots_results" in data.keys() and data["nshots_results"][f"{nshots}shot_results"] != {}:
            for key in data["nshots_results"].keys():
                if need_reeval:
                    zs_acc, intervene_acc, nshot_pred_answer = self.reeval((data["nshots_results"][key]))
                else:
                    zs_acc, intervene_acc = float(data["nshots_results"][key]["acc"]), float(data["nshots_results"][key]["+tv_acc"])
                all_results[category][dataset_name][f"{key.split('_')[0]}_acc"].append(zs_acc)
                all_results[category][dataset_name][f"{key.split('_')[0]}_tv"].append(intervene_acc)
                all_results[category][dataset_name][f"{key.split('_')[0]}_length"].append(
                    float(data["nshots_results"][key]["length"]))
                all_results[category][dataset_name][f"{key.split('_')[0]}_time"].append(
                    float(data["nshots_results"][key]["time"])/len(data["generation"]))
        
        used_states = []
        for g in data["generation"]:
            used_states+=g["chosen_states"]
        if need_reeval: return pred_results, used_states
        else: return None, used_states

    def _format_results_table(self, all_results):
        """Format results into a readable table"""
        headers = ["Category", "dataset"] + list(
            all_results[next(iter(all_results))][
                next(iter(all_results[next(iter(all_results))]))
            ].keys()
        )
        widths = [15, 30] + [15] * (len(headers) - 2)

        return headers, widths

    def _save_results_csv(self, headers, widths, all_results):
        """Save results to CSV file"""
        with open('results.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)

            all_averages = defaultdict(list)
            unseen_averages = defaultdict(list)

            for category, category_data in all_results.items():
                category_averages = defaultdict(list)

                for dataset, metrics in category_data.items():
                    values = [category, dataset] + [
                        self.safe_mean(
                            metrics[h]) * (100 if "time" not in h and "length" not in h and "state_num" not in h else 1)
                        for h in headers[2:]
                    ]
                    csvwriter.writerow(values)

                    for key, value in zip(headers[2:], values[2:]):
                        category_averages[key].append(value)

                if category != "unseen":
                    for key in headers[2:]:
                        all_averages[key].append(
                            self.safe_mean(category_averages[key]))
                else:
                    for key in headers[2:]:
                        unseen_averages[key].append(
                            self.safe_mean(category_averages[key]))

                avg_values = [f"{category} Average", ""] + [
                    self.safe_mean(category_averages[key]) for key in headers[2:]
                ]
                csvwriter.writerow(avg_values)
                csvwriter.writerow([])

            final_all_averages = {
                key: self.safe_mean(all_averages[key]) for key in headers[2:]
            }
            overall_avg_values = ["Overall Average", ""] + [
                final_all_averages[key] for key in headers[2:]
            ]
            csvwriter.writerow(overall_avg_values)

        return final_all_averages, all_averages, unseen_averages

    def _plot_layer_distribution(self, valid_best_layers):
        """Plot distribution of best layers"""
        bins = np.arange(0, 33, 1)
        plt.figure(figsize=(10, 6))
        plt.hist(valid_best_layers, bins=bins, edgecolor='black')
        plt.title('Best Layer Distribution', fontsize=16)
        plt.xlabel('Layer', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(bins)
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        plt.xlim(0, 32)
        plt.grid(True)
        plt.savefig("plots/best_layer_distribution_valid.png")
        plt.close()

    def analyze_results(self, results_dir, recall=0.8, tweet=False):
        """Main analysis function that processes all results"""
        all_results = defaultdict(self.nested_dict)
        valid_best_layers = []
        filtered_files = self._get_filtered_files(results_dir, recall)
        used_states_list = []
        # Process each dataset
        for category, dataset_list in self.datasets.items():
            for dataset_name in dataset_list:
                if tweet:
                    dataset_name = f"{dataset_name}_tweet"

                try:
                    file_name = next(
                        f for f in filtered_files if dataset_name in f)
                except StopIteration:
                    print(f"No file found for dataset: {dataset_name}")
                    continue
                
                with open(os.path.join(results_dir, file_name), "r") as f:
                    all_data = json.load(f)
                all_pred_results = []
                for data in all_data:
                    pred_results, used_states = self._process_data_entry(
                        data, all_results, category, dataset_name, results_dir)
                    all_pred_results.append(pred_results)
                    used_states_list += used_states
                    if 'valid_best_layer' in data:
                        valid_best_layers.append(data['valid_best_layer'])
                with open(os.path.join(results_dir, file_name.replace(".json", "parsed.json")), "w") as f:
                    json.dump(all_pred_results, f, indent=4)

        # Format and save results
        headers, widths = self._format_results_table(all_results)
        final_averages, all_averages, unseen_averages = self._save_results_csv(
            headers, widths, all_results
        )

        # Plot distribution
        # self._plot_layer_distribution(valid_best_layers)

        return final_averages, all_averages, unseen_averages, all_results, used_states_list

    def plot_recall_curve(self, results_dir, model="mamba"):
        """Plot accuracy vs recall curve"""
        recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
        accs = []

        for recall in recalls:
            eval_results, _, _, _ = self.analyze_results(
                results_dir, recall=recall)
            accs.append(eval_results['test_intervene_acc'])
            baseline = eval_results["test_zero_acc"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(recalls, accs, 'r-', linewidth=2, label='Intervene Accuracy')
        ax.axhline(baseline, linestyle='--', linewidth=2,
                   label='Zero-Shot Accuracy')

        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy vs. Recall', fontsize=14)
        ax.set_xlim(0.1, 1.1)
        ax.legend(loc='lower right', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=6, width=2)
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(f"plots/{model}_recall_vs_acc.png",
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_multiple_dirs(self, result_dirs):
        """分析多个结果目录并汇总结果"""
        category = ["nlu", "reasoning", "knowledge", "math", "safety"]
        unseen = ["glue_cola", "bbq_religion", "deepmind",
                  "mmlu_high_school_psychology", "bbh_logical_deduction_five_objects"]

        aggregate_results = self.nested_dict()
        unseen_results = self.nested_dict()
        # shots_results = defaultdict(list)
        chosen_states_nums = self.nested_dict()
        unseen_chosen_states_nums = self.nested_dict()
        time_dict = self.nested_dict()
        unseen_time_dict = self.nested_dict()
        used_states = []
        for results_dir in result_dirs:
            eval_results, all_averages, unseen_averages, all_results, used_states_list = self.analyze_results(
                results_dir)
            used_states += used_states_list

            # 处理chosen states numbers
            self._process_chosen_states(
                category, chosen_states_nums, all_averages)
            self._process_unseen_chosen_states(
                unseen, unseen_chosen_states_nums, all_results)

            # process time
            self._process_time_list(category, time_dict, all_averages)
            self._process_unseen_time_list(
                unseen, unseen_time_dict, all_results)
           
            
            self._process_results(eval_results, aggregate_results,
                                            category, all_averages, unseen_results, unseen_averages, all_results)    
           
        # 生成结果报告
        
        count_states = Counter(used_states)
        print(count_states)
        df = pd.DataFrame.from_dict(count_states, orient='index', columns=['count'])
        df.index = df.index.str.replace('_16shots', '')
        df = df.sort_values('count', ascending=True)

        plt.figure(figsize=(12, 8))
        plt.barh(df.index, df['count'])
        plt.xlabel('Usage Frequency', fontsize=16)
        plt.ylabel('Type of Task Vector', fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=12)

        plt.title('Usage Frequency Distribution of different task vectors', fontsize=20, pad=20)

        # 添加数值标签
        for i, v in enumerate(df['count']):
            plt.text(v, i, f' {v}', va='center')

        plt.tight_layout()
        plt.show()
        plt.savefig("used_states_distribution.png")
        self._generate_results_report(aggregate_results, unseen_results, chosen_states_nums,
                                      unseen_chosen_states_nums, time_dict, unseen_time_dict)

    def _process_time_list(self, category, time_dict, all_results):
        for i, c in enumerate(category):
            time_dict[c]["clean_time"].append(all_results["0shot_time"][i])
            time_dict[c]["intervene_time"].append(
                all_results["intervene_time"][i])
            if "retrieve_time" in all_results.keys():
                time_dict[c]["retrieve_time"].append(
                    all_results["retrieve_time"][i])
            if len(all_results["bm25_time"])>0 and len(all_results[f"{nshots}shot_time"])>0:
                time_dict[c]["bm25_infer_time"].append(all_results["bm25_time"][i])
                time_dict[c][f"{nshots}shot_time"].append(all_results[f"{nshots}shot_time"][i])

    def _process_unseen_time_list(self, unseen, unseen_time_dict, all_results):
        for i, c in enumerate(unseen):
            unseen_time_dict[c]["clean_time"] += all_results["unseen"][c]["0shot_time"]
            unseen_time_dict[c]["intervene_time"] += all_results["unseen"][c]["intervene_time"]
            unseen_time_dict[c]["retrieve_time"] += all_results["unseen"][c]["retrieve_time"]
            if len(all_results["unseen"][c]["bm25_time"]) > 0:
                unseen_time_dict[c]["bm25_infer_time"] += all_results["unseen"][c]["bm25_time"]
                unseen_time_dict[c][f"{nshots}shot_time"] += all_results["unseen"][c][f"{nshots}shot_time"]

    def _process_chosen_states(self, category, chosen_states_nums, all_averages):
        """处理chosen states的统计"""
        for i, c in enumerate(category):
            chosen_states_nums[c]["chosen_nums"].append(
                all_averages["chosen_state_num"][i])

    def _process_unseen_chosen_states(self, unseen, unseen_chosen_states_nums, all_results):
        """处理unseen数据集的chosen states统计"""
        for i, c in enumerate(unseen):
            unseen_chosen_states_nums[c]["chosen_nums"] += all_results['unseen'][c]["chosen_state_num"]

    def _process_results(self, eval_results, aggregate_results, category, 
                   all_averages, unseen_results, unseen_averages, all_results):
        """处理所有实验结果数据，包括普通项和tv项"""
        # 处理长度数据
        
        aggregate_results["Length"]["zs"].append(eval_results['zs_length'])
        aggregate_results["Length"]["Ours"].append(eval_results['zs_length'])
        aggregate_results["Length"]["bm25"].append(eval_results['bm25_length'])
        aggregate_results["Length"]["bm25_tv"].append(eval_results['bm25_length'])
        aggregate_results["Length"][f"{nshots}shot"].append(eval_results[f'{nshots}shot_length'])
        aggregate_results["Length"][f"{nshots}shot_tv"].append(eval_results[f'{nshots}shot_length'])
        
        # 处理每个类别的结果
        for i, c in enumerate(category):
            # 普通项
            aggregate_results[c]["bm25"].append(all_averages["bm25_acc"][i])
            aggregate_results[c]["zs"].append(all_averages['test_zero_acc'][i])
            aggregate_results[c]["Ours"].append(all_averages['test_intervene_acc'][i])
            
            # tv项
            aggregate_results[c]["bm25_tv"].append(all_averages["bm25_tv"][i])
            aggregate_results[c][f"{nshots}shot"].append(all_averages[f"{nshots}shot_acc"][i])
            aggregate_results[c][f"{nshots}shot_tv"].append(all_averages[f"{nshots}shot_tv"][i])

        # 处理unseen结果
        for i, u in enumerate(self.datasets['unseen']):
            # Length数据
            unseen_results["zs"]["Length"].append(
                sum(all_results['unseen'][u]["zs_length"]) /
                len(all_results['unseen'][u]["zs_length"])
            )
            unseen_results["Ours"]["Length"].append(
                sum(all_results['unseen'][u]["zs_length"]) /
                len(all_results['unseen'][u]["zs_length"])
            )
            unseen_results["bm25"]["Length"].append(unseen_averages["bm25_length"][0])
            unseen_results["bm25_tv"]["Length"].append(unseen_averages["bm25_length"][0])
            unseen_results[f"{nshots}shot"]["Length"].append(unseen_averages[f"{nshots}shot_length"][0])
            unseen_results[f"{nshots}shot_tv"]["Length"].append(unseen_averages[f"{nshots}shot_length"][0])
            
            # 准确率数据
            # 普通项
            unseen_results["zs"][u].append(
                sum(all_results['unseen'][u]["test_zero_acc"]) /
                len(all_results['unseen'][u]["test_zero_acc"]) * 100
            )
            unseen_results["Ours"][u].append(
                sum(all_results['unseen'][u]["test_intervene_acc"]) /
                len(all_results['unseen'][u]["test_intervene_acc"]) * 100
            )
            unseen_results["bm25"][u].append(
                sum(all_results['unseen'][u]["bm25_acc"]) /
                len(all_results['unseen'][u]["bm25_acc"]) * 100
            )
            # tv项
            unseen_results["bm25_tv"][u].append(
                sum(all_results['unseen'][u]["bm25_tv"]) /
                len(all_results['unseen'][u]["bm25_tv"]) * 100
            )
            
            unseen_results[f"{nshots}shot"][u].append(
                sum(all_results['unseen'][u][f"{nshots}shot_acc"]) /
                len(all_results['unseen'][u][f"{nshots}shot_acc"]) * 100
            )
            # tv项
            unseen_results[f"{nshots}shot_tv"][u].append(
                sum(all_results['unseen'][u][f"{nshots}shot_tv"]) /
                len(all_results['unseen'][u][f"{nshots}shot_tv"]) * 100
            )

        # 处理平均值
        # 普通项
        aggregate_results["avg"]["bm25"].append(eval_results['bm25_acc'])
        aggregate_results["avg"]['zs'].append(eval_results['test_zero_acc'])
        aggregate_results["avg"]["Ours"].append(eval_results['test_intervene_acc'])
        
        # tv项
        aggregate_results["avg"]["bm25_tv"].append(eval_results['bm25_tv'])
        aggregate_results["avg"][f"{nshots}shot"].append(eval_results[f'{nshots}shot_acc'])
        aggregate_results["avg"][f"{nshots}shot_tv"].append(eval_results[f'{nshots}shot_tv'])
        
        # unseen平均值
        # 普通项
        unseen_results['zs']['avg'].append(unseen_averages['test_zero_acc'][0])
        unseen_results['Ours']['avg'].append(unseen_averages['test_intervene_acc'][0])
        unseen_results['bm25']['avg'].append(unseen_averages['bm25_acc'][0])
        # tv项
        unseen_results['bm25_tv']['avg'].append(unseen_averages['bm25_tv'][0])
       
        unseen_results[f'{nshots}shot']['avg'].append(unseen_averages[f'{nshots}shot_acc'][0])
        # tv项
        unseen_results[f'{nshots}shot_tv']['avg'].append(unseen_averages[f'{nshots}shot_tv'][0])

    def _generate_results_report(self, aggregate_results, unseen_results,
                                 chosen_states_nums, unseen_chosen_states_nums,
                                 time_dict, unseen_time_dict):
        """生成结果报告"""
        # 创建DataFrame并格式化
        chosen_num_df = pd.DataFrame(chosen_states_nums)
        unseen_chosen_num_df = pd.DataFrame(unseen_chosen_states_nums)
        df = pd.DataFrame(aggregate_results)
        unseen_df = pd.DataFrame(unseen_results).T
        time_df = pd.DataFrame(time_dict).T
        unseen_time_df = pd.DataFrame(unseen_time_dict).T

        # 格式化函数
        def format_cell(cell, digit_num=1):
            if isinstance(cell, list):
                mean = np.mean(cell)
                std = np.std(cell)
                return f"{mean:.{digit_num}f} ± {std:.{digit_num}f}"
            return str(cell)

        # 应用格式化
        df_formatted = df.applymap(format_cell)
        unseen_df_formatted = unseen_df.applymap(format_cell)
        chosen_num_df_formatted = chosen_num_df.applymap(format_cell)
        unseen_num_df_formatted = unseen_chosen_num_df.applymap(format_cell)
        time_df_formatted = time_df.applymap(format_cell, digit_num=3)
        unseen_time_df_formatted = unseen_time_df.applymap(
            format_cell, digit_num=3)

        # 打印结果
        print(unseen_num_df_formatted)
        print("\n")
        print(chosen_num_df_formatted)
        print("\n")
        print(time_df_formatted)
        print("\n")
        print(unseen_time_df_formatted)
        print("\n")
        print(df_formatted)
        print("Unseen \n")
        print(unseen_df_formatted)

        # 保存结果
        df_formatted.to_csv("csvs/results.csv")
        unseen_df_formatted.to_csv("csvs/unseen_results.csv")
        unseen_num_df_formatted.to_csv("csvs/unseen_num.csv")
        chosen_num_df_formatted.to_csv("csvs/chosen_num.csv")
        time_df_formatted.to_csv("csvs/time.csv")
        unseen_time_df_formatted.to_csv("csvs/unseen_time.csv")

        # 生成性能分布图
        # self._plot_performance_distribution(aggregate_results, model)

    def _plot_performance_distribution(self, aggregate_results, model):
        """绘制性能分布图"""
        category = ["math", "nlu", "reasoning", "knowledge", "safety"]
        pre_performance_list = []
        post_performance_list = []

        for c in category:
            pre_performance_list.append(sum(aggregate_results[c]['zs'])/3)
            post_performance_list.append(sum(aggregate_results[c]['Ours'])/3)

        self._create_performance_plot(
            pre_performance_list, post_performance_list, model)

    def _create_performance_plot(self, pre_performance_list, post_performance_list, model):
        """创建性能对比图"""
        category = ['Math', 'NLU', 'Reasoning', 'Knowledge', 'Safety']
        bar_width = 0.4
        r1 = np.arange(len(category))
        r2 = [x + bar_width for x in r1]

        fig, ax = plt.subplots(figsize=(23, 23))
        ax.bar(r1, pre_performance_list, color='skyblue',
               width=bar_width, label='Zero-shot')
        ax.bar(r2, post_performance_list, color='slateblue',
               width=bar_width, label='ELICIT')

        self._format_performance_plot(ax, category, bar_width,
                                      pre_performance_list, post_performance_list)
        plt.savefig(f"plots/performance_distribution_{model}.png")
        plt.close()

    def _format_performance_plot(self, ax, category, bar_width,
                                 pre_performance_list, post_performance_list):
        """格式化性能图表"""
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=52)
        plt.xticks(fontsize=52)
        plt.yticks(fontsize=52)
        ax.set_xticks([r + bar_width/2 for r in range(len(category))])
        ax.set_xticklabels([c.capitalize() for c in category])
        ax.legend(fontsize=52)

        self._add_value_labels(ax)
        self._highlight_math_difference(ax, bar_width,
                                        pre_performance_list[0], post_performance_list[0])
        plt.tight_layout()

    @staticmethod
    def _add_value_labels(ax, spacing=5):
        """为柱状图添加数值标签"""
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing if y_value >= 0 else -spacing
            va = 'bottom' if y_value >= 0 else 'top'

            ax.annotate(
                f"{y_value:.1f}",
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center',
                va=va,
                fontsize=45
            )

    @staticmethod
    def _highlight_math_difference(ax, bar_width, math_pre, math_post):
        """突出显示数学类别的差异"""
        math_diff = math_post - math_pre
        rect = plt.Rectangle(
            (bar_width/2, math_pre),
            bar_width,
            math_diff,
            fill=False,
            edgecolor='red',
            linestyle='--',
            linewidth=10
        )
        ax.add_patch(rect)


def main():
    """主函数"""
    # # weight fv
    # model = sys.argv[2]
    # suffix = sys.argv[3]

    analyzer = ResultAnalyzer()
    model = "mistral"
    result_dirs = [
        # f"results/sep28_natural_1layer_local_mistral_math_seed100",
        # f"results/sep28_natural_1layer_local_mistral_math_seed10",
        # f"results/sep28_natural_1layer_local_mistral_math_seed42"
        # "rebuttal_results/nov13_natural_1layer_local_pythia6.9b_seed42",
        # "rebuttal_results/nov13_natural_1layer_local_pythia6.9b_seed100",
        # "rebuttal_results/nov13_natural_1layer_local_pythia6.9b_seed10"
        # "rebuttal_results/nov18_math_force_seed100",
        # "rebuttal_results/nov18_math_force_seed10",
        # "rebuttal_results/nov18_math_force_seed42",
        # "rebuttal_results/nov15_diversity_llama3_seed42",
        # "rebuttal_results/nov15_diversity_llama3_seed10",
        # "rebuttal_results/nov15_diversity_llama3_seed100",
        # "rebuttal_results/nov17_natural_1layer_local_pythia12b_seed10",
        # "rebuttal_results/nov17_natural_1layer_local_pythia12b_seed100",
        # "rebuttal_results/nov17_natural_1layer_local_pythia12b_seed42",
        # "results/sep24_natural_1layer_local_seed10",
        # "results/sep24_natural_1layer_local_seed100",
        # "results/sep24_natural_1layer_local_seed42"
        # "results/sep26_natural_1layer_local_seed10",
        # "results/sep26_natural_1layer_local_seed100",
        # "results/sep26_natural_1layer_local_seed42"
        # "rebuttal_results/nov_18_pythia-6.9b_group_k1_seed42",
        # "rebuttal_results/nov_18_pythia-6.9b_group_k2_seed10",
        # "rebuttal_results/nov_18_pythia-6.9b_group_k2_seed100"
        # "rebuttal_results/nov17_diversity_llama3_seed10",
        # "rebuttal_results/nov17_diversity_llama3_seed100",
        # "rebuttal_results/nov17_diversity_llama3_seed42"
        # "rebuttal_results/nov_17_llama3_70b_seed42",
        # "rebuttal_results/nov_17_llama3_70b_seed10",
        # "rebuttal_results/nov_17_llama3_70b_seed100"
        # "rebuttal_results/nov_18_pythia-6.9b_group_k2_seed10",
        # "rebuttal_results/nov_18_pythia-6.9b_group_k2_seed42",
        # "rebuttal_results/nov_18_pythia-6.9b_group_k2_seed100"
        # "rebuttal_results/nov18_llama_natural_all_layer"
        # "rebuttal_results/nov19_instruct_seed42"
        # "rebuttal_results/nov18_llama_seed42"
        # "rebuttal_results/nov19_instruct_generation_seed42"
        # "rebuttal_results/nov21_natural_1layer_local_pythia6.9b_seed42",
        # "iclr_results/feb5_natural_1layer_local_llama3_seed10",
        # "iclr_results/feb5_natural_1layer_local_llama3_seed100",
        # "iclr_results/feb11_natural_1layer_local_mamba_qa_seed10",
        # "iclr_results/feb11_natural_1layer_local_mamba_qa_seed100",
        f"iclr_feb_results/feb13_natural_1layer_local_{model}_seed42",
        f"iclr_feb_results/feb13_natural_1layer_local_{model}_seed10",
        f"iclr_feb_results/feb13_natural_1layer_local_{model}_seed100",
        # "iclr_results/feb13_natural_1layer_local_pythia2.8b_qa_seed10",
        # "iclr_results/feb13_natural_1layer_local_mistral_qa_seed100"
        # "iclr_test_results/feb11_natural_1layer_local_llama3_seed42"



    ]

    analyzer.analyze_multiple_dirs(result_dirs)


if __name__ == "__main__":
    main()
