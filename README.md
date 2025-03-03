<h2 align="center"> <a href="https://arxiv.org/pdf/2410.09343">ELICIT: LLM Augmentation via External In-Context Capability</a></h2>
<h5 align="center"> If our project helps you, please give us a star ⭐ and cite our <a href="#citation">paper</a>!</h2>
<h5 align="center">

## Overview
![overview](./assets/overview.png#pic_center=80x80)
We propose ELICIT, improving language model performance by:
1. **Building a Capability Library**: A collection of task-specific capabilities from in-domain datasets.
2. **Dynamic Capability Elicitation**: Using a trained retriever to dynamically select relevant capabilities for aribitary query.


## Experiment Codes
You can find our results [here](https://drive.google.com/drive/folders/1cWvWc4uSSvEnhzs03EtqJGptBGUitpcf?usp=drive_link).
### Building the Capability Library

The Capability Library is constructed using validation sets from in-domain tasks with 16-shot examples. Follow these steps:

1. Prepare datasets:
   ```bash
   python process_data.py --task <task_name>
   ```
   Example:
   ```bash
   python process_data.py --task arc_challenge
   ```

2. Collect libraries for different models:
   ```bash
   ./scripts/collect_tv.sh
   ```

### Dynamic Capability Elicitation

#### Train the Retriever
We provide a balanced dataset of 10,000 samples to train the retriever:
```bash
python train_retriever.py --output_model prompt_classifier_model.pth
```

#### Evaluate ELICIT
Once the retriever is trained, you can evaluate ELICIT using the collected library:
```bash
./scripts/eval_elicit.sh
```
To analyze results:
1. Update the evaluation directory in `analysis_results.py`.
2. Run the script:
   ```bash
   python analysis_results.py
   ```

## Citation
If you find this project helpful, please consider citing our work:

```bibtex
@inproceedings{
   wang2025elicit,
   title={{ELICIT}: {LLM} Augmentation Via External In-context Capability},
   author={Futing Wang and Jianhao Yan and Yue Zhang and Tao Lin},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=CI4sCBMXjP}
}
```