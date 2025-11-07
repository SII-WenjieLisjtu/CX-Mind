<div align="center"> <img src="docs/logo.png" alt="logo" height="150"> <h1 style="font-size: 16px; font-weight: bold;"> CX-Mind: A Pioneering Multimodal Large Language Model for Interleaved Reasoning in Chest X-ray via Curriculum-Guided Reinforcement Learning </h1> <br> <a href="https://arxiv.org/abs/2508.03733"> <img src="https://img.shields.io/badge/ArXiv-CXMind-brown?logo=arxiv" alt="Paper"> </a> <a href="https://huggingface.co/SII-JasperLi77/CX-Mind"> <img src="https://img.shields.io/badge/🤗 huggingface-Model-purple" alt="checkpoint"> </a> </div>

## CX-Mind

<figure style="margin:16px auto; text-align:center;">
  <img src="docs/overview.png"
       style="max-width:100%; width:900px; height:auto; border-radius:12px; box-shadow:0 4px 24px rgba(0,0,0,.08);" />
  <figcaption style="font-size:14px; color:#666; margin-top:8px;">
  </figcaption>
</figure>

Key insights:
- Large chest X-ray dataset with over 2 million entries across 23 datasets.
- Novel training strategy boosts medical knowledge and reasoning in models.
<figure style="margin:16px auto; text-align:center;">
  <img src="docs/Figure 2.png"
       style="max-width:100%; width:900px; height:auto; border-radius:12px; box-shadow:0 4px 24px rgba(0,0,0,.08);" />
  <figcaption style="font-size:14px; color:#666; margin-top:8px;">
  </figcaption>
</figure>
- First interleaved reasoning approach for clear medical model interpretability.
- CX-Mind surpasses top medical reasoning models in extensive benchmark tests.
- Real-world clinical dataset validates CX-Mind’s utility with expert reviews.



##  Quick Start


### Environment Setup
```bash
# Follow the Easy-R1 official installation procedure
git clone https://github.com/SII-zyj/CX-Mind-private.git
cd CX-Mind
pip install -e .
```
### Data Access & Privacy

This project uses publicly available medical imaging datasets such as **MIMIC-CXR** and **CheXpert**.  
Due to licensing and patient-privacy restrictions, **we do not redistribute any raw images, reports, or derived data**.  
Instead, we willprovide **data preprocessing scripts** that allow users to reproduce our dataset once they have obtained official access credentials.

You must first apply for and download each dataset from the original source:

- **MIMIC-CXR / MIMIC-CXR-JPG** → [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/) (credentialed access required)  
- **CheXpert** → [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/) (requires agreement to license terms)


### Start Training
We use Qwen-2.5-VL-7B-Instruct as our foundation model for RL training. 

We recommend using no less than 4 GPUs for close-ended training, and no less than 8 GPUs for open-ended training. 

step 1: need to process the two-stage (closed-ended and open-ended) RL datasets obtained separately.
Prepare data before starting training. We will provide our scripts to process training dataset.

Step 1: Start stage 1 training (closed-ended)
```bash
# First, build a ray cluster for all of the training nodes.
bash ./examples/qwen2_5_vl_7b_closeQA+think+ans_grpo.sh
```

Step 2: Start stage 2 training (open-ended)
```bash
# Second, we need to train the open-ended RL dataset using the pre-trained closed-ended model.
bash ./examples/qwen2_5_vl_7b_openQA+think+ans_grpo.sh
```

### Evaluation
We provide evaluation scripts to evaluate the performance of the trained models.
```bash
# For closed-ended evaluation, run the following command:
./eval/eval_close.py
# For open-ended evaluation, run the following command:
./eval/eval_open.py
```

**Important**:



<h2>Star Chart</h2>
<p align="center">
  <a href="https://star-history.com/#SII-WenjieLisjtu/CX-Mind&Date">
    <img src="https://api.star-history.com/svg?repos=SII-WenjieLisjtu/CX-Mind&type=Date" alt="Star History Chart" width="800">
  </a>
</p>

## Licence

This project is released under [Apache licence](./LICENSE).

## Citation

```
@article{li2025cx,
  title={CX-Mind: A Pioneering Multimodal Large Language Model for Interleaved Reasoning in Chest X-ray via Curriculum-Guided Reinforcement Learning},
  author={Li, Wenjie and Zhang, Yujie and Sun, Haoran and Li, Yueqi and Zhang, Fanrui and Xu, Mengzhe and Clausich, Victoria Borja and Mellin, Sade and Yang, Renhao and Wang, Chenrun and others},
  journal={arXiv preprint arXiv:2508.03733},
  year={2025}
}
```
