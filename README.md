
<div align="center">
    <h1 align="center"> GCPO: Group Critical-token Policy Optimization 
      for Autoregressive Image Generation
    </h1>
</div>

### 🗒️ TODO

- [ ] Release All Checkpoints
- [x] Release code

## 🤗 Model List
| Model          | Preference Alignment | GenEval          |
|:--------------:|:-------------------:|:----------------:|
| LlamaGen-T2I   | [🤗HPS](https://huggingface.co/zghhui/LlamaGen-T2I-GCPO)            |                  |
| Janus-Pro 1B   | [🤗HPS](https://huggingface.co/zghhui/Janus-Pro-1B-GCPO-HPS)            | [🤗Geneval](https://huggingface.co/zghhui/Janus-Pro-1B-GCPO-Geneval)     |
| Janus-Pro 7B   | [🤗HPS](https://huggingface.co/zghhui/Janus-Pro-7B-GCPO-HPS)            | [🤗Geneval](https://huggingface.co/zghhui/Janus-Pro-7B-GCPO-Geneval)     |



## 🔧 Environment SetUp
#### 1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/zghhui/GCPO.git
cd GCPO
```

#### 2. Install the training package:
We provide training codes for **LlamaGen** and **Janus-Pro**, and recommend installing the environments for each.

**For LlamaGen:**

```bash
conda create -n gcpo_llamagen python=3.10
conda activate gcpo_llamagen
pip install -r llamaGen/requirements.txt
```

**For Janus-Pro:**

```bash
conda create -n gcpo_janus python=3.10
conda activate gcpo_janus
pip install -r janus/requirements.txt
```

#### 3. Download Models
**For LlamaGen:**

```bash
huggingface-cli download zghhui/LlamaGen-T2I
huggingface-cli download google/flan-t5-xl
wget https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt
```

**For Janus-Pro:**

```bash
huggingface-cli download deepseek-ai/Janus-Pro-1B
huggingface-cli download deepseek-ai/Janus-Pro-7B
```

**For HPS Reward:**

```bash
huggingface-cli xswu/HPSv2
```

**For Geneval Reward**: 

- Please according to the instructions in [Flow-GRPO](https://github.com/yifan123/flow_grpo?tab=readme-ov-file) and [reward-server](https://github.com/yifan123/reward-server).

## 🚀 Training GCPO

### LlamaGen

```bash
cd llamaGen/src
bash scripts/rl_gcpo_hps.sh
```
Note:
- Remember to modify the t5_model path in `gcpo/llamaGen/simpar/model/llama_model.py` (line 1244)

### Janus-Pro

```bash
cd janus/src
bash scripts/run_gcpo_hps.sh
bash scripts/run_gcpo_geneval.sh
```
Notes:
- Please run geneval server before running geneval reward. The reward function is located in `utils/reward_geneval.py`, and the IP of server can be modified here.


## 💫 Inference
### LlamaGen
```bash
cd llamaGen
bash scripts/inference.sh
```

### Janus-Pro
```bash
cd janus/src
bash scripts/inference.sh
```



## 📧 Contact
If you have any comments or questions, please open a new issue.


## 🤗 Acknowledgments
Our training code is based on [T2I-R1](https://github.com/CaraJ7/T2I-R1), [SimpleAR](https://github.com/wdrink/SimpleAR), and [Flow-GRPO](https://github.com/yifan123/flow_grpo).

Thanks to all the contributors!
