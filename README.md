## SimCSE: Simple Contrastive Learning of Sentence Embeddings

This repository contains the code and pre-trained models for our paper [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821).

**************************** **Updates** ****************************

<!-- Thanks for your interest in our repo! -->

<!-- Probably you will think this as another *"empty"* repo of a preprint paper 🥱.
Wait a minute! The authors are working day and night 💪, to make the code and models available, so you can explore our state-of-the-art sentence embeddings.
We anticipate the code will be out * **in one week** *. -->

<!-- * 4/26: SimCSE is now on [Gradio Web Demo](https://gradio.app/g/AK391/SimCSE) (Thanks [@AK391](https://github.com/AK391)!). Try it out! -->
* 8/31: Our paper has been accepted to EMNLP! Please check out our [updated paper](https://arxiv.org/pdf/2104.08821.pdf) (with updated numbers and baselines). 
* 5/12: We updated our [unsupervised models](#model-list) with new hyperparameters and better performance.
* 5/10: We released our [sentence embedding tool](#getting-started) and [demo code](./demo).
* 4/23: We released our [training code](#training).
* 4/20: We released our [model checkpoints](#use-our-models-out-of-the-box) and [evaluation code](#evaluation).
* 4/18: We released [our paper](https://arxiv.org/pdf/2104.08821.pdf). Check it out!


## Quick Links

  - [Overview](#overview)
  - [Getting Started](#getting-started)
  - [Model List](#model-list)
  - [Use SimCSE with Huggingface](#use-simcse-with-huggingface)
  - [Train SimCSE](#train-simcse)
    - [Requirements](#requirements)
    - [Evaluation](#evaluation)
    - [Training](#training)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [SimCSE Elsewhere](#simcse-elsewhere)

## Overview
我们提出了一个简单的对比性学习框架，该框架同时适用于无标签和有标签的数据。无监督的SimCSE只需要一个输入句子，并在一个对比学习框架中预测自己，只用标准的dropout作为噪音。我们的有监督的SimCSE通过使用 "entailment "对作为正例，"contradiction "对作为困难性负样本，将来自NLI数据集的标注对纳入对比学习。下图是我们模型的说明。

![](figure/model.png)

## Getting Started
我们在SimCSE模型的基础上提供了一个易于使用的句子嵌入工具（详细用法见我们的[Wiki]（https://gi thub.com/princeton-nlp/SimCSE/wiki））。要使用该工具，首先要从PyPI安装`simcse`包
```bash
pip install simcse
```

Or directly install it from our code
```bash
python setup.py install
```
请注意，如果您想启用GPU编码，您应该安装支持CUDA的正确版本的PyTorch。请参阅[PyTorch官方网站](https://pytorch.org)了解相关说明。
安装完软件包后，你可以通过两行代码加载我们的模型
```python
from simcse import SimCSE
model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
```
See [model list](#model-list) for a full list of available models. 
然后你可以使用我们的模型来**将句子编码为嵌入**。
```python
embeddings = model.encode("A woman is reading.")
```

**计算两组句子之间的余弦相似度**。
```python
sentences_a = ['A woman is reading.', 'A man is playing a guitar.']
sentences_b = ['He plays guitar.', 'A woman is making a photo.']
similarities = model.similarity(sentences_a, sentences_b)
```

或者为一组句子建立索引并在其中**搜索
```python
sentences = ['A woman is reading.', 'A man is playing a guitar.']
model.build_index(sentences)
results = model.search("He plays guitar.")
```

我们也支持[faiss](https://gi thub.com/facebookresearch/faiss)，一个高效的相似度搜索库。只要按照这里的[说明](https://github.com/princeton-nlp/SimCSE/wiki/Installation)安装软件包，`simcse`将自动使用`faiss`进行高效搜索。

**警告**。我们发现`faiss`不能很好地支持Nvidia AMPERE GPU（3090和A100）。在这种情况下，你应该改用其他GPU或安装CPU版本的`faiss`包。
我们还提供了一个易于构建的[演示网站](./demo)来展示SimCSE如何用于句子检索。该代码基于[DensePhrases](https://arxiv.org/abs/2012.12624)' 
[repo](https://github.com/princeton-nlp/DensePhrases)和[demo](http://densephrases.korea.ac.kr)（非常感谢DensePhrases的作者）。

## Model List
我们已经发布的模型列举如下。你可以通过使用`simcse`包或使用[HuggingFace's Transformers]（https://github.com/huggingface/transformers）导入这些模型。

|              Model              | Avg. STS |
|:-------------------------------|:--------:|
|  [princeton-nlp/unsup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-base-uncased) |   76.25 |
| [princeton-nlp/unsup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/unsup-simcse-bert-large-uncased) |   78.41  |
|    [princeton-nlp/unsup-simcse-roberta-base](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-base)    |   76.57  |
|    [princeton-nlp/unsup-simcse-roberta-large](https://huggingface.co/princeton-nlp/unsup-simcse-roberta-large)   |   78.90  |
|   [princeton-nlp/sup-simcse-bert-base-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-base-uncased)  |   81.57  |
|  [princeton-nlp/sup-simcse-bert-large-uncased](https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased)  |   82.21  |
|     [princeton-nlp/sup-simcse-roberta-base](https://huggingface.co/princeton-nlp/sup-simcse-roberta-base)     |   82.52  |
|     [princeton-nlp/sup-simcse-roberta-large](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)    |   83.76  |

请注意，在采用了一组新的超参数（关于超参数，见[训练](#训练)部分）后，结果比我们在当前版本的论文中报告的要好一点。
**命名规则**。`unsup`和`sup`分别代表 "无监督"（在维基百科语料库上训练）和 "有监督"（在NLI数据集中训练）。

## Use SimCSE with Huggingface
除了使用我们提供的句子嵌入工具，你也可以用HuggingFace的 "transformer "轻松导入我们的模型。

```python
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

# Tokenize input texts
texts = [
    "There's a kid on a skateboard.",
    "A kid is skateboarding.",
    "A kid is inside the house."
]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Get the embeddings
with torch.no_grad():
    embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

# Calculate cosine similarities
# Cosine similarities are in [-1, 1]. Higher means more similar
cosine_sim_0_1 = 1 - cosine(embeddings[0], embeddings[1])
cosine_sim_0_2 = 1 - cosine(embeddings[0], embeddings[2])

print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[1], cosine_sim_0_1))
print("Cosine similarity between \"%s\" and \"%s\" is: %.3f" % (texts[0], texts[2], cosine_sim_0_2))
```

如果你在通过HuggingFace的API直接加载模型时遇到任何问题，你也可以从上表手动下载模型，并使用`model = AutoModel.from_pretrained({PATH TO THE DOWNLOAD MODEL})`。

## Train SimCSE
在下一节中，我们将描述如何通过使用我们的代码来训练一个SimCSE模型。

### Requirements
首先，按照[官方网站](https://pytorch.org)的说明安装PyTorch。为了忠实地再现我们的结果，
请使用与您的平台/CUDA版本相对应的正确的`1.7.1`版本。高于`1.7.1`的PyTorch版本也应该可以工作。
例如，如果您使用Linux和**CUDA11**（[如何检查CUDA版本](https://varhowto.com/check-cuda-version/)），请通过以下命令安装PyTorch。

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

如果您使用的是**CUDA**`<11`或**CPU**，请通过以下命令安装PyTorch。

```bash
pip install torch==1.7.1
```


然后运行下面的脚本来安装其余的依赖项。

```bash
pip install -r requirements.txt
```

### Evaluation
我们对句子嵌入的评估代码是基于[SentEval](https://github.com/facebookresearch/SentEval)的修改版。
它在语义文本相似性（STS）任务和下游迁移任务中对句子嵌入进行评估。对于STS任务，我们的评估采用 "全部 "设置，并报告Spearman的相关度。评估细节见[我们的论文](https://arxiv.org/pdf/2104.08821.pdf)（附录B）。

在评估之前，请通过运行以下程序下载评估数据集
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```
然后回到根目录，你可以使用我们的评估代码评估任何基于`transformers`的预训练模型。比如说。

```bash
python evaluation.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --pooler cls \
    --task_set sts \
    --mode test
```
预计它将以表格的形式输出结果。

```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 75.30 | 84.67 | 80.19 | 85.40 | 80.82 |    84.26     |      80.39      | 81.58 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```
评估脚本的参数如下。

* `--model_name_or_path`: 基于 "transformer "的预训练checkpoint的名称或路径。你可以直接使用上表中的模型, e.g., `princeton-nlp/sup-simcse-bert-base-uncased`.
* `--pooler`: 池化方法。现在我们支持
    * `cls` (default): 使用`[CLS]`token的表示。在表示之后应用一个线性+激活层（它在标准的BERT实现中）。如果你使用**监督的SimCSE**，你应该使用这个选项。
    * `cls_before_pooler`: 使用`[CLS]`token的表示法，没有额外的线性+激活。如果你使用**无监督的SimCSE**，你应该采取这个选项。
    * `avg`: 最后一层的平均嵌入值。如果你使用SBERT/RoBERTa的checkpoint（[论文](https://arxiv.org/abs/1908.10084)），你应该使用这个选项。
    * `avg_top2`: 最后两层的平均嵌入量。
    * `avg_first_last`: 第一层和最后一层的平均嵌入值。如果你使用vanilla BERT或RoBERTa，这个效果最好。
* `--mode`: 评估模式
    * `test` (默认)。默认的测试模式。为了忠实地再现我们的结果，你应该使用这个选项。
    * `dev`: 报告开发集的结果。注意，在STS任务中，只有`STS-B`和`SICK-R`有开发集，所以我们只报告它们的数字。它还采取快速模式进行迁移任务，所以运行时间比`测试`模式短得多（尽管数字略低）。
    * `fasttest`: 它与`test`相同，但有一个快速模式，所以运行时间更短，但报告的数字可能更低（只针对迁移任务）。
* `--task_set`: 对哪一组任务进行评估（如果设置，它将覆盖`--tasks`）。
    * `sts` (默认): 在STS任务上进行评估，包括`STS 12~16`、`STS-B`和`SICK-R`。这是评估句子嵌入质量的最常用的任务集。
    * `transfer`: 对迁移任务进行评估。
    * `full`: 对STS和迁移任务进行评估。
    * `na`: 通过`--tasks`手动设置任务。
* `--tasks`: 指定要评估的数据集。如果`--task_set`不是`na`，将被重写。完整的任务列表见代码。

### Training

**Data**
对于无监督的SimCSE，我们从英语维基百科中抽取100万个句子；对于有监督的SimCSE，我们使用SNLI和MNLI数据集。你可以运行`data/download_wiki.sh`和`data/download_nli.sh`来下载这两个数据集。

**Training scripts**
我们为无监督和有监督的SimCSE提供训练脚本的例子。在`run_unsup_example.sh`中，我们为无监督版本提供了一个单GPU（或CPU）的例子，在`run_sup_example.sh`中，我们为有监督版本提供了一个**多GPU的例子。两个脚本都调用`train.py`进行训练。我们在下面解释参数。
* `--train_file`: 训练文件路径。我们支持 "txt "文件（一行代表一个句子）和 "csv "文件（2栏：没有困难负样本的配对数据；3栏：有一个相应的困难负样本实例的配对数据）。你可以使用我们提供的维基百科或NLI数据，也可以使用你自己的相同格式的数据。
* `--model_name_or_path`: 预训练好的checkpoint开始使用。目前，我们支持BERT-base的模型 (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: 对比性损失的温度。
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: 我们发现，对于无监督的SimCSE来说，用MLP层训练模型，但不测试模型，效果更好。在训练无监督的SimCSE模型时，你应该使用这个参数。
* `--hard_negative_weight`: 如果使用困难负样本（即训练文件中有3列），这就是权重的对数。例如，如果权重是1，那么这个参数应该被设置为0（默认值）。
* `--do_mlm`: 是否使用MLM辅助目标。如果为真。
  * `--mlm_weight`: MLM目标的权重。
  * `--mlm_probability`: MLM目标的masked率。
 
* 有监督run_unsup_example.sh 
train.py --model_name_or_path bert-base-uncased --train_file data/nli_for_simcse.csv --output_dir result/my-sup-simcse-bert-base-uncased --num_train_epochs 3 --per_device_train_batch_size 128 --learning_rate 5e-5 --max_seq_length 32 --evaluation_strategy steps --metric_for_best_model stsb_spearman --load_best_model_at_end --eval_steps 125 --pooler_type cls --overwrite_output_dir --temp 0.05 --do_train --do_eval --fp16

所有其他参数都是标准的Huggingface的`transformers'训练参数。
一些经常使用的参数是。`--output_dir`, `--learning_rate`, `--per_device_train_batch_size`。
在我们的例子脚本中，我们还设置了在STS-B开发集上评估模型（需要在[evaluation](#evaluation)部分之后下载数据集）并保存最佳checkpoint。

对于本文的结果，我们使用Nvidia 3090 GPU和CUDA 11。使用不同类型的设备或不同版本的CUDA/其他软件可能会导致性能略有不同。

**Hyperparameters**
我们使用以下超参数器来训练SimCSE。

|               | Unsup. BERT | Unsup. RoBERTa | Sup.      |
|:--------------|:-----------:|:--------------:|:---------:|
| Batch size    | 64          | 512            | 512       |
| Learning rate (base)  | 3e-5 | 1e-5 | 5e-5 |
| Learning rate (large) | 1e-5 | 3e-5 | 1e-5 |


**Convert models**
我们保存的checkpoint与Huggingface的预训练checkpoint略有不同。运行`python simcse_to_huggingface.py --path {PATH_TO_CHECKPOINT_FOLDER}`来转换它。
之后，你可以通过我们的[评估](#evaluation)代码来评估它，或者直接使用它[out of the box](#use-our-models-out-of-the-box)。

## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Tianyu (`tianyug@cs.princeton.edu`) and Xingcheng (`yxc18@mails.tsinghua.edu.cn`). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation

Please cite our paper if you use SimCSE in your work:

```bibtex
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```

## SimCSE Elsewhere

We thank the community's efforts for extending SimCSE!

- [Jianlin Su](https://github.com/bojone) has provided [a Chinese version of SimCSE](https://github.com/bojone/SimCSE).
- [AK391](https://github.com/AK391) integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/SimCSE)
- [Nils Reimers](https://github.com/nreimers) has implemented a `sentence-transformers`-based [training code](https://colab.research.google.com/drive/1gAjXcI4uSxDE_IcvZdswFYVAo7XvPeoU?usp=sharing#scrollTo=UXUsikOc6oiB) for SimCSE.
