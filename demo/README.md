## Demo of SimCSE 
有几个演示可供人们使用我们预训练好的SimCSE进行游戏。

### Flask Demo
<div align="center">
<img src="../figure/demo.gif" width="750">
</div>

我们提供了一个基于[flask](https://github.com/pallets/flask)的简单网络演示，
以展示SimCSE是如何直接用于信息检索的。
该代码基于[DensePhrases](https://arxiv.org/abs/2012.12624)' [repo](https://github.com/princeton-nlp/DensePhrases)
和[demo](http://densephrases.korea.ac.kr)（非常感谢DensePhrases的作者们）。
要在本地运行这个flask演示，请确保SimCSE推理接口的设置。
```bash
git clone https://github.com/princeton-nlp/SimCSE
cd SimCSE
python setup.py develop
```
然后你可以使用`run_demo_example.sh`来启动这个演示。
作为默认设置，我们为从STS-B数据集中抽取的1000个句子建立索引。
请随意建立你自己的语料库的索引。你也可以安装[faiss](https://github.com/facebookresearch/faiss)来加快检索过程。

### Gradio Demo
[AK391](https://github.com/AK391)提供了SimCSE的[Gradio Web Demo](https://gradio.app/g/AK391/SimCSE)，
展示了预训练的模型如何预测两个句子之间的语义相似度。
