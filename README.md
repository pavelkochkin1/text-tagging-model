<h1 align="center" id="title">Text Tagging Model</h1>

<p id="description">Here we collected some online and offline models for text tagging.</p>

<h2>🚀 Demo</h2>

[https://colab.research.google.com/drive/1xlevLnqxd\_wCtXunGgf\_pSrdkz85jt49?usp=sharing](https://colab.research.google.com/drive/1xlevLnqxd_wCtXunGgf_pSrdkz85jt49?usp=sharing)



<h2>🧐 Features</h2>

Here're some of the project's best features:

*   Online model: Rake Based Model with 10-20 it/sec
*   Offline models: Bart based model with summarisation or attention. 1-5 it/sec

<h2>🛠️ Installation Steps:</h2>

<p>1. Installation</p>

```
pip install text-tagging-model
```

<p>2. import</p>

```
from text_tagging_model.models.rake_based_model import TagsExtractor
```

<p>3. Init tagger</p>

```
tagger = TagsExtractor()
```

<p>4. Get tags</p>

```
tagger.extract(some_text)
```
