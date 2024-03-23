# NLPModel
Sentiment language model based on review data.

---
language: en
tags:
- exbert
license: MIT
datasets:
- imdb
---

# Sentiment Analysis using `bert-base-uncased`.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> analysis = pipeline('sentiment-analysis', model='kreimben/bert-base-uncase-sentiment-analysis')
>>> analysis("That was great! 8 out of 10!")
```


## Training data

This model was trained using `IMDB` movie dataset in huggingface.


```bibtex
@article{JehwanKim2024BertSentimentAnalaysis,
  title={bert-sentiment-analysis},
  author={Jehwan Kim},
  year={2024},
}
```