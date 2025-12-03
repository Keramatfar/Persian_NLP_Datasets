Emotion Classification
| DataSet | Type | Classes | # Classes | # Samples | Domains | EmotionModel | Year | License | paper |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| [ShortPersianEmo](https://github.com/vkiani/ShortPersianEmo) | Emotion | happiness:1625, sadness:939, anger:1125, fear:380, and other:1403 | 5 | 5472 | Twitter, Digikala | RachaelJack | 2023 | GNU | Investigating shallow and deep learning techniques for emotion classification in short Persian texts |
| [JAMFA](https://github.com/Azadsee/JAMFA) | Emotion | Anger/Disgust, Happiness, Fear/Surprise, Sadness | 4 | 2241 | Fiction | EKMAN | 2022 | --- | Deep Emotion Detection Sentiment Analysis of Persian Literary Text |
| [PersianTweets](https://www.kaggle.com/datasets/behdadkarimi/persian-tweets-emotional-dataset) | Emotion | Anger:20069, Disgust:925, joy:28024, Fear:17624, Surprise:12859, Sadness:34328 | 6 | 113829 | Twitter | EKMAN | 2021 | Author's permission | - |
| [ArmanEmo](https://github.com/arman-rayan-sharif/arman-text-emotion?tab=readme-ov-file) | Emotion | Anger:1077, Fear:814, Happiness:893, Hatred:576, Sadness:1158, Wonder:884, Other:1874 | 7 | 7308 |  Twitter, Instagram, and Digikala | EKMAN | 2022 | non-commercial use | - | ARMANEMO: A PERSIAN DATASET FOR TEXT-BASED EMOTION DETECTION
| [EmoPars](https://github.com/nazaninsbr/persian-emotion-detection) | Emotion | Anger:1632, Fear:690, Happiness:692, Sadness:1770, Hatred:1256, Wonder:986 | 6 | 29997 | Twitter | EKMAN | 2021 | not specified | - | EmoPars: A Collection of 30K Emotion-Annotated Persian Social Media Texts



Sentiment Analysis
| DataSet | # Classes | # Samples | Domains | Year | License | Paper Title | Notes |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Not specified | 3 | 12055 | Twitter | 2021 | - | ParsBERT Post-Training for Sentiment Analysis of Tweets Concerning Stock Market | - |
| [JAMFA](https://github.com/Azadsee/JAMFA) | 2 | 2241 | Fiction | 2022 | Author's permission | Deep Emotion Detection Sentiment Analysis of Persian Literary Text | - |
| [Pars-ABSA](https://github.com/Titowak/Pars-ABSA) | 1:5114, 0:1827, -1:3061 | 10002 | Digikala | 2022 | not specified | Pars-ABSA: a Manually Annotated Aspect-based Sentiment Analysis Benchmark on Farsi Product Reviews | Aspect based  |
| [Snappfood](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis) | 1:35000, 0:35000 | 70000 | Snappfood | 2022 | not specified | - | - |
| [DeepSentiPers](https://github.com/JoyeBright/DeepSentiPers) | 2:988, 1:1623, 0:2409, -1:513, -2:28 | 19550 | Digikala | 2020 | not specified | DeepSentiPers: Novel Deep Learning Models Trained Over Proposed Augmented Persian Sentiment Corpus | - |
| Product reviews dataset | 2 | 3000 | Digikala | 2018 | not specified | Words are important: Improving sentiment analysis in the persian language by lexicon refining | - |
| Hotel reviews dataset | 2 | 3600 | hellokish | 2015 | not specified | Opinion Mining in Persian Language Using Supervised Algorithms | - |
| Not specified | 2 | 8373 | Sahamyab | 2019 | not specified | Tehran Stock Exchange Prediction Using Sentiment Analysis of Online Textual Opinions | Bullish or Bearish instead of positive or negative |
| Not specified | 3 | 12055 | Twitter | 2021 | not specified | ParsBERT Post-Training for Sentiment Analysis of Tweets Concerning Stock Market | - |
| [Multimodal-Persian-SA](https://github.com/mebasiri/Multimodal-Persian-Sentiment-Analysis) | 1:561, 0:439 | 1000 | Instagram, Telegram | 2021 | not specified | Sentiment Analysis of Persian Instagram Post: a Multimodal Deep Learning Approach | Multi-Modal, Text and image |
| InstaText | 3 | 8512 | Instagram | 2021 | not specified | Producing An Instagram Dataset for Persian Language Sentiment Analysis Using Crowdsourcing Method | - |

Named Entity Recognition
 DataSet | # Classes | # Samples | Domains | Year | License | Paper Title | Notes |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| [XTREME (PAN-X) NER](https://github.com/google-research/xtreme) | 3 | 40,000 samples (20k train / 10k val / 10k test) | Wikipedia | 2020 | CC BY-SA 4.0 | XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization | Persian subset of WikiAnn (tokens only, no raw sentences) |
| [Persian-NER](https://github.com/Text-Mining/Persian-NER) | 5 | ~25,000,000 tokens (~1,000,000 sentences) | Wikipedia (Persian) | 2023 | CC BY-SA / Open-source | Persian Wikipedia NER corpus | Standard NER corpus with 5 entity types (PER, ORG, LOC, EVT, DAT); community contributions (>1000 users) allowed to improve annotations via https://app.text-mining.ir

Natural Language Inference

| Dataset                                         | # Classes                                | # Samples | Domains                                       | Year | License                  | Paper Title                                                           | Notes                                                                                                                              |
| ----------------------------------------------- | ---------------------------------------- | --------- | --------------------------------------------- | ---- | ------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [FarsTail](https://github.com/dml-qom/FarsTail) | 3 (Entailment / Contradiction / Neutral) | 10,367    | Academic / Written (premise‑hypothesis pairs) | 2020 | Apache‑2.0 ([GitHub][1]) | *FarsTail: a Persian natural language inference dataset* ([arXiv][2]) | First relatively large-scale NLI dataset for Persian, with train/val/test split and both raw-text and indexed format ([GitHub][1]) |

Dependency Tree Bank

| Dataset                                         | # Classes                                | # Samples | Domains                                       | Year | License                  | Paper Title                                                           | Notes                                                                                                                              |
| ----------------------------------------------- | ---------------------------------------- | --------- | --------------------------------------------- | ---- | ------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [PerUDT](https://github.com/UniversalDependencies/UD_Persian-PerDT) | Treebank (Universal Dependencies) | 26,196 / 1,455 / 1,456 | Mixed (news / fiction / academic / web / blog) | 2023† | CC BY‑SA 4.0 | *Persian Universal Dependency Treebank (PerUDT)* | Automatic conversion of Persian Dependency Treebank (PerDT) into UD format with manual corrections, 29K sentences over multiple genres. |




[1]: https://github.com/dml-qom/FarsTail?utm_source=chatgpt.com "GitHub - dml-qom/FarsTail: FarsTail: a Persian natural language inference dataset"
[2]: https://arxiv.org/abs/2009.08820?utm_source=chatgpt.com "FarsTail: A Persian Natural Language Inference Dataset"



