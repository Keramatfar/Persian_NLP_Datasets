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
| [PerUDT](https://github.com/UniversalDependencies/UD_Persian-PerDT) | Treebank (Universal Dependencies) | 26,196 / 1,455 / 1,456 | Mixed (news / fiction / academic / web / blog) | 2023† | CC BY‑SA 4.0 | *The Persian Dependency Treebank Made Universal* | Automatic conversion of Persian Dependency Treebank (PerDT) into UD format with manual corrections, 29K sentences over multiple genres. |

Multilingual parallel corpus

| Dataset | # Classes | # Samples | Domains | Year | License | Paper Title | Notes |
| ------- | --------- | --------- | ------- | ---- | ------- | ----------- | ----- |
| [JW300](https://opus.nlpl.eu/JW300.php) | Parallel sentences | ~1.2M (fa-en) | Religious / translation | 2019 | CC BY | *JW300: A Wide-Coverage Parallel Corpus for Low-Resource Languages* | Large-scale parallel corpus from Jehovah’s Witness publications; Persian ↔ English included. |
| [Tatoeba](https://tatoeba.org/) | Parallel sentences | ~50K (fa-en) | Mixed / community sentences | 2022 | CC0 | *Tatoeba: A Collection of Example Sentences for Many Languages* | Manually contributed sentences; smaller but cleaner Persian ↔ English pairs. |
| [OPUS – OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles.php) | Parallel sentences | ~200K (fa-en) | Movie subtitles | 2021 | CC BY‑SA 4.0 | *OPUS: Open Parallel Corpus* | Informal conversational language; Persian ↔ English pairs available. |
| [CCMatrix](https://github.com/facebookresearch/CCMatrix) | Parallel sentences | ~50M (fa-en) | Web-crawled | 2020 | MIT | *CCMatrix: Mining Large-scale Parallel Corpora from the Web* | Massive web-crawled sentence pairs; noisy but very large Persian ↔ English corpus. |
| [mC4 (multilingual Common Crawl)](https://www.tensorflow.org/datasets/community_catalog/huggingface/mc4) | Parallel sentences | Variable | Web | 2020 | Apache 2.0 | *mC4: Multilingual Common Crawl Corpus* | Preprocessed Common Crawl; Persian text can be extracted for MT tasks. |

Benchmarks

| Dataset | # Tasks & Sample counts | Domains / Genres | Year | License | Paper Title | Notes |
| ------- | ---------------------- | ---------------- | ---- | ------- | ------------- | ----- |
| [ParsiNLU](https://github.com/persiannlp/parsinlu) | 6 tasks — Reading Comprehension: 1,300 • Multiple‑Choice QA: 2,460 • Sentiment Analysis: 2,423 • Textual Entailment: 2,700 • Question Paraphrasing: 4,644 • Machine Translation: 47,745 sentence pairs | Mixed (literary texts, reviews, QA, translations) | 2021 | CC BY‑4.0 | *ParsiNLU: A Suite of Language Understanding Challenges for Persian* | First large‑scale multi‑task Persian NLU benchmark. Includes MC‑QA, entailment, paraphrasing, sentiment, RC, and a large MT subset. |

Instruction Tuning

| Dataset                                                                                                   | # Tasks & Sample counts                                       | Domains / Genres                                                     | Year | License                               | Paper / Source       | Notes                                                                                                                       |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- | ---- | ------------------------------------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [Persian_instruct_dataset](https://github.com/mostafaamiri/Persian_instruct_dataset)                      | 4864 examples — mostly single-step instruction → output pairs | General / mixed (products, books, household items, descriptions, QA) | 2024 | Not specified                         | GitHub repo          | Semi-Alpaca style; quality is moderate, outputs often generic or partially cover inputs.                                    |
| [North_fa_llama3_Dataset](https://huggingface.co/datasets/payamvha/North_fa_llama3_Dataset)               | 15281 instruction-output pairs                                 | General knowledge, QA, open-domain                                   | 2024 | Apache‑2.0                            | Hugging Face dataset | Well-curated; includes instruction → output → category format; better coverage and diversity than Persian_instruct_dataset. |
| [persian-alpaca-reasoning-v1](https://huggingface.co/datasets/hosseinhimself/persian-alpaca-reasoning-v1) | ~20k instruction-output pairs with reasoning                  | General + reasoning, creative Q&A                                    | 2024 | Apache‑2.0                            | Hugging Face dataset | Instruction-following with reasoning / explanation outputs; useful for reasoning-augmented fine-tuning.                     |
| [xmanii/Maux-Persian-SFT-30k](https://huggingface.co/datasets/xmanii/Maux-Persian-SFT-30k)                | ~30k conversation samples                                     | Chat / dialogue (informal, general)                                  | 2023 | Apache‑2.0                            | Hugging Face dataset | Conversational SFT dataset for chatbots and assistants.                                                                     |
| [xmanii/Mauxi-SFT-Persian](https://huggingface.co/datasets/xmanii/Mauxi-SFT-Persian)                      | ~5k conversation threads                                      | Chat / dialogue                                                      | 2023 | Open / free for research & commercial | Hugging Face dataset | Smaller dialogue dataset for lightweight chat models; conversation-focused.                                                 |
| [xmanii/mauxitalk-persian](https://huggingface.co/datasets/xmanii/mauxitalk-persian)                      | ~10k conversation samples                                     | Chat / dialogue                                                      | 2023 | MIT                                   | Hugging Face dataset | Informal, open-domain conversational dataset for Persian chatbots.                                                          |
| [xmanii/Persian-Math-SFT](https://huggingface.co/datasets/xmanii/Persian-Math-SFT)                        | ~2k math Q&A pairs                                            | Education / STEM                                                     | 2023 | MIT                                   | Hugging Face dataset | Focused on Persian math questions; specialized SFT dataset for educational tutoring or math Q&A.                            |
| [sinarashidi/alpaca-persian](https://huggingface.co/datasets/sinarashidi/alpaca-persian)                  | ~35,000 examples — instruction → output pairs                 | General / mixed Persian instructions                                  | 2023 | Not specified / unclear                | Hugging Face repo    | Persian version of Stanford Alpaca; no explicit license; mostly single-turn instruction-response; provenance unclear.       |
| [Persian_instruct_dataset](https://github.com/mostafaamiri/Persian_instruct_dataset)                      | 4864 examples — mostly single-step instruction → output pairs | General / mixed (products, books, household items, descriptions, QA) | 2024 | Not specified                         | GitHub repo          | Semi-Alpaca style; quality is moderate, outputs often generic or partially cover inputs.                                     |
| [Farsinstract](https://github.com/Hojjat-Mokhtarabadi/FarsInstruct)                                              | 9374312 train - 1308596 test                  | Scientific / academic articles                                       | 2025 | CC BY-NC 4.0                           | Empowering Persian LLMs for Instruction Following: A Novel Dataset and Training Approach  | Focused on Persian scientific abstracts; good quality summaries, useful for abstractive summarization and instruction tuning. |

Text To Speech (TTS)
| Dataset                                                                 | # Classes / Labels                 | # Samples / Amount                                               | Domains                              | Year | License                            | Paper Title                                                                                 | Notes                                                                                                             |
| ----------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------- | ------------------------------------ | ---- | ---------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| [ParsVoice](https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice) | ~470+ speakers | 3,526.4 hrs raw → 1,803.9 hrs TTS-filtered | Audiobook / TTS | 2025 | (See HF license) | *ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis* | 3,526.4 hrs & 2,603,045 segments before filtering; 1,803.9 hrs & 1,147,718 segments after TTS-quality filtering; ~2,000 audiobook sources; clean multi-speaker Persian TTS corpus. |
| [Mana-TTS](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS) | Single-speaker audio | ~114 hours (fa) | Persian magazine narration / TTS | 2025 | CC0-1.0 | ManaTTS Persian: a recipe for creating TTS datasets for lower resource languages | Largest publicly available single-speaker Persian TTS corpus with high-quality aligned audio/text; includes open pipeline. :contentReference[oaicite:0]{index=0} |


Punctuation Resortaion
| Dataset | # Classes / Labels | # Samples / Amount | Domains | Year | License | Paper Title | Notes |
|---------|---------------------|---------------------|---------|------|---------|-------------|-------|
| [PersianPunc](https://huggingface.co/datasets/MohammadJRanjbar/PersianPunc) | Text pairs (unpunctuated → punctuated) | ~100K–1M examples | Persian punctuation restoration / ASR post-processing | 2025 | (See HF dataset card) | - | Dataset for restoring punctuation in Persian transcripts; designed for ASR pipelines and text normalization. |

Image captioning
| Dataset | # Classes / Labels | # Samples / Amount | Domains | Year | License | Paper Title | Notes |
|---------|------------------|------------------|---------|------|---------|-------------|-------|
| [Persian Image Captioning Dataset](https://www.kaggle.com/datasets/malekzadeharman/persian-image-captioning-dataset) | Image–caption pairs (no fixed label classes; paired descriptions) | ~1500 image–article pairs | News images & captions (Tasnim News Agency) | 2021 | Unknown / Not specified | Derived from “Persian Image Captioning Dataset (Lashkaryani, 2021)” | Images with corresponding Persian news captions crawled from a news agency; smaller dataset focused on real-world news imagery. |
| COCO‑Flickr Farsi (Navid Kanaani) | Image–caption pairs (no fixed label classes; paired descriptions) | Same as COCO & Flickr sizes (translated captions) *(original COCO: 123k train images + multiple captions)* | General images from COCO & Flickr with translated Farsi captions | 2021 | Unknown / Not specified | “COCO‑Flickr Farsi” (translated COCO & Flickr captions to Persian) | Uses standard COCO and Flickr datasets with captions translated into Persian; no fixed class taxonomy (captioning dataset). |

Visual question answering
| Dataset | # Classes / Labels | # Samples / Amount | Domains | Year | License | Paper Title | Notes |
|---------|---------------------|---------------------|---------|------|---------|-------------|-------|
| [ParsVQA‑Caps](https://www.kaggle.com/datasets/maryamsadathashemi/parsvqacaps/data) | Image–caption pairs (no fixed label classes; paired descriptions) | ~7.5k images and ~9k captions | General images with Persian captions, focused on Persian culture & language | 2022 (presented) / 2021–2022 (collection) | Unknown / Not specified | *ParsVQA‑Caps: A Benchmark for Visual Question Answering and Image Captioning in Persian* | First Persian benchmark combining VQA & captioning; image captioning portion contains ~7.5k images with ~9k human‑written captions. |


Face dataset with age and labels
| Dataset | # Classes / Labels | # Samples / Amount | Domains | Year | License | Paper Title | Notes |
|---------|---------------------|---------------------|---------|------|---------|-------------|-------|
| [ParsFace](https://github.com/Amirnoroozi/parsface) | Face identity metadata + images (no fixed “class labels” per se; identities can serve as labels) | ~6,000 Iranian personalities with face images and metadata | Face images of public figures (actors, politicians, athletes, etc.) | 2024 | CC0-1.0 | — | Contains names (Persian & English), ages, professions (Persian & English), gender, Wikipedia links, and associated face images; suitable for face recognition/attribute tasks |

License Plate Characters Detection
| Dataset | # Classes / Labels | # Samples / Amount | Domains | Year | License | Paper Title | Notes |
|---------|--------------------|--------------------|---------|------|---------|-------------|-------|
| [Iranis](https://github.com/alitourani/Iranis-dataset?utm_source=chatgpt.com) | 28 character classes (10 digits + 17 letters + 1 symbol) | ~83,844 cropped images | Farsi license plate characters extracted from real‑world plate images | 2021 | GPL‑3.0 (repo) / dataset available under academic terms | *Iranis: A Large‑scale Dataset of Farsi License Plate Characters* | Contains Farsi digits, letters, and a symbol from Iranian license plates; includes annotations for object detection/classification; suitable for OCR, recognition, and character generation tasks :contentReference[oaicite:0]{index=0} |



