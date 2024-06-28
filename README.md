
# applied-llm

Discover the latest resources on Applied LLMs. 

This collection includes the most relevant educational materials such as: 

- Foundational Research Papers
- Free LLM courses
- Popular Github repos
- Big tech articles featuring **successful LLM implementations in production**. 🚀


[![Substack](https://img.shields.io/badge/Substack-Subscribe-orange?style=flat&logo=substack&logoColor=white)](https://merinova.substack.com/about)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-blue.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/meri-bozulanova/)
![HitCount](http://hits.dwyl.com/meri-nova/applied-llm.svg)


Learn how organizations implement and leverage LLMs:

- **How** LLMs are deployed and integrated into systems 🔎
- **What** techniques and architectures worked ✅ (and sometimes, what didn't ❌)
- **Why** it works and what is the science behind it with research, literature, and references 📂
- **What** real-world results were achieved (so you can better assess ROI ⏰💰📈)


## Table of Contents

### Articles and Foundational Research Papers

1. [LLM Fundamentals](#llm-fundamentals)
2. [Training and Fine-tuning Techniques](#training-and-fine-tuning-techniques)
3. [Data Quality for LLMs](#data-quality-for-llms)
4. [Data Engineering for LLM](#data-engineering-for-llm)
5. [Deployment and Inference](#deployment-and-inference)
6. [Evaluation and Metrics](#evaluation-and-metrics)
7. [Prompt Engineering](#prompt-engineering)
8. [Vector Stores](#vector-stores)
9. [Tools and Frameworks](#tools-and-frameworks)
10. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
11. [Graph and LLMs](#graph-and-llms)
12. [Multimodal with LLMs](#multimodal-with-llms)
13. [Scaling and Optimization](#scaling-and-optimization)
14. [Ethical Considerations and Limitations](#ethical-considerations-and-limitations)

  # Learn and practice your skills 

15. [Courses and Tutorials](#courses-and-tutorials)
16. [GitHub Repositories](#github-repositories)
17. [LLM Tools for Developers](#llm-tools-for-developers)
18. [Team Structure and Strategy](#team-structure-and-strategy)
19. [Newsletters to follow](#newsletters)


---


## LLM Fundamentals

- [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Language Models are Few-Shot Learners (GPT-3 Paper)](https://arxiv.org/abs/2005.14165)
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
- [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)
- [Awesome-LLM: A curated list of Large Language Model resources](https://github.com/Hannibal046/Awesome-LLM)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

## Training and Fine-tuning Techniques

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751)
- [Fine-tuning GPT-2 from Human Preferences](https://arxiv.org/abs/1909.08593)


## Data Quality for LLMs

- [Data Quality for Machine Learning](https://eng.uber.com/data-quality-machine-learning/)
- [Improving Data Quality for Machine Learning Models](https://engineering.linkedin.com/blog/2021/improving-data-quality-for-machine-learning-models)
- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://pile.eleuther.ai/)
- [Data-Centric AI Resource Hub](https://datacentricai.org/)
- [Addressing Data Quality in Machine Learning](https://cloud.google.com/blog/products/ai-machine-learning/addressing-data-quality-in-machine-learning)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Curating High-Quality Datasets for Machine Learning](https://snorkel.ai/curating-high-quality-datasets-for-machine-learning/)


## Data Engineering for LLM

- [Building Scalable and Efficient Data Pipelines for Machine Learning](https://netflixtechblog.com/building-scalable-and-efficient-data-pipelines-for-machine-learning-f294fe800cb3)
- [Data Engineering for AI: Designing Efficient Pipelines for Language Models](https://medium.com/airbnb-engineering/data-engineering-for-ai-designing-efficient-pipelines-for-language-models-78e0283bd9f3)
- [Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-machine-learning/)
- [Data Engineering for NLP at Scale](https://engineering.atspotify.com/2022/06/data-engineering-for-nlp-at-scale/)
- [Building Data Pipelines for Large Language Models](https://www.databricks.com/blog/2023/03/20/building-data-pipelines-large-language-models.html)
- [Data Engineering Best Practices for Training Large Language Models](https://cloud.google.com/blog/products/ai-machine-learning/data-engineering-best-practices-for-training-large-language-models)
- [Efficient Data Processing for Language Model Pre-training](https://www.microsoft.com/en-us/research/publication/efficient-data-processing-for-language-model-pre-training/)


## Deployment and Inference

- [Efficient Large Language Model serving with FlexFlow](https://github.com/flexflow/FlexFlow)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://github.com/vllm-project/vllm)
- [TensorRT-LLM: Toolkit to optimize inference of LLMs](https://github.com/NVIDIA/TensorRT-LLM)
- [Datyabricks: Deploying Large Language Models in Production](https://www.databricks.com/blog/2023/04/13/deploying-large-language-models-in-production.html)
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
- [LLM Engineering Guide](https://github.com/stas00/ml-engineering/tree/master/llms)


## Evaluation and Metrics

- [Best Practices for LLM Evaluation](https://github.com/microsoft/promptbase)
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)
- [EleutherAI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [BIG-bench: Beyond the Imitation Game Benchmark](https://github.com/google/BIG-bench)
- [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [How We Evaluate Large Language Models at Anthropic](https://www.anthropic.com/index/how-we-evaluate-large-language-models)
- [Evaluating Large Language Models Trained on Code](https://openai.com/research/evaluating-large-language-models-trained-on-code)
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)


## Prompt Engineering

- [Google Cloud: Best practices for prompt engineering with LLMs](https://cloud.google.com/architecture/best-practices-for-prompt-engineering-with-llms)
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [OpenAI Cookbook: Techniques to improve reliability](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
  
## Vector Stores

- [Building a Vector Search Engine with Faiss](https://engineering.fb.com/2021/07/01/data-infrastructure/faiss/)
- [Introducing ScaNN: Efficient Vector Similarity Search](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html)
- [Milvus: An Open Source Vector Database for Scalable Similarity Search](https://medium.com/zilliz/milvus-an-open-source-vector-database-for-scalable-similarity-search-d512b2b9e40a)
- [Vector Similarity Search: From Basics to Production](https://www.pinecone.io/learn/vector-similarity-search/)
- [Weaviate: The Open Source Vector Database](https://weaviate.io/blog/weaviate-1-18-release)
- [Qdrant: Vector Database for the Next Generation of AI Applications](https://qdrant.tech/articles/qdrant-2-0/)

## Tools and Frameworks

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LangChain: Building applications with LLMs through composability](https://github.com/hwchase17/langchain)
- [OpenAI API](https://platform.openai.com/)
- [Anthropic Claude API](https://www.anthropic.com/product)
- [Llama.cpp: Inference of LLaMA model in pure C/C++](https://github.com/ggerganov/llama.cpp)



## Retrieval Augmented Generation (RAG)

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426)
- [Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2312.10997)
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909)
- [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282)
- [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299)
- [In-Context Retrieval-Augmented Language Models](https://arxiv.org/abs/2302.00083)

## Graph and LLMs

- [Combining Knowledge Graphs and Large Language Models](https://www.amazon.science/blog/combining-knowledge-graphs-and-large-language-models)
- [Graph Neural Networks and Language Models: A Powerful Combination](https://blog.twitter.com/engineering/en_us/topics/insights/2022/graph-neural-networks-and-language-models)
- [Knowledge Graphs and Language Models: Bridging the Gap](https://deepmind.com/research/open-source/knowledge-graphs-and-language-models)
- [Enhancing Language Models with Knowledge Graph Embeddings](https://www.microsoft.com/en-us/research/publication/enhancing-language-models-with-knowledge-graph-embeddings/)
- [Graph-augmented Learning for Language Understanding](https://ai.googleblog.com/2023/02/graph-augmented-learning-for-language.html)
- [Integrating Knowledge Graphs with Large Language Models](https://ai.stanford.edu/blog/integrating-knowledge-graphs-with-llms/)
- [Graph-based Neural Language Models](https://openai.com/research/graph-based-neural-language-models)

## Multimodal with LLMs

- [DALL·E 2: Extending Language Models to Images](https://openai.com/research/dall-e-2-extending-language-models-to-images)
- [PaLM-E: An Embodied Multimodal Language Model](https://ai.googleblog.com/2023/03/palm-e-embodied-multimodal-language.html)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model)
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)
- [Multimodal Few-Shot Learning with Frozen Language Models](https://www.microsoft.com/en-us/research/publication/multimodal-few-shot-learning-with-frozen-language-models/)
- [VisualBERT: A Simple and Performant Baseline for Vision and Language](https://arxiv.org/abs/)

  
## Scaling and Optimization

- [DeepSpeed: Deep learning optimization library](https://github.com/microsoft/DeepSpeed)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

## Ethical Considerations and Limitations

- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/10.1145/3442188.3445922)
- [Challenges and Applications of Large Language Models](https://arxiv.org/abs/2307.08723)
- [AI Ethics Guidelines Global Inventory](https://inventory.algorithmwatch.org/)
- [Ethical and social risks of harm from Language Models](https://arxiv.org/abs/2112.04359)


## Additional Resources

### Courses and Tutorials

1. [DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
2. [Coursera: Natural Language Processing Specialization](https://www.coursera.org/specializations/natural-language-processing)
3. [Fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/)
4. [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
5. [Hugging Face: NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
6. [Creme de la Creme of Free AI courses](https://github.com/SkalskiP/courses)

### GitHub Repositories

1. [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
2. [Applied ML](https://github.com/eugeneyan/applied-ml)
3. [Awesome Scalability](https://github.com/binhnguyennus/awesome-scalability)
4. [Made with ML](https://github.com/GokuMohandas/MadeWithML) 
5. [The Algorithms](https://github.com/TheAlgorithms/Python)
6. [TensorFlow Models](https://github.com/tensorflow/models)
7. [Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples)
8. [Sentence Transformers](https://github.com/UKPLab/sentence-transformers)

### LLM Tools for Developers

1. [OpenAI Playground](https://platform.openai.com/playground)
2. [Hugging Face Spaces](https://huggingface.co/spaces)
3. [Gradio: Build Machine Learning Web Apps](https://gradio.app/)
4. [Streamlit: The fastest way to build data apps](https://streamlit.io/)
5. [LangChain: Building applications with LLMs through composability](https://github.com/hwchase17/langchain)

### Team Structure and Strategy

1. [Google: Machine Learning: The High Interest Credit Card of Technical Debt](https://research.google/pubs/pub43146/)
2. [Spotify: How We Structure Our ML Teams](https://engineering.atspotify.com/2022/03/how-we-structure-our-ml-teams/)
3. [Uber: Scaling Machine Learning at Uber with Michelangelo](https://eng.uber.com/scaling-michelangelo/)
4. [Netflix: Human-Centric Machine Learning Infrastructure at Netflix](https://netflixtechblog.com/human-centric-machine-learning-infrastructure-at-netflix-9b6d21e661f9)
5. [Airbnb: Scaling Knowledge Access and Retrieval at Airbnb](https://medium.com/airbnb-engineering/scaling-knowledge-access-and-retrieval-at-airbnb-665b6ba21e95)

  ### Newsletters
  
1. [Merinova](https://merinova.substack.com/)
2. [Ahead of AI](https://magazine.sebastianraschka.com/)
3. [Breaking into Data](https://breakintodata.substack.com/)
4. [Underfitted](https://underfitted.svpino.com/)
5. [Marvelous MLOps](https://marvelousmlops.substack.com/)
6. [SemiAnalysis](https://www.semianalysis.com/)
