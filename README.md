
# Applied LLMs

[![Substack](https://img.shields.io/badge/Substack-Subscribe-orange?style=flat&logo=substack&logoColor=white)](https://merinova.substack.com/about)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-blue.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/meri-bozulanova/)
![HitCount](http://hits.dwyl.com/meri-nova/applied-llm.svg)

Discover the latest **LLM implementations in production**. 🚀

Learn how big tech companies and startups implement and leverage LLMs in 2024:

- **How** LLMs are deployed and integrated into large-scale applications 🔎
- **What** architectures and techniques worked when implementing LLM in the SWD cycle ✅ (Data Quality, Data Engineering, Serving, Monitoring 📈 etc)
- **What** real-world results were achieved (so you can better assess ROI ⏰💰)
- **Why** it works and what is the science behind it with research, literature, and references 📂

Feel free to contribute!


## Table of Contents


1. [Training and Fine-tuning Techniques](#training-and-fine-tuning-techniques)
2. [Data Quality for LLMs](#data-quality-for-llms)
3. [Data Engineering for LLM](#data-engineering-for-llm)
4. [Deployment](#deployment)
5. [Evaluation and Metrics](#evaluation-and-metrics)
6. [Prompt Engineering](#prompt-engineering)
7. [Vector Stores](#vector-stores)
8. [Tools and Frameworks](#tools-and-frameworks)
9. [Retrieval Augmented Generation (RAG)](#retrieval-augmented-generation-rag)
10. [Graph and LLMs](#graph-and-llms)
11. [Multimodal with LLMs](#multimodal-with-llms)
12. [Scaling and Optimization](#scaling-and-optimization)
13. [Ethical Considerations and Limitations](#ethical-considerations-and-limitations)

## Additional Resources 

14. [LLM Seminal Papers](#llm-seminal-papers)
15. [Courses and Tutorials](#courses-and-tutorials)
16. [GitHub Repositories](#github-repositories)
17. [LLM Tools for Developers](#llm-tools-for-developers)
18. [Team Structure and Strategy](#team-structure-and-strategy)
19. [Newsletters to follow](#newsletters)



---
# Main content 👇


## Training and Fine-tuning Techniques


- [Imbue: Training a 70B Model from Scratch](https://imbue.com/research/70b-intro/)
- [Google: PaLM 2 Technical Report](https://ai.google/static/documents/palm2techreport.pdf)
- [Anthropic: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [EleutherAI: The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)
- [Hugging Face: How we trained BLOOM, the world's largest open multilingual language model](https://huggingface.co/blog/bloom-megatron-deepspeed)
- [DeepMind: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- [NVIDIA Megatron-Turing NLG](https://developer.nvidia.com/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/)


## Data Quality for LLMs

## Data Quality and Contracts for LLM Deployment in Big Tech

- [Nvidia: Scale and Curate High-Quality Datasets for LLMs](https://developer.nvidia.com/blog/scale-and-curate-high-quality-datasets-for-llm-training-with-nemo-curator/)
- [Data Quality Error Detection with LLMs](https://towardsdatascience.com/automated-detection-of-data-quality-issues-54a3cb283a91)
- [IBM: How to ensure Data Quality and Reliability](https://www.ibm.com/blog/how-data-engineers-can-ensure-data-quality-value-and-reliability/)
- [NVIDIA: Curating custom datasets for LLM training](https://developer.nvidia.com/blog/curating-custom-datasets-for-llm-training-with-nvidia-nemo-curator/)

## Data Engineering for LLM

- [Harnessing the power of LLMs in Data Engineering](https://www.wednesday.is/writing-articles/the-future-of-llms-examining-the-impact-from-a-data-engineers-perspective)
- [Data Engineer 2.0](https://medium.com/adevinta-tech-blog/data-engineer-2-0-part-i-large-language-models-7b745c4683e4)
- [Data Collection Magic](https://huggingface.co/blog/JessyTsu1/data-collect) 
- [Data Engineers, here is how LLMs can make your life easier](https://builtin.com/articles/data-engineers-llms-easier)

## Deployment 

- [Efficient Large Language Model serving with FlexFlow](https://github.com/flexflow/FlexFlow)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://github.com/vllm-project/vllm)
- [TensorRT-LLM: Toolkit to optimize inference of LLMs](https://github.com/NVIDIA/TensorRT-LLM)
- [Datyabricks: Deploying Large Language Models in Production](https://www.databricks.com/blog/2023/04/13/deploying-large-language-models-in-production.html)
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/)
- [LLM Engineering Guide](https://github.com/stas00/ml-engineering/tree/master/llms)


## Evaluation and Metrics

- [OpenAI: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)
- [EleutherAI: HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)
- [Google: Beyond the Imitation Game - Measuring and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615)
- [DeepMind: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [Meta: MMLU - Massive Multitask Language Understanding](https://github.com/hendrycks/test)
- [Anthropic: Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/index/constitutional-ai-harmlessness-from-ai-feedback)
- [Microsoft: How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751)
- [BigScience: A Framework for Few-Shot Language Model Evaluation](https://arxiv.org/abs/2109.01652)
- [Best Practices for LLM Evaluation](https://github.com/microsoft/promptbase)

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


## Additional Educational Resources

## LLM Seminal Papers


1. [Attention Is All You Need (Transformer Paper)](https://arxiv.org/abs/1706.03762)
2. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
5. [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223)
6. [Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond](https://arxiv.org/abs/2304.13712)
7. [Awesome-LLM: A curated list of Large Language Model resources](https://github.com/Hannibal046/Awesome-LLM)
8. [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332)
9. [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)


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
