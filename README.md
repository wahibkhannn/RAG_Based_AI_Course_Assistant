<h1 align="center">
🚀 AI-Powered Multi-Modal Knowledge Retrieval System
</h1>

<p align="center">
Transform Videos & PDFs into Searchable AI Knowledge Bases using RAG, Qdrant, Gemini, Ollama & Whisper
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![RAG](https://img.shields.io/badge/RAG-System-green?style=for-the-badge)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-red?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-RTX_4050-success?style=for-the-badge)

</p>


## Overview

AI-Powered Multi-Modal Knowledge Retrieval System is an advanced Retrieval-Augmented Generation (RAG) platform capable of transforming long-form educational videos and PDF documents into searchable semantic knowledge repositories.

The system combines state-of-the-art speech recognition, vector embeddings, semantic retrieval, and Large Language Models to provide grounded, explainable, and source-cited answers.

Unlike conventional chatbots, responses are generated directly from retrieved evidence and linked back to their original timestamps or document pages.


## Architecture

```text
          ┌─────────────┐
          │ Video / PDF │
          └──────┬──────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Content Processing  │
      │ Whisper / PDF Parse │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Semantic Chunking   │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ BGE-M3 Embeddings   │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Qdrant Cloud DB     │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Semantic Retrieval  │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Context Expansion   │
      └─────────┬───────────┘
                │
                ▼
      ┌─────────────────────┐
      │ Gemini / Ollama     │
      └─────────┬───────────┘
                │
                ▼
         Source-Grounded
            Response


## Key Features

### Multi-Modal Ingestion
- Video Lecture Processing
- PDF Knowledge Extraction
- Automated Audio Extraction
- Speech-to-Text Conversion

### Semantic Search
- Vector Similarity Search
- Context Expansion
- Metadata-Aware Retrieval
- Confidence Scoring

### Hybrid LLM Inference
- Gemini 2.5 Flash
- Ollama Local Models
- Cloud + Local Flexibility

### Explainable AI
- Timestamp Citations
- Page-Level References
- Source Grounding
- Evidence-Based Responses



## Performance Benchmarks

| Metric | Result |
|----------|----------|
| Longest Lecture Tested | 59 Minutes |
| Transcript Segments Generated | 923 |
| Semantic Chunks Created | 90 |
| Processing Time | ~83 Seconds |
| Embedding Throughput | 48.7 Chunks/sec |
| Embedding Model | BAAI/bge-m3 |
| GPU | NVIDIA RTX 4050 |
| Vector Database | Qdrant Cloud |
| Transcription Model | Groq Whisper Large-v3 |

## Screenshots

### Upload Interface

![Upload](images/upload.png)

### AI Generated Responses

![Response](images/response.png)

### Source-Cited Retrieval

![Sources](images/sources.png)



## Tech Stack

### AI & NLP
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Semantic Search
- Information Retrieval

### Models
- Gemini 2.5 Flash
- Mistral-7B (Ollama) 
- Qwen:4B
- Deepseek-r1
- BAAI/bge-m3
- Groq Whisper Large-v3

### Infrastructure
- Qdrant Cloud
- CUDA
- Streamlit
- FFmpeg

### Programming
- Python

## Roadmap

- [ ] Hybrid Search (BM25 + Vector Search)
- [ ] Cross-Encoder Re-Ranking
- [ ] Multi-Document Collections
- [ ] Chat Memory
- [ ] Agentic Retrieval
- [ ] Knowledge Graph Integration
- [ ] Research Paper Mode
- [ ] Multi-Language Support

###Author Section
## Author

Mohammad Wahib Ashraf Khan

B.Tech CSE (Data Science)

Interests:
- Artificial Intelligence
- NLP
- Retrieval-Augmented Generation
- Machine Learning
- Computational Research
- Deep Learning
- Drug Discovery
- Computer Vision


GitHub: https://github.com/wahibkhannn
LinkedIn: https://linkedin.com/in/wahibkhannn

