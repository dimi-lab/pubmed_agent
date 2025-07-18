# 🔬 PubMed Agent AI

**AI-Powered Biomedical Research Assistant with RAG-Enhanced Analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/your-repo/pubmed-agent-ai)

PubMed Agent AI is an advanced biomedical research assistant that combines PubMed literature search with state-of-the-art AI analysis using BioMistral LLM and Retrieval-Augmented Generation (RAG) technology. Get intelligent, evidence-based answers to your medical and scientific questions backed by peer-reviewed research.

![PubMed Agent AI Demo](docs/images/demo-screenshot.png)

## ✨ Key Features

### 🧠 **Three Analysis Modes**
- **BioMistral Knowledge**: Direct AI responses using medical training data
- **PubMed + BioMistral**: Simple integration of literature search with AI analysis  
- **RAG + BioMistral**: Advanced semantic search with context-aware AI responses

### 🔬 **Advanced Capabilities**
- **Real-time PubMed Search**: Access to 35M+ biomedical research papers
- **Semantic Search**: RAG technology finds most relevant paper sections
- **Medical AI**: BioMistral-7B specialized for biomedical domain
- **Evidence-Based Responses**: All answers grounded in peer-reviewed literature
- **Interactive Chat**: Conversational interface for follow-up questions
- **Source Tracking**: Complete citations with PMIDs and DOIs

### 🚀 **Performance Optimized**
- **GPU Acceleration**: CUDA support for fast inference
- **Rate-Limited APIs**: Respectful PubMed access with backoff strategies
- **Vector Storage**: Efficient ChromaDB for semantic search
- **Caching**: Local storage of queries and responses

## 🛠️ Installation

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (optional but recommended)
- **8GB+ RAM** (16GB+ recommended for best performance)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/pubmed-agent-ai.git
cd pubmed-agent-ai
```

### 2. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Application dependencies
pip install -r requirements.txt

# Optional: For PubMed access
pip install metapub
```

### 3. Environment Configuration

Create a `.env` file with your credentials:

```bash
# Required for PubMed API access
PUBMED_EMAIL=your.email@example.com
PUBMED_API_KEY=your_pubmed_api_key

# Optional: For enhanced model downloads
HUGGINGFACE_TOKEN=your_hf_token

# System configuration
CHROMA_TELEMETRY=False
ANONYMIZED_TELEMETRY=False
```

### 4. Model Setup

The app will automatically download required models on first run:

- **BioMistral-7B**: ~14GB specialized medical LLM
- **Medical Embeddings**: ~1GB for semantic search
- **ChromaDB**: Vector database for RAG

## 🚀 Quick Start

### 1. Launch Application

```bash
streamlit run app.py
```

### 2. System Verification

1. **Check System Status**: Verify all components are loaded
2. **Run Quick Test**: Click "🧪 Test RAG Integration" in sidebar
3. **Try Example**: Use a sample biomedical question

### 3. Example Usage

**Question**: "How does CRISPR gene editing help treat cancer?"

**RAG Mode Response**:
- Searches PubMed for relevant papers
- Creates semantic embeddings of paper content  
- Finds most relevant sections using AI
- Generates evidence-based analysis with citations

## 📖 Usage Guide

### Research Modes

#### 🧠 **BioMistral Knowledge Mode**
- **Best for**: General medical questions, quick answers
- **Speed**: Fastest (~2-5 seconds)
- **Sources**: BioMistral's training knowledge
- **Use case**: "What is diabetes?" or "Explain DNA replication"

#### 📬 **PubMed + BioMistral Mode** 
- **Best for**: Literature-backed answers, recent research
- **Speed**: Medium (~10-30 seconds)
- **Sources**: Current PubMed papers + AI analysis
- **Use case**: "Recent advances in cancer immunotherapy"

#### 🚀 **RAG + BioMistral Mode** (Recommended)
- **Best for**: Complex questions, comprehensive analysis
- **Speed**: Slower (~30-60 seconds) 
- **Sources**: Semantic search + focused AI analysis
- **Use case**: "Compare effectiveness of different diabetes treatments"

### Advanced Features

#### **Interactive Chat**
- Follow-up questions in conversational format
- Context maintained across conversation
- Clarifications and deep-dive discussions

#### **Research History**
- Automatic saving of queries and responses
- Quick access to previous research
- Export functionality for reports

#### **Customization Options**
- Response length preferences (Short/Medium/Long)
- Number of papers to retrieve (5-50)
- Advanced search parameters

## 🔧 Configuration

### Performance Tuning

```python
# In app.py, adjust these settings:

# PubMed retrieval
MAX_PAPERS = 20  # Increase for more comprehensive analysis
RATE_LIMIT = 3   # Requests per second (respect NCBI limits)

# RAG parameters  
CHUNK_SIZE = 1000      # Text chunk size for embeddings
CHUNK_OVERLAP = 200    # Overlap between chunks
SIMILARITY_TOP_K = 5   # Number of relevant chunks to retrieve

# BioMistral settings
MAX_NEW_TOKENS = 512   # Response length limit
TEMPERATURE = 0.1      # Creativity vs accuracy (0.0-1.0)
```

### Hardware Optimization

#### **GPU Setup** (Recommended)
```bash
# Verify CUDA installation
nvidia-smi

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **CPU-Only Mode**
```python
# In components/llm.py, set:
device_map = "cpu"
load_in_4bit = False  # Disable quantization for CPU
```

## 🧪 Testing

### Run Test Suite

```bash
# Full system test
streamlit run test_app.py

# Or use built-in tests in main app
# Click "🧪 Test RAG Integration" in sidebar
```

### Manual Testing

```python
# Test individual components
python test_components.py

# Test PubMed connection
python test_pubmed.py

# Test BioMistral model
python test_biomistral.py
```

## 📁 Project Structure

```
pubmed-agent-ai/
├── app.py                          # Main Streamlit application
├── test_app.py                     # Comprehensive test suite
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .env.example                    # Environment template
│
├── components/
│   └── llm.py                      # BioMistral model configuration
│
├── backend/
│   ├── abstract_retrieval/
│   │   ├── pubmed_retriever.py     # PubMed API integration
│   │   ├── mock_retriever.py       # Testing mock data
│   │   └── retriever_interface.py  # Abstract base class
│   │
│   ├── rag_pipeline/
│   │   ├── chromadb_rag.py         # RAG implementation
│   │   ├── embeddings.py           # Embedding models
│   │   └── rag_interface.py        # RAG abstract interface
│   │
│   └── data_repository/
│       ├── local_storage.py        # Local data persistence
│       ├── models.py               # Data models
│       └── repository_interface.py # Storage interface
│
├── config/
│   └── logging_config.py           # Logging configuration
│
├── data/
│   ├── storage/                    # Local query storage
│   └── vector_store/               # ChromaDB vector storage
│
└── docs/
    ├── images/                     # Screenshots and diagrams
    └── examples/                   # Usage examples
```

## 🔗 API Integration

### PubMed E-utilities

```python
# Get your free API key from:
# https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

# Rate limits:
# - With API key: 10 requests/second
# - Without key: 3 requests/second
```

### HuggingFace Models

```python
# For private/gated models, get token from:
# https://huggingface.co/settings/tokens

# Models used:
# - BioMistral/BioMistral-7B (Medical LLM)
# - microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext (Embeddings)
```

## 📊 Performance Benchmarks

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **RAM** | 8GB | 16GB | 32GB+ |
| **GPU** | None | GTX 1660 | RTX 3080+ |
| **VRAM** | N/A | 6GB | 12GB+ |
| **Storage** | 20GB | 50GB | 100GB+ |
| **Internet** | 10 Mbps | 50 Mbps | 100 Mbps+ |

### Response Times

| Mode | Papers | Processing Time | Quality |
|------|--------|----------------|---------|
| **Knowledge Only** | 0 | 2-5s | Good |
| **PubMed Simple** | 10-20 | 10-30s | Better |
| **RAG Enhanced** | 10-20 | 30-60s | Best |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/your-username/pubmed-agent-ai.git
cd pubmed-agent-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black app.py backend/ components/
```

### Areas for Contribution

- 🔬 **New Medical Models**: Integration of specialized models
- 📊 **Data Visualization**: Charts and graphs for research insights  
- 🌐 **Multi-language**: Support for non-English papers
- 🔍 **Advanced Search**: Filters, date ranges, journal selection
- 📱 **Mobile UI**: Responsive design improvements
- 🧪 **Testing**: Expanded test coverage

## 📝 Citation

If you use PubMed Agent AI in your research, please cite:

```bibtex
@software{pubmed_agent_ai,
  title={PubMed Agent AI: RAG-Enhanced Biomedical Research Assistant},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/pubmed-agent-ai}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**Important**: PubMed Agent AI is a research tool and should not be used for clinical decision-making. Always consult qualified healthcare professionals for medical advice. The AI responses are based on available literature and may not reflect the most current medical standards or individual patient circumstances.

## 🆘 Support

### Getting Help

- 📖 **Documentation**: Check this README and inline help
- 🐛 **Issues**: [Report bugs](https://github.com/your-username/pubmed-agent-ai/issues)
- 💬 **Discussions**: [Ask questions](https://github.com/your-username/pubmed-agent-ai/discussions)
- 📧 **Email**: support@pubmed-agent-ai.com

### Common Issues

#### **BioMistral Not Loading**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### **PubMed API Errors**
```bash
# Check rate limits and API key
export PUBMED_API_KEY="your_key_here"
export PUBMED_EMAIL="your.email@example.com"
```

#### **ChromaDB Issues**
```bash
# Clear vector database
rm -rf data/vector_store/
mkdir -p data/vector_store/
```

#### **Memory Issues**
```python
# Reduce model precision in components/llm.py
load_in_4bit = True
torch_dtype = torch.float16
```

### FAQ

**Q: How accurate are the AI responses?**
A: Responses are based on peer-reviewed literature and specialized medical training. However, always verify critical information with healthcare professionals.

**Q: Can I use this for clinical practice?**
A: No, this is a research tool only. Not intended for clinical decision-making.

**Q: How current is the PubMed data?**
A: PubMed is updated daily. The app retrieves the most recent papers available.

**Q: Can I run this without internet?**
A: Limited functionality. You can use BioMistral Knowledge mode offline, but PubMed search requires internet access.

**Q: How do I get a PubMed API key?**
A: Visit [NCBI E-utilities](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/) for free API key registration.

## 🚀 Roadmap

### Version 2.0 (Planned)
- [ ] **Multi-modal Support**: Analysis of figures and tables
- [ ] **Clinical Trial Integration**: ClinicalTrials.gov data
- [ ] **Knowledge Graphs**: Relationship mapping between concepts
- [ ] **Real-time Alerts**: Notifications for new relevant papers
- [ ] **Collaborative Features**: Team research capabilities

### Version 1.5 (In Progress)
- [ ] **Enhanced UI**: Improved user experience design
- [ ] **Export Features**: PDF reports and citations
- [ ] **Advanced Analytics**: Research trend analysis
- [ ] **Mobile Optimization**: Better mobile device support

## 🏆 Acknowledgments

- **BioMistral Team**: For the specialized medical language model
- **NCBI/PubMed**: For providing free access to biomedical literature
- **HuggingFace**: For model hosting and transformation libraries
- **Streamlit**: For the excellent web application framework
- **ChromaDB**: For efficient vector storage and retrieval

---

<div align="center">

**Made with ❤️ for the biomedical research community**

[🌟 Star this project](https://github.com/your-username/pubmed-agent-ai) • [🐛 Report Bug](https://github.com/your-username/pubmed-agent-ai/issues) • [💡 Request Feature](https://github.com/your-username/pubmed-agent-ai/issues)

</div>