# Atlas 🎬

> AI-Powered Content Analysis Platform for Educational and Research Content

Atlas is a comprehensive platform that combines YouTube video analysis, academic paper research, and educational content generation into a unified AI-powered workflow.

## Features

### 🔍 YouTube Pipeline
- **Video Search**: Natural language search using YouTube Data API
- **Transcript Extraction**: Automatic subtitle fetching and processing
- **AI Summarization**: Technical content analysis with structured insights
- **Comparison Analysis**: Multi-video comparison with AI-powered insights

### 📚 Academic RAG System
- **Semantic Search**: Query academic papers using natural language
- **Citation Tracking**: Source papers with relevance scores
- **Vector Database**: LanceDB-powered semantic search
- **Paper Management**: Automatic PDF processing and indexing

### 📝 Educational Content
- **Assignment Generation**: AI-created hands-on learning exercises
- **Learning Objectives**: Structured educational outcomes
- **Progressive Tasks**: Step-by-step skill building activities
- **Assessment Criteria**: Clear success metrics and rubrics

### ⚡ Advanced Processing
- **Parallel Execution**: Concurrent processing for faster results
- **Real-time Tracking**: Progressive visualization of pipeline steps
- **Professional Interface**: Modern web UI with responsive design
- **Configurable Workers**: Adjustable concurrency for optimal performance

## Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- YouTube Data API key

### Installation
```bash
git clone https://github.com/ishandutta0098/atlas
cd atlas
pip install -r requirements.txt
```

### Environment Setup
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_key" >> .env
echo "YOUTUBE_API_KEY=your_youtube_key" >> .env
```

### Launch Atlas
```bash
python app.py
```

Access the web interface at `http://localhost:7860`

## Usage

### 1. YouTube Analysis
1. Enter a search query (e.g., "Python machine learning tutorial")
2. Configure max videos and workers
3. Click "Start Pipeline" to begin processing
4. View results: search → transcripts → summaries → comparison → assignments

### 2. Academic Papers Query
1. Ensure papers are in `papers/agents/` folder
2. Enter natural language query
3. Get AI responses with paper citations and excerpts

## Configuration

Key settings in `src/configs/config.yaml`:
- **Model**: OpenAI model selection
- **Workers**: Parallel processing configuration
- **API**: Timeout and retry settings
- **Paths**: Output directories and file locations

## Project Structure

```
atlas/
├── app.py                          # Main Gradio web interface
├── src/
│   ├── youtube_pipeline.py         # YouTube processing pipeline
│   ├── papers_rag.py              # Academic papers RAG system
│   ├── assignment_generator.py     # Educational content generator
│   ├── compare_youtube_outputs.py  # Video comparison analysis
│   └── configs/config.yaml         # Configuration settings
├── papers/agents/                  # Academic papers directory
└── requirements.txt               # Python dependencies
```

## License

MIT License - See LICENSE file for details