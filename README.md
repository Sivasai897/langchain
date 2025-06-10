# LangChain Learning Project

This repository contains a collection of examples and implementations using LangChain, a framework for developing applications powered by language models.

## Project Structure

The project is organized into several directories, each focusing on different aspects of LangChain:

- `1_chat_models/`: Examples of using different chat models with LangChain
- `2_prompt_templates/`: Implementation of various prompt templates
- `3_chains/`: Examples of LangChain chains and their usage
- `4_rag/`: Implementation of Retrieval-Augmented Generation (RAG) systems

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchain_learning
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project uses the following main dependencies:
- langchain-openai >= 0.1.8
- langchain >= 0.2.1
- langchain-community >= 0.2.1
- langchain-anthropic >= 0.1.15
- langchain-google-genai >= 1.0.5
- langchain-google-firestore >= 0.3.0
- chromadb >= 0.5.0
- sentence-transformers >= 3.0.0
- And other supporting libraries

## Environment Setup

Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Usage

Each directory contains specific examples and implementations. Navigate to the respective directory to explore different features:

1. Chat Models (`1_chat_models/`):
   - Examples of using different LLM providers
   - Chat model configurations and customizations

2. Prompt Templates (`2_prompt_templates/`):
   - Various prompt template implementations
   - Template customization and management

3. Chains (`3_chains/`):
   - Implementation of different LangChain chains
   - Chain composition and customization

4. RAG Systems (`4_rag/`):
   - Retrieval-Augmented Generation implementations
   - Vector store integration
   - Document processing and retrieval

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a new branch
3. Making your changes
4. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain team for the amazing framework
- All contributors and maintainers
