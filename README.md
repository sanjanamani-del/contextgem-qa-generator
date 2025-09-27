# ContextGem Q&A Dataset Generator

A system that automatically generates question-answer pairs from business documentation using ContextGem and GPT-5. Designed for creating fine-tuning datasets for language models.

## Overview

This tool takes business documents (cost guides, database schemas, process documentation) and generates diverse Q&A pairs suitable for training smaller language models. It uses ContextGem's structured extraction framework with GPT-5 for cost-effective, high-quality dataset generation.

**Key Features:**
- Generates 1-1000+ Q&A pairs from business documents
- Automatic batch processing (25 questions per batch)
- Zero exact duplicates with quality validation
- Outputs CSV and JSON formats for fine-tuning
- Cost-effective using GPT-5 instead of GPT-4

## Installation

### Prerequisites
- Python 3.10 or higher
- OpenAI API key with GPT-5 access

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/sanjanamani-del/contextgem-qa-generator.git
   cd contextgem-qa-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv contextgem-env
   # On Windows:
   contextgem-env\Scripts\activate
   # On macOS/Linux:
   source contextgem-env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key**
   ```bash
   # Option 1: Environment variable (recommended)
   export OPENAI_API_KEY="your-api-key-here"
   
   # Option 2: Create .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

5. **Prepare context files**
   ```bash
   mkdir context_files
   # Place your business documents (.txt, .md, .json) in this directory
   ```

6. **Update file path in script**
   Edit `contextgem_qa_generator.py` and change:
   ```python
   CONTEXT_DIR = "/path/to/your/context_files"
   ```

## Usage

1. **Basic generation**
   ```bash
   python contextgem_qa_generator.py
   ```

2. **Follow prompts**
   ```
   How many Q/A pairs do you want to generate? 100
   ```

3. **Output files**
   - `qa_dataset.csv` - For fine-tuning (prompt, completion columns)
   - `qa_dataset.json` - For analysis and review

## Performance

- **Small datasets (≤25 questions)**: 1-2 minutes
- **Medium datasets (100 questions)**: 3-4 minutes  
- **Large datasets (500+ questions)**: 15-20 minutes
- **Cost**: ~$2-5 per 100 questions using GPT-5

## Troubleshooting

### Common Issues

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'contextgem'` | Run `pip install contextgem` |
| `OpenAI API key not found` | Set `OPENAI_API_KEY` environment variable |
| `No context found` | Check context files directory path |
| `ContextGem generation failed` | Verify API key has GPT-5 access |

### Support
- Check that all context files contain text content
- Ensure stable internet connection for API calls
- Verify OpenAI account has sufficient credits
- For batch failures, the system will retry automatically

## System Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies
- **Network**: Stable internet for API calls
- **OS**: Windows 10+, macOS 10.14+, or Linux

## Example Output

The system generates business-relevant Q&A pairs like:

```
Q: How do you calculate the landed cost of a sourced component in the WINX MRP system?
A: Use WINX_MATERIAL_MASTER fields where SOURCE = 'Sourced': Component Cost = MATERIAL_COST + (MATERIAL_COST × TARIFF_RATE) + SHIPPING_COST...
```

## License

For internal business use. Contains proprietary business logic for ContextGem integration.
