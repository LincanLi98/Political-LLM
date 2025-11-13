markdown# Political-LLM Experiments

This repository contains three experimental setups to study how LLMs simulate voting behavior based on demographic and political profiles from the 2016 ANES (American National Election Studies) dataset.

## 📁 Directory Structure
```
├── FPP_ANES_2016_base/     # Experiment using ANES original political ideology
├── FPP_ANES_2016_NP/       # Experiment WITHOUT political ideology
└── FPP_ANES_2016_gen/      # Experiment using LLM-generated political ideology
```

## 🔬 Experiment Descriptions

### 1. **FPP_ANES_2016_base** (Baseline)
- Uses political ideology directly from ANES data (e.g., "extremely liberal", "moderate", "conservative")
- Identity profiles include all demographic information + ANES political ideology
- **Run command**: `python run.py --model <model_id>`

### 2. **FPP_ANES_2016_NP** (No Political Ideology)
- Excludes ALL political ideology information from identity profiles
- Tests voting behavior based purely on demographics (age, race, gender, state, etc.)
- **Run command**: `python run.py --model <model_id> --no-llm-ideology`

### 3. **FPP_ANES_2016_gen** (LLM-Generated Ideology)
- Asks LLM to generate political ideology for each identity profile
- Makes 2 API calls per identity: (1) generate ideology, (2) get vote
- Tests whether LLM can infer political ideology from demographics
- **Run command**: `python run.py --model <model_id>`

## 🚀 Quick Start

### Prerequisites

#### Required Python Packages
```bash
pip install boto3 openai pandas
```

#### Environment Variables

**For OpenAI models:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

**For AWS Bedrock models:**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"  # or your preferred region
```

### Running Experiments

#### 1. List Available Models
```bash
python run.py --list
```

#### 2. Run with OpenAI Models (Fast)
```bash
# Baseline experiment
cd FPP_ANES_2016_base
python run.py --model gpt-4o-mini

# No ideology experiment
cd ../FPP_ANES_2016_NP
python run.py --model gpt-4o-mini --no-llm-ideology

# LLM-generated ideology experiment
cd ../FPP_ANES_2016_gen
python run.py --model gpt-4o-mini
```

#### 3. Run with AWS Bedrock Models (Requires Rate Limiting)
```bash
# Use longer delays for Bedrock to avoid throttling
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0

# For FPP_ANES_2016_gen, use even longer delay (2 API calls per identity)
cd FPP_ANES_2016_gen
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 3.0
```

## 🤖 Supported Models

### OpenAI Models
- `gpt-4o` - Most capable, higher cost
- `gpt-4o-mini` - Balanced performance and cost (recommended)
- `gpt-3.5-turbo` - Fastest, lowest cost
- `o1-preview` - Advanced reasoning
- `o1-mini` - Efficient reasoning

### AWS Bedrock Models

#### Mistral Family
- `mistral.mistral-large-2402-v1:0` - Large model
- `mistral.mistral-7b-instruct-v0:2` - Efficient model

#### Mixtral Family
- `mistral.mixtral-8x7b-instruct-v0:1` - Mixture of experts

#### Llama 3.1 Family
- `meta.llama3-1-8b-instruct-v1:0` - Small, efficient
- `meta.llama3-1-70b-instruct-v1:0` - Large, high quality

#### Llama 3.2 Family
- `us.meta.llama3-2-1b-instruct-v1:0` - Very small
- `us.meta.llama3-2-3b-instruct-v1:0` - Small
- `us.meta.llama3-2-11b-instruct-v1:0` - Medium
- `us.meta.llama3-2-90b-instruct-v1:0` - Very large

## ⚙️ Command Line Options
```bash
# Show all available models
python run.py --list

# Specify model
python run.py --model 

# Set delay between requests (for Bedrock rate limiting)
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0

# Disable LLM-generated ideology (for FPP_ANES_2016_NP)
python run.py --model gpt-4o-mini --no-llm-ideology

# Disable candidate policy information
python run.py --model gpt-4o-mini --no-candidate-info
```

## 📊 Output Files

After running experiments, the following files are generated:
```
responses/
├── results.txt           # Complete AI responses for each identity
├── prompt_history.txt    # All prompts sent to the model
└── votes.txt            # Final voting statistics

republican_supporter/
└── identities.txt       # All identities that voted Republican

democratic_supporter/
└── identities.txt       # All identities that voted Democratic

nopreference_supporter/
└── identities.txt       # All identities with no preference
```

## 🔧 Configuration

### Changing Models
Edit `config.py` to modify the default model or add new models:
```python
# config.py
DEFAULT_MODEL = "gpt-4o-mini"  # Change default here
```

### Adjusting Rate Limits
For AWS Bedrock, adjust the delay parameter:
```bash
# Baseline and NP experiments (1 API call per identity)
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0

# Gen experiment (2 API calls per identity)
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 3.0
```

## 🐛 Troubleshooting

### Rate Limit Errors (Bedrock)
**Error**: `ThrottlingException: Too many requests`

**Solution**: Increase the delay between requests
```bash
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 5.0
```

### OpenAI API Key Not Found
**Error**: `OPENAI_API_KEY environment variable not set`

**Solution**: Set your API key
```bash
export OPENAI_API_KEY="sk-..."
```

### AWS Credentials Error
**Error**: `Unable to locate credentials`

**Solution**: Configure AWS credentials
```bash
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-west-2"
```

### Model Not Found
**Error**: `Invalid model_id`

**Solution**: Check available models
```bash
python run.py --list
```

## 📝 File Descriptions

### Core Files (Common to all experiments)

#### `anes.py`
- Loads ANES 2016 survey data from CSV
- Converts raw data into natural language identity descriptions
- **Different versions**:
  - `FPP_ANES_2016_base`: Includes political ideology from ANES
  - `FPP_ANES_2016_NP`: Excludes political ideology
  - `FPP_ANES_2016_gen`: Excludes ideology (will be LLM-generated)

#### `Identity.py`
- Main class for processing identities and collecting votes
- Sends prompts to LLM and extracts voting preferences
- Handles retries and error recovery

#### `Poligenerator.py` (only used in FPP_ANES_2016_gen)
- Generates political ideology descriptions using LLM
- Asks: "When it comes to politics, would you describe yourself as..."
- Inserts generated ideology into identity profile

#### `bedrock_client.py`
- Unified API client for both OpenAI and AWS Bedrock
- Handles model-specific prompt formatting
- Region selection for Bedrock models

#### `config.py`
- Lists all available models
- Defines default model
- Helper functions for model information

#### `run.py`
- Main execution script
- Processes all identities sequentially
- Outputs voting statistics
