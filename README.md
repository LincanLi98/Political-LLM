# Political-LLM Experiments

This repository contains three experimental setups to study how LLMs simulate voting behavior based on demographic and political profiles from the 2016 ANES (American National Election Studies) dataset.

## 📁 Directory Structure
```
├── FPP_ANES_2016_base/         # Experiment using ANES original political ideology
├── FPP_ANES_2016_NP/           # Experiment WITHOUT political ideology
├── FPP_ANES_2016_gen/          # Experiment using LLM-generated political ideology
│
├── FPP_MANIFESTO_2025_base/    # Cross-national experiment using Manifesto dataset with original annotated ideological positions
├── FPP_MANIFESTO_2025_NP/      # Cross-national experiment WITHOUT ideological features (neutral textual inputs only)
├── FPP_MANIFESTO_2025_gen/     # Cross-national experiment using LLM-generated ideology embeddings derived from manifesto texts
│
└── Evaluation_Tools/           #
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


### 4. FPP_MANIFESTO_2025_base (Cross-National Baseline)

- Uses the **Manifesto Project (2025 edition)** dataset containing annotated party manifestos across multiple countries and election years.
- Incorporates **original ideology annotations** provided by the Manifesto Corpus (RILE-based left–right positions).
- Tests how well the Political-LLM framework generalizes from U.S.-specific (ANES) to **cross-national contexts** using real ideology labels.
- Run command:
  ```bash
  cd FPP_MANIFESTO_2025_base
  python run.py --model <model_id>
  ```


### 5. FPP_MANIFESTO_2025_NP (Cross-National, No Political Ideology)

- Excludes all ideological information from manifesto texts.
- Tests whether LLMs can predict **party positions and vote patterns** purely from policy language, without explicit left–right ideology cues.
- Useful for analyzing the contribution of ideological content vs. factual policy descriptions in cross-country contexts.
- Run command:
  ```bash
  cd FPP_MANIFESTO_2025_NP
  python run.py --model <model_id> --no-llm-ideology
  ```


### 6. FPP_MANIFESTO_2025_gen (Cross-National, LLM-Generated Ideology)

- Uses **ManifestoBERTa** (fine-tuned multilingual XLM-RoBERTa model) to extract 56 policy-topic distributions for each party manifesto.
- Asks LLMs to infer ideological embeddings ("left", "center", "right") based on these topic profiles — simulating **LLM-generated ideology** in a multilingual political space.
- Makes 2 API calls per manifesto: (1) generate ideological embedding, (2) predict election or stance outcomes.
- Evaluates whether LLMs can generalize ideological reasoning across **countries, languages, and election years**.
- Run command:
  ```bash
  cd FPP_MANIFESTO_2025_gen
  python run.py --model <model_id>
  ```


###  Summary Table

| Experiment                  | Dataset                                | Ideology Source               | Description                              |
| --------------------------- | -------------------------------------- | ----------------------------- | ---------------------------------------- |
| FPP_ANES_2016_base          | ANES 2016 (U.S.)                       | Original ANES ideology labels | U.S. baseline with ground-truth ideology |
| FPP_ANES_2016_NP            | ANES 2016 (U.S.)                       | None                          | Demographics only (no ideology)          |
| FPP_ANES_2016_gen           | ANES 2016 (U.S.)                       | LLM-generated                 | LLM infers ideology from demographics    |
| FPP_MANIFESTO_2025_base     | Manifesto Corpus 2025 (Cross-national) | Original RILE annotations     | Ground-truth ideology across countries   |
| FPP_MANIFESTO_2025_NP       | Manifesto Corpus 2025                  | None                          | Cross-national text-only setup           |
| FPP_MANIFESTO_2025_gen      | Manifesto Corpus 2025                  | LLM-generated                 | Cross-national LLM-generated ideology    |

---



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

# --- MANIFESTO Experiments ---
# Cross-national baseline (using ground-truth RILE ideology)
cd ../../FPP_MANIFESTO_2025_base
python run.py --model gpt-4o-mini

# Cross-national, no ideology
cd ../FPP_MANIFESTO_2025_NP
python run.py --model gpt-4o-mini --no-llm-ideology

# Cross-national, LLM-generated ideology (ManifestoBERTa + LLM)
cd ../FPP_MANIFESTO_2025_gen
python run.py --model gpt-4o-mini
```

#### 3. Run with AWS Bedrock Models (Requires Rate Limiting)
```bash
# Use longer delays for Bedrock to avoid throttling
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0

# For FPP_ANES_2016_gen, use even longer delay (2 API calls per identity)
cd FPP_ANES_2016_gen
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 3.0

# Cross-national baseline (Manifesto 2025, ground-truth ideology)
cd ../../FPP_MANIFESTO_2025_base
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0

# Cross-national, no ideology
cd ../FPP_MANIFESTO_2025_NP
python run.py --model meta.llama3-1-70b-instruct-v1:0 --delay 2.0 --no-llm-ideology

# Cross-national, LLM-generated ideology (ManifestoBERTa + LLM)
cd ../FPP_MANIFESTO_2025_gen
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
- 

## 🧮 Evaluation and Analysis Tools

#### `evaluation.py`

- Provides a unified evaluation framework for all experiments.
- Computes core metrics (accuracy, F1, calibration error, ideology correlation) across experimental runs.
- Supports subgroup-level comparisons for fairness and robustness testing.

#### `fairness_report.py`

- Generates detailed subgroup fairness reports based on demographics (gender, age, education).
- Calculates within-group accuracy and bias metrics to assess representation balance.
- Outputs CSV and summary visualizations for inclusion in manuscript tables.

#### `uncertainty_quantification.py`

- Quantifies statistical uncertainty of simulation outcomes.
- Implements bootstrap resampling to estimate 95% confidence intervals for metrics (vote ratio, ideology alignment).
- Ensures transparency about simulation variability and reproducibility.

#### `Inclusivity_and_Transparency_Checklist.html / .pdf`

- Provides an auditable checklist for responsible deployment and reporting.
- Covers fairness, inclusivity, transparency, and accountability criteria based on *Model Cards* and *Datasheets* standards.
- Serves as a reproducibility and ethical compliance appendix for TMLR submission.
- 
