# bedrock_client.py

import os
import boto3
import json
from functools import lru_cache
from botocore.exceptions import ClientError

# Fallback default
DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

@lru_cache(maxsize=None)
def _get_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)

@lru_cache(maxsize=None)
def _get_openai_client():
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")


def invoke_model(
    model_id: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    stop: list[str] = None,
) -> dict:

    """
    Dispatch to the correct Bedrock model or OpenAI model in the right region,
    formatting prompt + payload and returning {'generation': text}.
    Raises ValueError if model_id isn't one of the supported variants.
    """
    key = model_id.lower()

    # Check if it's an OpenAI model
    is_openai = key.startswith("gpt-") or key.startswith("o1-")
    
    if is_openai:
        return _invoke_openai_model(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop
        )

    # Identify exactly which family we're calling
    is_mistral    = key.startswith("mistral.mistral-")
    is_mixtral    = key.startswith("mistral.mixtral-")
    is_llama3_8b  = key.startswith("meta.llama3-1-8b-instruct")
    is_llama3_70b = key.startswith("meta.llama3-1-70b-instruct")

    is_llama32_1b = key.startswith("us.meta.llama3-2-1b-instruct-v1:0")
    is_llama32_3b = key.startswith("us.meta.llama3-2-3b-instruct-v1:0")
    is_llama32_11b = key.startswith("us.meta.llama3-2-11b-instruct-v1:0")
    is_llama32_90b = key.startswith("us.meta.llama3-2-90b-instruct-v1:0")

    # Pick region (my specific AWS perms mean llama3 reside in us-west-2)
    if is_mistral or is_mixtral:
        region = DEFAULT_REGION
    elif is_llama3_8b or is_llama3_70b:
        region = "us-west-2"
    elif is_llama32_1b or is_llama32_3b or is_llama32_11b or is_llama32_90b:
        region = "us-east-1"
    else:
        raise ValueError(
            f"Invalid model_id '{model_id}'. "
            "Supported prefixes: "
            "'gpt-', 'o1-', "
            "'mistral.mistral-', 'mistral.mixtral-', "
            "'meta.llama3-1-8b-instruct-', 'meta.llama3-1-70b-instruct-', "
            "'us.meta.llama3-2-1b-instruct-', 'us.meta.llama3-2-3b-instruct-', "
            "'us.meta.llama3-2-11b-instruct-', 'us.meta.llama3-2-90b-instruct-'."
        )


    client = _get_client(region)

    if is_mistral:
        # Mistral Large
        formatted = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "prompt":      formatted,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

    elif is_mixtral:
        # Mixtral 8×7B Instruct
        formatted = f"<s>[INST] {prompt} [/INST]"
        payload = {
            "prompt":      formatted,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }

    elif is_llama3_8b:
        # Llama 3 8B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    elif is_llama3_70b:
        # Llama 3 70B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
    
    elif is_llama32_1b:
        # Llama 3.2 1B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    elif is_llama32_3b:
        # Llama 3.2 3B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    elif is_llama32_11b:
        # Llama 3.2 11B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
    
    elif is_llama32_90b:
        # Llama 3.2 90B Instruct
        formatted = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
            f"{prompt}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
        payload = {
            "prompt":      formatted,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }

    # Common knobs
    if top_p is not None:
        payload["top_p"] = top_p
    if (is_mistral or is_mixtral) and top_k is not None:
        payload["top_k"] = top_k
    if (is_mistral or is_mixtral) and stop is not None:
        payload["stop"]  = stop

    # Invoke!
    try:
        resp = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

    except ClientError as e:
        print(f"🛑 Bedrock access error for model: {model_id}")
        print(f"🧾 Region used: {region}")
        print(f"📤 Payload preview: {json.dumps(payload)[:200]}...")
        raise

    raw = json.loads(resp["body"].read().decode())

    # Pull out the text field
    if is_mistral or is_mixtral:
        outputs = raw.get("outputs", [])
        text = outputs[0].get("text", "") if outputs else ""
    else:
        text = raw.get("generation", "")

    return {"generation": text}


def _invoke_openai_model(
    model_id: str,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    top_p: float = None,
    stop: list[str] = None,
) -> dict:
    """
    Call OpenAI API and return response in the same format as Bedrock models.
    """
    try:
        client = _get_openai_client()
        
        # Build the API call parameters
        api_params = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add optional parameters
        if top_p is not None:
            api_params["top_p"] = top_p
        if stop is not None:
            api_params["stop"] = stop
        
        response = client.chat.completions.create(**api_params)
        
        text = response.choices[0].message.content
        return {"generation": text}
        
    except Exception as e:
        print(f"🛑 OpenAI API error for model: {model_id}")
        print(f"Error: {e}")
        raise