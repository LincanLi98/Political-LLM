# config.py
"""
模型配置文件
"""

# 可用的模型列表
AVAILABLE_MODELS = {
    # OpenAI 模型
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini",
    ],
    
    # AWS Bedrock - Mistral
    "mistral": [
        "mistral.mistral-large-2402-v1:0",
        "mistral.mistral-7b-instruct-v0:2",
    ],
    
    # AWS Bedrock - Mixtral
    "mixtral": [
        "mistral.mixtral-8x7b-instruct-v0:1",
    ],
    
    # AWS Bedrock - Llama 3.1
    "llama3.1": [
        "meta.llama3-1-8b-instruct-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
    ],
    
    # AWS Bedrock - Llama 3.2
    "llama3.2": [
        "us.meta.llama3-2-1b-instruct-v1:0",
        "us.meta.llama3-2-3b-instruct-v1:0",
        "us.meta.llama3-2-11b-instruct-v1:0",
        "us.meta.llama3-2-90b-instruct-v1:0",
    ]
}

# 默认模型
DEFAULT_MODEL = "gpt-4o-mini"

def get_model_family(model_id: str) -> str:
    """获取模型所属的家族"""
    for family, models in AVAILABLE_MODELS.items():
        if model_id in models:
            return family
    return "unknown"

def list_all_models():
    """列出所有可用模型"""
    print("\n🤖 Available Models:\n")
    for family, models in AVAILABLE_MODELS.items():
        print(f"📦 {family.upper()}:")
        for model_id in models:
            print(f"   • {model_id}")
        print()

def is_valid_model(model_id: str) -> bool:
    """检查模型ID是否有效"""
    for family, models in AVAILABLE_MODELS.items():
        if model_id in models:
            return True
    return False