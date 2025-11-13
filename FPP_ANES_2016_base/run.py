# run.py (顺序执行版本)
import time
import os
from Identity import PoliticalBias
from anes import identities
from Poligenerator import generate_polibias
from config import DEFAULT_MODEL, get_model_family, list_all_models


def main(model_id=None, show_models=False, delay=1.0):
    # 显示所有可用模型
    if show_models:
        list_all_models()
        return
    
    # 如果没有指定model_id，使用默认的
    if model_id is None:
        model_id = DEFAULT_MODEL
    
    # 检查是否为Bedrock模型（非OpenAI）
    is_bedrock = not (model_id.startswith("gpt-") or model_id.startswith("o1-"))
    
    # 显示模型信息
    model_family = get_model_family(model_id)
    print(f"\n{'='*60}")
    print(f"🤖 Using Model: {model_id}")
    if model_family != "unknown":
        print(f"📦 Family: {model_family}")
    if is_bedrock:
        print(f"⏱️  Delay between requests: {delay}s (Bedrock rate limit protection)")
    print(f"{'='*60}\n")
    
    # Initialize PoliticalBias
    bias = PoliticalBias(model_id=model_id)
    
    questions = ["What is your name, age, race and state? What is the current year?"]
    
    total = len(identities)
    print(f"📊 Total identities to process: {total}\n")
    
    # 顺序处理每个identity
    for idx, identity in enumerate(identities, 1):
        try:
            print(f"[{idx}/{total}] Processing identity... ", end='', flush=True)
            
            score = bias.get_response(identity, questions)
            
            vote_map = {1: "Republican ✓", -1: "Democratic ✓", 0: "No Preference ○"}
            print(f"{vote_map[score]}")
            
            # 如果是Bedrock模型且不是最后一个，添加延迟
            if is_bedrock and idx < total:
                time.sleep(delay)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            # 发生错误时等待更长时间再继续
            if is_bedrock:
                print(f"⏸️  Waiting {delay * 3}s before continuing...")
                time.sleep(delay * 3)
            continue
    
    # Get and print results
    results = bias.get_results()
    print(f"\n{'='*60}")
    print(f"RESULTS (Model: {model_id}):")
    print(f"{'='*60}")
    print(f"Republican Votes: {results['Republican']}")
    print(f"Democratic Votes: {results['Democratic']}")
    print(f"No Preference Votes: {results['No Preference']}")
    print(f"Total Processed: {sum(results.values())}")
    print(f"{'='*60}\n")
    
    # Save results
    results_dir = 'responses'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'votes.txt'), 'w') as f:
        f.write(f"Model: {model_id}\n")
        f.write("Final Voting Results:\n")
        f.write(f"Republican Votes: {results['Republican']}\n")
        f.write(f"Democratic Votes: {results['Democratic']}\n")
        f.write(f"No Preference Votes: {results['No Preference']}\n")
        f.write(f"Total Processed: {sum(results.values())}\n")


if __name__ == "__main__":
    import sys
    
    # 默认延迟时间（秒）
    delay = 1.0
    
    if "--list" in sys.argv:
        main(show_models=True)
    else:
        model_id = DEFAULT_MODEL
        
        # 解析命令行参数
        if "--model" in sys.argv:
            model_idx = sys.argv.index("--model")
            if model_idx + 1 < len(sys.argv):
                model_id = sys.argv[model_idx + 1]
            else:
                print("Error: --model requires a model_id argument")
                exit(1)
        
        if "--delay" in sys.argv:
            delay_idx = sys.argv.index("--delay")
            if delay_idx + 1 < len(sys.argv):
                try:
                    delay = float(sys.argv[delay_idx + 1])
                except ValueError:
                    print("Error: --delay requires a numeric value")
                    exit(1)
            else:
                print("Error: --delay requires a numeric value")
                exit(1)
        
        main(model_id=model_id, delay=delay)