import argparse
import json
import os
import logging
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import re
from typing import List

# API 配置
API_KEY = '' 

def setup_logging(output_file: str) -> logging.Logger:
    """设置日志记录"""
    log_dir = os.path.join(os.path.dirname(output_file), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, os.path.basename(output_file).replace('.json', '.log'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'), #mode='a', 
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_fidelity(source: str, target: str, translation: str, idiom: str, meaning: str, client) -> dict:
    """
    评测翻译保真度
    
    Args:
        source: 源句子
        target: 参考翻译
        translation: 机器翻译
        idiom: 习语
        meaning: 习语含义
        client: OpenAI客户端
        
    Returns:
        评测结果
    """
    try:
        # 构建提示
        prompt = f"""
Please evaluate the translation quality of the machine translation in conveying the given meaning.

Machine translation: {translation}

Meaning: {meaning}

Evaluation criteria:

Based on the machine translation, does it accurately convey the following meaning: {meaning}?

Please rate the following dimension (0-3 points) and provide a detailed explanation:

Fidelity: To what extent does the machine translation accurately convey the meaning provided?

*   0 points: Conveys a meaning that is opposite or contradictory to the provided meaning.
*   1 point: Does not convey any aspect of the provided meaning; the translation is non-idiomatic or literal and misses the intended sense.
*   2 points: Partially conveys the meaning.
*   3 points: Perfectly conveys the meaning.

Please provide your response in this JSON format:

```json
{{
"Fidelity": {{
"score": Rating from 0-3,
"justification": "[Brief justification for your rating]"
}}
}}
```
"""
        logging.info(f"评测保真度: {idiom}")
        # 调用LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            top_p = 1,
            response_format={"type": "json_object"},
            n=20
        )
        
        # 提取响应
        result = response.choices[0].message.content.strip()
        scores = json.loads(result)
        
        # 现在只有Fidelity一个维度，不需要计算平均分
        return scores
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"评测保真度时出错: {error_msg}")
        
        # 如果是超时错误，直接结束程序
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            logging.error("检测到API请求超时，程序将退出")
            import sys
            sys.exit(1)
            
        return {
            "Fidelity": {
                "score": 0,
                "justification": f"Error: {error_msg}"
            }
        }

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_usage_rate(meaning: str, translation: str, original_translation: str, client) -> dict:
    """
    评测翻译是否使用提供的hint
    
    Args:
        meaning: 习语含义
        translation: 机器翻译
        client: OpenAI客户端
        
    Returns:
        评测结果
    """
    try:
        # 构建提示
        prompt = f"""
You will receive a meaning (Provided Meaning), an original translation (Original Translation), and a new translation (Translation). Your task is to evaluate whether the Translation effectively uses the Provided Meaning, specifically by incorporating information or nuances present in the Provided Meaning that are *not* present in the Original Translation.

Provided Meaning: {meaning}
Original Translation: {original_translation}
Translation: {translation}

Evaluation Criteria:

*   If the Translation incorporates elements of the Provided Meaning that are absent in the Original Translation, give a score of 1, else 0.

Please respond using the following JSON format:
{{
"score": 0 or 1,
"explanation": "Analysis of how the Translation incorporates elements of the Provided Meaning, especially highlighting elements present in the Meaning but absent in the Original Translation.  Explain why the score was assigned.",
"evidence": "The portion of the Translation that uses the elements from the meaning, and a comparison to the corresponding (or absent) portion in the Original Translation."
}}
"""
        #logging.info(f"评测使用率: {prompt}")
        # 调用LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        # 提取响应
        result = response.choices[0].message.content.strip()
        return json.loads(result)
    
    except Exception as e:
        error_msg = str(e)
        logging.error(f"评测使用率时出错: {error_msg}")
        
        # 如果是超时错误，直接结束程序
        if "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            logging.error("检测到API请求超时，程序将退出")
            import sys
            sys.exit(1)
            
        return {
            "score": 0,
            "explanation": f"Error: {error_msg}",
            "evidence": ""
        }

def process_file(input_file: str, output_file: str, api_key: str, data_num: int = None, modes: List[str] = None) -> None:
    """
    处理输入文件，评测翻译结果
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        api_key: API密钥
    """
    # 设置日志
    logger = setup_logging(output_file)
    logger.info(f"开始处理文件: 输入文件={input_file}")
    print(f"output_file={output_file}")
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 设置API客户端
        client = OpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key=api_key,
            timeout=60  # 设置60秒超时时间
        )
        
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        logger.info(f"读取到 {len(input_data)} 条数据")

        # 限制读取的条目数
        if data_num is not None:
            input_data = input_data[:data_num]
        
        # 已评测的数据列表
        evaluated_data = []
        
        # 检查输出文件是否存在，如果存在则读取已评测数据
        start_idx = 0
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    evaluated_data = json.load(f)
                    start_idx = len(evaluated_data)
                    
                # 如果已评测的数据数量超过或等于要评测的数据数量，则无需继续评测
                if start_idx >= len(input_data):
                    logger.info(f"已评测所有数据，无需继续评测")
                    return
                    
                logger.info(f"发现已评测 {start_idx} 条数据，将从第 {start_idx+1} 条数据开始评测")
            except json.JSONDecodeError:
                logger.warning(f"输出文件 {output_file} 格式有误，将从头开始评测")
                # 确保输出文件存在且为有效的JSON数组
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False)
        else:
            # 创建空的输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False)
        
        logger.info(f"将评测从第 {start_idx+1} 条开始，共 {len(input_data)-start_idx} 条数据")

        # 记录每个模式的分数 - 只保留Fidelity
        scores_summary = {mode: {"Fidelity": []} for mode in modes}
        
        # 统计各模式下的Usage.score比例
        usage_score_counts = {mode: {"score_1": 0, "score_0": 0} for mode in modes}

        # 统计web_search_meaning模式下的relevance_flag比例
        web_search_meaning_relevance_flag_1_count = 0
        web_search_meaning_relevance_flag_0_count = 0
        web_search_meaning_usage_score_1_count = 0
        web_search_meaning_usage_score_0_count = 0

        # 遍历每条数据，从start_idx开始评测
        for idx in tqdm(range(start_idx, len(input_data)), desc="评测进度"):
            item = input_data[idx]
            try:
                # 获取基础信息
                source = item.get("source", "")
                target = item.get("target", "")
                idiom = item.get("idiom_in_source", "")
                
                # 初始化评分结果
                if "scores" not in item:
                    item["scores"] = {}
                
                # 获取习语含义 
                groundtruth_meaning = item["groundtruth_meaning"] 
                
                original_translation = item["translation"]["no_hint"]
                print(f"original_translation: {original_translation}")
                # 遍历每种翻译模式
                for mode in modes:
                    if mode in item.get("translation", {}):
                        translation = item["translation"][mode]
                        
                        # 评测保真度
                        logger.info(f"评测保真度: idx={idx}, mode={mode}")
                        fidelity_result = evaluate_fidelity(source, target, translation, idiom, groundtruth_meaning, client)
                        
                        # 存储保真度评测结果
                        if mode not in item["scores"]:
                            item["scores"][mode] = {}
                        item["scores"][mode] = fidelity_result

                        # 记录分数 - 只记录Fidelity分数
                        scores_summary[mode]["Fidelity"].append(fidelity_result["Fidelity"]["score"])
                        
                        # 评测使用率（除了no_hint模式）
                        meaning = ""
                        if mode != "no_hint":
                            # 根据不同模式选择不同的meaning
                            meaning = item.get(mode)
                            
                            if meaning:  # 确保meaning不为空
                                logger.info(f"评测使用率: idx={idx}, mode={mode}")
                                usage_result = evaluate_usage_rate(meaning, translation, original_translation, client)
                                
                                # 存储使用率评测结果
                                if "Usage" not in item["scores"][mode]:
                                    item["scores"][mode]["Usage"] = {}
                                item["scores"][mode]["Usage"] = usage_result

                                # 统计Usage.score
                                if usage_result["score"] == 1:
                                    usage_score_counts[mode]["score_1"] += 1
                                elif usage_result["score"] == 0:
                                    usage_score_counts[mode]["score_0"] += 1

                                # 统计hint模式的relevance_flag比例
                                if mode == "web_search_meaning":
                                    if usage_result["score"] == 1:
                                        web_search_meaning_usage_score_1_count += 1
                                        relevance_flag = item.get("web_search_meaning", {}).get("relevance_flag", None)
                                        if relevance_flag is not None:
                                            if relevance_flag == 1:
                                                web_search_meaning_relevance_flag_1_count += 1
                                    elif usage_result["score"] == 0:
                                        web_search_meaning_usage_score_0_count += 1
                                        relevance_flag = item.get("web_search_meaning", {}).get("relevance_flag", None)
                                        if relevance_flag is not None:
                                            if relevance_flag == 0:
                                                web_search_meaning_relevance_flag_0_count += 1
                
                # 将评测结果添加到已评测数据列表
                evaluated_data.append(item)
                
                # 每处理完一条数据，就保存已评测的数据
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluated_data, f, ensure_ascii=False, indent=4)
                logger.info(f"已完成第 {idx+1}/{len(input_data)} 条数据评测，并保存结果")
            
            except Exception as e:
                # 捕获 traceback，定位错误发生的文件和行号
                import sys, traceback
                tb = sys.exc_info()[2]
                frames = traceback.extract_tb(tb)
                # 最后一帧即为异常抛出的代码位置
                filename, line_no, func_name, text = frames[-1]
                
                error_msg = str(e)
                logger.error(
                    f"处理第 {idx+1} 条数据时出错: {error_msg} "
                    f"(文件 \"{filename}\" 第 {line_no} 行, 函数 \"{func_name}\")",
                    exc_info=True
                )
                
                # 发生错误时也保存已评测的数据
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(evaluated_data, f, ensure_ascii=False, indent=4)
                logger.info(f"在错误发生后保存当前结果")
                
                # 如果是超时错误，直接退出程序
                if "timed out" in error_msg.lower() or "timeout" in error_msg.lower() or "request timed out" in error_msg.lower():
                    logger.error(f"检测到超时错误，程序将退出")
                    sys.exit(1)

        # 计算并记录每个模式的平均分 - 只计算Fidelity
        for mode in scores_summary:
            if scores_summary[mode]["Fidelity"]:
                avg_fidelity = round(sum(scores_summary[mode]["Fidelity"]) / len(scores_summary[mode]["Fidelity"]), 2)
                logger.info(f"{mode} 模式的平均分: Fidelity={avg_fidelity}")
        
        # 计算并记录各模式的Usage.score比例
        for mode in usage_score_counts:
            total_count = usage_score_counts[mode]["score_1"] + usage_score_counts[mode]["score_0"]
            if total_count > 0:
                score_1_ratio = usage_score_counts[mode]["score_1"] / total_count
                logger.info(f"{mode} 模式中 Usage.score=1 的比例: {score_1_ratio:.2%}")

        # 计算并记录hint模式的relevance_flag比例
        if web_search_meaning_usage_score_1_count > 0:
            web_search_meaning_relevance_flag_1_ratio = web_search_meaning_relevance_flag_1_count / web_search_meaning_usage_score_1_count
            logger.info(f"web_search_meaning模式中 Usage.score=1 的数据中 web_search_meaning.relevance_flag=1 的比例: {web_search_meaning_relevance_flag_1_ratio:.2%};web_search_meaning模式中 Usage.score=1 的数据中 web_search_meaning.relevance_flag=0 的比例: {1-web_search_meaning_relevance_flag_1_ratio:.2%}")
        
        if web_search_meaning_usage_score_0_count > 0:
            web_search_meaning_relevance_flag_0_ratio = web_search_meaning_relevance_flag_0_count / web_search_meaning_usage_score_0_count
            logger.info(f"web_search_meaning模式中 Usage.score=0 的数据中 web_search_meaning.relevance_flag=1 的比例: {1-web_search_meaning_relevance_flag_0_ratio:.2%};web_search_meaning模式中 Usage.score=0 的数据中 web_search_meaning.relevance_flag=0 的比例: {web_search_meaning_relevance_flag_0_ratio:.2%}")
        
        logger.info(f"评测完成，结果已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}", exc_info=True)

def get_output_file_path(input_file: str) -> str:
    """
    根据输入文件路径生成输出文件路径
    
    Args:
        input_file: 输入文件路径
        
    Returns:
        输出文件路径
    """
    # 获取文件名
    file_name = os.path.basename(input_file)
    
    # 获取目录路径
    dir_path = os.path.dirname(input_file)
    
    # 替换路径中的 'results' 为 'gpt4_eval_results'
    if 'data' in dir_path:
        output_dir = dir_path.replace('results', 'gpt4_eval_results', 1)
    else:
        # 如果路径中没有 'data'，则在路径前添加 'results'
        output_dir = os.path.join('gpt4_eval_results', dir_path)
    
    # 组合输出文件路径
    output_file = os.path.join(output_dir, file_name)
    
    return output_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评测翻译结果")
    
    parser.add_argument("-i", "--input_file", type=str, required=True,
                        help="输入文件路径")
    parser.add_argument("-o", "--output_file", type=str,
                        help="输出文件路径，默认为替换输入路径中的'data'为'results'")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help="API密钥")
    parser.add_argument("--data_num", type=int, default=None,
                        help="读取前多少条数据进行评测，默认为全部")
    parser.add_argument("--modes", nargs="+", default=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                    choices=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                    help="Mode(s) for translation: no_hint, groundtruth_meaning, opposite_meaning, perturbation_meaning, literal_meaning, similar_literal_meaning, perturbation_literal_meaning")
    
    args = parser.parse_args()
    
    # 如果未指定输出文件，生成默认输出路径
    if not args.output_file:
        args.output_file = get_output_file_path(args.input_file)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # 处理文件
    process_file(args.input_file, args.output_file, args.api_key, args.data_num, args.modes)
