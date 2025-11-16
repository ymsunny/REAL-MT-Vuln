import argparse
import json
import csv
import os
import logging
import tqdm
import re
import numpy as np
from difflib import SequenceMatcher
from openai import OpenAI
import time
from typing import List, Dict, Any, Tuple, Optional
import socket, requests
from openai import OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

def setup_logger(log_file):
    import sys
    # 强制将 stdout/stderr 设为 utf-8，兼容 Windows 控制台
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 控制台处理程序
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # 文件处理程序，显式指定 utf-8 编码
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

# 读取文件

def read_file(input_file: str, lang_pair: str,
              json_lang_pairs: List[str],
              csv_lang_pairs: List[str]) -> List[Dict[str, Any]]:
     """根据语言对信息读取不同格式的文件"""
     data = []

     if lang_pair in json_lang_pairs:
         with open(input_file, 'r', encoding='utf-8') as f:
             data = json.load(f)

        # 验证 json 数据是否包含需要的字段
         for item in data:
             if "contains_idioms" not in item or "source" not in item or "english_target" not in item:
                 logging.warning(f"数据项缺少必要字段：{item}")

     elif lang_pair in csv_lang_pairs:
         with open(input_file, 'r', encoding='utf-8') as f:
             reader = csv.DictReader(f)
             for row in reader:
                 if "idiom" not in row or "sentence" not in row or "gold translation" not in row:
                     logging.warning(f"CSV行缺少必要字段：{row}")
                     continue
                 data.append(row)
     
     else:
         raise ValueError(f"不支持的语言对：{lang_pair}")
     
     return data

# 使用n-gram匹配source中的习语部分
def match_idiom_in_source(idiom: str, source: str) -> str:
    """
    查找source中与给定idiom最相似的短语
    返回匹配到的idiom_in_source
    """
    # 预处理习语和源文本，去除标点符号
    # 定义需要去除或替换的标点字符集
    punctuation_chars = '.,!?;:"\'[]()，。！？；：、""''（）'
    # 使用 re.escape 自动转义，避免手动写 \\[ \\] 等
    punctuation_pattern = '[' + re.escape(punctuation_chars) + ']'
    
    # 去除 idiom 中的标点
    idiom = re.sub(punctuation_pattern, '', idiom.strip())
    # 将源文本中的标点替换为空格，便于分词
    source_clean = re.sub(punctuation_pattern, ' ', source.strip())
    
    # 如果 idiom 在清洗后的 source 中
    if idiom in source_clean:
        return idiom
    
    # 转为小写进行匹配
    idiom_lower = idiom.lower()
    source_lower = source_clean.lower()
    
    # 如果小写匹配存在，找到原始大小写的版本
    if idiom_lower in source_lower:
        start_idx = source_lower.find(idiom_lower)
        return source_clean[start_idx:start_idx + len(idiom_lower)]
    
    # 使用滑动窗口在源文本中寻找最相似片段
    best_match = ""
    best_score = 0
    
    # 获取习语的单词数，用于设置搜索窗口大小
    idiom_words = idiom.split()
    min_window = max(1, len(idiom_words) - 1)
    max_window = min(len(idiom_words) + 3, 10)  # 限制窗口最大值
    
    # 对清理后的源文本进行分词
    words = source_clean.split()
    
    # 使用不同大小的窗口搜索
    for window_size in range(min_window, max_window + 1):
        for i in range(len(words) - window_size + 1):
            phrase = " ".join(words[i:i+window_size])
            similarity = SequenceMatcher(None, idiom_lower, phrase.lower()).ratio()
            
            if similarity > best_score:
                best_score = similarity
                best_match = phrase
    
    # 如果找不到合理匹配，返回原始习语
    if best_score < 0.3:
        return idiom
        
    return best_match

# 调用LLM API的公共函数
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # 捕获所有异常类型
    retry=retry_if_exception_type(Exception),
    # 最终不抛出，而是返回兜底结果
    reraise=False,
    retry_error_callback=lambda state: json.dumps({
        "meaning": "call failed"
    })
)
def call_llm_api(client, prompt, system_message="You are a helpful assistant.", temperature=0):
    """
    LLM API 调用，自动重试 (最多5次，指数退避)，超时参数统一用 timeout，
    重试用尽后返回一个固定 JSON 字符串，避免上层拿到空 или 异常。
    """
    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        timeout=60
    )
    return response.choices[0].message.content.strip()

# 使用LLM生成各种含义 - 使用封装的API调用
def generate_meanings(data_item: Dict[str, Any], client) -> Dict[str, str]:
    """生成各种含义 - 结构化实现，兼容不同 JSON 键"""
    result = {}
    
    idiom_in_source = data_item.get("idiom_in_source", "")
    source = data_item.get("source", "")
    target = data_item.get("target", "")
    
    meaning_types = [
        {
            "key": "groundtruth_meaning",
            "prompt_template": """
Provide the meaning of the idiom "{idiom_in_source}" as used in the source sentence, based on the context of the sentence and its translation.

Source Sentence: {source}
Translation: {target}

Respond only with the idiom's meaning in English without any explanation, formatted as a JSON object: 
{{
    "output": "meaning content"
}}
            """,
            "depends_on": None
        },
        {
            "key": "opposite_meaning",
            "prompt_template": """
Provide an opposite meaning for: {groundtruth_meaning}

Respond only with the opposite meaning in English without any explanation, formatted as a JSON object:
{{
    "output": "opposite meaning"
}}
            """,
            "depends_on": "groundtruth_meaning"
        },
        {
            "key": "perturbation_meaning",
            "prompt_template": """
Goal:
Create text with SUBTLE GRAMMATICAL ERRORS while preserving the core meaning of the input.
Input: {groundtruth_meaning}

Methods (MUST apply at least one):

1. Swap adjacent phrases/clauses to create awkward phrasing
(e.g., "New problems that will arise" → "Problems new that will arise")
(e.g., "I want to go home" → "I want home to go")

2. Move modifier to incorrect position
(e.g., "The quickly running child" → "The running quickly child")
(e.g., "She softly spoke words" → "She spoke softly words")

3. Slight word order inversion in short phrases
(e.g., "good ideas" → "ideas good")
(e.g., "to think clearly" → "to clearly think")
(e.g., "in front of" → "front in of")

Constraints
1. CRITICAL: Output MUST contain subtle grammatical errors
2. The output MUST be different from the input, even for very short phrases
3. Do NOT change the vocabulary - use the same words but in incorrect order/structure
4. For extremely short inputs (2-3 words), apply method #3 aggressively

**Important:** Provide your response in the following strict JSON format:
{{
    "output": "Your output with subtle grammatical errors"
}}
            """,
            "depends_on": "groundtruth_meaning"
        },
        {
            "key": "literal_meaning",
            "prompt_template": """
Given the idiom or phrase "{idiom_in_source}", please do the following:

1. Provide a literal translation of the idiom (word-for-word translation)
2. Generate a phrase with similar meaning to the literal translation (NOT the idiomatic meaning)

For example:
For the Chinese idiom "打草惊蛇" (da cao jing she):
- Literal translation: "to hit the grass and startle the snake"
- Similar phrase: "to disturb the vegetation and alarm the reptile"

For the Chinese idiom "画蛇添足" (hua she tian zu):
- Literal translation: "to draw a snake and add feet"
- Similar phrase: "to sketch a serpent and attach limbs"

Please respond in English and format it as this following JSON object:
{{
    "literal_translation": "the word-for-word translation",
    "similar_phrase": "phrase with similar meaning to the literal translation"
}}
            """,
            "depends_on": None
        },
        {
            "key": "perturbation_literal_meaning",
            "prompt_template": """
I have a literal translation of an idiom: "{literal_meaning}"

Please replace ONE key noun in this phrase with a different noun to significantly change its meaning.

For example:
- "to hit the grass and startle the snake" → "to hit the tree and startle the snake"
- "to draw a snake and add feet" → "to draw a snake and add wings"
- "to bite the bullet" → "to bite the apple"
- "hit your head" → "hit your face"

Guidelines:
1. Replace ONLY ONE noun (person, place, thing, or concept)
2. The substitution should significantly change the meaning
3. Keep the grammatical structure intact
4. Make sure the result is still grammatically correct

Please format your response as a JSON object:
{{
    "output": "The modified literal translation with exactly one noun replaced"
}}
            """,
            "depends_on": "literal_meaning"
        }
    ]
    
    for meaning_type in meaning_types:
        key = meaning_type["key"]
        try:
            if meaning_type["depends_on"] and meaning_type["depends_on"] not in result:
                logging.warning(f"跳过 {key} 生成：依赖项 {meaning_type['depends_on']} 不存在")
                continue

            params = {
                "idiom_in_source": idiom_in_source,
                "source": source,
                "target": target,
                **result
            }
            prompt = meaning_type["prompt_template"].format(**params)
            raw = call_llm_api(client, prompt)
            
            # 解析 JSON
            try:
                data = json.loads(raw)
            except Exception:
                logging.error(f"{key} 返回解析失败，raw={raw}")
                data = {}

            # 不同 key 对应不同的字段
            if key in ("groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "perturbation_literal_meaning"):
                # 前三项返回 {"output": "..."}
                result[key] = data.get("output", "")
            elif key == "literal_meaning":
                # literal_meaning 返回 {"literal_translation": "...", "similar_phrase": "..."}
                result["literal_meaning"] = data.get("literal_translation", "")
                result["similar_literal_meaning"] = data.get("similar_phrase", "")
            else:
                # 兜底
                result[key] = data.get("meaning", "")

            logging.info(f"生成 {key}: {result.get(key)}")

        except Exception as e:
            logging.error(f"生成 {key} 时出错: {e}")
            # 保证字段存在
            if key == "literal_meaning":
                result.setdefault("literal_translation", "")
                result.setdefault("similar_phrase", "")
            else:
                result.setdefault(key, "")
    
    return result


# 主函数
def main():
    # 预定义API密钥
    key = ''
    
    parser = argparse.ArgumentParser(description="习语数据预处理脚本")
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    parser.add_argument("--lang_pair", type=str, required=True, help="语言对，如fi-en或fa-en")
    parser.add_argument("--api_key", type=str, default=key, help="OpenAI API密钥")
    parser.add_argument("--data_num", type=int, default=None, help="读取数据的前多少条，默认为全部")

    parser.add_argument(
        "--json_lang_pair_list",
        type=str,
        nargs="+",
        default=["fi-en","fr-en","ja-en"],
        help="所有对应 JSON 格式文件的语言对列表，默认 ['fi-en']"
    )
    parser.add_argument(
        "--csv_lang_pair_list",
        type=str,
        nargs="+",
        default=["fa-en","en-fa","ko-en"],
        help="所有对应 CSV 格式文件的语言对列表，默认 ['fa-en']"
    )
    
    args = parser.parse_args()
    
    # 设置OpenAI API密钥
    # 设置API客户端
    client = OpenAI(
        base_url="https://api2.aigcbest.top/v1",
        api_key=args.api_key
    )
    
    # 设置输出文件和日志文件
    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    output_file = os.path.join(input_dir, f"{args.lang_pair}_meaning.json")
    log_file = os.path.join(input_dir, f"{args.lang_pair}_meaning.log")
    
    # 设置日志
    logger = setup_logger(log_file)
    logger.info(f"开始处理文件: {args.input_file}, 语言对: {args.lang_pair}")
    
    # 读取文件
    data = read_file(
        args.input_file,
        args.lang_pair,
        args.json_lang_pair_list,
        args.csv_lang_pair_list
      )
    logger.info(f"成功读取{len(data)}条数据")
    
    # 限制处理数据数量
    if args.data_num is not None:
        data = data[:args.data_num]
        logger.info(f"将只处理前{args.data_num}条数据")
    
    # 处理数据
    result_data = []
    
    for item in tqdm.tqdm(data, desc="处理数据"):
        # 根据语言对获取 idiom 字段
        if args.lang_pair in args.json_lang_pair_list:
            idiom = item.get("contains_idioms", "")
            source = item.get("source", "")
            target = item.get("english_target", "")
        elif args.lang_pair in args.csv_lang_pair_list:
            idiom = item.get("idiom", "")
            source = item.get("sentence", "")
            target = item.get("gold translation", "")
        else:
            idiom = ""
            
        # 匹配source中的习语部分
        idiom_in_source = match_idiom_in_source(idiom, source)
        logger.info(f"source: {source}")
        logger.info(f"idiom_in_source: {idiom_in_source}")
        
        # 创建结果数据项
        result_item = {
            "source": source,
            "target": target,
            "idiom": idiom,
            "idiom_in_source": idiom_in_source
        }
        
        # 生成各种含义
        meanings = generate_meanings(result_item, client)
        result_item.update(meanings)
        
        result_data.append(result_item)
        
        # 添加延迟以避免超出API限制
        time.sleep(1)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"处理完成，结果已保存至 {output_file}")

if __name__ == "__main__":
    main()
