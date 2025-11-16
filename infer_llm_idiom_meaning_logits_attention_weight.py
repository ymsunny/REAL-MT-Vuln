from typing import Dict, List, Optional, Union, Any
import json
import torch
import argparse
import os
import csv
import logging
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time
from transformers import BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria

# ====== 日志设置 ======
def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = logging.getLogger(__name__)

# ====== 模型配置类 ======
class ModelConfig:
    def __init__(
        self,
        model_name: str,
        model_path: str = None,
        tokenizer_path: str = None,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        do_sample: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        self.model_name = model_name
        self.model_path = model_path or model_name
        self.tokenizer_path = tokenizer_path or self.model_path
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.do_sample = do_sample
        self.device = device
        self.kwargs = kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "device": self.device,
            **self.kwargs
        }

# ====== 提示模板类 ======
class PromptTemplate:
    @staticmethod
    def get_template(prompt_num: str, prompt_dict: Dict[str, str], source_lang: str, target_lang: str) -> str:
        template = prompt_dict.get(prompt_num, prompt_dict["1"])
        return template.replace("[SRC]", source_lang).replace("[TGT]", target_lang)

    @classmethod
    def translation_template(cls, source_lang: str, target_lang: str, prompt_num: str, prompt_dict: Dict[str, str],
                             sentence: str, idiom: str = None, meaning: str = None) -> str:
        template_text = cls.get_template(prompt_num, prompt_dict, source_lang, target_lang)
        if meaning and meaning.strip():
            prompt = f"""
{template_text}
Sentence: {sentence}
Hint: '{idiom}' means '{meaning}'
**Important:** Provide your response in the following strict JSON format. Do not include any other text or explanations.
{{
    "translation": "Your final translation"
}}
"""
        else:
            prompt = f"""
{template_text}
Sentence: {sentence}
Hint:
**Important:** Provide your response in the following strict JSON format. Do not include any other text or explanations.
{{
    "translation": "Your final translation"
}}
"""
        return prompt


# ====== 模型管理器 ======
class ModelManager:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def load_model(self, config: ModelConfig) -> None:
        model_key = config.model_name
        if model_key not in self.models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
                print("已加载Fast Tokenizer")
            except Exception as e:
                print(f"加载Fast Tokenizer失败: {e}, 使用普通Tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=False)

            model_args = {
                "device_map": config.device,
                "torch_dtype": torch.float32
            }
            if "qwen" in config.model_name.lower():
                model_args["attn_implementation"] = "eager"

            model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_args)
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer

    def get_model(self, model_name: str):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name], self.tokenizers[model_name]

    def unload_model(self, model_name: str):
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            torch.cuda.empty_cache()


# ====== 停止条件类 ======
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


# ====== 翻译任务类 ======
class TranslationTask:
    def __init__(
        self,
        model_manager: ModelManager,
        source_lang: str = "en",
        target_lang: str = "zh",
        prompt_num: str = "1",
        prompt_dict: Dict[str, str] = None
    ):
        self.model_manager = model_manager
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prompt_num = prompt_num
        self.prompt_dict = prompt_dict or {
            "1": "Translate the following sentences from [SRC] to [TGT].",
            "2": "Translate the following sentences from [SRC] to [TGT] based on the hint.",
            "3": "Translate the following sentences from [SRC] to [TGT] based on the hint, ensuring that the translation accurately conveys the meaning and context of the original sentence.",
            "5": "Please provide the [TGT] translation for the following sentences."
        }

         # 初始化logger
        self.logger = logging.getLogger(__name__)

    def set_translation_config(self, source_lang: str, target_lang: str, prompt_num: str = None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        if prompt_num:
            self.prompt_num = prompt_num

    def calculate_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    def extract_json_response(self, response_text):
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                return response_text, False
            json_str = response_text[json_start:json_end]
            result = eval(json_str)
            translation = result.get("translation", response_text)
            return translation.strip(), True
        except Exception as e:
            logger.warning(f"JSON解析失败: {str(e)}")
            return response_text, False

    def _check_stop_condition(self, next_token_str, tokenizer):
        """检查是否满足停止条件"""
        # 更完整的停止条件
        stop_token_str = ['}', '<|endoftext|>', '</s>', '<|im_end|>']
        return any(stop_token in next_token_str for stop_token in stop_token_str)

    def build_translation_prompts(self, sentence, idiom, meaning, src, tgt, prompt_num, prompt_dict):
        """构建翻译提示
        
        Args:
            sentence: 需要翻译的句子
            idiom: 句子中的习语
            meaning: 习语的含义
            src: 源语言
            tgt: 目标语言
            prompt_num: 提示模板编号
            prompt_dict: 提示模板字典
            
        Returns:
            prompt: 带有Hint的提示
        """
        # 获取源语言和目标语言的全称
        src_full = lang_dict.get(src, src)
        tgt_full = lang_dict.get(tgt, tgt)
        
        # 构建提示
        template = PromptTemplate.get_template(prompt_num, prompt_dict, src_full, tgt_full)
        
        # 添加更明确的JSON格式要求
        if meaning and meaning.strip():
            return f"""{template}
Sentence: {sentence}
Hint: '{idiom}' means '{meaning}'

Please provide your translation in the following JSON format:
{{
    "translation": "your translation here"
}}"""
        else:
            return f"""{template}
Sentence: {sentence}
Hint:

Please provide your translation in the following JSON format:
{{
    "translation": "your translation here"
}}"""
    
    def _find_token_positions(self, text, text_start_pos, text_end_pos, offset_mapping, all_tokens, prompt):
        """查找文本对应的token位置"""
        token_positions = []
        
        if offset_mapping:
            # 使用offset_mapping精确查找
            for idx, (start, end) in enumerate(offset_mapping):
                # 如果token与文本有重叠
                if end > text_start_pos and start < text_end_pos:
                    token_positions.append(idx)
        else:
            # 使用近似匹配方法
            context_before = prompt[max(0, text_start_pos-10):text_start_pos]
            context_after = prompt[text_end_pos:min(len(prompt), text_end_pos+10)]
            
            # 在token序列中查找近似位置
            tokens_text = all_tokens
            
            # 尝试在token文本中找到文本的开始和结束附近的上下文
            for i in range(len(tokens_text)):
                # 构建一个滑动窗口
                window = " ".join(tokens_text[i:i+20]).lower()
                if text.lower() in window:
                    # 从这个位置开始，构建连接文本并查找
                    connected_text = ""
                    token_boundaries = []
                    
                    for j, token in enumerate(tokens_text[i:i+20]):
                        token_boundaries.append((len(connected_text), len(connected_text) + len(token)))
                        connected_text += token
                    
                    text_lower = text.lower()
                    connected_lower = connected_text.lower()
                    local_pos = connected_lower.find(text_lower)
                    
                    if local_pos != -1:
                        local_end = local_pos + len(text_lower)
                        # 找出覆盖文本的所有token
                        for j, (start, end) in enumerate(token_boundaries):
                            if end > local_pos and start < local_end:
                                token_positions.append(i + j)
                        break
        
        return token_positions
    
    def locate_tokens_in_prompt(self, prompt, tokenizer, idiom=None, meaning=None):
        """
        在输入文本中定位习语和意义的token位置
        """
        # 获取整个输入序列的 tokens
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'][0]
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # 使用fast tokenizer的映射功能
        offset_mapping = None
        if hasattr(tokenizer, 'is_fast') and tokenizer.is_fast:
            encoding = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
            offset_mapping = encoding.offset_mapping[0].tolist()
        
        idiom_tokens_global = []
        meaning_tokens_global = []
        
        # 定位习语和意义在提示中的位置
        text_positions = {
            'idiom': (idiom, []),
            'meaning': (meaning, [])
        }
        
        for key, (text, token_positions) in text_positions.items():
            if not text:
                continue
                
            self.logger.info(f"查找{key}: '{text}'")
            
            # 在原始提示文本中查找文本位置
            text_start_pos = prompt.lower().find(text.lower())
            if text_start_pos == -1:
                continue
                
            text_end_pos = text_start_pos + len(text)
            self.logger.info(f"在原始文本中找到{key}，位置: {text_start_pos}-{text_end_pos}")
            
            # 使用offset_mapping或近似匹配找到对应的token位置
            text_tokens = self._find_token_positions(
                text, text_start_pos, text_end_pos, 
                offset_mapping, all_tokens, prompt
            )
            
            if key == 'idiom':
                idiom_tokens_global = text_tokens
                if idiom_tokens_global:
                    idiom_tokens_text = [all_tokens[i] for i in idiom_tokens_global]
                    self.logger.info(f"习语 '{idiom}' 对应的token位置: {idiom_tokens_global}")
                    self.logger.info(f"对应的tokens: {idiom_tokens_text}")
            else:
                meaning_tokens_global = text_tokens
                if meaning_tokens_global:
                    meaning_tokens_text = [all_tokens[i] for i in meaning_tokens_global]
                    self.logger.info(f"Meaning '{meaning}' 对应的token位置: {meaning_tokens_global}")
                    self.logger.info(f"对应的tokens: {meaning_tokens_text}")
        
        # 合并idiom和meaning的token位置
        attention_focus_tokens = list(set(idiom_tokens_global + meaning_tokens_global))
        attention_focus_tokens.sort()  # 按顺序排列
        
        if attention_focus_tokens:
            self.logger.info(f"需要关注的token位置（习语+meaning）: {attention_focus_tokens}")
            if len(attention_focus_tokens) > 0:
                focus_tokens_text = [all_tokens[i] for i in attention_focus_tokens if i < len(all_tokens)]
                self.logger.info(f"需要关注的tokens: {focus_tokens_text}")
        
        return attention_focus_tokens, idiom_tokens_global, meaning_tokens_global, all_tokens

    def process_attention_outputs(self, outputs, inputs, all_attention_weights):
        """处理模型输出中的注意力权重
        
        Args:
            outputs: 模型输出
            inputs: 模型输入
            all_attention_weights: 已收集的注意力权重列表
            
        Returns:
            更新后的注意力权重列表
        """
        if not hasattr(outputs, 'attentions') or outputs.attentions is None:
            return all_attention_weights
        
        # 确保outputs.attentions是元组且非空
        if not isinstance(outputs.attentions, tuple) or len(outputs.attentions) == 0:
            return all_attention_weights
        
        try:
            # 获取注意力权重
            attention_weights = torch.stack(outputs.attentions)  # [num_layers, batch, num_heads, tgt_len, src_len]
            
            # 获取当前生成token的注意力
            last_token_attn = attention_weights[..., -1:, :]  # [..., 1, src_len]
            
            # 只保留对输入序列的注意力
            input_length = inputs['input_ids'].shape[1]
            orig_input_attn = last_token_attn[..., :input_length]
            
            # 添加到列表中
            all_attention_weights.append(orig_input_attn)
            
        except Exception as e:
            self.logger.warning(f"处理注意力权重时出错: {str(e)}")
        
        return all_attention_weights

    def get_src_tgt_alignment(self, attention_weights, prompt_tokens, tgt_tokens, idiom_token_positions=None):
        """
        分析注意力权重，得到源语言tokens与目标语言tokens之间的对应关系
        Args:
            attention_weights: 注意力权重张量，形状为 [num_layers, num_heads, tgt_len, src_len]
            prompt_tokens: 源语言 tokens 列表（包括上下文）
            tgt_tokens: 目标语言生成的 tokens 列表
            idiom_token_positions: 习语在提示文本中的token位置列表（可选）

        Returns:
            attentions: 包含注意力对齐信息的字典：
                - "alignment": {tgt_idx: {"output_token": str, "attention_input_tokens": [{"input_idx": int, "input_token": str, "score": float}, ...]}, ...}
                - "idiom_meaning_translation": 如果存在，表示可能包含习语/meaning翻译的目标token索引范围
        """
        if attention_weights is None or len(attention_weights.shape) < 4:
            self.logger.warning("无效的注意力权重")
            return {"alignment": {}}

        try:
            # 获取最后一层注意力并取平均值 across heads
            last_layer_attn = attention_weights[-1]  # [num_heads, tgt_len, src_len]
            mean_attn = last_layer_attn.mean(dim=0)   # [tgt_len, src_len]

            # 初始化返回结构
            alignment = {}
            idiom_meaning_indices = []

            for tgt_idx in range(len(tgt_tokens)):
                if tgt_idx >= mean_attn.shape[0]:
                    break

                attn_dist = mean_attn[tgt_idx]  # [src_len]
                topk = min(5, len(prompt_tokens))  # 取 top-5 注意力
                scores, indices = torch.topk(attn_dist, topk)

                # 构建当前目标token的关注点列表
                attention_inputs = []
                attends_idiom_or_meaning = False

                for score, idx in zip(scores.tolist(), indices.tolist()):
                    input_token = prompt_tokens[idx] if idx < len(prompt_tokens) else "PAD"
                    attention_inputs.append({
                        "input_idx": idx,
                        "input_token": input_token,
                        "score": score
                    })

                    # 检查是否关注了 idiom 或 meaning 的位置
                    if idiom_token_positions and idx in idiom_token_positions:
                        attends_idiom_or_meaning = True
                        idiom_meaning_indices.append(tgt_idx)

                # 记录当前目标token的注意力分布
                alignment[tgt_idx] = {
                    "output_token": tgt_tokens[tgt_idx],
                    "attention_input_tokens": attention_inputs,
                    "attends_idiom_or_meaning": attends_idiom_or_meaning
                }

            # 尝试识别连续的习语翻译片段（如果有的话）
            idiom_meaning_translation = None
            if idiom_meaning_indices:
                # 寻找最长连续序列
                idiom_meaning_indices.sort()
                longest_seq = []
                current_seq = [idiom_meaning_indices[0]]

                for i in range(1, len(idiom_meaning_indices)):
                    if idiom_meaning_indices[i] == idiom_meaning_indices[i - 1] + 1:
                        current_seq.append(idiom_meaning_indices[i])
                    else:
                        if len(current_seq) > len(longest_seq):
                            longest_seq = current_seq
                        current_seq = [idiom_meaning_indices[i]]
                if len(current_seq) > len(longest_seq):
                    longest_seq = current_seq

                # 构建翻译片段信息
                if longest_seq:
                    translation_tokens = [tgt_tokens[i] for i in longest_seq if i < len(tgt_tokens)]
                    idiom_meaning_translation = {
                        "token_indices": longest_seq,
                        "tokens": translation_tokens,
                        "translation_text": ' '.join(translation_tokens)
                    }

            # 返回最终结果
            result = {
                "alignment": alignment
            }
            if idiom_meaning_translation:
                result["idiom_meaning_translation"] = idiom_meaning_translation

            return result

        except Exception as e:
            self.logger.error(f"处理注意力对齐时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"alignment": {}}

    def process_collected_attention_weights(
    self,
    all_attention_weights,
    inputs,
    tokenizer,
    generated_tokens,
    attention_focus_tokens=None
):
        """
        处理收集到的注意力权重，生成结构化的注意力分析结果
        """
        if not all_attention_weights:
            return None

        try:
            stacked_attn = torch.cat(all_attention_weights, dim=3)
            squeezed_attn = stacked_attn.squeeze(1)  # 去除 batch 维度
            input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            attention_result = self.get_src_tgt_alignment(
                squeezed_attn, input_tokens, generated_tokens, attention_focus_tokens
            )
            return attention_result

        except Exception as e:
            self.logger.error(f"处理注意力权重时出错: {str(e)}")
            return None
    

    def translate(self, prompt: str, model_config: ModelConfig, idiom: str = None, meaning: str = None, store_attention: bool = True):
        model, tokenizer = self.model_manager.get_model(model_config.model_name)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_config.device)
        attention_focus_tokens, idiom_tokens_global, meaning_tokens_global, all_tokens = self.locate_tokens_in_prompt(
            prompt, tokenizer, idiom, meaning
        )

        all_attention_weights = []
        generated_ids = inputs['input_ids'].clone()
        attention_mask = inputs['attention_mask'].clone()
        generated_tokens = []
        tokens_entropies = []

        with torch.no_grad():
            for _ in range(model_config.max_tokens):
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_attentions=store_attention
                )
                next_token_logits = outputs.logits[:, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                entropy = self.calculate_entropy(probs)
                tokens_entropies.append(float(entropy.cpu().numpy()))

                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                next_token_str = tokenizer.convert_ids_to_tokens(next_token)[0]
                generated_tokens.append(next_token_str)

                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

                if self._check_stop_condition(next_token_str, tokenizer):
                    break

        response = tokenizer.decode(generated_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        translation_text, _ = self.extract_json_response(response)

        attention_result = None
        avg_sentence_entropy = sum(tokens_entropies) / len(tokens_entropies) if tokens_entropies else 0.0
        idiom_translation_entropy = None

        if store_attention and all_attention_weights:
            attention_result = self.process_collected_attention_weights(
                all_attention_weights, inputs, tokenizer, generated_tokens, attention_focus_tokens
            )

        return {
            "text": translation_text,
            "uncertainty": {"avg_sentence_entropy": avg_sentence_entropy},
            "attention": attention_result
        }


# ====== 注意力分析辅助函数 ======
def compute_attention_stats(attention_result, idiom_positions, meaning_positions):
    if not attention_result or "alignment" not in attention_result:
        return None, None, None

    idiom_scores = []
    meaning_scores = []
    context_scores = []

    for tgt_idx, info in attention_result["alignment"].items():
        for token_info in info.get("attention_input_tokens", []):
            src_idx = token_info["input_idx"]
            score = token_info["score"]

            if src_idx in idiom_positions:
                idiom_scores.append(score)
            elif src_idx in meaning_positions:
                meaning_scores.append(score)
            else:
                # 真正的context位置：既不是idiom也不是meaning的位置
                context_scores.append(score)

    avg_idiom_score = sum(idiom_scores) / len(idiom_scores) if idiom_scores else None
    avg_meaning_score = sum(meaning_scores) / len(meaning_scores) if meaning_scores else None
    avg_context_score = sum(context_scores) / len(context_scores) if context_scores else None

    return avg_idiom_score, avg_meaning_score, avg_context_score


def save_attention_to_csv(data, file_path, src, tgt):
    if not data:
        print(f"没有数据可写入 {file_path}")
        return

    # 从文件路径中获取目录和文件名
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name, ext = os.path.splitext(base_name)
    
    # 构建包含语言对信息的新文件名
    new_file_name = f"{file_name}_{src}-{tgt}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    # 确保目录存在
    os.makedirs(dir_name, exist_ok=True)

    fieldnames = data[0].keys()
    with open(new_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"已将注意力得分保存至 {new_file_path}")


# ====== 批量翻译主流程 ======
def batch_translate(
    input_file: str,
    output_file: str,
    model_name: str,
    src: str,
    tgt: str,
    prompt_num: str,
    prompt_dict: Dict[str, str],
    meaning_lang: str,
    data_num: int = None,
    modes: List[str] = ["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
    threshold: float = None,
    store_attention: bool = True
):
    start_time = time.time()
    log_file = setup_logging(input_file, output_file)
    logger.info(f"开始批量翻译任务: 源语言={src}, 目标语言={tgt}, 模型={model_name}")
    logger.info(f"翻译模式: {modes}")

    model_manager = ModelManager()
    model_config = ModelConfig(model_name=model_name, max_tokens=4096, do_sample=False)
    model_manager.load_model(model_config)

    translation_task = TranslationTask(model_manager, src, tgt, prompt_num, prompt_dict)

    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = json.load(f)
    if data_num:
        inputs = inputs[:data_num]

    results = []
    idiom_attention_data = []
    context_attention_data = []

    for i, item in enumerate(tqdm(inputs, desc="翻译进度")):
        sentence = item.get("source")
        result_entry = {k: v for k, v in item.items()}
        idiom = item.get("idiom_in_source")

        if "translation" not in result_entry:
            result_entry["translation"] = {}
        if "sentence_uncertainty" not in result_entry:
            result_entry["sentence_uncertainty"] = {}
        if "idiom_translation_entropy" not in result_entry:
            result_entry["idiom_translation_entropy"] = {}
        if "idiom_meaning_translation" not in result_entry:
            result_entry["idiom_meaning_translation"] = {}

        for mode in modes:
            meaning = item.get(mode, '')
            prompt = translation_task.build_translation_prompts(sentence, idiom, meaning, src, tgt, prompt_num, prompt_dict)
            translation_result = translation_task.translate(prompt, model_config, idiom=idiom, meaning=meaning, store_attention=store_attention)

            result_entry["translation"][mode] = translation_result["text"]
            result_entry["sentence_uncertainty"][mode] = translation_result["uncertainty"]["avg_sentence_entropy"]
            result_entry["idiom_translation_entropy"][mode] = None
            result_entry["idiom_meaning_translation"][mode] = None


            model, tokenizer = model_manager.get_model(model_config.model_name)
            # 获取 idiom/context token 位置
            _, idiom_tokens_global, meaning_tokens_global, _ = translation_task.locate_tokens_in_prompt(prompt, tokenizer, idiom, meaning)
            avg_idiom_attn, avg_meaning_attn, avg_context_attn = compute_attention_stats(
                translation_result["attention"], idiom_tokens_global, meaning_tokens_global
            )

            # 记录注意力得分
            idiom_attention_data.append({
                "sample_id": i,
                "condition": mode,
                "idiom_attention_score": avg_idiom_attn,
                "meaning_attention_score": avg_meaning_attn,
                "sentence_entropy": translation_result["uncertainty"]["avg_sentence_entropy"]
            })
            context_attention_data.append({
                "sample_id": i,
                "condition": mode,
                "context_attention_score": avg_context_attn,
                "sentence_entropy": translation_result["uncertainty"]["avg_sentence_entropy"]
            })

        results.append(result_entry)
        torch.cuda.empty_cache()

    # 保存结果
    save_results(results, output_file, logger)
    
    # 构建注意力得分文件的基础路径
    attention_dir = os.path.join(os.path.dirname(output_file), "attention_scores")
    os.makedirs(attention_dir, exist_ok=True)
    
    # 保存注意力得分，使用不同的基础文件名
    idiom_attention_path = os.path.join(attention_dir, "idiom_attention_scores.csv")
    context_attention_path = os.path.join(attention_dir, "context_attention_scores.csv")
    
    save_attention_to_csv(idiom_attention_data, idiom_attention_path, src, tgt)
    save_attention_to_csv(context_attention_data, context_attention_path, src, tgt)

    model_manager.unload_model(model_config.model_name)
    logger.info("已卸载模型并释放内存")
    logger.info(f"总耗时: {time.time() - start_time:.2f}s")


# ====== 工具函数 ======
def seconds_to_hms(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds


def setup_logging(input_file, output_file):
    log_dir = os.path.join(os.path.dirname(output_file), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, os.path.basename(output_file).replace('.json', '.log'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    return log_file


def save_results(results, output_file, logger):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"结果已保存到 {output_file}")


# ====== 主程序入口 ======
lang_dict = {
    "en": "English", "zh": "Chinese", "de": "German", "fr": "French", "fi": "Finnish",
    "ja": "Japanese", "ru": "Russian", "hi": "Hindi"
}

DEFAULT_PROMPT_DICT = {
    "1": "Translate the following sentences from [SRC] to [TGT].",
    "2": "Translate the following sentences from [SRC] to [TGT] based on the hint.",
    "3": "Translate the following sentences from [SRC] to [TGT] based on the hint, ensuring that the translation accurately conveys the meaning and context of the original sentence.",
    "5": "Please provide the [TGT] translation for the following sentences."
}


def parse_args():
    parser = argparse.ArgumentParser(description="Translation with LLM models")

     # 定义支持的语言列表
    LANGUAGES = ["en", "zh", "de", "fr", "fi",  "ja", "cs", "ru", "uk","fa","ko","hi"]

    parser.add_argument("--input_file", type=str, help="输入文件路径 (JSON 格式)")
    parser.add_argument("--output_dir", type=str, default="translation_results", help="输出目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="模型名称/路径")
    parser.add_argument("--meaning_lang", type=str, choices=LANGUAGES, default="en",
                        help="Language of meaning to use from meaning object (english, german, etc.)")
    parser.add_argument("--src", type=str, default="en", help="源语言")
    parser.add_argument("--tgt", type=str, default="zh", help="目标语言")
    parser.add_argument("--prompt_num", type=str, default="1", help="提示模板编号")
    parser.add_argument("--no_attention", action="store_true", help="禁用注意力分析")
    parser.add_argument("--data_num", type=int, default=None, help="要处理的数据条数")
    parser.add_argument("--mode", nargs="+", default=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                        choices=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                        help="Mode(s) for translation: no_hint, groundtruth_meaning, opposite_meaning, perturbation_meaning, literal_meaning, similar_literal_meaning, perturbation_literal_meaning")
    return parser.parse_args()


def get_output_file_path(input_file, model_name_short, meaning_lang, prompt_num, output_dir=None):
    """根据输入文件构建输出文件路径
    
    Args:
        input_file: 输入文件路径
        model_name_short: 模型简称
        meaning_lang: 意义语言代码
        prompt_num: 提示模板编号
        output_dir: 自定义输出目录
        
    Returns:
        output_file: 输出文件完整路径
    """
    input_file_basename = os.path.basename(input_file)
    input_file_dir = os.path.dirname(input_file)
    
    # 确定输出目录
    if output_dir:
        output_file_dir = output_dir
    else:
        # 将路径中的data替换为results
        output_file_dir = input_file_dir.replace('/data/', '/results/', 1)
        if output_file_dir == input_file_dir:  # 如果没有/data/路径，尝试替换开头
            if input_file_dir.startswith('data/'):
                output_file_dir = 'results/' + input_file_dir[5:]
            elif input_file_dir.startswith('./data/'):
                output_file_dir = './results/' + input_file_dir[7:]
            else:
                # 使用默认输出目录
                output_file_dir = "translation_results"
    
    # 确保输出目录存在
    os.makedirs(output_file_dir, exist_ok=True)
    
    # 构建输出文件名
    output_file_basename = os.path.splitext(input_file_basename)[0]
    output_file_name = f"{output_file_basename}_{meaning_lang}_{model_name_short}_{prompt_num}.json"
    
    return os.path.join(output_file_dir, output_file_name)

if __name__ == "__main__":
    args = parse_args()
    output_file = get_output_file_path(args.input_file, args.model_name.split('/')[-1], args.meaning_lang, args.prompt_num, args.output_dir)
    batch_translate(
        input_file=args.input_file,
        output_file=output_file,
        model_name=args.model_name,
        src=args.src,
        tgt=args.tgt,
        prompt_num=args.prompt_num,
        prompt_dict=DEFAULT_PROMPT_DICT,
        meaning_lang=args.meaning_lang,
        data_num=args.data_num,
        modes=args.mode,
        threshold=None,
        store_attention=not args.no_attention
    )