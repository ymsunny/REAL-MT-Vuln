from typing import Dict, List, Optional, Union, Any
import json
import torch
import argparse
import os
import csv
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
import time  # 导入time模块
import torch
from vllm import LLM, SamplingParams

# 设置日志配置函数
def setup_logger(log_file):
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger()

class ModelConfig:
    """模型配置类，存储模型相关的所有配置"""
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
        """将配置转换为字典"""
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
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """从字典创建配置"""
        return cls(**config_dict)

class PromptTemplate:
    """提示模板类，处理不同类型的提示模板"""
    def __init__(self, template: str):
        self.template = template
        
    def format(self, **kwargs) -> str:
        """格式化提示模板"""
        return self.template.format(**kwargs)
    
    @staticmethod
    def get_template(prompt_num: str, prompt_dict: Dict[str, str], source_lang: str, target_lang: str) -> str:
        """根据提示编号获取模板"""
        template = prompt_dict.get(prompt_num, prompt_dict["1"])  # 默认使用第一个模板
        return template.replace("[SRC]", source_lang).replace("[TGT]", target_lang)
    
    @classmethod
    def translation_template(cls, source_lang: str, target_lang: str, prompt_num: str, prompt_dict: Dict[str, str], sentence: str, idiom: str=None, meaning: str = None) -> str:
        """创建翻译任务的提示模板"""
        template_text = cls.get_template(prompt_num, prompt_dict, source_lang, target_lang)
       
        if meaning != '':
            prompt = f"""
    {template_text}

    Sentence: {sentence}
    Hint: "{idiom}" means "{meaning}"

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
    
    @classmethod
    def custom_template(cls, template_str: str) -> "PromptTemplate":
        """创建自定义提示模板"""
        return cls(template_str)

class ModelManager:
    """模型管理器，负责加载和管理不同的模型"""
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
    def load_model(self, config: ModelConfig) -> None:
        """加载模型和分词器"""
        model_key = config.model_name
        
        if model_key not in self.models:
            try:
                # 尝试加载Fast Tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tokenizer_path, 
                    use_fast=True,
                    token=""
                )
                print("已加载Fast Tokenizer")
            except Exception as e:
                # 如果失败，回退到普通Tokenizer
                print(f"加载Fast Tokenizer失败: {str(e)}, 使用普通Tokenizer")
                tokenizer = AutoTokenizer.from_pretrained(
                    config.tokenizer_path, 
                    use_fast=False,
                    token=""
                )
            
            # 为Qwen系列模型设置特殊参数
            model_args = {
                "device_map": config.device,
                "torch_dtype": torch.float32,
                "token": ""
            }
            
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path, 
                **model_args
            )
            
            self.models[model_key] = model
            self.tokenizers[model_key] = tokenizer
    
    def get_model(self, model_name: str):
        """获取模型和分词器"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def unload_model(self, model_name: str):
        """卸载模型以释放内存"""
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            torch.cuda.empty_cache()

from transformers import StoppingCriteriaList
from transformers.generation.stopping_criteria import StoppingCriteria

class StopOnTokens(StoppingCriteria):
    """用于停止生成的自定义标准"""
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
class TranslationTask:
    """翻译任务类，处理不同语言之间的翻译"""
    def __init__(
        self,
        model_manager: ModelManager,
        source_lang: str = "en",  # 新增：默认源语言
        target_lang: str = "zh",  # 新增：默认目标语言
        prompt_num: str = "1",    # 新增：默认提示模板编号
        prompt_dict: Dict[str, str] = None  # 新增：提示模板字典
    ):
        self.model_manager = model_manager
        self.source_lang = source_lang  # 存储源语言
        self.target_lang = target_lang  # 存储目标语言
        self.prompt_num = prompt_num    # 存储提示模板编号
        
        # 使用默认提示模板字典或传入的字典
        self.prompt_dict = prompt_dict or {
            "1": "Translate the following sentences from [SRC] to [TGT].",
            "2": "Translate the following sentences from [SRC] to [TGT] based on the hint.",
            "3": "Translate the following sentences from [SRC] to [TGT] based on the hint, ensuring that the translation accurately conveys the meaning and context of the original sentence.",
            "5": "Please provide the [TGT] translation for the following sentences."
        }
        
        # 初始化logger
        self.logger = logging.getLogger(__name__)
        
    # 设置语言和提示模板的方法
    def set_translation_config(self, source_lang: str, target_lang: str, prompt_num: str = None):
        """设置翻译配置，包括源语言、目标语言和提示模板编号"""
        self.source_lang = source_lang
        self.target_lang = target_lang
        if prompt_num:
            self.prompt_num = prompt_num
            
    def calculate_entropy(self, probs):
        """计算概率分布的熵
        
        Args:
            probs: 概率分布张量
            
        Returns:
            entropy: 熵值
        """
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

    def extract_json_response(self, response_text):
        """从响应文本中提取JSON信息
        
        Args:
            response_text: 模型返回的原始文本
            
        Returns:
            translation_text: 提取的翻译文本
            is_valid_json: 是否成功解析为有效JSON
        """
        try:
            # 提取JSON部分并解析
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                return response_text, False
                
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            translation = result.get("translation", response_text)
            
            if isinstance(translation, str):
                translation = translation.replace('\n', ' ').strip('\n').strip()
            
            return translation, True
        except json.JSONDecodeError:
            self.logger.warning(f"JSON解析失败: {response_text}")
            
            # 尝试从格式不规范的JSON中提取translation字段
            try:
                start_index = response_text.find('"translation":') 
                if start_index != -1:
                    start_index += len('"translation":')
                    # 查找下一个引号对
                    content_start = response_text.find('"', start_index) + 1
                    content_end = response_text.find('"', content_start)
                    if content_end > content_start:
                        translation = response_text[content_start:content_end]
                        return translation.replace('\n', ' ').strip(), False
            except Exception:
                pass
                
            return response_text, False
        except Exception as e:
            self.logger.error(f"提取翻译时出错: {str(e)}")
            return response_text, False


    def _check_stop_condition(self, next_token_str, tokenizer):
        """检查是否满足停止条件"""
        stop_token_str = ['}', '<|endoftext|>']
        return any(stop_token in next_token_str for stop_token in stop_token_str)

    def _extract_idiom_translation_entropy(self, attention_result, tokens_entropies):
        """从注意力结果中提取习语翻译部分的平均不确定性"""
        #print(f"attention_result is {attention_result}")
        if (attention_result and 
            "idiom_meaning_translation" in attention_result):
            longest_seq_indices = attention_result["idiom_meaning_translation"].get("token_indices", [])
            if longest_seq_indices:
                longest_seq_entropies = [tokens_entropies[idx] for idx in longest_seq_indices 
                                        if idx < len(tokens_entropies)]
                if longest_seq_entropies:
                    return sum(longest_seq_entropies) / len(longest_seq_entropies)
        
        return None

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
        prompt = PromptTemplate.translation_template(
            source_lang=src_full,
            target_lang=tgt_full,
            prompt_num=prompt_num,
            prompt_dict=prompt_dict,
            sentence=sentence,
            idiom=idiom,
            meaning=meaning
        )
        
        return prompt
    
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
        """
        处理模型输出中的注意力权重
        
        Args:
            outputs: 模型输出
            inputs: 模型输入
            all_attention_weights: 存储注意力权重
            
        Returns:
            all_attention_weights: 更新后的注意力权重
        """
        try:
            # 获取所有层的注意力权重
            if isinstance(outputs.attentions, tuple) and len(outputs.attentions) > 0:
                attention_weights = torch.stack(outputs.attentions) # 将所有层的注意力权重堆叠成一个张量,扩展维度
                # 检查张量的形状
                if attention_weights.dim() > 0 and attention_weights.size(0) > 0:
                    # 获取当前生成token（序列中的最后一个token）对所有输入token的注意力分布
                    # attention_weights形状: [num_layers, batch_size, num_heads, seq_length_tgt, seq_length_src]
                    last_token_attn = attention_weights[:, :, :, -1:, :]  # 选择最后生成的token的注意力
                    #self.logger.info(f"提取的当前token注意力形状: {last_token_attn.shape}")
                    
                    # 获取输入的原始长度(不包括生成的token)
                    input_length = inputs['input_ids'].shape[1]
                    
                    # 只保留对原始输入的注意力，忽略后续生成部分
                    orig_input_attn = last_token_attn[..., :input_length]
                    
                    #self.logger.info(f"提取的当前token对原始输入的注意力形状: {orig_input_attn.shape}")
                    all_attention_weights.append(orig_input_attn)
                    
            else:
                self.logger.warning("注意力输出为空或格式不支持")
        except Exception as e:
            self.logger.warning(f"处理attention输出时出现错误: {str(e)}")
            # 继续执行，不让注意力处理错误影响翻译主流程
            
        return all_attention_weights
    
    def get_src_tgt_alignment(self, attention_weights, prompt_tokens, tgt_tokens, idiom_token_positions=None):
        """
        分析注意力权重，得到源语言tokens与目标语言tokens之间的对应关系
        
        Args:
            attention_weights: 注意力权重，形状为 [num_layers, num_heads, tgt_len, src_len]
            prompt_tokens: 提示文本的tokens
            tgt_tokens: 目标语言tokens
            idiom_token_positions: 习语在提示文本中的token位置列表，如果包含meaning时表示习语+meaning的所有token位置
            
        Returns:
            attention对应关系字典，键为目标语言token索引，值为源语言token索引列表
        """
        try:
            # 检查注意力权重有效性
            if attention_weights is None or attention_weights.dim() < 3:
                self.logger.warning(f"注意力权重维度不足: {attention_weights.shape if attention_weights is not None else 'None'}")
                return {}
                
            if len(prompt_tokens) == 0 or len(tgt_tokens) == 0:
                self.logger.warning(f"源或目标tokens为空: prompt_len={len(prompt_tokens)}, tgt_len={len(tgt_tokens)}")
                return {}
            
            # 打印注意力权重的形状以帮助调试
            self.logger.info(f"注意力权重形状: {attention_weights.shape}")
            
            # 获取最后一层的注意力权重平均值
            if attention_weights.size(0) > 0:  # 确保有层
                # 正确获取最后一层
                last_layer_idx = attention_weights.size(0) - 1  # 真正的最后一层
                self.logger.info(f"使用第 {last_layer_idx+1} 层注意力")
                last_layer = attention_weights[last_layer_idx]
                
                # 检查层的形状
                self.logger.info(f"last_layer形状: {last_layer.shape}")
                
                # 确保张量具有足够的维度用于注意力头的平均
                if last_layer.dim() >= 2:
                    # 确定正确的维度来平均注意力头
                    if last_layer.dim() >= 3:
                        last_layer_weights = last_layer.mean(dim=0)  # 平均所有头的注意力
                    else:
                        # 如果层没有头维度，直接使用
                        last_layer_weights = last_layer
                    
                    self.logger.info(f"平均后的注意力形状: {last_layer_weights.shape}")
                else:
                    self.logger.warning(f"注意力层维度不足: {last_layer.shape}")
                    return {}
            else:
                self.logger.warning("注意力权重没有层")
                return {}
            
            # 为每个目标token找出最关注的提示token
            attentions = {}
            if "alignment" not in attentions:
                attentions["alignment"] = {} 
            idiom_meaning_translation_tokens = []  # 存储可能对应习语或meaning翻译的token索引
            
            for tgt_idx in range(len(tgt_tokens)):
                # 确保目标索引在注意力张量范围内
                if tgt_idx < last_layer_weights.shape[0]:
                    attn_dist = last_layer_weights[tgt_idx]
                    
                    # 安全地获取前k个最高注意力分数的提示token索引
                    src_len = attn_dist.size(0)  # 获取注意力分布的实际长度
                    if src_len == 0:
                        # 如果没有prompt token，则跳过
                        continue


                     # 首次尝试使用top-5
                    initial_top_k = 4
                    
                    # 获取top-5的注意力值和索引 #第一个一般是无关的token
                    # 获取初始top-k的注意力值和索引
                    top_k = min(initial_top_k, src_len)
                    top_k_values, top_k_indices = torch.topk(attn_dist, top_k)
                    top_k_indices_list = top_k_indices.tolist()

                    
                    # 检查是否关注习语或meaning
                    attends_idiom_or_meaning = False
                    if idiom_token_positions:
                        for idx in top_k_indices_list:
                            if idx in idiom_token_positions:
                                attends_idiom_or_meaning = True
                                idiom_meaning_translation_tokens.append(tgt_idx)
                                
                                try:
                                    token = tgt_tokens[tgt_idx].encode('utf-8').decode('utf-8')
                                except UnicodeEncodeError as e:
                                    token = f"Encoding Error: {e}"  # 处理编码错误
                                except UnicodeDecodeError as e:
                                    token = f"Decoding Error: {e}"  # 处理解码错误

                                self.logger.info(f"目标token [{tgt_idx}] {token} 关注习语或Meaning位置 {idx}")
                                break
                    
                    # 构建对应关系 - 保持原有结构
                    attentions["alignment"][tgt_idx] = {
                        "output_token": tgt_tokens[tgt_idx],
                        "attention_input_tokens": [
                            {"input_idx": idx, "input_token": prompt_tokens[idx] if idx < len(prompt_tokens) else "PAD", 
                             "score": float(attn_dist[idx])}
                            for idx in top_k_indices_list if idx < len(prompt_tokens)
                        ]
                    }
                    
                    # 添加额外的标记，但不改变原有结构
                    if attends_idiom_or_meaning:
                        attentions["alignment"][tgt_idx]["attends_idiom_or_meaning"] = True
            

            # 如果没有找到习语相关的token，逐步扩大搜索范围
            max_top_k = src_len
            if idiom_token_positions and not idiom_meaning_translation_tokens:
                self.logger.info(f"使用初始top-{initial_top_k}未找到关注习语的token，开始扩大搜索范围")
                
                current_top_k = initial_top_k + 2  # 从初始值+2开始，如从3到5
                
                while current_top_k <= max_top_k and not idiom_meaning_translation_tokens:
                    self.logger.info(f"尝试使用top-{current_top_k}扩大搜索")
                    
                    # 重新扫描所有token，使用更大的top_k
                    for tgt_idx in range(len(tgt_tokens)):
                        if tgt_idx < last_layer_weights.shape[0]:
                            attn_dist = last_layer_weights[tgt_idx]
                            
                            src_len = attn_dist.size(0)
                            if src_len == 0:
                                continue
                            
                            # 使用扩大的top_k
                            expanded_top_k = min(current_top_k, src_len)
                            _, expanded_indices = torch.topk(attn_dist, expanded_top_k)
                            expanded_indices_list = expanded_indices.tolist()
                            
                            # 仅检查新增的索引范围，避免重复检查
                            new_indices = expanded_indices_list[initial_top_k:]
                            
                            if idiom_token_positions:
                                for idx in new_indices:
                                    if idx in idiom_token_positions:
                                        attends_idiom_or_meaning = True
                                        idiom_meaning_translation_tokens.append(tgt_idx)
                                        self.logger.info(f"扩展搜索后: 目标token [{tgt_idx}] {tgt_tokens[tgt_idx]} 关注习语或Meaning位置 {idx}")
                                        
                                        # 更新alignment数据中的attention_input_tokens
                                        if tgt_idx in attentions["alignment"]:
                                            attentions["alignment"][tgt_idx]["attends_idiom_or_meaning"] = True
                                            # 可以选择是否更新attention_input_tokens列表，这里保持原样
                                        break
                    
                    # 如果仍未找到，继续扩大范围
                    if not idiom_meaning_translation_tokens:
                        current_top_k += 5  # 可以更激进地增加步长，如每次+5

            # 处理习语翻译信息，但作为额外数据添加，不改变原有结构
            if idiom_token_positions and idiom_meaning_translation_tokens:
                # 对idiom_meaning_translation_tokens排序
                idiom_meaning_translation_tokens.sort()
                
                # 寻找最长连续序列
                longest_seq = []
                current_seq = []
                
                for i, idx in enumerate(idiom_meaning_translation_tokens):
                    if not current_seq or idx == current_seq[-1] + 1:
                        current_seq.append(idx)
                    else:
                        # 新的不连续序列开始
                        if len(current_seq) > len(longest_seq):
                            longest_seq = current_seq.copy()
                        current_seq = [idx]
                
                # 检查最后一个序列
                if len(current_seq) > len(longest_seq):
                    longest_seq = current_seq
                
                #print(f"longest_seq is {longest_seq}")
                # 标记最长连续序列为习语翻译
                if longest_seq:
                    self.logger.info(f"习语的最长连续翻译片段: {longest_seq}")
                    # 获取对应的token文本
                    idiom_meaning_translation_text = [tgt_tokens[idx] for idx in longest_seq]
                    self.logger.info(f"习语的翻译片段: {' '.join(idiom_meaning_translation_text)}")
                    
                    # 将信息添加到额外字段，而不是修改原有结构
                    attentions["idiom_meaning_translation"] = {
                        "token_indices": longest_seq,
                        "tokens": idiom_meaning_translation_text
                    }
            
            return attentions
        except Exception as e:
            self.logger.error(f"注意力对齐分析出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}


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
        
        Args:
            all_attention_weights: 所有收集到的注意力权重
            inputs: 输入数据
            tokenizer: 分词器
            generated_tokens: 生成的token列表
            attention_focus_tokens: 需要特别关注的token位置
            
        Returns:
            attention_results: 根据注意力分析结果得到的源语言tokens与目标语言tokens之间的对应关系,并且找到习语翻译的位置
        """
        attention_results = None
        try:
            # 提取生成过程中的输入序列
            input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # 确保有注意力权重可用
            if len(all_attention_weights) > 0:
                # 尝试合并注意力权重
                try:
                    # 确保所有注意力权重的形状一致
                    valid_weights = []
                    for attn in all_attention_weights:
                        if attn is not None and attn.dim() > 3:  # 确保至少有[layer, batch, head, src]维度
                            valid_weights.append(attn)
                        else:
                            self.logger.warning(f"跳过无效的注意力权重: {attn.shape if attn is not None else 'None'}")
                    
                    if not valid_weights:
                        raise ValueError("没有有效的注意力权重")
                        
                    # 沿着目标序列长度维度(dim=3)合并多个生成步骤的注意力权重
                    # 从 [num_layers, batch_size, num_heads, 1, seq_src] * num_tokens
                    # 变为 [num_layers, batch_size, num_heads, num_tokens, seq_src]
                    stacked_attn = torch.cat(valid_weights, dim=3)
                    self.logger.info(f"合并后的注意力权重形状: {stacked_attn.shape}")
                    
                    # 压缩批次维度(dim=1)，因为批次大小始终为1
                    # 从 [num_layers, batch=1, num_heads, num_tokens, seq_src]
                    # 变为 [num_layers, num_heads, num_tokens, seq_src]
                    squeezed_attn = stacked_attn.squeeze(1)
                    self.logger.info(f"压缩批次维度后的注意力权重形状: {squeezed_attn.shape}")
                    
                    # 确保squeezed结果形状正确
                    if squeezed_attn.dim() < 3:  # 确保至少有layer, tgt, src三个维度
                        raise ValueError(f"Squeezed注意力权重维度不足: {squeezed_attn.shape}")
                        
                    # 获取源语言和目标语言tokens之间的对应关系
                    attention_results = self.get_src_tgt_alignment(
                        squeezed_attn, 
                        input_tokens,  # 使用全部输入tokens
                        generated_tokens,
                        attention_focus_tokens  # 传递习语token位置信息
                    )
                    
                    return attention_results
                except Exception as e:
                    self.logger.error(f"处理注意力权重时出错: {str(e)}")
                    attention_results = {
                        "error": f"处理注意力权重时出错: {str(e)}",
                        }
            else:
                self.logger.warning("注意力权重列表为空")
                attention_results = {
                    "error": "注意力权重列表为空",
                    }
                
        except Exception as e:
            self.logger.error(f"根据注意力分析结果得到的源语言tokens与目标语言tokens之间的对应关系时出错: {str(e)}")
            attention_results = {
                "error": f"根据注意力分析结果得到的源语言tokens与目标语言tokens之间的对应关系时出错: {str(e)}",
            }
        
        return attention_results
    
    def adaptive_alpha_selection(self, parameter_entropy, context_entropy, base_alpha=0.5):
        """
        Adaptively select alpha based on entropy values.

        Args:
            parameter_entropy: Entropy of parametric distribution
            context_entropy: Entropy of context distribution
            base_alpha: Base alpha value
        Returns:
            Adaptive alpha value
        """
        # Lower entropy means higher confidence
        parameter_confidence = 1 / (1 + parameter_entropy)
        context_confidence = 1 / (1 + context_entropy)

        # Normalize confidences
        total_confidence = parameter_confidence + context_confidence
        normalized_context_confidence = context_confidence / total_confidence

        # Adjust alpha dynamically
        adaptive_alpha = base_alpha * normalized_context_confidence * 2

        return max(0, min(1, adaptive_alpha))
    
    def translate(self, prompt: str, model_config: ModelConfig) -> str:
        """使用批处理优化的翻译方法"""
        model, tokenizer = self.model_manager.get_model(model_config.model_name)
        self.logger.info(f"prompt is {prompt}")

        # 获取prompt_no_hint
        hint_start = prompt.find("Hint:")
        if hint_start != -1:
            next_newline = prompt.find("\n", hint_start)
            if next_newline != -1:
                prompt_no_hint = prompt[:hint_start + 5] + prompt[next_newline:]
            else:
                prompt_no_hint = prompt[:hint_start + 5]
        else:
            prompt_no_hint = prompt
        
        self.logger.info(f"prompt_no_hint is {prompt_no_hint}")

        # 同时准备两种输入
        inputs_with_hint = tokenizer(prompt, return_tensors="pt", padding=True).to(model_config.device)
        inputs_no_hint = tokenizer(prompt_no_hint, return_tensors="pt", padding=True).to(model_config.device)
        
        # 找到最大长度
        max_length = max(inputs_with_hint['input_ids'].shape[1], inputs_no_hint['input_ids'].shape[1])
        
        # 对较短的序列进行填充
        if inputs_with_hint['input_ids'].shape[1] < max_length:
            pad_length = max_length - inputs_with_hint['input_ids'].shape[1]
            inputs_with_hint['input_ids'] = torch.nn.functional.pad(inputs_with_hint['input_ids'], (0, pad_length), value=tokenizer.pad_token_id)
            inputs_with_hint['attention_mask'] = torch.nn.functional.pad(inputs_with_hint['attention_mask'], (0, pad_length), value=0)
        
        if inputs_no_hint['input_ids'].shape[1] < max_length:
            pad_length = max_length - inputs_no_hint['input_ids'].shape[1]
            inputs_no_hint['input_ids'] = torch.nn.functional.pad(inputs_no_hint['input_ids'], (0, pad_length), value=tokenizer.pad_token_id)
            inputs_no_hint['attention_mask'] = torch.nn.functional.pad(inputs_no_hint['attention_mask'], (0, pad_length), value=0)
        
        # 将两种输入合并为一个batch
        batch_input_ids = torch.cat([inputs_with_hint['input_ids'], inputs_no_hint['input_ids']], dim=0)
        batch_attention_mask = torch.cat([inputs_with_hint['attention_mask'], inputs_no_hint['attention_mask']], dim=0)
        
        # 初始化生成参数
        max_length = inputs_with_hint['input_ids'].shape[1] + model_config.max_tokens
        generated_ids = batch_input_ids.clone()
        attention_mask = batch_attention_mask.clone()

        with torch.no_grad():
            # 首先强制生成 "{"
            open_brace_token = tokenizer.encode("{", add_special_tokens=False)[0]
            next_token_batch = torch.tensor([[open_brace_token]], device=model_config.device).repeat(2, 1)
            generated_ids = torch.cat([batch_input_ids, next_token_batch], dim=1)
            attention_mask = torch.cat([batch_attention_mask, attention_mask.new_ones((2, 1))], dim=1)
            generated_text = "{"

            for _ in range(model_config.max_tokens - 1):
                # 一次性对batch进行前向传播
                outputs = model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                # 分别获取with_hint和no_hint的logits
                next_token_logits_with_hint = outputs.logits[0, -1, :]  # 第一个样本(with hint)的logits
                next_token_logits_no_hint = outputs.logits[1, -1, :]    # 第二个样本(no hint)的logits

                # 计算完整的概率分布和熵
                probs_with_hint = torch.softmax(next_token_logits_with_hint, dim=-1)
                probs_no_hint = torch.softmax(next_token_logits_no_hint, dim=-1)

                entropy_with_hint = -torch.sum(probs_with_hint * torch.log(probs_with_hint + 1e-10))
                entropy_no_hint = -torch.sum(probs_no_hint * torch.log(probs_no_hint + 1e-10))

                self.logger.info(f"entropy_with_hint is {entropy_with_hint}, entropy_no_hint is {entropy_no_hint}")

                # 根据熵值选择使用哪个logits
                if entropy_with_hint < entropy_no_hint:
                    # 如果使用Hint的不确定性更低，则进行插值
                    self.alpha = self.adaptive_alpha_selection(
                        entropy_no_hint,
                        entropy_with_hint,
                        base_alpha=0.5
                    )
                    self.logger.info(f"Alpha: {self.alpha}")
                    
                    # 进行概率分布的插值
                    modulated_dist = (1 - self.alpha) * probs_no_hint + self.alpha * probs_with_hint
                    next_token_logits = torch.log(modulated_dist + 1e-10)
                else:
                    next_token_logits = next_token_logits_no_hint

                # 选择下一个token
                if model_config.do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 更新生成序列 (对两个输入都使用相同的next_token)
                next_token_batch = next_token.repeat(2, 1)  # 复制token到batch size=2
                generated_ids = torch.cat([generated_ids, next_token_batch], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((2, 1))], dim=1)

                # 检查停止条件
                if self._check_stop_condition(tokenizer.convert_ids_to_tokens(next_token)[0], tokenizer):
                    break

                if generated_ids.shape[1] >= max_length:
                    break

        # 只解码with_hint的输出部分
        response = tokenizer.decode(generated_ids[0, inputs_with_hint['input_ids'].shape[1]:], skip_special_tokens=True)
        self.logger.info(f"response is {response}")
        
        translation_text, _ = self.extract_json_response(response)
        self.logger.info(f"translation_text is {translation_text}")
        
        return {"text": translation_text}

def seconds_to_hms(total_seconds):
    """将秒转换为时分秒格式"""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds

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
    topK: int = 10,
    modes: List[str] = ["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
    threshold: float = None,
    store_attention: bool = True
) -> None:
    """批量翻译文件内容并保存结果"""
    # 记录开始时间和设置日志
    start_time = time.time()
    log_file = setup_logging(input_file, output_file)
    logger = logging.getLogger()
    
    logger.info(f"开始批量翻译任务: 源语言={src}, 目标语言={tgt}, 模型={model_name}")
    logger.info(f"翻译模式: {modes}")
    
    # 创建模型和翻译任务
    model_manager, model_config, translation_task = setup_translation_environment(
        model_name, src, tgt, prompt_num, prompt_dict, logger
    )
    
    # 读取输入数据
    inputs, total_count = load_input_data(input_file, data_num, logger)
    
    # 进行批量翻译
    results = process_batch_translation(
        inputs, translation_task, model_config, src, tgt, prompt_num, prompt_dict,
        modes, topK, threshold, store_attention, output_file, logger
    )
    
    # 保存最终结果
    save_results(results, output_file, logger)
    
    # 清理资源
    model_manager.unload_model(model_config.model_name)
    logger.info("已卸载模型并释放内存")
    
    # 记录结束时间和总耗时
    log_completion_time(start_time, logger)

def setup_logging(input_file, output_file):
    """设置日志"""
    log_dir = os.path.join(os.path.dirname(output_file), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, os.path.basename(output_file).replace('.json', '.log'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"日志文件: {log_file}")
    return log_file

def setup_translation_environment(model_name, src, tgt, prompt_num, prompt_dict, logger):
    """设置翻译环境"""
    model_manager = ModelManager()
    model_config = ModelConfig(
        model_name=model_name,
        max_tokens=4096,
        do_sample=False
    )
    
    logger.info("正在加载模型...")
    model_manager.load_model(model_config)
    logger.info("模型加载完成")

    translation_task = TranslationTask(
        model_manager=model_manager,
        source_lang=src,
        target_lang=tgt,
        prompt_num=prompt_num,
        prompt_dict=prompt_dict
    )
    translation_task.logger = logger
    
    return model_manager, model_config, translation_task

def load_input_data(input_file, data_num, logger):
    """加载输入数据"""
    with open(input_file, 'r', encoding='utf-8') as f:
        inputs = json.load(f)
    
    logger.info(f"加载了 {len(inputs)} 条数据")
    
    if data_num is not None and data_num > 0:
        inputs = inputs[:data_num]
        logger.info(f"限制处理数据量为 {data_num} 条")
    
    return inputs, len(inputs)

def extract_think_content(text):
    """提取<think></think>标签之间的内容
    
    Args:
        text: 原始文本
        
    Returns:
        think_content: 思考内容
        remaining_text: 剩余文本
    """
    think_content = ""
    remaining_text = text
    
    think_start = text.find("<think>")
    think_end = text.find("</think>")
    print(f"think_start is {think_start}, think_end is {think_end}")
    
    if think_start != -1 and think_end != -1 and think_end > think_start:
        think_content = text[think_start + 7:think_end].strip()
        # 移除think标签及其内容
        remaining_text = text[:think_start].strip() + " " + text[think_end + 8:].strip()
    
    return think_content, remaining_text


def extract_json_response1(response_text, logger):
    """从响应文本中提取JSON信息，提取最后一对大括号中的内容
    
    Args:
        response_text: 模型返回的原始文本
        logger: 日志记录器
        
    Returns:
        translation_text: 提取的翻译文本
        is_valid_json: 是否成功解析为有效JSON
    """
    try:
        # 如果输入已经是字典类型，直接处理
        if isinstance(response_text, dict):
            translation = response_text.get("translation", str(response_text))
            if isinstance(translation, str):
                translation = translation.replace('\n', ' ').strip('\n').strip()
            return translation, True

        # 使用正则表达式找到最后一个完整的JSON对象
        import re
        json_pattern = r'{[^{}]*}'
        matches = list(re.finditer(json_pattern, response_text))
        
        if not matches:
            return response_text, False
            
        # 获取最后一个匹配的JSON
        json_str = matches[-1].group()
        
        try:
            # 首先尝试用json.loads
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # 如果失败，尝试用ast.literal_eval
            import ast
            result = ast.literal_eval(json_str)
            
        translation = result.get("translation", response_text)
        
        if isinstance(translation, str):
            translation = translation.replace('\n', ' ').strip('\n').strip()
        
        return translation, True
        
    except Exception as e:
        logger.error(f"提取翻译时出错: {str(e)}")
        
        # 如果JSON解析失败，尝试直接提取translation字段
        try:
            matches = list(re.finditer(r'"translation":\s*"([^"]*)"', response_text))
            if matches:
                # 获取最后一个匹配
                last_match = matches[-1]
                return last_match.group(1).strip(), True
        except Exception:
            pass
            
        return response_text, False

def process_batch_translation(inputs, translation_task, model_config, src, tgt, prompt_num, prompt_dict,
                            modes, topK, threshold, store_attention, output_file, logger):
    """处理批量翻译"""
    results = []
    
    for i, item in enumerate(tqdm(inputs, desc="翻译进度")):
        sentence = item.get("source")
        result_entry = {k: v for k, v in item.items()}
        
        idiom = item.get("idiom_in_source")
        logger.info(f"项目 {i+1} 的习语: {idiom}")
        
        # 初始化结果结构
        if "translation" not in result_entry:
            result_entry["translation"] = {}
        
        
        # 处理每个翻译模式
        for mode in modes:
            try:
                # 如果是no_hint模式，跳过处理
                if mode == "no_hint":
                    logger.info("跳过no_hint模式")
                    continue

                meaning = item.get(mode,'') # mode 键不存在 # 输出: ''
                
                # 构建提示
                prompt = translation_task.build_translation_prompts(
                    sentence, idiom, meaning, src, tgt, prompt_num, prompt_dict
                )
                
                # 翻译
                translation_result = translation_task.translate(
                    prompt, model_config
                )
                    
                # 存储结果
                result_entry["translation"][mode] = translation_result["text"]
               
                #print(f"result_entry is {result_entry}")

                logger.debug(f"模式 {mode} 的翻译结果: {translation_result}")
            except Exception as e:
                logger.error(f"项目 {i+1} 模式 {mode} 翻译失败: {str(e)}")
                result_entry["translation"][mode] = f"ERROR: {str(e)}"
        
        results.append(result_entry)
        
        # 清理GPU缓存，防止内存溢出
        torch.cuda.empty_cache()
        logger.info(f"项目 {i+1} 处理完成，已清理GPU缓存")
        
        # 定期保存中间结果
        save_interim_results(results, output_file, i, len(inputs))
    
    return results

def save_results(results, output_file, logger):
    """保存翻译结果"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    logger.info(f"批量翻译完成，结果已保存到 {output_file}")

def log_completion_time(start_time, logger):
    """记录完成时间"""
    end_time = time.time()
    elapsed_time = end_time - start_time
    h, m, s = seconds_to_hms(elapsed_time)
    logger.info(f"总耗时: {elapsed_time:.2f}秒 ({h}小时{m}分钟{s}秒)")


def create_indexed_dictionary(generated_tokens, tokens_entropies, tokens_probs):
    """
    将三个列表合并成一个索引字典，并将结果赋值给 "all_uncertainty" 键。

    Args:
    generated_tokens: 包含生成 token 的列表。
    tokens_entropies: 包含 token 熵值的列表。
    tokens_probs: 包含 token 概率的列表。

    Returns:
    一个字典，其中包含 "all_uncertainty" 键，其值为索引字典。
    """

    if not (len(generated_tokens) == len(tokens_entropies) == len(tokens_probs)):
        raise ValueError("三个列表的长度必须相同。")

    indexed_dict = {}
    for i in range(len(generated_tokens)):
        indexed_dict[i] = {
            "token": generated_tokens[i],
            "entropy": tokens_entropies[i],
            "probability": tokens_probs[i]
        }

    return indexed_dict

def extract_json_response(self, response_text):
        """从响应文本中提取JSON信息
        
        Args:
            response_text: 模型返回的原始文本
            
        Returns:
            translation_text: 提取的翻译文本
            is_valid_json: 是否成功解析为有效JSON
        """
        try:
            # 提取JSON部分并解析
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start == -1 or json_end <= json_start:
                return response_text, False
                
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            translation = result.get("translation", response_text)
            
            if isinstance(translation, str):
                translation = translation.replace('\n', ' ').strip('\n').strip()
            
            return translation, True
        except json.JSONDecodeError:
            self.logger.warning(f"JSON解析失败: {response_text}")
            
            # 尝试从格式不规范的JSON中提取translation字段
            try:
                start_index = response_text.find('"translation":') 
                if start_index != -1:
                    start_index += len('"translation":')
                    # 查找下一个引号对
                    content_start = response_text.find('"', start_index) + 1
                    content_end = response_text.find('"', content_start)
                    if content_end > content_start:
                        translation = response_text[content_start:content_end]
                        return translation.replace('\n', ' ').strip(), False
            except Exception:
                pass
                
            return response_text, False
        except Exception as e:
            self.logger.error(f"提取翻译时出错: {str(e)}")
            return response_text, False
        
# 定义语言字典
lang_dict = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "fi": "Finnish",
    "es": "Spanish",
    "ru": "Russian",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "id": "Indonesian",
    "ceb": "Cebuano",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "hi": "Hindi"
}

# 定义默认提示模板字典
DEFAULT_PROMPT_DICT = {
    "1": "Translate the following sentences from [SRC] to [TGT].",
    "2": "Translate the following sentences from [SRC] to [TGT] based on the hint.",
    "3": "Translate the following sentences from [SRC] to [TGT] based on the hint, ensuring that the translation accurately conveys the intendedmeaning of the source sentence.",
    "4": "Translate the following sentences from [SRC] to [TGT] based on the hint, and ignore all potentially misleading contextual information and maintain absolute fidelity to the source sentence."
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Translation with LLM models")

    # 定义支持的语言列表
    LANGUAGES = ["en", "zh", "de", "fr", "fi",  "ja", "cs", "ru", "uk","fa","ko","hi"]
    
    # 基本参数
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", 
                        help="Model name/path to use for translation")
    parser.add_argument("--src", type=str, choices=LANGUAGES, default="en", 
                        help="Source language code (en, zh, de, fr, etc.)")
    parser.add_argument("--tgt", type=str, choices=LANGUAGES, default="zh", 
                        help="Target language code (en, zh, de, fr, etc.)")
    parser.add_argument("--prompt_num", type=str, default="1", 
                        help="Prompt template number to use from prompt_dict")
    parser.add_argument("--meaning_lang", type=str, choices=LANGUAGES, default="en",
                        help="Language of meaning to use from meaning object (english, german, etc.)")
    parser.add_argument("--data_num", type=int, default=None,
                        help="Number of data entries to process from input file (default: all)")
    parser.add_argument("--mode", nargs="+", default=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                        choices=["no_hint", "groundtruth_meaning", "opposite_meaning", "perturbation_meaning", "literal_meaning", "similar_literal_meaning", "perturbation_literal_meaning"],
                        help="Mode(s) for translation: no_hint, groundtruth_meaning, opposite_meaning, perturbation_meaning, literal_meaning, similar_literal_meaning, perturbation_literal_meaning")
    parser.add_argument("--threshold", type=float, default=None,
                        help="不确定性阈值，只有当 entropy_no_hint > threshold 时才使用 hint")
    
    # 注意力分析参数
    parser.add_argument("--no_attention", action="store_true",
                        help="禁用注意力分析，节省内存和计算时间")
    parser.add_argument("--idiom_in_source", type=str,
                        help="源语言中需要特别关注的惯用语或短语")
    

    parser.add_argument("--top_p", type=float, default=1.0, 
                        help="Top-p sampling parameter")
    parser.add_argument("--do_sample", action="store_true", 
                        help="Whether to use sampling")
    parser.add_argument("--max_tokens", type=int, default=4096, 
                        help="Maximum tokens to generate")
    
    # 文件处理参数
    parser.add_argument("-i", "--input_file", type=str, 
                        help="Input file path for batch translation (JSON format)")
    parser.add_argument("-o", "--output_dir", type=str, default="translation_results", 
                        help="Output directory for translation results")
    # 单句翻译参数
    parser.add_argument("--sentence", type=str, 
                        help="Single sentence to translate")
    parser.add_argument("--meaning", type=str, 
                        help="meaning for the sentence")
    
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
        output_file_dir = input_file_dir.replace('/data/', '/ckplug_results/', 1)
        if output_file_dir == input_file_dir:  # 如果没有/data/路径，尝试替换开头
            if input_file_dir.startswith('data/'):
                output_file_dir = 'ckplug_results/' + input_file_dir[5:]
            elif input_file_dir.startswith('./data/'):
                output_file_dir = './ckplug_results/' + input_file_dir[7:]
            else:
                # 使用默认输出目录
                output_file_dir = "ckplug_results"
    
    # 确保输出目录存在
    os.makedirs(output_file_dir, exist_ok=True)
    
    # 构建输出文件名
    output_file_basename = os.path.splitext(input_file_basename)[0]
    output_file_name = f"{output_file_basename}_{meaning_lang}_{model_name_short}_{prompt_num}.json"
    
    return os.path.join(output_file_dir, output_file_name)

def save_interim_results(results, output_file, current_index, total_count):
    """保存中间结果
    
    Args:
        results: 当前结果列表
        output_file: 最终输出文件路径
        current_index: 当前处理的索引
        total_count: 总数据量
    """
    if (current_index + 1) % 100 == 0:
        # 构建中间文件名
        interim_file = output_file.replace('.json', f'_interim_{current_index+1}.json')
        
        # 保存结果
        with open(interim_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        logging.info(f"已处理 {current_index+1}/{total_count} 条数据，中间结果已保存到 {interim_file}")

if __name__ == '__main__':
    # 记录总体开始时间
    total_start_time = time.time()
    
    default_dir = os.path.dirname(os.path.abspath(__file__))
    # 解析命令行参数
    args = parse_args()
    
    print("开始执行翻译程序")
    print(f"命令行参数: {args}")
    
    # 获取提示模板字典
    prompt_dict = DEFAULT_PROMPT_DICT
    
    # 确定是否启用注意力分析
    store_attention = not args.no_attention
    if store_attention:
        print("已启用注意力分析")
    else:
        print("已禁用注意力分析")
    
    # 根据参数决定翻译模式
    if args.input_file:
        # 获取模型短名称
        model_name_short = args.model_name.split('/')[-1]
        
        # 构建输出文件路径
        input_file_basename = os.path.basename(args.input_file)
        input_file_dir = os.path.dirname(args.input_file)
        
        # 将路径中的data替换为results
        output_file_dir = input_file_dir.replace('/data/', 'ckplug_results/', 1)
        if output_file_dir == input_file_dir:  # 如果没有/data/路径，尝试替换开头
            if input_file_dir.startswith('data/'):
                output_file_dir = 'ckplug_results/' + input_file_dir[5:]
            elif input_file_dir.startswith('./data/'):
                output_file_dir = 'ckplug_results/' + input_file_dir[7:]
            else:
                # 使用指定的输出目录
                output_file_dir = args.output_dir
        
        # 确保输出目录存在
        os.makedirs(output_file_dir, exist_ok=True)
        
        # 构建输出文件名
        output_file_basename = os.path.splitext(input_file_basename)[0]
        output_file_name = f"{output_file_basename}_{args.meaning_lang}_{model_name_short}_ckplug_{args.prompt_num}.json"
        output_file = os.path.join(output_file_dir, output_file_name)
        
        print(f"输出文件路径: {output_file}")
    
        # 批量翻译模式
        batch_translate(
            input_file=args.input_file,
            output_file=output_file,
            model_name=args.model_name,
            src=args.src,
            tgt=args.tgt,
            prompt_num=args.prompt_num,
            prompt_dict=prompt_dict,
            meaning_lang=args.meaning_lang,
            data_num=args.data_num,
            modes=args.mode,
            threshold=args.threshold,
            store_attention=store_attention
        )
    else:
        # 测试模式
        sentence = "There is tornado coming, so batten down the hatches!"
        meaning = "to get ready for trouble"
        idiom = "batten down the hatches"
        
        prompt = PromptTemplate.translation_template(
            source_lang=lang_dict["fi"],
            target_lang=lang_dict["en"],
            prompt_num=2,
            prompt_dict=DEFAULT_PROMPT_DICT,
            sentence=sentence,
            idiom=idiom,
            meaning=meaning
        )
        
        print(f"提示: {prompt}")
        
        
        logger = logging.getLogger()
        logger.info(f"翻译模式: 测试")
        
        # 创建模型和翻译任务
        model_manager, model_config, translation_task = setup_translation_environment(
                "Qwen/Qwen2.5-7B-Instruct", "fi", "en", 2, DEFAULT_PROMPT_DICT, logger
            )
        # 翻译
        translation_result = translation_task.translate(
            prompt, model_config
        )
        print(f"翻译结果: {translation_result}")
    
    # 记录总体结束时间和总耗时
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    h, m, s = seconds_to_hms(total_elapsed_time)
    print(f"程序执行完成，总耗时: {total_elapsed_time:.2f}秒 ({h}小时{m}分钟{s}秒)")
