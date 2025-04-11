from typing import List, Dict
from ..base import IntentProviderBase
from plugins_func.functions.play_music import initialize_music_handler
from config.logger import setup_logging
import re
import json
import hashlib
import time

TAG = __name__
logger = setup_logging()


class IntentProvider(IntentProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.llm=None
        # 添加缓存管理
        self.intent_cache = {}  # 缓存意图识别结果
        self.cache_expiry = 600  # 缓存有效期10分钟
        self.cache_max_size = 100  # 最多缓存100个意图

    def clean_cache(self):
        """清理过期缓存"""
        now = time.time()
        # 找出过期键
        expired_keys = [
            k
            for k, v in self.intent_cache.items()
            if now - v["timestamp"] > self.cache_expiry
        ]
        for key in expired_keys:
            del self.intent_cache[key]

        # 如果缓存太大，移除最旧的条目
        if len(self.intent_cache) > self.cache_max_size:
            # 按时间戳排序并保留最新的条目
            sorted_items = sorted(
                self.intent_cache.items(), key=lambda x: x[1]["timestamp"]
            )
            for key, _ in sorted_items[: len(sorted_items) - self.cache_max_size]:
                del self.intent_cache[key]

    def parse_text(self,text):
        # 定义正则表达式模式来匹配 <tags>, <tool_call>, <content> 及其内容
        tags_pattern = r'<tags>(.*?)</tags>'
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        content_pattern = r'<content>(.*?)</content>'
        # 使用正则表达式查找匹配的内容
        tags_match = re.search(tags_pattern, text, re.DOTALL)
        tool_call_match = re.search(tool_call_pattern, text, re.DOTALL)
        content_match = re.search(content_pattern, text, re.DOTALL)
        # 提取匹配的内容，如果没有匹配到则返回空字符串
        tags = tags_match.group(1).strip() if tags_match else ""
        tool_call = tool_call_match.group(1).strip() if tool_call_match else ""
        content = content_match.group(1).strip() if content_match else ""
        # 将提取的内容存储在字典中
        tools=json.loads(tool_call)
        if len(tools)==0:
            return {
            "tags": tags,
            "content": content
            }
        result = {
        "tags": tags,
        "function_call": tools[0],
        "content": content
        }
        return result

    async def detect_intent(self, conn, dialogue_history: List[Dict], text: str) -> str:
        if not self.llm:
            raise ValueError("LLM provider not set")

        # 记录整体开始时间
        total_start_time = time.time()

        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用意图识别模型: {model_info}")

        # 计算缓存键
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # 检查缓存
        if cache_key in self.intent_cache:
            cache_entry = self.intent_cache[cache_key]
            # 检查缓存是否过期
            if time.time() - cache_entry["timestamp"] <= self.cache_expiry:
                cache_time = time.time() - total_start_time
                logger.bind(tag=TAG).debug(
                    f"使用缓存的意图: {cache_key} -> {cache_entry['intent']}, 耗时: {cache_time:.4f}秒"
                )
                return cache_entry["intent"]

        # 清理缓存
        self.clean_cache()

        user_prompt = text
        music_config = initialize_music_handler(conn)
        music_file_names = music_config["music_file_names"]
        #prompt_music = f"{self.promot}\n<start>{music_file_names}\n<end>"

        tools=conn.func_handler.get_functions()
        tools=list(x["function"] for x in tools)
        tools_string = json.dumps(tools,ensure_ascii=False)

        system_prompt = f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You may call one or more tools to assist with the user query. The tools you can use are as follows:
        {tools_string}
        Response in INTENT_MODE."""
        logger.bind(tag=TAG).debug(f"User prompt: {text}")

        # 记录预处理完成时间
        preprocess_time = time.time() - total_start_time
        logger.bind(tag=TAG).debug(f"意图识别预处理耗时: {preprocess_time:.4f}秒")

        # 使用LLM进行意图识别
        llm_start_time = time.time()
        logger.bind(tag=TAG).debug(f"开始LLM意图识别调用, 模型: {model_info}")

        intent = self.llm.response_no_stream(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        # 记录LLM调用完成时间
        llm_time = time.time() - llm_start_time
        logger.bind(tag=TAG).debug(
            f"LLM意图识别完成, 模型: {model_info}, 调用耗时: {llm_time:.4f}秒"
        )

        # 记录后处理开始时间
        postprocess_start_time = time.time()

        # 清理和解析响应
        intent = json.dumps(self.parse_text(intent.strip()),ensure_ascii=False)
        # 尝试解析为JSON
        try:
            intent_data = json.loads(intent)
            # 如果包含function_call，则格式化为适合处理的格式
            if "function_call" in intent_data:
                function_data = intent_data["function_call"]
                function_name = function_data.get("name")
                function_args = function_data.get("arguments", {})

                # 记录识别到的function call
                logger.bind(tag=TAG).info(
                    f"识别到function call: {function_name}, 参数: {function_args}"
                )

                # 添加到缓存
                self.intent_cache[cache_key] = {
                    "intent": intent,
                    "timestamp": time.time(),
                }

                # 后处理时间
                postprocess_time = time.time() - postprocess_start_time
                logger.bind(tag=TAG).debug(f"意图后处理耗时: {postprocess_time:.4f}秒")

                # 确保返回完全序列化的JSON字符串
                return intent
            else:
                # 添加到缓存
                self.intent_cache[cache_key] = {
                    "intent": intent,
                    "timestamp": time.time(),
                }

                # 后处理时间
                postprocess_time = time.time() - postprocess_start_time
                logger.bind(tag=TAG).debug(f"意图后处理耗时: {postprocess_time:.4f}秒")

                # 返回普通意图
                return intent
        except json.JSONDecodeError:
            # 后处理时间
            postprocess_time = time.time() - postprocess_start_time
            logger.bind(tag=TAG).error(
                f"无法解析意图JSON: {intent}, 后处理耗时: {postprocess_time:.4f}秒"
            )
            # 如果解析失败，默认返回继续聊天意图
            return '{"intent": "继续聊天"}'
