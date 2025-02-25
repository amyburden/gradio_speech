import os
from openai import OpenAI
import asyncio
import time
text = '''在人工智能飞速发展的今天，向世界展示你的AI模型变得越来越重要。这就是Gradio发挥作用的地方：一个简单、直观、且强大的工具，让初学者到专业开发者的各个层次的人都能轻松展示和分享他们的AI模型
'''
class Chat:
    def __init__(self):
        self.client = OpenAI(
            # 请用知识引擎原子能力API Key将下行替换为：api_key="sk-xxx",
            api_key="sk-S8pWgILPtfd0aPhgNoD0Jrk4DLWlHtCX5goyFS9uriOBPKJG", # 如何获取API Key：https://cloud.tencent.com/document/product/1772/115970
            base_url="https://api.lkeap.cloud.tencent.com/v1",
        )

    def get_answer(self,s='你好'):
        for item in text:
            time.sleep(0.05)
            yield item