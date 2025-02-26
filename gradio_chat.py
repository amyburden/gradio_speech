import gradio as gr
import speech_recognition as sr
from gtts import gTTS
# from deepseekv3 import Chat
from llm_fake import Chat
from qwen import Chat
import asyncio
import librosa
import numpy as np
from io import BytesIO

def float32_2_int16(data):
    data = data / (np.abs(data).max()+1e-10)
    data = data * 32767
    data = data.astype(np.int16)
    return data

# 语音转文本
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
        return text
    except sr.UnknownValueError:
        return "无法识别语音"
    except sr.RequestError:
        return "请求失败"

# 调用LLM生成回答
qwen = Chat()
async def generate_response(text, queue1, queue2=None):
    result = ""
    for item in qwen.get_answer(text):
        result += item
        if item in [',','.','，','。', '!','！','\n']:
            await queue1.put(result)
            result = ""
        if queue2 is None:
            yield item
        await asyncio.sleep(0.03)
        # queue2.put(item)
    print(result)
    if len(result)>0:
        await queue1.put(result)
    await queue1.put(None) # Signal end of generation


# 文本转语音
from concurrent.futures import ThreadPoolExecutor
async def text_to_speech(queue1, queue2):
    with ThreadPoolExecutor() as pool:
        while True:
            text_chunk = await queue1.get()  # 从缓存队列中读取文本块
            if text_chunk is None:  # 检查结束信号
                break
            print(f"TTS processing: {text_chunk}")
            tts = gTTS(text=text_chunk, lang='zh-cn')
            audio_file = "output.mp3"
            tts.save(audio_file)
            # Read audio data using librosa
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            # Convert to NumPy array (if needed)
            numpy_array = np.array(audio_data)
            await queue2.put(numpy_array)

            print(f"TTS processed: {text_chunk}")

    await queue2.put(None)


# 处理整个流程

async def process_audio(audio_file_stt, image_file=None):
    queue1 = asyncio.Queue(maxsize=20)  # 设置缓存队列的最大长度
    # queue2 = asyncio.Queue(maxsize=20)  # 设置缓存队列的最大长度
    queue3 = asyncio.Queue(maxsize=2)  # 设置缓存队列的最大长度
    response_audio = np.zeros(shape=(1,))
    sample_rate = 24000
    # 语音转文本
    text = speech_to_text(audio_file_stt)
    if text in ["无法识别语音", "请求失败"]:
        yield text, text, (sample_rate, response_audio)
        return
    # text = '你好，你是什么模型。'
    # # 调用LLM生成回答
    response_text_gen = generate_response(text, queue1)
    # # 文本转语音
    # response_audio_gen = text_to_speech(queue1, queue2)

    # 启动两个模型的任务
    # task1 = asyncio.create_task(generate_response(text, queue1, queue2))
    task2 = asyncio.create_task(text_to_speech(queue1, queue3))
    # 同时从两个生成器中获取输出
    all_text = ""
    while True:
        # 获取 Model 1 的输出
        await asyncio.sleep(0.05)
        try:
            # 获取 Model 1 的输出
            response_text = await response_text_gen.__anext__()
            all_text += response_text
        except StopAsyncIteration:
            response_text = "Model 1: Done"

        # 获取 Model 2 的输出
        if not queue3.empty():
            response_audio = await queue3.get()
            if response_audio is None:  # 生成器结束
                response_audio = "Model 2: Done"
                await task2
                task2.cancel()  # 取消任务
            else:
                response_audio = np.asarray(response_audio, 'float64')
        else:
            response_audio = np.zeros(shape=(1,))
            pass

        # 返回当前的输出
        print(all_text, response_audio)

        # 如果两个模型都完成了，退出循环
        if response_text == "Model 1: Done" and isinstance(response_audio, str):
            queue1.empty()
            queue3.empty()
            response_audio = np.zeros((1,))
            text = ''
            all_text = ""
            break
        yield text, all_text, (sample_rate, float32_2_int16(response_audio))

if __name__ == '__main__':

    # 创建Gradio界面
    iface = gr.Interface(
        fn=process_audio,
        inputs=[gr.Audio(type="filepath"), gr.Image(type="filepath")],
        outputs=[gr.Textbox(label="stt的文本"),
                 gr.Textbox(label="生成的文本"),
                 gr.Audio(label="生成的语音", streaming=True, autoplay=True)],
        live=True
    )
    # 启动 Gradio 应用
    iface.launch(share=False)



