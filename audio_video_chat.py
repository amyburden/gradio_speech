from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass, field
from logging import getLogger
from threading import Event
from typing import Any, Literal, cast

import numpy as np
import time
from fastrtc.pause_detection import ModelOptions, PauseDetectionModel, get_silero_model
from fastrtc.tracks import EmitType, StreamHandler, AsyncAudioVideoStreamHandler
from fastrtc.utils import AdditionalOutputs, create_message, split_output, wait_for_item, audio_to_float32, audio_to_int16
logger = getLogger(__name__)
import cv2
from gradio.utils import get_space
from PIL import Image
import speech_recognition as sr
import os
os.environ['HF_TOKEN'] = 'hf_xxxxx'

from fastrtc import (Stream, get_stt_model, get_tts_model, AdditionalOutputs, WebRTC, get_cloudflare_turn_credentials_async)
import gradio as gr


# --------------------- part 1 llm model ----------------------------
#
#
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="float16", device_map="cuda:0"
)
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")


def chat(prompt, img_list):
    # Messages containing a images list as a video and a text query
    # return 'thanks for using'
    fps = 1.0
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": img_list,
                    "max_pixels": 112 * 112,
                    "fps": fps,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    return output_text


## ---------------  part 2 webrtc part ---------------------

# stt_model = get_stt_model()
tts_model = get_tts_model()
r = sr.Recognizer()

@dataclass
class AlgoOptions:
    """Algorithm options."""

    audio_chunk_duration: float = 0.6
    started_talking_threshold: float = 0.2
    speech_threshold: float = 0.1

async def iterate(generator: Generator) -> Any:
    return next(generator)

@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 24000
    pause_detected: bool = False
    started_talking: bool = False
    responding: bool = False
    stopped: bool = False
    buffer: np.ndarray | None = None
    responded_audio: bool = False
    interrupted: asyncio.Event = field(default_factory=asyncio.Event)

    def new(self):
        return AppState()

class Videoaudio(AsyncAudioVideoStreamHandler):
    """
       A stream handler that processes incoming audio, detects pauses,
       and triggers a reply function (`fn`) when a pause is detected.

       This handler accumulates audio chunks, uses a Voice Activity Detection (VAD)
       model to determine speech segments, and identifies pauses based on configurable
       thresholds. Once a pause is detected after speech has started, it calls the
       provided generator function `fn` with the accumulated audio.

       It can optionally run a `startup_fn` at the beginning and supports interruption
       of the reply function if new audio arrives.

       Attributes:
           fn (ReplyFnGenerator): The generator function to call when a pause is detected.
           startup_fn (Callable | None): An optional function to run at startup.
           algo_options (AlgoOptions): Configuration for the pause detection algorithm.
           model_options (ModelOptions | None): Configuration for the VAD model.
           can_interrupt (bool): Whether incoming audio can interrupt the `fn` execution.
           expected_layout (Literal["mono", "stereo"]): Expected audio channel layout.
           output_sample_rate (int): Sample rate for the output audio from `fn`.
           input_sample_rate (int): Expected sample rate of the input audio.
           model (PauseDetectionModel): The VAD model instance.
           state (AppState): The current state of the pause detection logic.
           generator (Generator | AsyncGenerator | None): The active generator instance from `fn`.
           event (Event): Threading event used to signal pause detection.
           loop (asyncio.AbstractEventLoop): The asyncio event loop.
       """

    def __init__(
            self,
            expected_layout: Literal["mono", "stereo"] = "mono",
            output_sample_rate: int = 24000,
            output_frame_size: int | None = None,  # Deprecated
            input_sample_rate: int = 24000,
            model: PauseDetectionModel | None = None,
    ):
        """
        Initializes the ReplyOnPause handler.

        Args:
            fn: The generator function to execute upon pause detection.
                It receives `(sample_rate, audio_array)` and optionally `*args`.
            startup_fn: An optional function to run once at the beginning.
            algo_options: Options for the pause detection algorithm.
            model_options: Options for the VAD model.
            can_interrupt: If True, incoming audio during `fn` execution
                will stop the generator and process the new audio.
            expected_layout: Expected input audio layout ('mono' or 'stereo').
            output_sample_rate: The sample rate expected for audio yielded by `fn`.
            output_frame_size: Deprecated.
            input_sample_rate: The expected sample rate of incoming audio.
            model: An optional pre-initialized VAD model instance.
        """
        super().__init__(expected_layout,
                         output_sample_rate,
                         input_sample_rate=input_sample_rate)
        self.event = asyncio.Event()
        self.quit = asyncio.Event()
        # video part
        # from queue import Queue
        n_img_llm = 10
        # self.fps = 1 # do not touch this parameter
        self.video_queue_in = asyncio.Queue(maxsize=n_img_llm)
        self.video_queue_out = asyncio.Queue()
        # self.audio = np.array([])
        self.audio_queue_out = asyncio.Queue()
        self.last_frame_time = 0
        self.lastest_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        ## pause detect model
        self.model = model or get_silero_model()
        self.algo_options = AlgoOptions()
        self.state = AppState()
        self.model_options = None

    def copy(self):
        """Creates a new instance of ReplyOnPause with the same configuration."""
        return Videoaudio(
        )

    def reset(self):
        super().reset()
        self.state = AppState()
        self.event.clear()

    async def _save_files(self, audio, images):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._sync_save_files(audio, images)
        )

    def _sync_save_files(self, audio, images):
        for i, img in enumerate(images):
            cv2.imwrite(f'./cache/tmp{i}.png', np.asarray(img))
            # img.save(f'./cache/tmp{i}.png')
        with open('./cache/tmp.wav', "wb") as f:
            import soundfile as sf
            sf.write(f.name, audio, 24000)

    def stt(self, audio):
        audio_int16 = audio_to_int16(audio[1])
        audio_data = sr.AudioData(
            audio_int16.tobytes(),
            sample_rate=audio[0],
            sample_width=2  # int16 是 2 字节
        )
        prompt = r.recognize_google(audio_data)
        return prompt

    def chat(self, prompt, images):
        # time.sleep(0.1)
        return chat(prompt, images)[0]
        # return prompt

    async def response(self, audio, images):

        self.chat_bot = self.latest_args[1] if len(self.latest_args)>1 else None

        loop = asyncio.get_event_loop()
        images = await loop.run_in_executor(
            None,
            lambda: [Image.fromarray(item) for item in images]
        )

        loop = asyncio.get_event_loop()
        # prompt = await loop.run_in_executor(
        #     None,
        #     lambda: stt_model.stt((self.input_sample_rate, audio[None]))
        # )
        prompt = await loop.run_in_executor(
            None,
            lambda: self.stt((self.input_sample_rate, audio[None]))
        )

        if self.chat_bot is not None:
            self.chat_bot.append({"role": "user", "content": prompt})

        print('stt result:', prompt)
        if len(prompt) < 1:
            return
        loop = asyncio.get_event_loop()
        text_response = await loop.run_in_executor(
            None,
            lambda: self.chat(prompt, images)
        )
        if self.chat_bot is not None:
            self.chat_bot.append({"role": "assistant", "content": text_response})

        async for audio_chunk in tts_model.stream_tts(text_response):
            self.audio_queue_out.put_nowait(audio_chunk[1])

    async def start_up(self):
        print('runing inference in start up')
        while not self.quit.is_set():
            await self.event.wait()
            print('got something to save in start up')

            img_list = []
            while not self.video_queue_in.empty():
                img_list.append(await self.video_queue_in.get())
            if len(img_list)==0:
                self.reset()
                continue
            await self._save_files(self.state.stream, img_list)
            await self.response(self.state.stream, img_list)
            self.reset()

    def _determine_pause(self, audio: np.ndarray, sampling_rate: int, state: AppState) -> bool:
        """
        Analyzes an audio chunk to detect if a significant pause occurred after speech.

        Uses the VAD model to measure speech duration within the chunk. Updates the
        application state (`state`) regarding whether talking has started and
        accumulates speech segments.

        Args:
            audio: The numpy array containing the audio chunk.
            sampling_rate: The sample rate of the audio chunk.

        Returns:
            True if a pause satisfying the configured thresholds is detected
            after speech has started, False otherwise.
        """
        duration = len(audio) / sampling_rate

        # if duration >= self.algo_options.audio_chunk_duration:
        dur_vad, _ = self.model.vad((sampling_rate, audio), self.model_options)
        logger.debug("VAD duration: %s", dur_vad)
        if (
                dur_vad > self.algo_options.started_talking_threshold
                and not state.started_talking
        ):
            state.started_talking = True
            logger.debug("Started talking")
            self.send_message_sync(create_message("log", "started_talking"))
        if state.started_talking:
            if state.stream is None:
                state.stream = audio
            else:
                state.stream = np.concatenate((state.stream, audio))
        state.buffer = None
        if dur_vad < self.algo_options.speech_threshold and state.started_talking:
            return True
        return False

    async def _process_audio(self, frame):
        array = audio_to_float32(frame)
        array = array.squeeze()
        if self.state.buffer is None:
            self.state.buffer = array
        else:
            self.state.buffer = np.concatenate((self.state.buffer, array))
        if len(self.state.buffer)/self.input_sample_rate > self.algo_options.audio_chunk_duration:
            loop = asyncio.get_event_loop()
            pause_detected = await loop.run_in_executor(
                None,
                lambda: self._determine_pause(self.state.buffer,
                                              self.input_sample_rate,
                                              self.state))
            self.state.pause_detected = pause_detected
        else:
            self.state.pause_detected = False


    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """
        Receives an audio frame from the stream.

        Processes the audio frame using `process_audio`. If a pause is detected,
        it sets the `event`. If interruption is enabled and a reply is ongoing,
        it closes the current generator and clears the processing queue.

        Args:
            frame: A tuple containing the sample rate and the audio frame data.
        """
        # print('receive audio')
        if not self.event.is_set():
            t1 = time.time()
            await self._process_audio(frame)
            # try:
            #     print('receive audio:', len(self.state.buffer) / 24000)
            # except:
            #     pass
            if self.state.pause_detected:
                self.event.set()
            # await asyncio.sleep(0.02)
            # print('using:', time.time()-t1,'s')
        else:
            # print('wait for response')
            await asyncio.sleep(0.2)


    async def emit(self):
        """
        Produces the next output chunk from the reply queue.
        """
        while True:
            try:
                # print('send audio')
                array = await asyncio.wait_for(self.audio_queue_out.get(), timeout=0.1)
                print(self.chat_bot)
                return (self.output_sample_rate, np.asarray(array, np.float32)), AdditionalOutputs(self.chat_bot)

            except (TimeoutError, asyncio.TimeoutError):
                # print('send audio None')
                await asyncio.sleep(0.2)
                continue


    async def video_receive(self, frame: np.ndarray):
        '''
        receive video from front end
        :param frame:
        :return:
        '''
        # save image every 1 second
        # print('time gap:', time.time() - self.last_frame_time)
        # t1 = time.time()

        if time.time() - self.last_frame_time > 1 and (not self.event.is_set()):
            # print('receive video')

            self.last_frame_time = time.time()
            try:
                self.video_queue_in.put_nowait(np.copy(frame))
            except asyncio.QueueFull:
                # Handle full queue case
                await self.video_queue_in.get() # remove oldest one
                self.video_queue_in.put_nowait(np.copy(frame))
        self.lastest_frame = frame
        # print('receive video using:', time.time() - t1, 's')
        # print(self.lastest_frame.shape)
        # await asyncio.sleep(0.01)
        # print('video queue1:', self.video_queue_out.qsize(), '\t video queue2:', self.video_queue_in.qsize())

    async def video_emit(self):
        # while True:
        # print('send video')
        # t1 = time.time()
        try:
            frame = self.lastest_frame
            return frame#, AdditionalOutputs(self.chat_bot)
        except Exception as e:
            raise e
        # print('send video using:', time.time() - t1, 's')


    async def shutdown(self) -> None:
        self.quit.set()
        self.quit.clear()




chatbot = gr.Chatbot(type="messages")

stream = Stream(Videoaudio(),
                additional_outputs_handler=lambda a, b: b,
                additional_inputs=[chatbot],
                additional_outputs=[chatbot],
                # rtc_configuration=credentials,
                rtc_configuration=get_cloudflare_turn_credentials_async,
                modality="audio-video",
                mode="send-receive",
                ui_args={"title": "LLM Voice Chat kneron"})

css = """
#video-source {max-width: 600px !important; max-height: 600 !important;}
"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <div style='display: flex; align-items: center; justify-content: center; gap: 20px'>
        <div style="background-color: var(--block-background-fill); border-radius: 8px">
            <img src="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png" style="width: 100px; height: 100px;">
        </div>
        <div>
            <h1>Gen AI SDK Voice Chat</h1>
            <p>Speak with Gemini using real-time audio + video streaming</p>
            <p>Powered by <a href="https://gradio.app/">Gradio</a> and <a href=https://freddyaboulton.github.io/gradio-webrtc/">WebRTC</a>⚡️</p>
            <p>Get an API Key <a href="https://support.google.com/googleapi/answer/6158862?hl=en">here</a></p>
        </div>
    </div>
    """
    )
    with gr.Row() as row:
        with gr.Column():
            webrtc = WebRTC(
                label="Video Chat",
                modality="audio-video",
                mode="send-receive",
                elem_id="video-source",
                rtc_configuration=get_cloudflare_turn_credentials_async
                if get_space()
                else None,
                icon="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png",
                pulse_color="rgb(255, 255, 255)",
                icon_button_color="rgb(255, 255, 255)",
            )
        # with gr.Column():
        #     image_input = gr.Image(
        #         label="Image", type="numpy", sources=["upload", "clipboard"]
        #     )
        # with gr.Column():
        #     chatbot = chatbot
        chatbot.render()
        webrtc.stream(
            Videoaudio(),
            inputs=[webrtc, chatbot],
            outputs=[webrtc],
            time_limit=60 if get_space() else None,
            concurrency_limit=2 if get_space() else None,
        )
        webrtc.on_additional_outputs(
            lambda prev, current: current,
            concurrency_limit=2,  # type: ignore
            inputs=[chatbot],
            outputs=[chatbot],
        )

stream.ui = demo
stream.ui.launch(share=True)
if __name__ == "__main__":
    stream.ui.launch(server_port=7860)
