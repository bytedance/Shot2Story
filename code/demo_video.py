import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
import lavis.tasks as tasks

from lavis.common.config import Config
from lavis.common.dist_utils import get_rank
from lavis.common.registry import registry
from lavis.conversation.conversation import Chat, CONV_VISION_MS, CONV_VISION_MS_TEXT

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="lavis/projects/blip2/eval/demo.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

#TODO
task = tasks.setup_task(cfg)
dataset = task.build_datasets(cfg)

print(cfg.__dict__)
pre_cfg = cfg.config.preprocess 

vis_processor_cfg = pre_cfg.vis_processor.eval
print(vis_processor_cfg.__dict__)
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
#vis_processors, txt_processors = load_preprocess(pre_cfg) 
#vis_processor = vis_processors['eval']
chat = Chat(model, vis_processor, task=task, dataset=dataset, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your image first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list

def gradio_reset_history(chat_state, img_list):
    if chat_state is not None:
        role = chat_state.messages[0][0]
        sum_video = chat_state.messages[0][1].split('Based on the video, please answer ')[0] + 'Based on the video, please answer '
        chat_state.messages = [[role, sum_video]]
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), chat_state

def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION_MS.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def upload_vid(gr_vid, text_input, chat_state, temperature=0.1, input_splits=""):
    if gr_vid is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION_MS.copy()
    if input_splits == 'Automatic detection':
        input_splits = ''
    img_list = []
    llm_message = chat.upload_video_ms_standalone(gr_vid, chat_state, img_list, input_splits=input_splits)
    chat.ask("Please describe this video in detail.", chat_state)
    summary = chat.answer(conv=chat_state,
                            num_beams=1,
                            temperature=temperature,
                            max_new_tokens=650,
                            max_length=2048)[0][0]
    print(gr_vid, summary)
    chat_state = CONV_VISION_MS_TEXT.copy()
    chat_state.append_message(chat_state.roles[0], f"The video content is: {summary}\n\nYou should answer the question concisely. Based on the video, please answer ")
        
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    # llm_message = chat.answer(conv=chat_state,
    #                           num_beams=num_beams,
    #                           temperature=temperature,
    #                           max_new_tokens=300,
    #                           max_length=2048)[0][0]
    llm_message = chat.answer_text(conv=chat_state,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2048)[0][0]
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list

# New Title
title = """<h1 align='center'>Demo for <a href="mingfei.info/shot2story">Shot2Story</a></h1>"""

# Updated Description with highlighted project names
description = "<h3>This is the demo of <b>SUM-shot</b> model. Answers are based on our detailed video summaries. Upload your videos and start chatting! </h3>"

# Same Links and badges
article = """
<div style='display: flex; justify-content: start; align-items: center; gap: 20px;'>
  <a href='https://mingfei.info/shot2story'>
    <img src='https://img.shields.io/badge/Project-Page-Green'>
  </a>
  <a href='https://github.com/bytedance/Shot2Story'>
    <img src='https://img.shields.io/badge/Github-Code-blue'>
  </a>
  <a href='https://mingfei.info/files/paper_shot2story20k.pdf'>
    <img src='https://img.shields.io/badge/Paper-PDF-red'>
  </a>
</div>
<div>
  <h4>🌟 Current Approach:</h4>
  <p>- Take a look at our <a href="https://github.com/bytedance/Shot2Story/DATA.md">Shot2Story data</a>, with 20k human-annotated video summaries, 80k video shot captions and 40k narration captions.</p>
  <p>- Detailed and grounded video summaries are powerful, versatile in various video related tasks. It showcases the robust capabilities of our model.</p>

  <h4>🚀 Future Plans</h4>
  <p>- Chat-SUM-shot is on the way for grounded and powerful video dialogue! Stay tuned!</p>
</div>
<div style='margin-top: 20px; background-color: rgba(240, 248, 255, 0.7); padding: 15px; border-radius: 10px;'>
  <h4 style='color: #2a9df4;'>🎉 Playtime Guide</h4>
  <p style='color: #333;'>😄 For a more comprehensive understanding, try specifying reasonable starting and ending timestamps for the shots. Enjoy!</p>
  <p style='color: #333;'>😄 For grounded video-text understanding, set the temperature for story generation to 0.1.</p> 
  <p style='color: #333;'>😄 For creative video-text understanding, set the temperature for story generation to a higher value, e.g., 1.0.</p>
</div>
"""
#   <p>😄 For more comprehensive understanding, try to specify the starting and ending timestamps for the shots. Either reasonable or not, for your joy!</p>

#TODO show examples below

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            #image = gr.Image(type="pil")
            video = gr.Video(label="video_input")
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            gr.Markdown("### Manual video shots (Optional)")
            input_splits = gr.Textbox(label='Optional manual shots', placeholder="If you want to manually specify shots, please enter each shot on a new line with format 'start_second end_second'", lines=3)
            
            gr.Markdown("### Story generation")
            with gr.Row():
                sum_temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
            
            gr.Markdown("### Chatting")
            with gr.Row():
                num_beams = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=1,
                    step=1,
                    interactive=True,
                    label="beam search numbers)",
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label='SUM-shot')
            text_input = gr.Textbox(label='User', placeholder='Please upload your image first', interactive=False)
            clear_hist = gr.Button("Clear history")
    
    examples = gr.Examples(
            examples=[
                # ["examples/videos/v_-EIsT868Trw_0_1373.mp4", "0 3.1\n3 11.5\n11.5 24.2\n24.2 45", "Please describe this video in detail."],
                # ["examples/videos/v_-EIsT868Trw_0_1373.mp4", "", "Please describe this video in detail."],
                # ["examples/videos/v_-EIsT868Trw.mp4", "0 3.1\n3 11.5\n11.5 24.2\n24.2 45\n45 78\n78 82\n82 102\n102 157", "Please describe this video in detail."],
                # ["examples/videos/v_-EIsT868Trw.mp4", "0 3.1\n3 11.5\n11.5 24.2\n24.2 45\n45 78\n78 82\n82 157", "Please describe this video in detail."],
                ["examples/videos/v_-EIsT868Trw.mp4", "0 3.1\n3 11.5\n11.5 24.2\n24.2 45\n45 82\n82 157", "What is the woman doing?"],
                ["examples/videos/v_-EIsT868Trw.mp4", "0 75\n75 157", "What is the woman doing?"],
                ["examples/videos/v_-EIsT868Trw.mp4", "Automatic detection", "What is the woman doing?"],
                ["examples/videos/v_cCDffwsJvsY.mp4", "0 3\n3 5.5\n5.5 22.3\n22.3 32.2\n32.2 52.1\n52.1 65\n65 70.4\n70.4 81.2\n81.2 86.2\n86.2 90\n90 95.7\n95.7 103.8\n103.8 111", "What are the steps the person takes in the video?"],
                ["examples/videos/v_cCDffwsJvsY.mp4", "Automatic detection", "What are the steps the person takes in the video?"],
                ["examples/videos/aV14BKrGai8_45_202.mp4", "0 21.2\n21.2 28.5\n28.5 38.0\n38.0 51.2\n51.2 61.3\n61.2 73.3\n73.3 82.2\n82.2 90.1\n90.1 99.5\n99.5 108\n108 113.6\n113.6 116.7\n116.7 122\n122 128.8\n129.2 143\n143 156", "What objects appears in the video? List them all, and describe where they are."],
                ["examples/videos/aV14BKrGai8_45_202.mp4", "Automatic detection", "What objects appears in the video? List them all, and describe where they are."]
            ],
            inputs=[video, input_splits, text_input],
        )
    
    upload_button.click(upload_vid, [video, text_input, chat_state, sum_temperature, input_splits], [video, text_input, upload_button, chat_state, img_list])
    
    text_input.submit(gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]).then(
        gradio_answer, [chatbot, chat_state, img_list, num_beams, temperature], [chatbot, chat_state, img_list]
    )
    clear.click(gradio_reset, [chat_state, img_list], [chatbot, video, text_input, upload_button, chat_state, img_list], queue=False)
    clear_hist.click(gradio_reset_history, [chat_state, img_list], [chatbot, text_input, chat_state], queue=False)

demo.queue()
demo.launch(share=True, enable_queue=True)
