import argparse
import time
import whisper
import os
import ffmpeg
from PIL import Image
import numpy as np
import random
from transnetv2_pytorch import TransNetV2
import random as rnd

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector

from decord.ndarray import NDArray
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader

from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    answer_prompt: str = None
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            if self.answer_prompt:
                ret += self.answer_prompt
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message, update=True):
        if len(self.messages) > 0 and self.messages[-1][0] == role:
            self.messages[-1][1] = ' '.join([self.messages[-1][1].strip(), message.strip()])
        else:
            self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

im_list =""
for im in range(8):
# for im in range(32):
    im_list += "<Img>ImageContent{}</Img>".format(im)

CONV_VISION = Conversation(
    system="Give the following video: {}. ".format(im_list) +
           "You will be able to see the images once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_MS = Conversation(
    system="You will see the frames of the video provided to you. Please answer the question based on the video and your knowledge.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

CONV_VISION_MS_TEXT = Conversation(
    system="You should identify a specific object, place, person and way in the video based on its description. You must answer my question. The answer is definitely contained in the provided video description. ",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

def find_scenes(video_path: str, threshold=21.0):
    # create a video_manager point to video file
    video_manager = VideoManager([video_path])

    # initialize a SceneManager and add ContentDetector algorithm (set threshold as 27)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    # start the video_manager and perform scene detection
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # obtain list of detected scenes as timecodes
    scene_list = scene_manager.get_scene_list(video_manager.get_base_timecode())
    
    # print the timecode of scenes
    print(f"Scenes from {video_path}:")
    for i, scene in enumerate(scene_list):
        print(f"    Scene {i+1}: Start {scene[0]} / End {scene[1]}")
    return [[shot_[0].frame_num, shot_[1].frame_num] for shot_ in scene_list]


def read_video_frames_resized(video_fn, resize_width, resize_height):
    # Get video metadata to extract FPS
    probe = ffmpeg.probe(video_fn)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    fps = eval(video_stream['avg_frame_rate']) if video_stream else 30

    # Read the video frames into a pipe at original size
    out, _ = (
        ffmpeg
        .input(video_fn)
        .filter('scale', resize_width, resize_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Calculate the number of frames
    # frame_size = original_width * original_height * 3  # 3 for RGB
    frame_size = resize_width * resize_height * 3  # 3 for RGB
    num_frames = len(out) // frame_size

    # Convert the raw video data to a NumPy array
    video_data = np.frombuffer(out, np.uint8).reshape([num_frames, resize_height, resize_width, 3])

    # Process frames
    resized_tensors = []
    for frame in video_data:
        # Resize frame as tensor
        resized_frame = torch.from_numpy(frame).permute(2, 0, 1)
        resized_tensors.append(resized_frame)

    return resized_tensors, fps

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(np.uint8)

    scenes = []
    t, t_prev, start = -1, 0, 0
    for i, t in enumerate(predictions):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            scenes.append([start, i])
        t_prev = t
    if t == 0:
        scenes.append([start, i])

    # just fix if all predictions are 1
    if len(scenes) == 0:
        return np.array([[0, len(predictions) - 1]], dtype=np.int32)

    return np.array(scenes, dtype=np.int32)

def get_len(x, fps):
    assert len(x) == 2, "split range should have a length of 2"
    return (x[1] - x[0]) / fps

def combine_splits(s1,s2):
    assert s1[1] < s2[0] 
    return (s1[0], s2[1])

def get_asr(asr_model, video_fn, fps, shot_list):
    audio = whisper.load_audio(video_fn)
    audio = whisper.pad_or_trim(audio)
    res_ = asr_model.transcribe(audio)
    
    vlen = max([shot[1] for shot in shot_list])
    
    asr = []
    asr_segs = res_['segments']
    for shot_cap in shot_list:
        start, end = shot_cap[0]/fps, shot_cap[1]/fps
        
        clip_start = start  # replace with actual value
        clip_end = end  # replace with actual value

        loosen_amount = 0.2
        # Loosen the clip start and end timestamps
        clip_start = max(clip_start-loosen_amount, 0.0) # replace loosen_amount with actual value
        clip_end = min(clip_end+loosen_amount, vlen/fps)  # replace loosen_amount with actual value

        clip_sentences = []
        for sentence in asr_segs:
            sentence_start = sentence['start']  # replace with actual key
            sentence_end = sentence['end']  # replace with actual key
            if sentence_start <= clip_end and sentence_end >= clip_start:
                clip_sentences.append(sentence['text'].strip())
        
        text = ' '.join(clip_sentences)
        asr.append(text)
    return asr, ' '.join([asr_['text'].strip() for asr_ in asr_segs])

def evaluate_input_shots(input_text, fps, total_frames):
    shot_lines = input_text.split('\n')
    shot_list = []
    # total_frames = 0

    for line in shot_lines:
        if not line.strip():
            continue  # Skip empty lines

        try:
            start_sec, end_sec = map(float, line.split())
            if start_sec < 0 or end_sec < 0:
                print(f"Invalid input: Negative timestamp found in '{line}'.")
                return True, None
            if start_sec >= end_sec:
                print(f"Invalid input: Start timestamp must be less than end timestamp in '{line}'.")
                return True, None
            
            start_frame = start_sec * fps
            end_frame = end_sec * fps
            shot_list.append((int(start_frame), int(end_frame)))
            total_frames = max(total_frames, end_frame)
        except ValueError:
            print(f"Invalid input: Each line should contain exactly two non-negative floats, found '{line}'.")
            return True, None

    # Call the function to evaluate shot detection method
    return False, shot_list

def evaluate_shot_detection(total_frames, shot_list, fps):
    # video_duration_minutes = (total_frames / fps) / 60
    num_shots = len(shot_list)
    avg_shot_length_frames = total_frames / num_shots if num_shots else 0
    avg_shot_length_seconds = avg_shot_length_frames / fps

    # Define thresholds
    min_shots_threshold = total_frames / (30 * fps)  # One shot per 30 seconds as an example threshold
    max_avg_shot_duration_seconds = 60  # 60 seconds as an example threshold

    # Criteria evaluation
    too_few_shots = num_shots < min_shots_threshold
    avg_shot_too_long = avg_shot_length_seconds > max_avg_shot_duration_seconds

    if too_few_shots or avg_shot_too_long:
        return True
    else:
        return False

def get_split(video_fn, transform, dataset, transnet_model, asr_model, sampling='headtail', input_splits=None):
    # Usage example
    # video_fn = '/mnt/bn/kinetics-lp-maliva-v6/data/hdvila/tcs/collation_final_videos/dR9jfG9Zr5A.2.mp4'
    desired_width = 48
    desired_height = 27
    frames_for_shots, fps = read_video_frames_resized(video_fn, desired_width, desired_height)
    # frames = torch.stack(frames).permute(0,2,3,1)
    frames_for_shots = torch.stack(frames_for_shots).unsqueeze(dim=0).permute(0,1,3,4,2)
    
    vr = VideoReader(video_fn, height=224, width=224)
    whole_vlen = len(vr)
    
    model = transnet_model

    if input_splits is not None:
        require_shot_detection, new_splits = evaluate_input_shots(input_splits, fps, whole_vlen)
    else:
        require_shot_detection = True

    if require_shot_detection:
        with torch.no_grad():
            # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
            # input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
            single_frame_pred, all_frame_pred = model(frames_for_shots.cuda())
            
            single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
            all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
            
            scenes = predictions_to_scenes(single_frame_pred[0])
            new_splits = []
            th = 1.0
            for x in scenes:
                # print(x)
                if len(new_splits) == 0:
                    new_splits.append(x.tolist())
                elif get_len(new_splits[-1], fps) < th:
                    new_splits[-1] = combine_splits(new_splits[-1], x)
                elif get_len(x, fps) < th:
                    new_splits[-1] = combine_splits(new_splits[-1], x)
                else:
                    new_splits.append(x.tolist())
            
            new_splits = [x for x in new_splits]
            # print(new_splits)
        
        resplit_video = evaluate_shot_detection(whole_vlen, new_splits, fps)
        if resplit_video:
            resplit_video_splits = find_scenes(video_fn, 18)
            print('New scene detection results', len(new_splits), len(resplit_video_splits))
            if len(resplit_video_splits) > len(new_splits):
                new_splits = resplit_video_splits
        
    # Assuming 'shots' is a list of tuples (start_frame, end_frame)
    shot_lengths = [(i, end - start, start, end) for i, (start, end) in enumerate(new_splits)]
    shot_lengths.sort(key=lambda x: x[1]) # Sort by shot length

    if require_shot_detection:
        if resplit_video:
            max_num_shots = 8
        else:
            max_num_shots = 16
    else:
        max_num_shots = 16

    while len(shot_lengths) > max_num_shots:
        # Choose the shortest shot to merge
        shortest_shot = shot_lengths.pop(0)

        # Determine adjacent shots
        prev_shot = next((s for s in shot_lengths if s[3] == shortest_shot[2]), None)
        next_shot = next((s for s in shot_lengths if s[2] == shortest_shot[3]), None)

        # Decide which adjacent shot to merge with
        if prev_shot and next_shot:
            merge_with = prev_shot if prev_shot[1] < next_shot[1] else next_shot
        elif prev_shot:
            merge_with = prev_shot
        elif next_shot:
            merge_with = next_shot
        else:
            continue  # No adjacent shot to merge with

        # Merge the shots
        new_start = min(shortest_shot[2], merge_with[2])
        new_end = max(shortest_shot[3], merge_with[3])
        new_length = new_end - new_start

        # Update the shot list
        shot_lengths.remove(merge_with)
        shot_lengths.append((merge_with[0], new_length, new_start, new_end))
        shot_lengths.sort(key=lambda x: x[1]) # Re-sort by shot length

    # Sort back into original order and extract start and end frames
    final_shots = sorted(shot_lengths, key=lambda x: x[0])
    final_shots = [(start, end) for _, _, start, end in final_shots]

    new_splits = final_shots

    samples = dataset['bdmsvdc_multishot_minigpt_caption']['20k_val'].get_sample_nfrms(len(new_splits), "")
    shot_splits = [new_splits[i] for i in samples["shot_ids"]]
    if len(samples["shot_split"]) != len(new_splits):
        frames = [fi for shot in shot_splits for fi in range(shot[0], shot[1])]

    shot_frames = []
    for shot_id, split, n_frms in zip(samples["shot_ids"], shot_splits, samples["shot_split"]):
        vlen = split[1]-split[0]+1
        #print('video len', vlen)
        start, end = 0, vlen
        #n_frms = min(n_frms, vlen)
        if n_frms > vlen:
            indices = np.arange(start, end, vlen / n_frms).astype(int)
        elif sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int)
        elif sampling == "headtail":
            half = n_frms // 2
            another_half = n_frms - half
            sampled_cnt = [half, another_half]
            random.shuffle(sampled_cnt)
            indices_h = sorted(rnd.sample(range(vlen // 2), sampled_cnt[0]))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), sampled_cnt[1]))
            indices = indices_h + indices_t
        else:
            raise NotImplementedError

        # get_batch -> T, H, W, C
        #print(frms)
        #print(frms.shape)
        indices = [int(i) if int(i) < vlen else vlen-1 for i in indices]
        indices = sorted(indices)[:n_frms]
        indices = [i + split[0] for i in indices]
        try:
            frms = vr.get_batch(indices)
            # frms = torch.stack([frames[split[0]+i] for i in indices])
            if isinstance(frms, torch.Tensor):
                frms = frms.permute(1,0,2,3).float() 
            elif isinstance(frms, NDArray):
                frms = torch.from_numpy(frms.asnumpy()).permute(3,0,1,2).float()
        except Exception as e:
            # print(indices, len(vr), n_frms)
            # print(video_path)
            # indices = [int(i) if int(i) < len(vr) else rnd.sample(range(vlen),1)[0] for i in indices]
            # print(indices)
            print(e)
        assert frms.shape[1]==n_frms, f"{frms.shape}, {len(frms)}, {indices}, {vlen}, {n_frms}"
        # frms = torch.from_numpy(vr.get_batch(indices).asnumpy()).permute(3, 0, 1, 2).float()  # (C, T, H, W)
        frms = transform(frms)
        shot_frames.append(frms)
        
    asr, whole_asr = get_asr(asr_model, video_fn, fps, shot_splits)
    assert shot_frames[0].shape[0] == 3
    video = torch.cat(shot_frames, dim=1)
    samples['asrs'] = [dataset['bdmsvdc_multishot_minigpt_caption']['20k_val'].text_processor(asr_) if len(asr_) != 0 else '' for asr_ in asr ]
    samples['video'] = video
    samples['whole_asr'] = dataset['bdmsvdc_multishot_minigpt_caption']['20k_val'].text_processor(whole_asr)
    samples = prepare_sample(samples)
    
    samples['whole_asr'] = [samples['whole_asr']]
    samples['video'] = samples['video'].unsqueeze(dim=0)
    return samples


class Chat:
    def __init__(self, model, vis_processor, task=None, dataset=None, device='cuda:0'):
        self.device = device
        self.model = model
        self.task = task
        self.dataset = dataset
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.n_frms = self.vis_processor.n_frms
        
        self.transnet_model = TransNetV2()
        state_dict = torch.load("/mnt/bn/kinetics-lp-maliva/playground_projects/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth")
        self.transnet_model.load_state_dict(state_dict)
        self.transnet_model.eval().cuda()
        
        self.asr_model = whisper.load_model("large-v2", 'cuda')

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, num_captions=1, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, max_new_token_adapt=False):
        conv.append_message(conv.roles[1], None)
        embs = self.get_context_emb_multishot(conv, self.samples)

        if max_new_token_adapt:
            max_new_tokens = min(max_length - embs.shape[1], max_new_tokens)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_captions
        )
        output_texts = []
        for output in outputs:
            output_token = output
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_texts.append(output_text)
        conv.messages[-1][1] = output_texts[0]
        return output_texts, output_token.cpu().numpy()

    def answer_text(self, conv, num_captions=1, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, max_new_token_adapt=False):
        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        seg_tokens = [[
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=False).to(self.model.device).input_ids
                for i, seg in enumerate(prompt_segs_)
            ] for prompt_segs_ in [[prompt]]]
        seg_embs = [[self.model.llama_model.model.embed_tokens(seg_t).squeeze(dim=0) for seg_t in seg_bs_t] for seg_bs_t in seg_tokens] 
        embs = torch.stack([torch.cat(seg_embs_, dim=0) for seg_embs_ in seg_embs])

        if max_new_token_adapt:
            max_new_tokens = min(max_length - embs.shape[1], max_new_tokens)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_captions
        )
        output_texts = []
        for output in outputs:
            output_token = output
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_texts.append(output_text)
        conv.messages[-1][1] = output_texts[0]
        return output_texts, output_token.cpu().numpy()

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

        image_emb, _ = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def upload_video_single_frame(self, video, conv, img_list):
        if isinstance(video, str):  # is a image path
            #raw_image = Image.open(image).convert('RGB')
            video = self.vis_processor(video).unsqueeze(0).to(self.device)
        elif isinstance(video, torch.Tensor):
            if len(video.shape) == 3:
                video = video.unsqueeze(0)
            video = video.to(self.device)
        else:
            raise Exception("video should be either str or torch.Tesnor")
        video = torch.permute(video, (0,2,1,3,4))

        image_emb, _ = self.model.encode_img(video[0])
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg
    
    def upload_video_ms_standalone(self, video, conv, img_list, collate_way='none', step=1, input_splits=""):
        assert isinstance(video, str)
        assert collate_way == 'none'
        input_splits = None if len(input_splits) == 0 else input_splits
        
        dataset = self.dataset
        self.samples = get_split(video, self.vis_processor.transform, dataset, self.transnet_model, self.asr_model, sampling='headtail', input_splits=input_splits)
        # dataset
        prompt = self.task.get_prompt(self.samples)[0].split('###Human: ')[1].split('###Assistant: ')[0]
        conv.append_message(conv.roles[0], prompt)
        
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg
    
    '''
    collate_way: how the different frames are processed and collated. 'none', 'raw_avg', 'feat_avg'
    step: the number of sampled frames been collated
    '''
    def upload_video(self, video, conv, img_list, collate_way='none', step=1, shot_list=[]):
        if isinstance(video, str):  # is a image path
            if len(shot_list) != 0:
                shot_names = shot_list
                num_shots = len(shot_names)
                total_frames_to_sample = self.n_frms
                
                # Calculate the number of frames each shot should get
                frames_per_shot = total_frames_to_sample // num_shots
                # Calculate the number of remaining frames
                remaining_frames = total_frames_to_sample % num_shots
                # Create a list of frames per shot
                frames_to_sample = [frames_per_shot] * num_shots
                # Distribute the remaining frames evenly among the shots
                for i in range(remaining_frames):
                    frames_to_sample[i] += 1
                # print(frames_to_sample)
                
                vis_root = os.path.dirname(video)
                shot_videos = []
                for n_frms, shot_name in zip(frames_to_sample, shot_names):
                    self.vis_processor.n_frms = n_frms
                    video_path = os.path.join(vis_root, shot_name)
                    shot_videos.append(self.vis_processor(video_path))
                assert shot_videos[0].shape[0] == 3
                # print([s.shape for s in shot_videos])
                video = torch.cat(shot_videos, dim=1)
                video = video.unsqueeze(0).to(self.device)

                shot_split = frames_to_sample
                num_shot = len(shot_split)
                id_text_mapping = {0: 'first', 1: 'second', 2:'third', 3:'forth', 4:'fifth', 5:'sixth', 
                    6:'seventh', 7:'eighth'}
                prompt_prefix = f'This is a video with {num_shot} shots. '
                for shot_idx, shot_frms in enumerate(shot_split):
                    prompt_prefix += f'The {id_text_mapping[shot_idx]} shot is '
                    for i in range(shot_frms):
                        prompt_prefix += '<Img><ImageHere></Img>'
                    prompt_prefix += '. '
                im_list_str = prompt_prefix
            else:
                #raw_image = Image.open(image).convert('RGB')
                video = self.vis_processor(video).unsqueeze(0).to(self.device)
                im_list_str = ""
                for i in range(self.n_frms//step):
                    im_list_str += "<Img><ImageHere></Img>"
        elif isinstance(video, torch.Tensor):
            if len(video.shape) == 3:
                video = video.unsqueeze(0)
            video = video.to(self.device)
        else:
            raise Exception("video should be either str or torch.Tesnor")
        
        if collate_way == 'none':
            # N C T H W to N T C H W
            bs = 1
            n_frms = video.size(2)
            video = torch.permute(video, (0,2,1,3,4))
            # to NxT, C H W
            video = video.reshape((bs*n_frms, ) + video.size()[2:])
            video_emb = self.model.encode_img(video)
            #video_emb = video_emb.view((bs, n_frms * video_emb.size(1), video_emb.size(2)))
            for i in range(n_frms):
                img_list.append(video_emb[i].unsqueeze(0))
            # im_list_str = ""
            # for i in range(n_frms//step):
            #     im_list_str += "<Img><ImageHere></Img>"
            conv.append_message(conv.roles[0], im_list_str)
            msg = "Received."
            # self.conv.append_message(self.conv.roles[1], msg)
        elif collate_way == 'raw_avg':
            # N C T H W to N T C H W
            bs = 1
            video = [v.mean(dim=2).unsqueeze(dim=2) for v in video.split(step, dim=2)]
            # print(video.shape)
            video = torch.cat(video, dim=2)
            # print(video.shape)
            n_frms = video.size(2)
            video = torch.permute(video, (0,2,1,3,4))
            # to NxT, C H W
            video = video.reshape((bs*n_frms, ) + video.size()[2:])
            video_emb = self.model.encode_img(video)
            #video_emb = video_emb.view((bs, n_frms * video_emb.size(1), video_emb.size(2)))
            for i in range(n_frms):
                img_list.append(video_emb[i].unsqueeze(0))
            im_list_str = ""
            for i in range(n_frms):
                im_list_str += "<Img><ImageHere></Img>"
            conv.append_message(conv.roles[0], im_list_str)
            msg = "Received."
            # self.conv.append_message(self.conv.roles[1], msg)
        elif collate_way == 'feat_avg':
            # N C T H W to N T C H W
            bs = 1
            assert video.shape[0] == 1
            n_frms = video.size(2)
            video = torch.permute(video, (0,2,1,3,4))
            # to NxT, C H W
            video = video.reshape((bs*n_frms, ) + video.size()[2:])
            video_emb = self.model.encode_img(video)
            #video_emb = video_emb.view((bs, n_frms * video_emb.size(1), video_emb.size(2)))
            img_list.extend(v.mean(dim=0).unsqueeze(dim=0) for v in video_emb.split(step, dim=0))
            # for i in range(n_frms):
            #     img_list.append(video_emb[i].unsqueeze(0))
            im_list_str = ""
            for i in range(n_frms//step):
                im_list_str += "<Img><ImageHere></Img>"
            conv.append_message(conv.roles[0], im_list_str)
            msg = "Received."
            # self.conv.append_message(self.conv.roles[1], msg)
        return msg
    
    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, f"Unmatched numbers of image placeholders and images, {prompt} {len(prompt_segs)}, {len(img_list) + 1}."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def get_context_emb_multishot(self, conv, samples):
        prompt = conv.get_prompt()
        # print(samples['video_name'])
        video = samples['video']
        # N C T H W to N T C H W
        # print(video.shape)
        bs = video.size(0)
        # assert bs == 1
        n_frms = video.size(2)
        video = torch.permute(video, (0,2,1,3,4))
        # to NxT, C H W
        video = video.reshape((bs*n_frms, ) + video.size()[2:])
        # print("L297", video.shape)
        
        img_list = []
        video_emb = self.model.encode_img(video)
        # print("L300", video_emb.shape)
        video_emb = video_emb.view((bs, n_frms * video_emb.size(1), video_emb.size(2)))
        # print("302", video_emb.shape)
        atts_img = torch.ones(video_emb.size()[:-1], dtype=torch.long).to(video.device)
        video_emb = video_emb.view((bs, n_frms, -1, video_emb.size(2)))
        # print("305", video_emb.shape)
        # print(video_emb.shape)
        video_emb, atts_img = self.prompt_wrap(video_emb, atts_img, [prompt])
        return video_emb

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        # print(prompt)
        if prompt:
            batch_size = img_embeds.shape[0]
            n_frms = img_embeds.shape[1]
            prompt_segs = [prompt_.split('<ImageHere>') for prompt_ in prompt]
            assert len(prompt_segs[0])-1 == n_frms, f"{len(prompt_segs)}, {n_frms}"
            seg_tokens = [[
                    self.model.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device).input_ids
                    for i, seg in enumerate(prompt_segs_)
                ] for prompt_segs_ in prompt_segs]
            # # seg_tokens = torch.load('/mnt/bn/kinetics-lp-maliva/playground_projects/BLIP/demo_tokens.pth')
            # # seg_tokens = [s.to(self.device) for s in seg_tokens]
            # seg_embs = [model.llama_model.model.embed_tokens(seg_t).expand(batch_size, -1, -1) for seg_t in seg_tokens] 
            # # im_list = [im_emb.unsqueeze(0) for im_emb in img_embeds]
            # im_list = [img_embeds[:,fi,...] for fi in range(n_frms)]
            # mixed_embeds = [emb for pair in zip(seg_embs[:-1], im_list) for emb in pair] + [seg_embs[-1]]
            
            # im_list = [img_embeds[:,fi,...] for fi in range(n_frms)]
            seg_embs = [[self.model.llama_model.model.embed_tokens(seg_t).squeeze(dim=0) for seg_t in seg_bs_t] for seg_bs_t in seg_tokens] 
            # mixed_embeds = [[torch.cat([seg_e_bs, im_e_bs], dim=0) for seg_e_bs, im_e_bs in zip(*pair)] for pair in zip(seg_embs[:-1], im_list)] + [seg_embs[-1]]
            mixed_embeds = [[torch.cat([seg_e, im_e], dim=0) for im_e, seg_e in zip(img_bs_embs, seg_bs_embs[:-1])] + [seg_bs_embs[-1]] for img_bs_embs, seg_bs_embs in zip(img_embeds, seg_embs)]
            mixed_embeds = [torch.cat(mixed_embeds_, dim=0) for mixed_embeds_ in mixed_embeds]
            wrapped_img_embeds = pad_sequence(mixed_embeds, batch_first=True)
            wrapped_atts_img = (wrapped_img_embeds != 0)[:,:,0].long()
            
            # wrapped_img_embeds = torch.cat(mixed_embeds, dim=1)
            # wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img  