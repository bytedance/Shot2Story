"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import re
import json

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger
from lavis.common.dist_utils import is_dist_avail_and_initialized, main_process
from transformers import StoppingCriteria, StoppingCriteriaList

from collections import OrderedDict
from nlgeval import NLGEval

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask):
    def __init__(self, cfg, model_cfg, report_metric=True):
        super().__init__()
        self._device = None
        self.config = cfg
        self.model_cfg = model_cfg
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.asr_audio = self.model_cfg.asr_audio
        self.audio_target = self.model_cfg.audio_target
        self.whole_video = self.model_cfg.whole_video
        self.multishot = self.model_cfg.multishot
        self.fix_total = self.model_cfg.get('fix_total', False)
        self.whole_asr = self.model_cfg.get('whole_asr', False)
        self.flexible_sampling = self.model_cfg.get('flexible_sampling', False)
        self.flexible_sampling_frames = [16,8,4,2,1,1]
        self.flexible_sampling_thrs = [0,4,8,16,32,64,128]
        
        self.flexible_sampling_mapping = {}
        for i in range(1, len(self.flexible_sampling_thrs)):
            for f_n in range(self.flexible_sampling_thrs[i-1], self.flexible_sampling_thrs[i]):
                self.flexible_sampling_mapping[f_n] = self.flexible_sampling_frames[i-1]
        
        num_frms = self.model_cfg.num_frms
        self.system_prompt = self.model_cfg.get('system_prompt', "Given a video, " +
           "you will be able to see the frames once I provide it to you. Please answer my questions.")
        self.system_prompt = "" if self.system_prompt is None else self.system_prompt
        # im_list = ""
        # for im in range(num_frms):
        #     im_list += "<Img>ImageContent{}</Img>".format(im)
        # self.system_prompt = self.model_cfg.get('system_prompt', "Give the following video: {}. ".format(im_list) +
        #    "You will be able to see the images once I provide it to you. Please answer my questions.",)
        # self.system_prompt = ""
        #    "you will be able to see the frames once I provide it to you. Please answer my questions.")
        self.sep = self.model_cfg.end_sym
        self.prompt_template = self.model_cfg.prompt_template
        
        self.multishot_prompt = self.model_cfg.get('multishot_prompt', "This is a video with {num_shot} shots. ")
        self.multishot_prompt = "This is a video with {num_shot} shots. " if self.multishot_prompt is None else self.multishot_prompt
        
        self.multishot_secondary_prompt = self.model_cfg.get('multishot_secondary_prompt', "The {shot_idx_text} shot is ")
        self.multishot_secondary_prompt = "The {shot_idx_text} shot is " if self.multishot_secondary_prompt is None else self.multishot_secondary_prompt
        
        self.per_shot_asr_words_limit = 30
        self.whole_video_asr_words_limit = 80
        
        # if self.audio_target:
        #     self.answer_prompt = "In the audio, "
        # else:
        #     self.answer_prompt = "The video shows"
        # self.answer_prompt = ""
        self.answer_prompt = self.model_cfg.get('answer_prompt', "") 
        self.answer_prompt = "" if self.answer_prompt is None else self.answer_prompt
        
        # text_input = " "
        # if self.asr_audio:
        #     text_input += "In the audio, I heat that: {asr}. "
        #     if self.audio_target:
        #         text_input += "Please provide a detailed description of the audio in the video." # Do not include details that you are not sure of and the content that not relates to the visual content. "#For example, if there is text in the image, do not include the content of the text if they are not clearly shown. "
        #     else:
        #         text_input += "Please describe this video according to the audio and visual content." # Do not include details that you are not sure of. For example, if there is text in the image, do not include the content of the text if they are not clearly shown. "
        # else:
        #     text_input = "Please provide a detailed description of the video."# Do not include details that you are not sure of. For example, if there is text in the image, do not include the content of the text if they are not clearly shown. "
        # self.question_prompt = text_input
        self.question_prompt = self.model_cfg.get('question_prompt', "Please provide a detailed description of the video.")
        self.question_prompt = "" if self.question_prompt is None else str(self.question_prompt)
        
        if self.multishot:
            prompt_prefix = '{promp_img}'
        else:
            if self.whole_video and not self.fix_total:
                prompt_prefix = '{promp_img}'
            else:
                prompt_prefix = ''
                for i in range(num_frms):
                    prompt_prefix += '<Img><ImageHere></Img>'
        # filted_prompt = ' '.join([prompt_prefix, self.question_prompt])
        filted_prompt = prompt_prefix + str(self.question_prompt)
        self.inference_prompt = self.prompt_template.format(filted_prompt) +  self.answer_prompt
        self.inference_prompt = self.system_prompt + self.inference_prompt
        print('Inference Prompt Example \n{}'.format(self.inference_prompt))
        
        # self.num_beams = num_beams
        # self.max_len = max_len
        # self.min_len = min_len
        # self.evaluate = evaluate

        self.report_metric = report_metric
        
    @classmethod
    def setup_task(cls, cfg):
        model_cfg = cfg.model_cfg
        run_cfg = cfg.run_cfg

        # num_beams = run_cfg.num_beams
        # max_len = run_cfg.max_len
        # min_len = run_cfg.min_len
        # evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(cfg, model_cfg,
            report_metric=report_metric,
            )
        
    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            prompt = self.get_prompt(samples)
            eval_output = self.valid_step(model=model, samples=samples, prompt=prompt)
            results.extend(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwself):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics
    
    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        def build_nlgEvalObj():
            # need to add java into environ variable
            os.environ["PATH"] += os.pathsep + "/opt/tiger/yarn_deploy/jdk/bin/"
            nlgEvalObj = NLGEval(
                no_overlap=False,
                no_skipthoughts=True,
                no_glove=True,
                metrics_to_omit=None,
            )

            return nlgEvalObj

        nlgEvalObj = build_nlgEvalObj()
        eval_result = json.load(open(eval_result_file))
        eval_result = sorted(eval_result, key=lambda x: x['image_id'])
        # eval_result = sorted(eval_result.items(), key=lambda x: x[0])
        # eval_result = [{'image_id': vid_cap[0], 'caption': vid_cap[1][0][0]} for vid_cap in eval_result]
        
        all_ids = [res['image_id'] for res in eval_result]
        # print(len(all_ids))
        gt_path = '/mnt/bn/kinetics-lp-maliva-v6/data/hdvila/annotations/{}.json'.format(split_name)
        annos = json.load(open(gt_path))
        if self.audio_target:
            annos = [anno for anno in annos if anno['audio_caption'] != '']
        assert len(annos) == len(eval_result)
        # if len(annos) > len(eval_result):
        #     annos = annos[:len(eval_result)]
        all_gts = OrderedDict()
        for item in annos:
            if self.audio_target:
                all_gts[item['image_id']] = [item['audio_caption']]
            else:
                all_gts[item['image_id']] = [item['whole_caption' if self.whole_video else 'caption']]

        # all_caption_lists = [v for k,v in all_gts.items()]
        all_caption_lists = [all_gts[k] for k in all_ids]
        all_caption_lists = [
                                list(itms) for itms in zip(*all_caption_lists)
                                                                        ]
        result_captions = [res['caption'] for res in eval_result]
        metrics_nlg = nlgEvalObj.compute_metrics(
                    ref_list=all_caption_lists, hyp_list=result_captions
                )
        metrics_nlg['agg_metrics'] = metrics_nlg['CIDEr']
        print(metrics_nlg)
        
        return metrics_nlg

    def get_prompt(self, samples):
        prompt = self.inference_prompt
        if self.multishot:
            shot_split = samples["shot_split"]
            num_shot = len(shot_split)
            id_text_mapping = {
                0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth', 5: 'sixth',
                6: 'seventh', 7: 'eighth', 8: 'ninth', 9: 'tenth', 10: 'eleventh', 11: 'twelfth', 
                12: 'thirteenth', 13: 'fourteenth', 14: 'fifteenth', 15: 'sixteenth', 
                16: 'seventeenth', 17: 'eighteenth', 18: 'nineteenth', 19: 'twentieth', 
                20: 'twenty-first', 21: 'twenty-second', 22: 'twenty-third', 
                23: 'twenty-fourth', 24: 'twenty-fifth', 25: 'twenty-sixth',
                26: 'twenty-seventh', 27: 'twenty-eighth', 28: 'twenty-ninth', 
                29: 'thirtieth', 30: 'thirty-first', 31: 'thirty-second', 
                32: 'thirty-third', 33: 'thirty-fourth', 34: 'thirty-fifth',
                35: 'thirty-sixth', 36: 'thirty-seventh', 37: 'thirty-eighth',
                38: 'thirty-ninth', 39: 'fortieth', 40: 'forty-first',
                41: 'forty-second', 42: 'forty-third', 43: 'forty-fourth',
                44: 'forty-fifth', 45: 'forty-sixth', 46: 'forty-seventh',
                47: 'forty-eighth', 48: 'forty-ninth', 49: 'fiftieth',
                50: 'fifty-first', 51: 'fifty-second', 52: 'fifty-third',
                53: 'fifty-fourth', 54: 'fifty-fifth', 55: 'fifty-sixth',
                56: 'fifty-seventh', 57: 'fifty-eighth', 58: 'fifty-ninth',
                59: 'sixtieth', 60: 'sixty-first', 61: 'sixty-second',
                62: 'sixty-third', 63: 'sixty-fourth'
            }
            key = re.findall(r"\{(.+?)\}", self.multishot_prompt)
            if len(key) == 1:
                if key[0] == "num_shot":
                    prompt_prefix = self.multishot_prompt.format(num_shot=num_shot)
                else:
                    raise NotImplementedError
            elif len(key) == 0:
                prompt_prefix = self.multishot_prompt
            else:
                raise NotImplementedError
            del key
            for shot_idx, shot_frms in enumerate(shot_split):
                # prompt_prefix += f'The {id_text_mapping[shot_idx]} shot is '
                shot_idx_text = id_text_mapping[shot_idx]
                key = re.findall(r"\{(.+?)\}", self.multishot_secondary_prompt)
                if len(key) == 1:
                    if key[0] == "shot_idx_text":
                        prompt_prefix += self.multishot_secondary_prompt.format(shot_idx_text=shot_idx_text)
                    else:
                        raise NotImplementedError
                elif len(key) == 0:
                    prompt_prefix += self.multishot_secondary_prompt
                else:
                    raise NotImplementedError
                del key
                for i in range(shot_frms):
                    prompt_prefix += '<Img><ImageHere></Img>'
                prompt_prefix += '. '
        else:
            prompt_prefix = self.multishot_prompt
            if self.whole_video and not self.fix_total:
                assert self.whole_video
                shot_split = samples["shot_split"]
                for i in range(sum(shot_split)):
                    prompt_prefix += '<Img><ImageHere></Img>'
                prompt_prefix += '. '
        # print(prompt_prefix, samples['shot_split'])
        if self.asr_audio:
            if self.whole_video:
                asr = samples["whole_asr"]
                words_limit = self.whole_video_asr_words_limit
                assert len(asr) == 1
            else:
                if self.whole_asr:
                    asr = samples["whole_asr"]
                else:
                    asr = samples["asr"]
                words_limit = self.per_shot_asr_words_limit
                assert isinstance(asr, list)
                asr = [' '.join(asr_.split(' ')[:words_limit]) for asr_ in asr]
            if self.multishot:
                assert self.whole_video
                prompt = [prompt.format(asr=asr_, promp_img=prompt_prefix) for asr_ in asr]
            else:
                if self.whole_video and not self.fix_total:
                    prompt = [prompt.format(asr=asr_, promp_img=prompt_prefix) for asr_ in asr]
                else:
                    prompt = [prompt.format(asr=asr_) for asr_ in asr]
        else:
            bs = samples['video'].shape[0]
            if self.multishot:
                prompt = [prompt.format(promp_img=prompt_prefix)] * bs
            elif self.whole_video and not self.fix_total:
                prompt = [prompt.format(promp_img=prompt_prefix)] *bs
            else:
                prompt = [prompt] * bs
                
        return prompt
          
    def prompt_wrap(self, model, img_embeds, atts_img, prompt):
        # print(prompt)
        if prompt:
            batch_size = img_embeds.shape[0]
            n_frms = img_embeds.shape[1]
            prompt_segs = [prompt_.split('<ImageHere>') for prompt_ in prompt]
            assert len(prompt_segs[0])-1 == n_frms, f"{len(prompt_segs)}, {n_frms}"
            seg_tokens = [[
                    model.llama_tokenizer(
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
            seg_embs = [[model.llama_model.model.embed_tokens(seg_t).squeeze(dim=0) for seg_t in seg_bs_t] for seg_bs_t in seg_tokens] 
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
    
    def get_context_emb(self, model, samples, prompt):
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
        video_emb = model.encode_img(video)
        # print("L300", video_emb.shape)
        video_emb = video_emb.view((bs, n_frms * video_emb.size(1), video_emb.size(2)))
        # print("302", video_emb.shape)
        atts_img = torch.ones(video_emb.size()[:-1], dtype=torch.long).to(video.device)
        video_emb = video_emb.view((bs, n_frms, -1, video_emb.size(2)))
        # print("305", video_emb.shape)
        # print(video_emb.shape)
        video_emb, atts_img = self.prompt_wrap(model, video_emb, atts_img, prompt)
        return video_emb
        
        # for i in range(n_frms):
        #     img_list.append(video_emb[i].unsqueeze(0))
        
        # prompt_segs = prompt.split('<ImageHere>')
        # assert len(prompt_segs) == n_frms + 1, f"Unmatched numbers of image placeholders and images, {prompt} {len(prompt_segs)}, {len(img_list) + 1}."
        # seg_tokens = [
        #     model.llama_tokenizer(
        #         seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
        #     # only add bos to the first seg
        #     for i, seg in enumerate(prompt_segs)
        # ]
        # seg_tokens = torch.load('/mnt/bn/kinetics-lp-maliva/playground_projects/BLIP/demo_tokens.pth')
        # seg_tokens = [s.to(self.device) for s in seg_tokens]
        # seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        # mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        # mixed_embs = torch.cat(mixed_embs, dim=1)
        # return mixed_embs

    def valid_step(self, model, samples, prompt, max_length=2048, max_new_tokens=600, max_new_token_adapt=True, num_beams=3, min_length=8, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, num_captions=1):
        # print(self.fix_total, samples['video'].shape)
        embs = self.get_context_emb(model, samples, prompt)
        # print(embs.shape)
        video_names = samples['video_name']

        if max_new_token_adapt:
            max_new_tokens = min(max_length - embs.shape[1], max_new_tokens)
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        outputs = model.llama_model.generate(
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
        # print("This is outputs", outputs)
        results = []
        for output, video_name in zip(outputs, video_names):
            output_token = output
            if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
            if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
            output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            output_text = output_text.split('###')[0]  # remove the stop sign '###'
            output_text = output_text.split('Assistant:')[-1].strip()
            output_text = f"{self.answer_prompt} {output_text}"
            # print(output_text)
            results.append({"caption": output_text, "image_id": video_name})
    
        return results