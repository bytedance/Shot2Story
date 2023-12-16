import logging
import random
import re

import torch
from torch.cuda.amp import autocast as autocast
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer


@registry.register_model("video_minigpt4")
class VideoMiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/blip2/video_minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        low_resource=False,  # use 8 bit and put vit in cpu
        end_sym='\n',
        num_frms=8,
        asr_audio=False,
        audio_target=False,
        visual_target=True,
        whole_video=False,
        multishot=False,
        av_target=False,
        prompt_order="vt",
        system_prompt="",
        answer_prompt="",
        multishot_prompt="This is a video with {num_shot} shots. ",
        multishot_secondary_prompt="The {shot_idx_text} shot is ",
        fix_total=False,
        mix_multishot=False,
        asr_target=False,
        whole_asr=False,
        filter_asr=False,
    ):
        super().__init__()
        
        self.per_shot_asr_words_limit = 80
        self.whole_video_asr_words_limit = 80
        
        self.prompt_order = prompt_order
        self.system_prompt = "" if system_prompt is None else str(system_prompt)
        self.answer_prompt = "" if answer_prompt is None else str(answer_prompt)
        self.multishot_prompt = "This is a video with {num_shot} shots. " if multishot_prompt is None else str(multishot_prompt)
        self.multishot_secondary_prompt = "The {shot_idx_text} shot is " if multishot_secondary_prompt is None else str(multishot_secondary_prompt)

        self.asr_audio = asr_audio
        self.audio_target = audio_target
        self.visual_target = visual_target
        self.whole_video = whole_video
        self.multishot = multishot
        self.fix_total = fix_total
        self.mix_multishot = mix_multishot
        self.asr_target = asr_target
        self.whole_asr = whole_asr
        self.filter_asr = filter_asr
        
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        print(q_former_model)
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = raw_prompts
            # manually add <ImageHere> tag
            if self.multishot:
                prompt_prefix = '{promp_img}'
            else:
                if self.whole_video and not self.fix_total:
                    prompt_prefix = '{promp_img}'
                else:
                    prompt_prefix = ''
                    for i in range(num_frms):
                        prompt_prefix += '<Img><ImageHere></Img>'
            if prompt_order == "vt":
                filted_prompts = ["".join([prompt_prefix, prompt]) for prompt in raw_prompts]
            elif prompt_order == "tv":
                filted_prompts = ["".join([prompt, prompt_prefix]) for prompt in raw_prompts]
            elif prompt_order == "random":
                filted_prompts = [[prompt, prompt_prefix] for prompt in raw_prompts]
                [random.shuffle(prompt_pair) for prompt_pair in filted_prompts]
            else:
                raise NotImplementedError
            # filted_prompts = ["".join([prompt_prefix, prompt]) for prompt in raw_prompts]
            self.prompt_list = [system_prompt + prompt_template.format(p) + answer_prompt for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
        return inputs_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            n_frms = img_embeds.shape[1]
            prompt_segs = [prompt_.split('<ImageHere>') for prompt_ in prompt]
            assert len(prompt_segs[0])-1 == n_frms
            seg_tokens = [[
                    self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device).input_ids
                    for i, seg in enumerate(prompt_segs_)
                ] for prompt_segs_ in prompt_segs]
            # seg_tokens_bs_dim_inside = [[seg_tokens[seg_bs_idx][seg_prompt_idx] for seg_bs_idx in range(len(prompt_segs))] for seg_prompt_idx in range(len(prompt_segs[0]))]
            # # seg_embs = [self.llama_model.model.embed_tokens(seg_t).expand(batch_size, -1, -1) for seg_t in seg_tokens] 
            # seg_embs = [[self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_bs_t] for seg_bs_t in seg_tokens_bs_dim_inside] 
            # # im_list = [im_emb.unsqueeze(0) for im_emb in img_embeds]
            # im_list = [img_embeds[:,fi,...] for fi in range(n_frms)]
            # mixed_embeds = [[torch.cat([seg_e_bs, im_e_bs], dim=0) for seg_e_bs, im_e_bs in zip(*pair)] for pair in zip(seg_embs[:-1], im_list)] + [seg_embs[-1]]
            
            # im_list = [img_embeds[:,fi,...] for fi in range(n_frms)]
            seg_embs = [[self.llama_model.model.embed_tokens(seg_t).squeeze(dim=0) for seg_t in seg_bs_t] for seg_bs_t in seg_tokens] 
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

    def forward(self, samples):
        # whole video contains the case that self.multishot is set to True
        #image = samples["image"]
        video = samples['video']
        # N C T H W to N T C H W
        # print(video.shape)
        bs = video.size(0)
        if self.mix_multishot:
            assert bs==1
        n_frms = video.size(2)
        video = torch.permute(video, (0,2,1,3,4))
        # to NxT, C H W
        video = video.reshape((bs*n_frms, ) + video.size()[2:])

        # resize image shape
        img_embeds = self.encode_img(video)
        img_embeds = img_embeds.view((bs, n_frms * img_embeds.size(1), img_embeds.size(2)))
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(video.device)
        
        id_text_mapping = {0: 'first', 1: 'second', 2:'third', 3:'forth', 4:'fifth', 5:'sixth', 
            6:'seventh', 7:'eighth'}
        if self.prompt_list:
            prompt = random.choices(self.prompt_list, k=bs)
            # prompt = [random.choice(self.prompt_list)] * bs
            # if"{shot_idx_text}" in prompt:
            #     assert self.mix_multishot
            # else:
            #     cur_mix_multishot = False
            cur_mix_multishot = ["{shot_idx_text}" in prompt_ for prompt_ in prompt]
            if self.multishot:
                shot_split = samples["shot_split"]
                num_shot = len(shot_split)
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
                    shot_idx_text = id_text_mapping[shot_idx]
                    # prompt_prefix += f'The {id_text_mapping[shot_idx]} shot is '
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
                    del shot_idx
            else:
                prompt_prefix = self.multishot_prompt
                if self.whole_video and not self.fix_total:
                    assert self.whole_video
                    for i in range(n_frms):
                        prompt_prefix += '<Img><ImageHere></Img>'
                    prompt_prefix += '. '
                
            if self.asr_audio:
                if self.whole_video:
                    asr = samples["whole_asr"]
                    words_limit = self.whole_video_asr_words_limit
                    assert len(asr) == 1
                    # asr = asr[0] if isinstance(asr, list) else asr
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
                    num_shot = len(samples['text_input_shots'])
                    shot_idx = random.sample(range(num_shot), 1)[0]
                    shot_idx_text = id_text_mapping[shot_idx]
                    # if cur_mix_multishot:
                    #     prompt = [prompt_.format(asr=asr_, shot_idx_text=shot_idx_text, promp_img=prompt_prefix) for asr_,prompt_ in zip(asr, prompt)]
                    # else:
                    #     prompt = [prompt_.format(asr=asr_, promp_img=prompt_prefix) for asr_ in asr]
                    prompt_comb = []
                    for asr_,prompt_,cur_mix_multishot_ in zip(asr, prompt, cur_mix_multishot):
                        if cur_mix_multishot_:
                            assert self.mix_multishot
                            prompt_comb.append(prompt_.format(asr=asr_, shot_idx_text=shot_idx_text, promp_img=prompt_prefix))
                        else:
                            prompt_comb.append(prompt_.format(asr=asr_, promp_img=prompt_prefix))
                    prompt = prompt_comb
                else:
                    if self.whole_video and not self.fix_total:
                        num_shot = len(samples['text_input_shots'])
                        shot_idx = random.sample(range(num_shot), 1)[0]
                        shot_idx_text = id_text_mapping[shot_idx]
                        prompt_comb = []
                        for asr_,prompt_,cur_mix_multishot_ in zip(asr, prompt, cur_mix_multishot):
                            if cur_mix_multishot_:
                                assert self.mix_multishot
                                prompt_comb.append(prompt_.format(asr=asr_, shot_idx_text=shot_idx_text, promp_img=prompt_prefix))
                            else:
                                prompt_comb.append(prompt_.format(asr=asr_, promp_img=prompt_prefix))
                        prompt = prompt_comb
                    else:
                        if self.whole_video:
                            num_shot = len(samples['text_input_shots'])
                            shot_idx = random.sample(range(num_shot), 1)[0]
                            shot_idx_text = id_text_mapping[shot_idx]
                        prompt_comb = []
                        for asr_,prompt_,cur_mix_multishot_ in zip(asr, prompt, cur_mix_multishot):
                            if cur_mix_multishot_:
                                assert self.mix_multishot
                                prompt_comb.append(prompt_.format(asr=asr_, shot_idx_text=shot_idx_text))
                            else:
                                prompt_comb.append(prompt_.format(asr=asr_))
                        prompt = prompt_comb
            else:
                if self.multishot:
                    prompt = [prompt_.format(promp_img=prompt_prefix) for prompt_ in prompt] * bs
                elif self.whole_video and not self.fix_total:
                    prompt = [prompt_.format(promp_img=prompt_prefix) for prompt_ in prompt] *bs
                # else:
                #     prompt = [prompt] *bs
                
            # print(prompt, bs, n_frms, img_embeds.shape, len(prompt.split('<ImageHere>')))
            img_embeds = img_embeds.view((bs, n_frms, -1, img_embeds.size(2)))
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        if self.visual_target:
            text_target = samples["text_input"]
            if sum(cur_mix_multishot) > 1:
                raise NotImplementedError
                # print(samples["text_input_shots"])
                # print(samples["text_input"])
                text_target = samples["text_input_shots"][shot_idx]
        else:
            text_target = samples["text_audio_input"]
        
        if self.asr_target:
            assert self.audio_target
            text_target = [f"{asr_} {ac_}" for asr_, ac_ in zip(samples['asr'], text_target)]
        
        text = [t + self.end_sym for t in text_target]
        # if self.asr_audio:
        #     asr = [t + self.end_sym for t in samples["asr"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(video.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(video.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        # print('bos embeds', bos_embeds.size())
        # print('img_embeds', img_embeds.size())
        # print('to regress embeds', to_regress_embeds.size())
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}
    
    def generate(self, samples):
        #image = samples["image"]
        video = samples['video']
        # N C T H W to N T C H W
        bs = video.size(0)
        n_frms = video.size(2)
        video = torch.permute(video, (0,2,1,3,4))
        # to NxT, C H W
        video = video.reshape((bs*n_frms, ) + video.size()[2:])

        # resize image shape
        img_embeds = self.encode_img(video)
        img_embeds = img_embeds.view((bs, n_frms * img_embeds.size(1), img_embeds.size(2)))
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(video.device)
        if self.prompt_list:
            prompt = self.prompt_list[0]
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"



        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        # to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        with self.maybe_autocast():
            output_ids = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length = self.max_txt_len
            )
            output_ids = output_ids[inputs_embeds.size(1):]
        output_text = self.llama_tokenizer.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ouput_text = [text.strip() for text in output_text]
        return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        num_frms = cfg.get("num_frms", 8)
        visual_target = cfg.get("visual_target", True)
        audio_target = cfg.get("audio_target", False)
        asr_audio = cfg.get("asr_audio", False)
        whole_video  = cfg.get("whole_video", False)
        multishot  = cfg.get("multishot", False)
        av_target = cfg.get("av_target", False)
        fix_total = cfg.get("fix_total", False)
        mix_multishot = cfg.get("mix_multishot", False)
        asr_target = cfg.get("asr_target", False)
        whole_asr = cfg.get("whole_asr", False)
        
        prompt_order = cfg.get("prompt_order", "vt")
        system_prompt = cfg.get("system_prompt", "")
        answer_prompt = cfg.get("answer_prompt", "")
        multishot_prompt = cfg.get("multishot_prompt", "This is a video with {num_shot} shots. ")
        multishot_secondary_prompt = cfg.get("multishot_secondary_prompt", "The {shot_idx_text} shot is ")
        
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            num_frms=num_frms,
            visual_target=visual_target,
            audio_target=audio_target,
            asr_audio=asr_audio,
            whole_video=whole_video,
            multishot=multishot,
            av_target=av_target,
            prompt_order=prompt_order,
            system_prompt=system_prompt,
            answer_prompt=answer_prompt,
            multishot_prompt=multishot_prompt,
            multishot_secondary_prompt=multishot_secondary_prompt,
            fix_total=fix_total,
            mix_multishot=mix_multishot,
            asr_target=asr_target,
            whole_asr=whole_asr,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
