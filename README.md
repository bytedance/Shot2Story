# Shot2Story: A New Benchmark for Comprehensive Understanding of Multi-shot Videos

<a href='https://mingfei.info/shot2story/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://github.com/bytedance/Shot2Story/blob/master/DATA.md'><img src='https://img.shields.io/badge/Github-Data-red'></a>
<a href='https://huggingface.co/spaces/mhan/Shot2Story'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://arxiv.org/abs/2312.10300'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
<a href='https://github.com/bytedance/Shot2Story/tree/master/code'><img src='https://img.shields.io/badge/Github-Code-green'></a>

We are excited to release a new video-text benchmark and extendable codes for multi-shot video understanding. Our 20k version of dataset includes detailed long summaries for 20k videos and shot captions for 80k video shots.

**Stay tuned for more exciting data release and new features!**

---

## What's new 👀 <a name="news"></a>

<!-- This section includes any recent updates or changes to the dataset. It may also include information about related events or projects, such as challenges or competitions using the dataset. This section is frequently updated, so please check back often for the latest news. -->
🌟 Update (16/12/2023): [Paper](https://arxiv.org/pdf/2312.10300) and [Demo](https://huggingface.co/spaces/mhan/Shot2Story) for SUM-shot model. It showcases the power and versatility of detailed and grounded video summaries. Dive into the demo and share your experiences with us! **Chat-SUM-shot** is on the way! Stay tuned!🎥📝🚀

🌟 Update (12/12/2023): [Code](https://github.com/bytedance/Shot2Story/tree/master/code) for video summarization and shot captioning, in the sub-directory `code` of this repo. Dive into these new features and share your experiences with us! 🎥📝🚀

🌟 Update (30/11/2023): [Data](https://github.com/bytedance/Shot2Story/blob/master/DATA.md) of Shot2Story-20K. Check them out and stay tuned for more exciting updates! 💫🚀

---

## Demo <a name="demo"></a>

We build a demo for SUM-shot model hosted in [Space](https://huggingface.co/spaces/mhan/Shot2Story). Please have a look and explore what it is capable of. Issues are welcomed! **Chat-SUM-shot model is on the way!**

Some hints to play with our demo: 

*   🎉 Start with our provided **demo videos**, some of which are sampled from ActivityNet, not included in our training data.
*   🚀 Please upload videos **less than 20MB**. Enjoy!
*   😄 For a more comprehensive understanding, try specifying reasonable starting and ending timestamps for the shots. Enjoy!
*   😄 Setting temperature to 0.1 for the most grounded understanding and question-answering.
*   😄 Setting temperature to greater value for the creative grounded understanding and question-answering.


https://github.com/bytedance/Shot2Story/assets/18671115/cb53dc72-3d1f-48a7-ba07-a41573311ac9


---

## Table of Contents

1. [🌟 What's new 👀](#news)
2. [Demo](#demo)
3. [Introduction](#introduction)
4. [Dataset Glance](#dataset-glance)
5. [Baselines and Tasks](#baselines)
6. [License](#license)
7. [Citation](#citation)
8. [Contact](#contact)

---

## Introduction <a name="introduction"></a>

<!-- This section provides a brief overview of the dataset, its purpose, and its potential applications. It also includes a brief history of the dataset's creation and any changes or updates that have been made over time. -->
A short clip of video may contain progression of multiple events and an interesting story line. A human needs to capture both the event in every shot and associate them together to understand the story behind it. In this work, we present a new multi-shot video understanding benchmark Shot2Story with detailed shot-level captions and comprehensive video summaries. To facilitate better semantic understanding of videos, we provide captions for both visual signals and human narrations. We design several distinct tasks including single-shot video and narration captioning, multi-shot video summarization, and video retrieval with shot descriptions. Preliminary experiments show some challenges to generate a long and comprehensive video summary.

<p align="center"> <br> <img src="assets/S2S_demo.png" alt="Dataset Glance"/> <br> </p>

---


## Dataset Glance <a name="dataset-glance"></a>

Our dataset comprises 20k video clips sourced from HD-VILA-100M. Each clip is meticulously annotated with single-shot video captions, narration captions, video summaries, extracted ASR texts, and shot transitions. Please refer to [DATA.md](./DATA.md) for video and annotation preparation.

The dataset includes an average of 4.0 shots per video, resulting in a total of 80k video shots, each with detailed video caption and narration caption annotations. The average length of our video summaries is 201.8, while the average length of a video is 16s.

For more comprehensive details, please refer to the plots below.
<p align="center">
    <br>
    <img src="assets/dataset_stats.png"/>
    <br>
<p>

---

## Baselines and Tasks <a name="baselines"></a>

To benchmark the advances of multi-modal video understanding, we designed several distinctive tasks using our dataset, including single-shot captioning, multi-shot summarization, and video retrieval with shot description. We design and implemented several baseline models using a frozen vision encoder and an LLM, by prompting the LLM with frame tokens and ASR (Automatic Speech Recognition) text. 

[Code](./code/README.md) here for running the project.

<p align="center">
    <br>
    <img src="assets/code_demo.png"/>
    <br>
<p>


---

## License <a name="license"></a>

Our code is licensed under a [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt). 

Our text annotations are released under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). They are available strictly for non-commercial research. More guidelines of dataset can be found in [here](./DATA.md#license).


---

## Citation <a name="citation"></a>

If you find this repo useful for your research, please consider citing the paper
```
@article{han2023shot2story20k,
      title={Shot2Story20K: A New Benchmark for Comprehensive Understanding of Multi-shot Videos}, 
      author={Mingfei Han and Linjie Yang and Xiaojun Chang and Heng Wang},
      journal={arXiv preprint arXiv:2311.17043},
      year={2023}
}
```

<!-- ## Acknowledgements <a name="acknowledgements"></a> -->


---

## Contact <a name="contact"></a>

If you have any questions or concerns about our dataset, please don't hesitate to contact us. You can raise an issue or reach us at hanmingfei@bytedance.com. We welcome feedback and are always looking to improve our dataset.

---

<!-- This README is intended to be a comprehensive guide to the dataset, but it may not cover every detail or use case. Users are encouraged to contact the dataset's creators or maintainers with any questions or concerns. -->


We extend our thanks to the teams behind [HD-VILA-100M](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md), [BLIP2](https://github.com/salesforce/LAVIS), [Whisper](https://github.com/openai/whisper), [MiniGPT-4](https://minigpt-4.github.io/), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) and [LLaMA](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/). Our work builds upon their valuable contributions. Please acknowledge these resources in your work.
