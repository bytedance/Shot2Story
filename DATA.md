
We are excited to release a new video-text benchmark for multi-shot video understanding. This release contains a 20k version of our dataset. It includes detailed long summaries for 20k videos and shot captions for 80k video shots.

🌟 **_More Annotations are on the Way!_** 🌟


## Annotation Format <a name="annotation-format"></a>

Our annotations are in JSON format, with each video as a JSON object:

- **video, image_id, nvid:** Video file name.
- **id:** Unique video ID.
- **whole_caption:** Video summary.
- **whole_ASR:** Full-video ASR from [Whisper Large-v2](https://github.com/openai/whisper).
- **video_names:** Array of video shot names.
- **audio_captions:** Array of narration captions per shot.
- **captions:** Array of video captions per shot.
- **ASR:** Array of ASR outputs from [Whisper Large-v2](https://github.com/openai/whisper) per shot.

Example:

```json
[
    {
    "video": "video_name.mp4",
    "image_id": "video_name.mp4",
    "id": 0,
    "whole_caption": "summary",
    "whole_ASR": "ASR output",
    "nvid": "video_name.mp4",
    "video_names": ["shot_name1.mp4", "shot_name2.mp4"],
    "audio_captions": ["narration1", "narration2"],
    "captions": ["caption1", "caption2"],
    "ASR": ["ASR shot1", "ASR shot2"]
    },
    ...
]
```


---

## Videos Downloading <a name="videos-downloading"></a>

We do not supply raw videos. Instead, we provide:

1. **Access Information:** YouTube video ID, chapter ID, and start-end timestamps from HD-VILA-100M are in `./data/annotations/20k_meta.csv`.
2. **Download Scripts:** Use our Python scripts in `./data/scripts/download_videos.py` to download videos. Ensure you have necessary permissions.
3. **Video Preparation:** Use our code in `./data/scripts/process_videos.py` to prepare video clips and single-shot videos.


---

## Opt-Out Approach <a name="opt-out-approach"></a>

We uphold the rights of individuals and copyright holders. If you are featured in any of our video annotations or hold copyright to a video and wish to have its annotation removed from our dataset, please reach out to us. Send an email to hanmingfei@bytedance.com with the subject line beginning with *Shot2Story-optout*, or raise an issue with the same title format. We commit to reviewing your request promptly and taking suitable action.

---

## License <a name="license"></a>

Our text annotations are licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](https://creativecommons.org/licenses/by-nc-sa/4.0/). They are available strictly for non-commercial research.

Please note, our dataset does not include the original videos. Users must refer to [HD-VILA-100M](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md) for video access. By downloading our annotations, you agree to these terms. Respect for video copyright holders is paramount. Ensure your use of the videos aligns with the original source's terms.

---

We extend our thanks to the teams behind [HD-VILA-100M](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md) and [Whisper](https://github.com/openai/whisper). Our work builds upon their valuable contributions. Please acknowledge these resources in your work.