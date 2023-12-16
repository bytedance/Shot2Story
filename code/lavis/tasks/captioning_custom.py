"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict
try:
    from nlgeval import NLGEval
except Exception:
    print("nlgeval not installed")
from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import torch.distributed as dist

@registry.register_task("captioning_custom")
class CaptionCustomTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": int(img_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
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
        
        all_ids = [res['image_id'] for res in eval_result]
        print(len(all_ids))
        gt_path = '/export/home/.cache/lavis/msrvtt/annotations/cap_{}_merged.json'.format(split_name)
        annos = json.load(open(gt_path))
        if len(annos) > len(eval_result):
            annos = annos[:len(eval_result)]
        all_gts = OrderedDict()
        for item in annos:
                all_gts[item['image_id']] = item['caption']

        all_caption_lists = [v for k,v in all_gts.items()]
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


