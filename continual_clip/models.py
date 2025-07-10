

import pdb
from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn
import types
from loraclip import lora_clip
from clip.model import VisionTransformer as CLIPVisionTransformer
from torch.nn import functional as F
from .utils import get_class_ids_per_task, get_class_names
import random
import numpy as np





def forward_clip(self, image, text, return_feature=False):
    image_features = self.encode_image(image)
    text_features = self.encode_text(text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    if return_feature:
        return logits_per_image, logits_per_text, image_features, text_features


    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text



class VisionClassifier(nn.Module):
    def __init__(self, in_features, num_classes, weight_init=None, activation=None):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes, bias=False)
        self.fc = nn.Parameter(self.fc.weight.data)
        if weight_init is not None:
            self.fc.data = weight_init
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.Identity()
    
    def add_weight(self, weight):
        self.fc = nn.Parameter(torch.cat([self.fc, weight], dim=0))

    def set_weight(self, weight):
        self.fc = nn.Parameter(weight)


    def forward(self, x):
        # normalize the weights
        x = F.normalize(x, p=2, dim=-1)
        weight = F.normalize(self.fc, p=2, dim=-1)
        x = F.linear(x, weight)
        x = self.activation(x)
        return x

        

class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        # self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)


        #lora_clip
        self.model, self.transforms = lora_clip.load(cfg.model_name, device=device, jit=jit, r=cfg.lora_rank, lora_mode=cfg.lora_mode)
        # for name, param in self.model.named_parameters():
        #     if 'adapter_mlp' in name:
        #         param.requires_grad = True
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable: {name}")
        self.model.forward = types.MethodType(forward_clip, self.model)
        ori_state = self.model.state_dict()

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.current_task = -1
        self.only_reset_B = cfg.only_reset_B
        self.freeze_A = cfg.freeze_A
    

    def cur_text_features(self):
        f = self.model.encode_text(self.text_tokens)
        f = f / f.norm(dim=1, keepdim=True)
        return f

    def inference(self, image, text_tokens):
        text_features = self.model.encode_text(text_tokens)
        image_features = self.model.visual(image.type(self.model.dtype), all_tokens=False, adapt=self.attention_adapter)
        # pdb.set_trace()

        # image_features = self.attention_adapter(image_features.type(torch.float32))[:, 0, :]

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def forward(self, image, test=False, all_test=False, return_feature=False,replay=None):
        if test:
            # pdb.set_trace()
            with torch.no_grad():
                if all_test:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.all_text_tokens, return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.all_text_tokens)
                    # logits_per_image = self.inference(image, self.all_text_tokens)
                else:
                    if return_feature:
                        logits_per_image, _, image_features, __ = self.model(image, self.text_tokens, return_feature=return_feature)
                    else:
                        logits_per_image, _ = self.model(image, self.text_tokens)
                # pdb.set_trace()
                probs = logits_per_image.softmax(dim=-1)
        else:

            if return_feature:
                __, _, image_features, text_features = self.model(image, self.text_tokens, return_feature=return_feature)
                return image_features, text_features
            if replay is not None:
                logits_per_image, _ = self.model(image, self.text_tokens)
                # text_features_for_replay = self.model.encode_text(self.text_tokens[:-self.cfg.increment])
                text_features_for_replay = self.model.encode_text(self.text_tokens)
                text_features_for_replay = text_features_for_replay / text_features_for_replay.norm(dim=1, keepdim=True)
                replay_features = replay / replay.norm(dim=1, keepdim=True)
                replay_logits = replay_features @ text_features_for_replay.t() * 100
            else:
                logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image
                
        if return_feature:
            text_features = self.model.encode_text(self.all_text_tokens)
            return probs, image_features, text_features

        if replay is not None:
            return probs, replay_logits
        return probs

    def adaptation(self, task_id, reset=False):
        self.current_task +=1
        if reset and self.current_task>0:
            ori_state = torch.load('ori_state.pth')
            if self.only_reset_B:
                now_state = self.model.state_dict()
                lora_params = {k: v for k, v in ori_state.items() if 'lora_B' in k}
                now_state.update(lora_params)
            else:
                now_state = ori_state
            self.model.load_state_dict(now_state)
        if self.freeze_A and self.current_task>0:
            for name, param in self.model.named_parameters():
                if 'lora_A' in name:
                    param.requires_grad = False
            
        self.current_task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += self.current_task_class_names
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self.current_task_text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_task_class_names]
        ).to(self.device)
        if self.current_task == 0:
            class_names = []
            for i in range(self.cfg.task_num):
                class_names += get_class_names(self.classes_names, self.class_ids_per_task[i])
            self.all_class_names = class_names
            self.all_text_tokens = clip.tokenize(
                [self.prompt_template.format(c) for c in self.all_class_names]
            ).to(self.device)





def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncrementalCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    
