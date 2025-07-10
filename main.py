
import os
import json
import pdb
import hydra
import logging
from omegaconf import DictConfig

import torch
import statistics
from torch.utils.data import DataLoader
import torch.nn.functional as F
from continuum.metrics import Logger
import random
import numpy as np
from collections import defaultdict

from tqdm import tqdm
from continual_clip import utils
from continual_clip.models import load_model, VisionClassifier
from continual_clip.datasets import build_cl_scenarios
from sklearn.cluster import KMeans
from continuum import rehearsal
import copy
from torchvision import transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def intra_cls(logits, y, classes):
    y = y - classes
    logits1 = logits[:, classes:]
    return F.cross_entropy(logits1, y, reduction='none')

def get_finetuning_dataset(dataset, memory, finetuning='balanced', oversample_old=1, task_id=0):
    if finetuning == 'balanced':
        x, y, t = memory.get()

        if oversample_old > 1:
            old_indexes = np.where(t < task_id)[0]
            assert len(old_indexes) > 0
            new_indexes = np.where(t >= task_id)[0]

            indexes = np.concatenate([
                np.repeat(old_indexes, oversample_old),
                new_indexes
            ])
            x, y, t = x[indexes], y[indexes], t[indexes]

        new_dataset = copy.deepcopy(dataset)
        new_dataset._x = x
        new_dataset._y = y
        new_dataset._t = t
    return new_dataset


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def activation(x):
    return torch.exp(-10*(1-x))



def run_class_incremental(cfg, device):

    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    train_dataset, _ = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    # pdb.set_trace()
    model.classes_names = classes_names
    if cfg.visual_agent:
        if cfg.model_name == "ViT-L/14":
            vision_agent = VisionClassifier(768, cfg.increment, activation=None)
        else:
            vision_agent = VisionClassifier(512, cfg.increment, activation=None)
    

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])

    p1 = 0
    p2 = 0
    negative_records = 0
    trainable_params = {k: v for k, v in model.named_parameters() if v.requires_grad}
    # pdb.set_trace()
    torch.save(trainable_params, f'ori_params.pth')

    if cfg.real_replay:
        memory = rehearsal.RehearsalMemory(
            memory_size=2000,
            herding_method="random"
        )
    for task_id, _ in enumerate(eval_dataset):

        # negative_records = 0

        torch.cuda.empty_cache()
        if task_id == 0:
            targets_bais = 0
        else:
            targets_bais = cfg.initial_increment + (task_id - 1) * cfg.increment
        
        logging.info(f"Evaluation for task {task_id} has started.")
        model.adaptation(task_id, reset=cfg.reset)

        # 将model的参数保存
        trainable_params = {k: v for k, v in  model.named_parameters() if v.requires_grad}
        torch.save(trainable_params, f'trainable_params.pth')

        trainable_params = torch.load(f'ori_params.pth')
        model.load_state_dict(trainable_params, strict=False)

        # 计算未经训练时正类别和负类别的输出平均值
        model.eval()  # 切换到评估模式
        positive_outputs = []
        negative_outputs = []

        val_gap_loader = DataLoader(train_dataset[task_id], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)

        with torch.no_grad():
            for inputs, targets, t in val_gap_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                one_hot_targets = torch.nn.functional.one_hot(targets, outputs.shape[1]).float()
                positive_outputs.append((outputs * one_hot_targets).sum(dim=1).mean())
                mask = 1 - one_hot_targets
                negative_outputs.append(((outputs * mask).sum(dim=1) / mask.sum(dim=1)).mean())
        positive_mean = sum(positive_outputs) / len(positive_outputs)
        negative_mean = sum(negative_outputs) / len(negative_outputs)
        # if  task_id == 0:
        negative_records = negative_mean
        # if task_id == 0:
        logit_size = cfg.increment if task_id>0 else cfg.initial_increment
        bias_logit = torch.full((logit_size,), negative_mean, device=device)
        bias_logit[0] = positive_mean
        # pdb.set_trace()
        # pdb.set_trace()
        logging.info(f"positive_records: {positive_mean}")
        logging.info(f"negative_records: {negative_mean}")
        # pdb.set_trace()
        trainable_params = torch.load(f'trainable_params.pth')
        model.load_state_dict(trainable_params, strict=False)

        model.train()
        if task_id > 0 and cfg.real_replay:
            mem_x, mem_y, mem_t = memory.get()
            t_data = train_dataset[task_id]
            t_data.add_samples(mem_x, mem_y, mem_t)
        else:
            t_data = train_dataset[task_id]
        # train_loader = DataLoader(train_dataset[:task_id+1], batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
        train_loader = DataLoader(t_data, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)

        epochs = cfg.epochs

        if epochs>0:
            # filter out the parameters that require grad
            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(params, lr=cfg.lr) 
            # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=cfg.lr*0.01)   
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
        torch.cuda.empty_cache()
        for i_epoch in range(epochs):

            for bach_i, (inputs, targets, t) in enumerate(train_loader):
                loss_c = torch.tensor(0.0).to(device)
                loss = torch.tensor(0.0).to(device)

                replay_loss = torch.tensor(0.0).to(device)
                torch.cuda.empty_cache()


                # targets = targets - targets_bais
                inputs, targets = inputs.to(device), targets.to(device)

                outputs =  model(inputs)
                # image_f, text_f = model(inputs, return_feature=True)
                if task_id >0:
                    if cfg.real_replay:
                        mask_replay = (targets < targets_bais)
                        old_targets = targets[mask_replay].clone()
                        old_outputs = outputs[mask_replay].clone()
                        targets = targets[~mask_replay]
                        outputs = outputs[~mask_replay]
                        replay_loss = intra_cls(old_outputs, old_targets, 0).mean()*0.1
                    loss_c = intra_cls(outputs,targets,targets_bais).mean() + replay_loss
                    pass
                else:
                    loss_c = torch.nn.functional.cross_entropy(outputs, targets) 
                loss += loss_c
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if bach_i % 10 == 0:
                    logging.info(f"Epoch {i_epoch + 1}/{epochs} | Batch {bach_i + 1}/{len(train_loader)} | Loss: {loss.item()} | Loss_c: {loss_c.item()}")
            scheduler.step()

                

            # torch.cuda.empty_cache()
            # positive_outputs = []
            # negative_outputs = []
            # with torch.no_grad():
            #     model.eval()
            #     for inputs, targets, t in val_gap_loader:
            #         inputs, targets = inputs.to(device), targets.to(device)
            #         outputs = model(inputs)
            #         # pdb.set_trace()
            #         one_hot_targets = torch.nn.functional.one_hot(targets, outputs.shape[1]).float()
            #         positive_outputs.append((outputs * one_hot_targets).sum(dim=1).mean())
            #         mask = 1 - one_hot_targets
            #         negative_outputs.append(((outputs * mask).sum(dim=1) / mask.sum(dim=1)).mean())
            #     model.train()
            # positive_mean = sum(positive_outputs) / len(positive_outputs)
            # negative_mean = sum(negative_outputs) / len(negative_outputs)
            # all_mean = (sum(positive_outputs)+ sum(positive_outputs))/ (len(positive_outputs)+len(negative_outputs))

            # logging.info(f"positive_mean: {positive_mean}")
            # logging.info(f"negative_mean: {negative_mean}")
            # torch.cuda.empty_cache()

        if cfg.real_replay:
            memory.add(*train_dataset[task_id].get_raw_samples(), None)

        if cfg.balance_ft and cfg.real_replay and task_id > 0:
            balance_data = get_finetuning_dataset(t_data, memory, 'balanced')
            balance_loader = DataLoader(balance_data, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
            epochs = cfg.balance_epochs

            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(params, lr=cfg.lr*0.01) 
            # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)  
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=cfg.lr*0.001)   
            for i_epoch in range(epochs):
                for bach_i, (inputs, targets, t) in enumerate(balance_loader):
                    loss_c = torch.tensor(0.0).to(device)
                    loss = torch.tensor(0.0).to(device)

                    replay_loss = torch.tensor(0.0).to(device)
                    torch.cuda.empty_cache()

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs =  model(inputs)
                    # image_f, text_f = model(inputs, return_feature=True)
                    loss_c = torch.nn.functional.cross_entropy(outputs, targets)
                    loss += loss_c
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if bach_i % 10 == 0:
                        logging.info(f"Epoch {i_epoch + 1}/{epochs} | Batch {bach_i + 1}/{len(balance_loader)} | Loss: {loss.item()} | Loss_c: {loss_c.item()}")
                    # break
                scheduler.step()

        # if task_id > 0:
        #     alpha = 0.2  # EMA 
        #     print("EMA")
        #     with torch.no_grad():
        #         for name, param in model.named_parameters():
        #             if param.requires_grad and name in initial_params:
        #                 param.copy_(alpha * initial_params[name] + (1 - alpha) * param)
        if cfg.visual_agent:
            # pdb.set_trace()
            torch.cuda.empty_cache()
            model.eval()
            e_num = cfg.visual_agent_epochs
            vision_agent_loader = DataLoader(train_dataset[task_id], batch_size=cfg.visual_agent_batch_size, shuffle=True, num_workers=cfg.num_workers)
            features_dict = {}
            with torch.no_grad():
                for inputs, targets, t in vision_agent_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    _, features, __ = model(inputs, test=True, return_feature=True)
                    for feature, target in zip(features, targets):
                        target = target.item()
                        if target not in features_dict:
                            features_dict[target] = []
                        features_dict[target].append(feature.cpu())
            mean_features = []
            for target in sorted(features_dict.keys()):
                features = torch.stack(features_dict[target])
                mean_feature = features.mean(dim=0)
                mean_features.append(mean_feature.unsqueeze(0))
            mean_features = torch.cat(mean_features).to(device)
            if task_id > 0:
                vision_agent.add_weight(mean_features)
                pass
            else:
                vision_agent.set_weight(mean_features)
                pass
            optimizer = torch.optim.Adam(vision_agent.parameters(), lr=cfg.visual_agent_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, e_num*len(vision_agent_loader), eta_min=cfg.visual_agent_lr*0.01)
            for e in range(e_num):
                bach_i = -1
                for inputs, targets, t in vision_agent_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    # pdb.set_trace()
                    with torch.no_grad():
                        outputs, _ = model(inputs, return_feature=True)
                    # pdb.set_trace()
                    outputs = vision_agent(outputs)
                    # pdb.set_trace()
                    loss = intra_cls(outputs,targets,targets_bais).mean()
                    # loss = F.cross_entropy(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    bach_i+=1
                    if bach_i % 10 == 0:
                        logging.info(f"Epoch {e + 1}/{e_num} | Batch {bach_i + 1}/{len(vision_agent_loader)} | Loss: {loss.item()}")
                    scheduler.step()
            
            if cfg.balance_ft and cfg.real_replay and task_id > 0:
                balance_data = get_finetuning_dataset(t_data, memory, 'balanced')
                balance_loader = DataLoader(balance_data, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.num_workers)
                epochs = cfg.balance_epochs

                optimizer = torch.optim.Adam(vision_agent.parameters(), lr=cfg.visual_agent_lr*0.1)
                # optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(balance_loader), eta_min=cfg.lr*0.01)
                for i_epoch in range(epochs):   
                    for bach_i, (inputs, targets, t) in enumerate(balance_loader):
                        loss_c = torch.tensor(0.0).to(device)
                        loss = torch.tensor(0.0).to(device)

                        replay_loss = torch.tensor(0.0).to(device)
                        torch.cuda.empty_cache()

                        inputs, targets = inputs.to(device), targets.to(device)
                        with torch.no_grad():
                            outputs, _ = model(inputs, return_feature=True)
                        # pdb.set_trace()
                        outputs = vision_agent(outputs)
                        loss_c = torch.nn.functional.cross_entropy(outputs, targets)
                        loss += loss_c
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        if bach_i % 10 == 0:
                            logging.info(f"Epoch {i_epoch + 1}/{epochs} | Batch {bach_i + 1}/{len(balance_loader)} | Loss: {loss.item()} | Loss_c: {loss_c.item()}")
                        # break
                        scheduler.step()


        if cfg.all_test:
            eval_loader = DataLoader(eval_dataset[:cfg.task_num], batch_size=cfg.batch_size)
        else:
            eval_loader = DataLoader(eval_dataset[:task_id + 1], batch_size=cfg.batch_size)
        # eval_loader = DataLoader(eval_dataset[:10], batch_size=cfg.batch_size)
        image_feature_list = []
        targets_list = []
        model.eval()
        text_feature_list = []
        torch.cuda.empty_cache()
        correct_per_class = defaultdict(int)
        total_per_class = defaultdict(int)
        for inputs, targets, task_ids in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            with torch.no_grad():
                if cfg.visual_agent:
                    a = 1
                    b = 4
                    
                    outputs, image_feature, text_feature  = model(inputs, test=True, all_test=cfg.all_test, return_feature=True)
                    vision_outputs = vision_agent(image_feature)

                    outputs_softmax = F.softmax(outputs, dim=1)
                    vision_outputs_softmax = F.softmax(vision_outputs, dim=1)
                    
                    combined_outputs = (a*outputs_softmax + b*vision_outputs_softmax) / (a + b)
                    
                    metric_logger.add([combined_outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
                    preds = combined_outputs.cpu().argmax(dim=1)
                    for l,p in zip(targets.cpu(), preds):
                        label = l.item()
                        total_per_class[label] += 1
                        if l == p:
                            correct_per_class[label] += 1
                else:
                    outputs = model(inputs, test=True, all_test=cfg.all_test)
                    metric_logger.add([outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test")
        class_acc = {}
        for clas in total_per_class:
            acc = correct_per_class[clas] / total_per_class[clas]
            class_acc[clas] = acc
        avg_acc = np.mean(list(class_acc.values()))
        


        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * metric_logger.accuracy, 2),
                'avg_acc': round(100 * metric_logger.average_incremental_accuracy, 2),
                'forgetting': round(100 * metric_logger.forgetting, 6),
                'acc_per_task': [round(100 * acc_t, 2) for acc_t in metric_logger.accuracy_per_task],
                'bwt': round(100 * metric_logger.backward_transfer, 2),
                'fwt': round(100 * metric_logger.forward_transfer, 2),
            }) + '\n')
            metric_logger.end_task()
    torch.save(model.state_dict(), f'final_model.pth')
    with open(cfg.log_path, 'a+') as f:
        f.write(json.dumps({
            'last': round(acc_list[-1], 2), 
            'avg': round(statistics.mean(acc_list), 2)
        }) + '\n')



def run_domain_incremental(cfg, device):
        
    model = model = load_model(cfg, device)
    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    model.tokenize(classes_names)

    with open(cfg.log_path, 'w+') as f: 
        pass

    logger = Logger(list_subsets=["test"])
    logging.info(f">>> Evaluation scenario length is {len(eval_dataset)}")
    for task_id, _ in enumerate(eval_dataset):

        dataset_val = eval_dataset[:task_id + 1]
        eval_loader = DataLoader(dataset_val, batch_size=cfg.batch_size)
        for input, target, task_ids in tqdm(eval_loader):
            input, target = input.to(device), target.to(device)
            output = torch.from_numpy(model(input))
            logger.add([output.cpu().argmax(dim=1), target.cpu(), task_ids], subset='test')

        with open(cfg.log_path, 'a+') as f:
            f.write(json.dumps({
                'task': task_id,
                'acc': round(100 * logger.accuracy, 2),
            }) + '\n')
            
        logger.end_task()   

def run_task_agnostic():
    pass



@hydra.main(config_path=None, config_name=None, version_base="1.1") 
def continual_clip(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    with open(cfg.log_path, 'w+') as f: 
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.scenario == "class":
        run_class_incremental(cfg, device)

    elif cfg.scenario == "domain":
        run_domain_incremental(cfg, device)

    elif cfg.scenario == "task-agnostic":
        NotImplementedError("Method has not been implemented. Soon be added.")

    else:
        ValueError(f"You have entered `{cfg.scenario}` which is not a defined scenario, " 
                    "please choose from {{'class', 'domain', 'task-agnostic'}}.")



    
        

















if __name__ == "__main__":
    continual_clip()
