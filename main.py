from data_utils import ZOLDatesetReader
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score,precision_score
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import math
from DIST import DIST,BertConfig
from logs import logger
import numpy as np
import ot
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

#寄
class OptimalTransportLoss(nn.Module):
    def __init__(self, reg=0.1):
        super().__init__()
        self.reg = reg

    def forward(self, preds, targets):
        # 将标签转换为one-hot编码形式
        targets_one_hot = F.one_hot(targets, num_classes=preds.size(1)).float()
        preds = F.softmax(preds, dim=1)
        # 计算成本矩阵
        C = torch.cdist(preds, targets_one_hot, p=2)
        # 使用POT库计算最优传输计划
        ot_plan = ot.emd([], [], C.cpu().detach().numpy(), self.reg)
        ot_plan = torch.from_numpy(ot_plan).to(preds.device)
        # 计算最优传输损失
        loss = torch.sum(ot_plan * C)
        return loss

class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, lr_max, warmup_steps, last_epoch=-1):
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # 线性预热
            lr = self.lr_max * (self.last_epoch / self.warmup_steps)
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr_max * (0.5 * (1.0 + math.cos(math.pi * progress)))
        return [lr for _ in self.optimizer.param_groups]

if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='DIST')
    parser.add_argument('--dataset', default='zol_cellphone', type=str, help='restaurant, laptop, zol_cellphone')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_epoch', default=3, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--eval_step', default=100, type=int,help='when pass every x step in train ,run eval function')
    parser.add_argument('--config_file', default='./pretrains/bert_config.json', type=str)
    parser.add_argument('--config_file_cat', default='./pretrains/bert_config_cat.json', type=str)

    parser.add_argument('--max_seq_len', default=128, type=int)
    parser.add_argument('--max_aspect_len', default=16, type=int)
    parser.add_argument('--max_img_len', default=36, type=int)

    parser.add_argument('--device', default=0, type=str)

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_dataset = ZOLDatesetReader(dataset=opt.dataset,
                                  max_seq_len=opt.max_seq_len,
                                  max_aspect_len=opt.max_aspect_len,
                                  max_img_len=opt.max_img_len,)

    train_data_loader = DataLoader(dataset=my_dataset.train_data, batch_size=opt.batch_size,
                                        shuffle=True)  # 数据读取
    dev_data_loader = DataLoader(dataset=my_dataset.dev_data, batch_size=opt.batch_size,
                                      shuffle=False)
    test_data_loader = DataLoader(dataset=my_dataset.test_data, batch_size=opt.batch_size,
                                       shuffle=False)

    bert_config = BertConfig.from_json_file(opt.config_file)
    bert_config_cat = BertConfig.from_json_file(opt.config_file_cat)
    model = DIST(pretrain='./pretrains',config=bert_config,config_cat=bert_config_cat)
    model.to(opt.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params}')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            0.01
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=opt.learning_rate, eps=1e-6)  # 确保这里设置了初始学习率 优化器
    num_train_steps = int(len(train_data_loader) * opt.num_epoch)
    total_steps = num_train_steps
    warmup_steps = int(total_steps * 0.1)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, total_steps, opt.learning_rate,
                                              warmup_steps)  # 10%线性 90%余弦退火到0
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = SamplesLoss(loss="sinkhorn", p=2, blur=0.3, scaling=0.5)
    train_step = 0
    max_acc = 0
    best_f1 = 0
    best_acc = 0


    def train(model, train_data_loader, dev_data_loader, test_data_loader):
        global train_step
        model.train()
        for epoch in range(opt.num_epoch):
            with tqdm(total=len(train_data_loader.dataset), desc='Training', unit='sample') as pbar:
                for batch in train_data_loader:
                    text_raw_indices = batch['text_raw_indices'].to(opt.device)
                    text_raw_indices_token = batch['text_raw_indices_token'].to(opt.device)
                    text_raw_indices_mask = batch['text_raw_indices_mask'].to(opt.device)
                    aspect_indices = batch['aspect_indices'].to(opt.device)
                    aspect_indices_token = batch['aspect_indices_token'].to(opt.device)
                    aspect_indices_mask = batch['aspect_indices_mask'].to(opt.device)
                    imgs = batch['imgs'].to(opt.device)
                    img_mask = batch['img_mask'].to(opt.device)
                    labels = batch['polarity'].to(opt.device)
                    optimizer.zero_grad()

                    with autocast():
                        outputs = model(sentence_ids=text_raw_indices, sentence_mask=text_raw_indices_mask,
                                        target_ids=aspect_indices, target_mask=aspect_indices_mask,
                                        sentence_token_type_ids=text_raw_indices_token,
                                        target_token_type_ids=aspect_indices_token,
                                        img_feature=imgs,
                                        img_mask=img_mask)
                        loss = criterion(outputs, labels)

                    # 使用 GradScaler 进行反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    scheduler.step()
                    # 显示loss
                    pbar.set_postfix({'loss': loss.item()})
                    pbar.update(text_raw_indices.size(0))
                    # 显示学习率
                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({'loss': loss.item(), 'lr': lr})
                    # 每100步执行一次验证
                    train_step = train_step + 1
                    if train_step % opt.eval_step == 0:
                        validate(model, dev_data_loader, test_data_loader)
        validate(model, dev_data_loader, test_data_loader)


    def validate(model, dev_data_loader, test_data_loader):
        global max_acc
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(dev_data_loader.dataset), desc='Validating', unit='sample', position=0) as pbar:
                for batch in dev_data_loader:
                    text_raw_indices = batch['text_raw_indices'].to(opt.device)
                    text_raw_indices_mask = batch['text_raw_indices_mask'].to(opt.device)
                    aspect_indices = batch['aspect_indices'].to(opt.device)
                    aspect_indices_mask = batch['aspect_indices_mask'].to(opt.device)
                    imgs = batch['imgs'].to(opt.device)
                    img_mask = batch['img_mask'].to(opt.device)
                    labels = batch['polarity'].to(opt.device)
                    aspect_indices_token = batch['aspect_indices_token'].to(opt.device)
                    text_raw_indices_token = batch['text_raw_indices_token'].to(opt.device)
                    with autocast():
                        outputs = model(sentence_ids=text_raw_indices, sentence_mask=text_raw_indices_mask,
                                        target_ids=aspect_indices, target_mask=aspect_indices_mask,
                                        sentence_token_type_ids=text_raw_indices_token,
                                        target_token_type_ids=aspect_indices_token,
                                        img_feature=imgs,
                                        img_mask=img_mask)
                        loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_loss += loss.item()
                    pbar.set_postfix({'val_loss': loss.item()})
                    pbar.update(text_raw_indices.size(0))
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)
        logger.info(f'Validation Loss: {total_loss / len(dev_data_loader)}')
        logger.info(f'Validation F1 Score: {f1}')
        logger.info(f'Validation Accuracy: {val_acc}')
        logger.info(f'Validation precision_score: {precision}')
        if val_acc >= max_acc:
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), "./pytorch_model.pth")
            test(model, test_data_loader)
            max_acc = val_acc


    def test(model, test_data_loader):
        global best_f1
        global best_acc
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(test_data_loader.dataset), desc='Testing', unit='sample', position=0) as pbar:
                for batch in test_data_loader:
                    text_raw_indices = batch['text_raw_indices'].to(opt.device)
                    text_raw_indices_mask = batch['text_raw_indices_mask'].to(opt.device)
                    aspect_indices = batch['aspect_indices'].to(opt.device)
                    aspect_indices_mask = batch['aspect_indices_mask'].to(opt.device)
                    imgs = batch['imgs'].to(opt.device)
                    img_mask = batch['img_mask'].to(opt.device)
                    labels = batch['polarity'].to(opt.device)
                    aspect_indices_token = batch['aspect_indices_token'].to(opt.device)
                    text_raw_indices_token = batch['text_raw_indices_token'].to(opt.device)
                    with autocast():
                        outputs = model(sentence_ids=text_raw_indices, sentence_mask=text_raw_indices_mask,
                                        target_ids=aspect_indices, target_mask=aspect_indices_mask,
                                        img_feature=imgs,
                                        sentence_token_type_ids=text_raw_indices_token,
                                        target_token_type_ids=aspect_indices_token,
                                        img_mask=img_mask)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    pbar.update(text_raw_indices.size(0))
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        if (best_acc < acc):
            best_f1 = f1
            best_acc = acc

        logger.info(f'Test F1 Score: {f1}')
        logger.info(f'Test Accuracy: {acc}')
        logger.info(f'Test precision_score: {precision}')


    # 开始训练
    train(model, train_data_loader, dev_data_loader,test_data_loader)
    # 完成训练后执行测试
    test(model, test_data_loader)

