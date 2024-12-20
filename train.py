import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from models.unlearning_model import SiamUL, LossKVKC, RandomResponse
from utils import load_model, save_model, accuracy, evaluate, freeze_bnlayer
        
class Trainer():
    def __init__(self, model, datal_train, datal_vali, 
                 optimizer, train_scheduler, warmup_scheduler,
                 criterion, warm_epoches,
                 device, maxnorm, writer_path, path=None):
        
        self.datal_train = datal_train
        self.datal_vali = datal_vali
        self.model = model
        
        self.optimizer = optimizer
        self.train_scheduler = train_scheduler
        self.warmup_scheduler = warmup_scheduler

        self.criterion = criterion
        self.warm_epoches = warm_epoches
        
        self.device = device
        self.maxnorm = maxnorm
        
        self.writer = SummaryWriter(writer_path)
        self.path = path
        
    def train(self, epoches, current_epoch=0, save_model=True, 
              is_vali=True, is_display=True, load_model=True):
        if load_model:
            current_epoch = self.__load()

        self.model.to(self.device)
        
        n_iter =  0
        best_acc = 0.0
        for epoch in range(current_epoch, epoches):
            
            epoch_start = time.time()
            # Training
            re = self.train_one_epoch(n_iter, epoch)

            if epoch > self.warm_epoches:
                self.train_scheduler.step(epoch)
                
            acc = self.validate()

            epoch_end = time.time()
            time_for_one_epoch = epoch_end - epoch_start
            
            if save_model and ((epoch == 0) or (acc > best_acc)):
                best_acc = acc
                self.__save(epoch)
                
            if is_display:
                print('\n \n Time for one epoch',':', time_for_one_epoch)
                batches_start = time.time()
                print('[%d:%d]\t Losses:%.4f\t'% (epoch, epoches, re.item()))
                print('ACC>>>>>:', acc)
   
        
    def train_one_epoch(self, n_iter, epoch):
        self.model.train()
        for i_batch, (x, labs) in enumerate(self.datal_train):
            x = x.to(self.device)
            labs = labs.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.__compute_loss(out, x,labs)
                
            loss.backward()
            
            if self.maxnorm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.maxnorm)
            self.optimizer.step()
            
            self.writer.add_scalar('train/losses', loss.item(), n_iter)
            n_iter += 1

            if epoch <= self.warm_epoches:
                self.warmup_scheduler.step()
            
        return loss
            
    
    def validate(self):
        acc = 0
        with torch.no_grad():
            self.model.eval()

            for x, labs in self.datal_vali:
                x = x.to(self.device) 
                labs = labs.to(self.device)
                
                out = self.model(x)
                acc += accuracy(out, labs)
                
        return acc / len(self.datal_vali)
                
                
    def __compute_loss(self, out, x, labs):
        return self.criterion(out, labs)
                
    def __load(self):
        return load_model(self.model, self.optimizer, self.train_scheduler, self.path)
       
    def __save(self, epoch):
        save_model(epoch, self.model, self.optimizer, self.train_scheduler, self.path)  
        

def my_unlearn(backbone_model, 
               datal_forget, datal_few, testl_remain, testl_forget,
               ratioes:list, num_classes:int,
               lam:float, device, total_epoches:int, learning_rate:float, few_shot=False,
               is_shift=True, concentrate_iters=50, path_to_backup=None, skip=False):
    # model
    model = SiamUL(backbone_model, num_classes)
    
    # criterion
    criterion_KV = LossKVKC(lam)
    criterion_KC = LossKVKC(lam, forget=False)
    

    RR = RandomResponse(ratioes, num_classes)
    
    # optim 
    optimizer = torch.optim.SGD(model.parameters(), 
                                learning_rate,
                                momentum=0.9,
                                weight_decay=1e-4)
    # Train
    model.train()
    model.to(device)
    freeze_bnlayer(model)

    c_iter = 0

    remain_accs = []
    forget_accs = []
    while few_shot and concentrate_iters > c_iter:
        x, y = next(iter(datal_few))
        y = y.to(device)
        x1 = x[0].to(device)
        x2 = x[1].to(device)
                
        p1, p2, z1, z2, zd1, zd2 = model(x1, x2)
        loss, t_1, t_2 = criterion_KC(p1, p2, z1, z2, 
                                       zd1, zd2, y, y)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Concenstrate (%d:%d)\t Remain Losses:%.4f\t, Similarity:%.4f\t, crossEntr:%.4f\t'% (c_iter, concentrate_iters, loss.item(), t_1.item(),t_2.item()))
        c_iter += 1

    # print('==========================================')
    if path_to_backup is not None:
        with open('{0}.pt'.format(path_to_backup), 'wb') as f:
            torch.save(
                            {
                            'model_states': backbone_model.state_dict(), 
                            # 'optimizer_states': optimizer.state_dict(),
                            }, f
                    ) 
    
    for epoch in tqdm(range(total_epoches)):
        epoch_start = time.time()
        remain_accs.append(
            evaluate(backbone_model, testl_remain, device).item()
        )
        forget_accs.append(
            evaluate(backbone_model, testl_forget, device).item()
        )
        for x, y in datal_forget:
            if skip:
                print("[!!!] skip knowledge vor")
            else:
                y_hat_1 = RR(y)
                y_hat_2 = RR(y)
                
                x1 = x[0].to(device)
                x2 = x[1].to(device)
                y_hat_1 = y_hat_1.to(device)
                y_hat_2 = y_hat_2.to(device)
                
                p1, p2, z1, z2, zd1, zd2 = model(x1, x2)
                loss, t_1, t_2 = criterion_KV(p1, p2, z1, z2, 
                                zd1, zd2, y_hat_1, y_hat_2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print('[%d:%d]\t Forget Losses:%.4f\t, Similarity:%.4f\t, crossEntr:%.4f\t'% (epoch, total_epoches, loss.item(), t_1.item(),t_2.item()))
            
            if few_shot:
                x, y = next(iter(datal_few))
                y = y.to(device)
                x1 = x[0].to(device)
                x2 = x[1].to(device)
                
                p1, p2, z1, z2, zd1, zd2 = model(x1, x2)
                loss, t_1, t_2 = criterion_KC(p1, p2, z1, z2, 
                                 zd1, zd2, y, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        
        if path_to_backup is not None:
            with open('{0}.pt'.format(path_to_backup), 'wb') as f:
                torch.save(
                            {
                            'model_states': backbone_model.state_dict(), 
                            }, f
                    )
    return remain_accs, forget_accs

    