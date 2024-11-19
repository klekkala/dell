import os
import math
import torch
import time
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

import re
from .utils import *
from .model import MYNET
from .sampler import CategoriesSampler
import task_mapper.atari_dataloader as Dataset

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=0, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model

class FSCILTrainer():
    def __init__(self, args):
        super().__init__()

        self.args = args

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        # self.trlog['max_acc'] = [0.0] * args.sessions
        self.trlog['max_acc'] = [0.0]

        self.args.base_class = 5
        self.args.num_classes = 155 
        # self.args.way = 5
        self.args.way = 1
        self.args.shot = 1

        self.args.Dataset = Dataset
        self.args.num_gpu = 1
        self.args.start_session = 0
        self.args.dataroot = '/home/student/deepvp/'
        self.args.gamma = 1

        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        self.set_save_path()
        self.set_up_model()
        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        print(MYNET)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if self.args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader_meta()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader

    def get_base_dataloader_meta(self):
        txt_path = "data/" + "session_" + str(0 + 1) + '.txt'
        
        class_index = np.arange(self.args.base_class)
        trainset = self.args.Dataset.Atari_Dataset(root=self.args.dataroot, train=True, index_path=txt_path)
        testset = self.args.Dataset.Atari_Dataset(root=self.args.dataroot, train=False, index=class_index)

        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way,
                                    self.args.episode_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                  pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        print('get_new_dataloader', session)
        # txt_path = "data/" + "session_" + str(session + 1) + '.txt'
        txt_path = "data/" + "session_" + str(session) + '.txt'

        trainset = self.args.Dataset.Atari_Dataset(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
            #                                           num_workers=8, pin_memory=True)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=0, pin_memory=True)
        else:
            # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
            #                                           shuffle=True,
            #                                           num_workers=8, pin_memory=True)
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=0, pin_memory=True)

        class_new = self.get_session_classes(session-1)

        testset = self.args.Dataset.Atari_Dataset(root=self.args.dataroot, train=False,
                                                      index=class_new)


        # testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
        #                                          num_workers=8, pin_memory=True)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)

        return trainset, trainloader, testloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):
        for i in range(self.args.low_way):
            # random choose rotate degree
            rot_list = [90, 180, 270]
            sel_rot = random.choice(rot_list)
            if sel_rot == 90:  # rotate 90 degree
                # print('rotate 90 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
            elif sel_rot == 180:  # rotate 180 degree
                # print('rotate 180 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
            elif sel_rot == 270:  # rotate 270 degree
                # print('rotate 270 degree')
                proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
        return proto_tmp, query_tmp

    def get_optimizer_base(self):

        # optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
        #                              {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
        #                             momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        optimizer = torch.optim.SGD([{'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def train(self, session):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        # for session in range(args.start_session, args.sessions):

        train_set, trainloader, testloader = self.get_dataloader(session)
            
        self.model = self.update_param(self.model, self.best_model_dict)

        if session == 0:  # load base class train img label

            print('new classes for this session:\n', np.unique(train_set.targets))
            optimizer, scheduler = self.get_optimizer_base()

            for epoch in range(args.epochs_base):
                start_time = time.time()
                # train base session
                self.model.eval()
                tl, ta = self.base_train(self.model, trainloader, optimizer, scheduler, epoch, args)

                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)

                self.model.module.mode = 'avg_cos'

                if args.set_no_val: # set no validation
                    save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                    torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    tsl, tsa = self.test(self.model, testloader, args, session)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                        epoch, lrc, tl, ta, tsl, tsa))
                    result_list.append(
                        'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                else:
                    # take the last session's testloader for validation
                    vl, va = self.validation()

                    # save better model
                    if (va * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))
                    self.trlog['val_loss'].append(vl)
                    self.trlog['val_acc'].append(va)
                    lrc = scheduler.get_last_lr()[0]
                    print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                        epoch, lrc, tl, ta, vl, va))
                    result_list.append(
                        'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))

                self.trlog['train_loss'].append(tl)
                self.trlog['train_acc'].append(ta)

                print('This epoch takes %d seconds' % (time.time() - start_time),
                        '\nstill need around %.2f mins to finish' % (
                                (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                scheduler.step()

            # always replace fc with avg mean
            self.model.load_state_dict(self.best_model_dict)
            self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
            best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            torch.save(dict(params=self.model.state_dict()), best_model_dir)

            self.model.module.mode = 'avg_cos'
            tsl, tsa = self.test(self.model, testloader, args, session)
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

            result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

        else:  # incremental learning sessions

            print("training session: [%d]" % session)
            self.model.load_state_dict(self.best_model_dict)

            self.model.module.mode = self.args.new_mode
            self.model.eval()
            trainloader.dataset.transform = testloader.dataset.transform
            self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

            tsl, tsa = self.test(self.model, testloader, args, session-1)

            # save better model
            if len(self.trlog['max_acc']) <= session:
                # Extend the list to accommodate the current session index, filling with 0.0
                self.trlog['max_acc'].extend([0.0] * (session + 1 - len(self.trlog['max_acc'])))
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))

            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('Saving model to :%s' % save_model_dir)
            print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

            result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))

        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, 'results_new.txt'), result_list)

    def get_cur_session_num(self, data_root):
        session_files = [f for f in os.listdir(data_root) if f.startswith("session_") and f.endswith(".txt")]

        session_numbers = []
        for session_file in session_files:
            match = re.match(r'session_(\d+)\.txt', session_file)
            if match:
                session_numbers.append(int(match.group(1)))

        if session_numbers:
            max_number = max(session_numbers)
        else:
            max_number = 0
        
        return max_number
    
    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, testloader, self.args, session)

        return vl, va

    def base_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        scaler = GradScaler()

        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)

        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        for i, batch in enumerate(tqdm_gen, 1):
            data, true_label = [_.cuda() for _ in batch]

            k = args.episode_way * args.episode_shot
            proto, query = data[:k], data[k:]
            # sample low_way data
            proto_tmp = deepcopy(
                proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
                :args.low_shot,
                :args.low_way, :, :, :].flatten(0, 1))
            query_tmp = deepcopy(
                query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[:,
                :args.low_way, :, :, :].flatten(0, 1))
            # random choose rotate degree
            proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)

            model.module.mode = 'encoder'

            with autocast():
                data = model(data)
                proto_tmp = model(proto_tmp)
                query_tmp = model(query_tmp)

                # k = args.episode_way * args.episode_shot
                proto, query = data[:k], data[k:]

                # actual_size = proto.shape[0]
                # print(proto.shape)

                # if actual_size != k:
                #     raise ValueError(f"Expected batch size {k}, but got {actual_size}")

                proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
                query = query.view(args.episode_query, args.episode_way, query.shape[-1])

                proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
                query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])

                proto = proto.mean(0).unsqueeze(0)
                proto_tmp = proto_tmp.mean(0).unsqueeze(0)

                proto = torch.cat([proto, proto_tmp], dim=1)
                query = torch.cat([query, query_tmp], dim=1)

                proto = proto.unsqueeze(0)
                query = query.unsqueeze(0)

                logits = model.module._forward(proto, query)

                total_loss = F.cross_entropy(logits, label)

                acc = count_acc(logits, label)

            # Scale the loss and backpropagate using GradScaler
            scaler.scale(total_loss).backward()

            # Update the weights
            scaler.step(optimizer)
            scaler.update()

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()
            scheduler.step()

            
        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def test(self, model, testloader, args, session):
        test_class = args.base_class + session * args.way
        print('test_class', test_class)
        model = model.eval()
        vl = Averager()
        va = Averager()
        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                print(data.shape, test_label)

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                proto = proto.unsqueeze(0).unsqueeze(0)

                logits = model.module._forward(proto, query)
                print('logits', logits)

                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()

        return vl, va
    
    def eval(self, obs):
        args = self.args
        data_root = '/home/student/dell/data'
        cur_session_num = self.get_cur_session_num(data_root)
        test_class = args.base_class + (cur_session_num - 1) * args.way
        model = self.model.eval()

        with torch.no_grad():  # Disable gradient computation for testing
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float() 
            # Move the input observation to the device
            data = obs.cuda().unsqueeze(0)  # Add batch dimension if it's a single image
            
            # Run the model in encoder mode to get the query embeddings
            model.module.mode = 'encoder'
            query = model(data)  # Get embedding for the input observation
            query = query.unsqueeze(0).unsqueeze(0)  # Reshape to match expected dimensions

            # Get the prototype vectors (pre-computed class embeddings) for the current session
            proto = model.module.fc.weight[:test_class, :].detach()
            proto = proto.unsqueeze(0).unsqueeze(0)

            # Compute the logits (cosine similarity between query and prototypes)
            logits = model.module._forward(proto, query)

            # Get the predicted class (argmax of the logits) and the associated confidence
            predicted_class = torch.argmax(logits, dim=-1)
            confidence = torch.softmax(logits, dim=-1)
            predicted_confidence = confidence[torch.arange(logits.size(0)), predicted_class]
            print('confidence', confidence)

        return predicted_confidence.item(), predicted_class.item(), test_class, cur_session_num + 1

    def set_save_path(self):
        self.args.save_path = '/home/student/dell/checkpoint/atari/cec'
        ensure_path(self.args.save_path)
        return None
