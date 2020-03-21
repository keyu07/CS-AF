import torch
from torch.autograd import Variable
import numpy as np
import time
import argparse
from sklearn.metrics import accuracy_score
from get_data import train_loader, val_loader, test_loader
import torch.optim as optim
from Model import nets

networks = {'resnext101':    320,
            'pnasnet':       331, 'nasnet':       331,
            'xception':      299, 'inceptionv4':  299, 'incepresv2':   299,
            'se_resnext101': 224, 'resnet152':    224, 'dpn':          224, 'senet154':     224}

class skin_train():
    def __init__(self, flags):
        self.device = torch.device('cuda:{}'.format(flags.gpu) if torch.cuda.is_available() else 'cpu')
        print('--------------------Using device:', self.device)
        print('flags: ', flags, '\n')        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.model = nets(flags.network, flags.num_class).to(self.device)
        self.opt = optim.SGD(self.model.parameters(), lr=flags.lr, momentum=0.9)
        self.train_loader = train_loader(networks[flags.network], flags.train_batchsz)
        self.val_loader = val_loader(networks[flags.network], flags.train_batchsz)
        self.test_loader = test_loader(networks[flags.network], flags.train_batchsz)
        self.best_acc = 0

    def val_test(self):
        self.model.eval()
        with torch.no_grad():
            label_test = np.array([])
            pred_test = np.array([])
            for i, (data, label) in enumerate(self.val_loader):
                data, label = Variable(data).to(self.device), \
                              Variable(label).to(self.device)
                label_test = np.append(label_test, label.data.cpu().numpy())
                predict = self.model(data)
                pred_label = torch.max(predict, 1)[1].data.cpu().numpy()
                pred_test = np.append(pred_test, pred_label)
            accuracy_val = accuracy_score(label_test, pred_test)
        return accuracy_val

    def final_test(self):
        self.model.eval()
        with torch.no_grad():
            label_test = np.array([])
            pred_test = np.array([])
            for i, (data, label) in enumerate(self.test_loader):
                data, label = Variable(data).to(self.device), \
                              Variable(label).to(self.device)
                label_test = np.append(label_test, label.data.cpu().numpy())
                predict = self.model(data)
                pred_label = torch.max(predict, 1)[1].data.cpu().numpy()
                pred_test = np.append(pred_test, pred_label)
            accuracy_test = accuracy_score(label_test, pred_test)
        return accuracy_test

    def train(self, flags):
        self.model.train()
        scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 20, gamma=0.1)
        for epoch in range(flags.epochs):
            for i, (x, y) in enumerate(self.train_loader):
                data, label = Variable(x).to(self.device), \
                              Variable(y).to(self.device)

                predict = self.model(data)
                loss = self.loss_fn(predict, label)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if (i+1) % 100 == 0:
                    print('epoch:', epoch+1,
                          '| iteration:', i+1,
                          '| train loss:', loss.data.cpu().numpy())
            accuracy = self.val_test()
            print('epoch:', epoch+1, '/ {}'.format(flags.epochs),
                  '| loss:', loss.data.cpu().numpy(),
                  '| accuracy:', accuracy)
            if accuracy >= self.best_acc:
                self.best_acc = accuracy
                torch.save(self.model, './models/full_{}.pth'.format(flags.network))
                print('----validation accuracy:', accuracy)
                count = 0
            elif accuracy < self.best_acc:
                count = count + 1
                if count == 7:
                    break
            scheduler.step()
        test_acc = self.final_test()
        return self.best_acc, test_acc

def args():
    main_arg_parser = argparse.ArgumentParser(description="parser")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")

    train_arg_parser.add_argument("--gpu", type=int, default=0,
                                  help="assign gpu index")
    train_arg_parser.add_argument("--epochs", type=int, default=54,
                                  help="train how many epochs")
    train_arg_parser.add_argument("--train_batchsz", type=int, default=32,
                                  help="batch size for training")
    train_arg_parser.add_argument("--num_class", type=int, default=8,
                                  help="number of class")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help='learning rate of the model')
    train_arg_parser.add_argument("--network", choices=networks, type=str, default='efficient',
                                  help='network name')
    return train_arg_parser.parse_args()


def main():
    flags = args()
    model_ = skin_train(flags=flags)
    val_acc, test_acc = model_.train(flags=flags)
    return val_acc, test_acc

if __name__ == "__main__":

    val_acc, test_acc = main()

    with open('./accuracy.txt', 'a') as file:
        file.write('accuracy of {}:'.format(args().network))
        file.write(str(val_acc), str(test_acc))
        file.write('\n')
