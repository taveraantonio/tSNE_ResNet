import time
import copy
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model.resnet import resnet101
from loader import loader_helper
from logger import TensorboardXLogger
import os

num_epochs = 50
batch_size = 16
save_every_iter = 1000
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
save_dir = 'snapshots'
MODEL_NAME = 'current.pth'
MODEL_LOG_NAME = 'current_log.txt'
resume = False
merge_idda_classes = False
N_CLASSES = 105 if not merge_idda_classes else 4
experiment_name = str(N_CLASSES)+'_classes'


model_path = os.path.join(save_dir, experiment_name, MODEL_NAME)
if resume and os.path.isfile(model_path):
    model = torch.load(model_path)
    model.eval()
    with open(os.path.join(save_dir, experiment_name, MODEL_LOG_NAME), 'r') as fp:
        words = fp.readline().split()
        starting_iter = int(words[2])
        starting_epoch = int(words[5])
    print("Resuming model at epoch %d and iter %d, with %d classes" % (starting_epoch, starting_iter, N_CLASSES))
else:
    model = resnet101(pretrained=True, num_classes=N_CLASSES)
    starting_iter = 0
    starting_epoch = 0
    print("Training ResNet-101 from scratch with %d classes" % N_CLASSES)
model.cuda()
# Cross entropy loss takes the logits directly, so we don't need to apply softmax in our CNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=6)


train_loader, val_loader = loader_helper.get_loaders(batch_size=batch_size, merge_idda_classes=merge_idda_classes)


since = time.time()

val_acc_history = []

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

logger = TensorboardXLogger('tensorboard')

for epoch in range(starting_epoch, num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:#['train', 'val']:
        if phase == 'train':
            print('\n-- Training epoch %d' % int(epoch+1))
            model.train()  # Set model to training mode

        else:
            print('-- Validating epoch %d' % int(epoch+1))
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        loader = train_loader if phase == 'train' else val_loader
        # Iterate over data.

        for iter, input in enumerate(loader):
            #if phase == 'train' and iter < starting_iter:
            #    continue
            image = Variable(input[0]).cuda()
            label = Variable(input[1]).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                features, output = model(image)
                loss = criterion(output, label)

                _, preds = torch.max(output, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item()# * image.size(0)
            running_corrects += torch.sum(preds == label)
            if phase == 'train' and iter % save_every_iter == 0 and iter > 0:
                save_path = os.path.join(save_dir, experiment_name)
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
                torch.save(model, os.path.join(save_path, MODEL_NAME))
                with open(os.path.join(save_path, MODEL_LOG_NAME), 'w') as fp:
                    fp.write("iter = {:6d} epoch = {:d} | train loss = {:.4f}".format(iter, epoch, running_loss/(iter+1)))
            if phase == 'train' and iter % 50 == 0 and iter>0:
                print("iter = {:6d} / {:d} | train loss = {:.4f}".format(iter, int(len(loader.dataset)//batch_size), running_loss/(iter+1)))  #todo: dividere per valore del batch size
            if phase == 'val' and iter % 50 == 0 and iter>0:
                print("iter = {:6d} / {:d} | val acc = {:.2f}%".format(iter, len(loader.dataset), running_corrects.item() * 100 / (iter+1)))  #todo: dividere per valore del batch size
        starting_iter = 0
        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc = running_corrects.item() * 100 / len(loader.dataset)

        print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))
        if phase == 'train':
            save_path = os.path.join(save_dir, experiment_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, MODEL_NAME))
            with open(os.path.join(save_path, MODEL_LOG_NAME), 'w') as fp:
                fp.write(
                    "iter = {:6d} epoch = {:d} | train loss = {:.4f}".format(iter, epoch, epoch_loss))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = os.path.join(save_dir, experiment_name)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(model, os.path.join(save_path, 'best.pth'))
            with open(os.path.join(save_path, 'current_log_val.txt'), 'w') as fp:
                fp.write(
                    "epoch = {:d} | val accuracy = {:.2f}%".format(epoch, epoch_acc))
        if phase == 'val':
            val_acc_history.append(epoch_acc)
            scheduler.step(epoch_loss)

    print()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))

# load best model weights
model.load_state_dict(best_model_wts)
#return model, val_acc_history

