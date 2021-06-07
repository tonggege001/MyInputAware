import torch
from torch.nn import MSELoss, CrossEntropyLoss
import torch.nn.functional as F
import tqdm
from data import get_data
from model import ResNet18, Generator


LR = 0.01
EPS = 1e-7          # used for diversity loss in case of divide zero
EPOCHS = 300
TARGET_LABEL = 8
P1 = 0.05
P2 = 0.05
BATCH_SIZE = 100
lamda = 1.0
DENSITY = 0.032


def adjust_learning_rate(opt, epoch):
    lr = LR * (0.1 ** (epoch // 100))
    for param_group in opt.param_groups:
        param_group['lr'] = lr


def make_first(x_batch1, netG, netM):
    x_trigger = netG(x_batch1)
    mask = netM(x_batch1)
    x_first = x_batch1 * (1 - mask) + mask * x_trigger
    return x_first


def make_second(x_batch1, x_batch2, netG, netM):
    x_trigger = netG(x_batch2)
    mask = netM(x_batch2)
    x_second = x_batch1 * (1 - mask) + mask * x_trigger
    return x_second


def train_mask(epoch, train_loader1, train_loader2, netM, optM, mse):
    log = {
        "loss_div": 0.0,
        "loss_norm": 0.0,
        "loss": 0.0
    }
    for (x_batch1, y_batch1), (x_batch2, y_batch2) in tqdm.tqdm(zip(train_loader1, train_loader2), desc=f"Epoch: {epoch}"):
        x_batch1, y_batch1 = x_batch1.to(device), y_batch1.to(device)
        x_batch2, y_batch2 = x_batch2.to(device), y_batch2.to(device)
        mask1, mask2 = netM(x_batch1), netM(x_batch2)

        distance_image = mse(x_batch1, x_batch2)
        distance_pattern = mse(mask1, mask2)
        loss_div = torch.sqrt(distance_image) / torch.sqrt(distance_pattern+EPS)
        loss_norm = torch.mean(F.relu(mask1 - DENSITY))

        loss = loss_div + 100 * loss_norm
        optM.zero_grad()
        loss.backward()
        optM.step()

        log["loss_div"] += loss_div
        log["loss_norm"] += loss_norm
        log["loss"] += loss
    return log




def train_one_epoch(epoch, train_loader1, train_loader2, netD, optD, criterion, netG, optG, netM, optM, mse):
    fake_label = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    torch.fill_(fake_label, TARGET_LABEL)
    fake_label = fake_label.to(device)
    log = {
        "loss_first": 0.0,
        "loss_second": 0.0,
        "loss_third": 0.0,
        "loss_reg": 0.0,
        "loss": 0.0,
    }
    for (x_batch1, y_batch1), (x_batch2, y_batch2) in tqdm.tqdm(zip(train_loader1, train_loader2),
                                                               desc=f"Epoch: {epoch}"):
        x_batch1, y_batch1 = x_batch1.to(device), y_batch1.to(device)
        x_batch2, y_batch2 = x_batch2.to(device), y_batch2.to(device)

        first_num = int(x_batch1.size(0) * P1)
        second_num = int(x_batch1.size(0) * P2)

        x_first = make_first(x_batch1[:first_num], netG, netM)
        x_second = make_second(x_batch1[first_num:first_num + second_num],
                               x_batch2[first_num:first_num + second_num], netG, netM)

        out_first = netD(x_first)
        loss_first = criterion(out_first, fake_label[:out_first.size(0)])

        out_second = netD(x_second)
        loss_second = criterion(out_second, y_batch1[first_num:first_num + second_num])

        out_third = netD(x_batch1[first_num + second_num:])
        loss_third = criterion(out_third, y_batch1[first_num + second_num:])

        loss_reg = torch.sqrt(mse(x_batch1, x_batch2)) / torch.sqrt(mse(netG(x_batch1), netG(x_batch2)))
        loss = loss_first + loss_second + loss_third + lamda * loss_reg

        optD.zero_grad()
        optG.zero_grad()
        loss.backward()
        optD.step()
        optG.step()

        log["loss_first"] += loss_first
        log["loss_second"] += loss_second
        log["loss_third"] += loss_third
        log["loss_reg"] += loss_reg
        log["loss"] += loss
    return log


def test(test_loader1, test_loader2, netD, netG, netM):
    fake_label = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    torch.fill_(fake_label, TARGET_LABEL)
    fake_label = fake_label.to(device)
    log = {
        "correct_first": 0.0,
        "correct_second": 0.0,
        "correct_third": 0.0,
        "total": 0.0,
    }

    with torch.no_grad():
        for (x_batch1, y_batch1), (x_batch2, y_batch2) in tqdm.tqdm(zip(test_loader1, test_loader2)):
            x_batch1, y_batch1 = x_batch1.to(device), y_batch1.to(device)
            x_batch2, y_batch2 = x_batch2.to(device), y_batch2.to(device)

            x_first = make_first(x_batch1, netG, netM)
            x_second = make_second(x_batch1, x_batch2, netG, netM)

            out_first = netD(x_first)
            out_second = netD(x_second)
            out_third = netD(x_batch1)

            log["correct_first"] += torch.sum(torch.eq(torch.argmax(out_first, dim=1), fake_label[:out_first.size(0)]))
            log["correct_second"] += torch.sum(torch.eq(torch.argmax(out_second, dim=1), y_batch1))
            log["correct_third"] += torch.sum(torch.eq(torch.argmax(out_third, dim=1), y_batch1))
            log["total"] += x_batch1.size(0)

    return log


def train():
    train_loader1, test_loader1 = get_data(BATCH_SIZE)
    train_loader2, test_loader2 = get_data(BATCH_SIZE)

    netD = ResNet18().to(device)
    optD = torch.optim.SGD(netD.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    criterion = CrossEntropyLoss().to(device)
    netG = Generator().to(device)
    optG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))
    netM = Generator(ch_out=1).to(device)
    optM = torch.optim.Adam(netM.parameters(), lr=LR, betas=(0.5, 0.9))
    mse = MSELoss(reduction='mean').to(device)

    # Training Mask
    print("=================================  Training Mask Generator  ================================")
    for epoch in range(25):
        log_mask = train_mask(epoch, train_loader1, train_loader2, netM, optM, mse)
        print("Epoch: {}, loss: {}, loss_div: {}, loss_norm: {}".format(epoch, log_mask["loss_div"],
                                                                        log_mask["loss_norm"], log_mask["loss"]))
    torch.save(netM, "model/netM.pkl")
    netM.requires_grad_(False)
    for epoch in range(EPOCHS):
        adjust_learning_rate(optG, epoch)
        adjust_learning_rate(optD, epoch)
        adjust_learning_rate(optM, epoch)
        log_train = train_one_epoch(epoch, train_loader1, train_loader2, netD, optD, criterion, netG, optG, netM, optM,
                                    mse)
        log_test = test(test_loader1, test_loader2, netD, netG, netM)

        print("Epoch: {}, loss: {}, loss_first: {}, loss_second: {}, loss_third: {}, loss_reg: {}, "
              "correct_first: {}, correct_second: {}, correct_third: {}".format(
            epoch, log_train["loss"], log_train["loss_first"], log_train["loss_second"], log_train["loss_third"],
            log_train["loss_reg"], log_test["correct_first"] / log_test["total"],
                                   log_test["correct_second"] / log_test["total"],
                                   log_test["correct_third"] / log_test["total"]
        ))

        if (epoch + 1) % 10 == 0:
            torch.save(netD, "model/netD.pkl")
            torch.save(netG, "model/netG.pkl")



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train()
