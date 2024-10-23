# yz, 1020/2024, celeba 

import sys
import torch
import shutil
import random
import argparse
import torch.backends.cudnn as cudnn

from os import makedirs
from os.path import join
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image as tv_save_image
from tqdm import tqdm
from dataset.general import get_dataset
from models.general import get_reconstructor_model, get_protector_model
from utils.logger import get_logger
from utils.general import get_result_path, args2json, AverageMeter
from utils.loss import get_loss_fn

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train reconstructor")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--print-freq", default=200, type=int)
    parser.add_argument("--recon-arch")
    parser.add_argument("--recon-loss")
    parser.add_argument("--recon-lr", type=float, default=1e-3)
    parser.add_argument("--protector-arch")
    parser.add_argument("--protector-lr", type=float, default=1e-3)
    parser.add_argument('--protector-weight')
    parser.add_argument('--std', type=float)

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_path = get_result_path(dataset_name=args.dataset,
                                  task_arch=args.recon_arch,
                                  seed=args.seed,
                                  result_folder_name='train_recon')

    logger = get_logger(result_path)

    python_version = sys.version.replace('\n', ' ')
    # logger.info(f"Python version : {python_version}")
    # logger.info(f"Torch version : {torch.__version__}")
    # logger.info(f"Cudnn version : {torch.backends.cudnn.version()}")
    # logger.info(f"Model Path : {result_path}")

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    args2json(args, result_path)

    file_save_path = join(result_path, 'code')
    makedirs(file_save_path, exist_ok=True)
    shutil.copy(sys.argv[0], join(file_save_path, sys.argv[0]))

    data_train, data_test, _, _, _ = get_dataset(
        dataset_name=args.dataset)
    dataloader_train = DataLoader(data_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
    dataloader_test = DataLoader(data_test,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    recon_net = get_reconstructor_model(args.recon_arch).to(device)
    protector_net = get_protector_model(
        args.protector_arch, std=args.std).to(device)
    protector_net.load_state_dict(torch.load(args.protector_weight)[
                                   'state_dict_protector'])
    protector_net.eval()
    
    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers)
    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.workers)

    recon_loss_fn = get_loss_fn(args.recon_loss)
    optimizer_recon = Adam(recon_net.parameters(),
                           args.recon_lr, betas=(0.9, 0.999))
    loss_recon_avgm = AverageMeter()

    for epoch in range(args.epochs):
        loss_recon_avgm.reset()
        recon_net.train()

        for batch_idx, (x, _) in tqdm(enumerate(dataloader_train), total=len(dataloader_train),
                                      dynamic_ncols=True):
            x = x.to(device)

            with torch.no_grad():
                obf = protector_net(x)
            x_recon = recon_net(obf)
            loss = recon_loss_fn(x_recon, x)

            recon_net.zero_grad()
            loss.backward()
            optimizer_recon.step()

            loss_recon_avgm.update(loss.item(), x.size(0))

            if (batch_idx + 1) % args.print_freq == 0:
                logger.info('Train Epoch: {}\t'
                            'LR Recon {lr:.5f} '
                            'LossA {loss.val:.4f} ({loss.avg:.4f})'.format(
                                epoch,
                                lr=optimizer_recon.param_groups[0]["lr"],
                                loss=loss_recon_avgm))

        def save_example(x, x_recon, tag, tag_protector='', x_recon_train=None):
            x_save_path = join(
                result_path, f'Epoch{epoch:03d}_{tag}{tag_protector}.jpg')
            if x_recon_train is not None:
                x_recon = torch.cat((x_recon, x_recon_train))
            tv_save_image(torch.cat((x, x_recon)), x_save_path, nrow=x.size(0))

        save_example(x[:10], x_recon[:10], 'train')
        dataiter_test = iter(dataloader_test)
        x, _ = next(dataiter_test)
        x = x[:10].to(device)
        with torch.no_grad():
            xp = protector_net(x)
            x_recon = recon_net(xp)
            recon_net.train()
            x_recon_train = recon_net(xp)
        save_example(x, x_recon, 'val', x_recon_train=x_recon_train)

        save_path = join(result_path, 'PMSL_checkpoint.pth')
        torch.save(
            {
                'state_dict_recon': recon_net.state_dict(),
                'optimizer_recon': optimizer_recon.state_dict(),
            }, save_path)


if __name__ == "__main__":
    main()