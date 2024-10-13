# yz, 1011/2024

import sys
import shutil
import random
import argparse
import torch
import torch.backends.cudnn as cudnn

from os import makedirs
from os.path import join
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from dataset.general import get_dataset
from models.general import get_classification_model, get_protector_model
from utils.logger import get_logger
from utils.general import get_result_path, args2json, AverageMeter
from utils.metrics import get_metrics
from utils.loss import get_loss_fn
from validate import validate

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train Attacker")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--print-freq", default=200, type=int)

    parser.add_argument("--task-arch")
    parser.add_argument("--task-loss")
    parser.add_argument("--task-lr", type=float, default=1e-3)
    parser.add_argument('--task-metrics', type=str, nargs="+")
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
                                  task_arch=args.task_arch,
                                  seed=args.seed,
                                  result_folder_name='train_adversary')

    logger = get_logger(result_path)

    python_version = sys.version.replace('\n', ' ')
    logger.info(f"Python version : {python_version}")
    logger.info(f"Torch version : {torch.__version__}")
    logger.info(f"Cudnn version : {torch.backends.cudnn.version()}")
    logger.info(f"Model Path : {result_path}")

    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        logger.info("{} : {}".format(key, value))

    args2json(args, result_path)

    file_save_path = join(result_path, 'code')
    makedirs(file_save_path, exist_ok=True)
    shutil.copy(sys.argv[0], join(file_save_path, sys.argv[0]))

    data_train, data_test, label_divider, task_nc, _ = get_dataset(
        dataset_name=args.dataset)
    dataloader_train = DataLoader(data_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.workers)
    dataloader_test = DataLoader(data_test,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.workers)

    task_net = get_classification_model(
        args.task_arch, num_classes=task_nc).to(device)
    protector_net = get_protector_model(
        args.protector_arch, std=args.std).to(device)
    protector_net.load_state_dict(torch.load(args.protector_weight)[
                                   'state_dict_protector'])
    protector_net.eval()

    task_loss_fn = get_loss_fn(args.task_loss)

    optimizer_task = Adam(task_net.parameters(),
                          args.task_lr, betas=(0.9, 0.999))

    metrics_train = get_metrics(args.task_metrics)
    metrics_test = get_metrics(args.task_metrics)

    loss_avgm = AverageMeter()

    best_acc = 0
    for epoch in range(args.epochs):
        task_net.train()
        loss_avgm.reset()
        metrics_train.reset()

        for batch_idx, (x, y) in tqdm(enumerate(dataloader_train), total=len(dataloader_train),
                                      dynamic_ncols=True):
            x = x.to(device)
            y, _ = label_divider(y)
            y = y.to(device)

            with torch.no_grad():
                x = protector_net(x)

            pred = task_net(x)
            loss = task_loss_fn(pred, y)
            loss_avgm.update(loss.item(), x.size(0))
            metrics_train.update(pred, y)
            optimizer_task.zero_grad()
            loss.backward()
            optimizer_task.step()

            if (batch_idx + 1) % args.print_freq == 0:
                logger.info('+++Train+++ Epoch: {} LR: {lr:.5f}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                            'Metrics: {metrics.val}\n\t({metrics.avg})'.format(
                                epoch,
                                lr=optimizer_task.param_groups[0]["lr"],
                                loss=loss_avgm,
                                metrics=metrics_train))

        validate(val_loader=dataloader_test,
                 device=device,
                 task_net=task_net,
                 metrics=metrics_test,
                 protector_net=protector_net,
                 is_task=True,
                 label_divider=label_divider)
        logger.info(('+++Test+++ Epoch: {} Metrics: {metrics.avg}'
                     .format(epoch, metrics=metrics_test)))

        main_metric_val = metrics_test.get_main_metric()
        if main_metric_val > best_acc:
            best_acc = main_metric_val
            save_path = join(result_path, 'checkpoint_best.pth')
            torch.save(
                {
                    'state_dict_task': task_net.state_dict(),
                    'optimizer_task': optimizer_task.state_dict(),
                }, save_path)
            logger.info(f'Best acc renewed: {best_acc}')

        save_path = join(result_path, 'checkpoint.pth')
        torch.save(
            {
                'state_dict_task': task_net.state_dict(),
                'optimizer_task': optimizer_task.state_dict(),
            }, save_path)


if __name__ == "__main__":
    main()
