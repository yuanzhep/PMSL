# yz, 1020/2024

from tqdm import tqdm
from torch import no_grad

@no_grad()
def validate(val_loader, device, task_net, metrics, protector_net=None, is_task=True,
             label_divider=None):
    metrics.reset()
    if protector_net:
        protector_net.eval()
    task_net.eval()
    
    for _, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader), dynamic_ncols=True):
        x = x.to(device)
        y = y.to(device)
        if is_task:
            y = label_divider(y)[0]
        else:
            y = label_divider(y)[1]

        if protector_net:
            x = protector_net(x)
        output = task_net(x)
        metrics.update(output, y)