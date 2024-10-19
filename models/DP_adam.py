# yz, 1017/2024

import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
from federated_dp.private_trainer import PrivateFederatedTrainer

class DPAdamTrainer(PrivateFederatedTrainer):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs
    ) -> None:
        super().__init__(
            args, logging, device, server_model, train_sites, val_sites, client_weights, **kwargs
        )

        assert self.sample_rate == 1, "Only support sample rate 1 at this moment"

        self.mt = None
        self.vt = None
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.adam_tau = 1e-9

        self.adam_lr = args.adam_lr

    def communication_grad(self, server_model, models, client_weights):
        with torch.no_grad():
            aggregated_grads = self._private_grad_aggregation(client_weights)

            if self.mt is None:
                self.mt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            if self.vt is None:
                self.vt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            for idx, key in enumerate(server_model.state_dict().keys()):
                assert self.mt[idx].shape == aggregated_grads[idx].shape
                if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                    continue

                self.mt[idx].mul_(self.beta1).add_(aggregated_grads[idx], alpha=1 - self.beta1)
                self.vt[idx].mul_(self.beta2).add_(
                    torch.pow(aggregated_grads[idx], 2), alpha=1 - self.beta2
                )

                mt_h = self.mt[idx]
                denom = torch.sqrt(self.vt[idx]) + self.adam_tau

                aggregated_grads[idx].copy_(mt_h.mul(self.adam_lr).div(denom))

            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.add_(aggregated_grads[idx])

                    if self.args.ema > 0.0:
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.mul_(self.args.ema).add_(
                                (1 - self.args.ema) * server_model.state_dict()[key]
                            )

            if not self.args.ema > 0.0:
                for client_idx in range(len(client_weights)):
                    models[client_idx].load_state_dict(server_model.state_dict())

        return server_model, models