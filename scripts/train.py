import os
import time
import logging
import hydra
import torch
import dill
import copy
import random
import wandb
import tqdm
import numpy as np
import torch.distributed as dist
from termcolor import cprint
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.core.hydra_config import HydraConfig
from fgo.dataset.base_dataset import BaseDataset
from fgo.env_runner.base_runner import BaseRunner
from fgo.policy.base_policy import BasePolicy
from fgo.common.checkpoint_util import TopKCheckpointManager
from fgo.common.pytorch_util import dict_apply, optimizer_to
from fgo.model.common.ema_model import EMAModel
from fgo.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)
logger = logging.getLogger(__name__)

class TrainWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = []

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.global_step = 0
        self.epoch = 1
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ------------------------------
        # DDP initialization
        # ------------------------------
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_distributed = (world_size > 1 and local_rank >= 0)
        if self.is_distributed:
            self.local_rank = local_rank
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
            dist.init_process_group(backend="nccl", init_method="env://")
            dist.barrier()
            print(f"[pid {os.getpid()}] Initialized process group: rank {dist.get_rank()}/{dist.get_world_size()}; "
                  f"device {self.device}")
        else:
            self.local_rank = 0
            if torch.cuda.is_available() and ("cuda" in cfg.training.device):
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(cfg.training.device)
            print(f"[pid {os.getpid()}] Non-distributed mode, device {self.device}")

        # ------------------------------
        # Configure model
        # ------------------------------
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.model.to(self.device)
        # wrap model with DDP (if distributed)
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # ------------------------------
        # Configure dataset
        # ------------------------------
        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        normalizer.to(self.device)
        
        dl_kwargs = dict(cfg.dataloader)
        if self.is_distributed: # use DistributedSampler if distributed
            train_sampler = DistributedSampler(dataset, shuffle=True)
            dl_kwargs.pop("shuffle", None)
            self.train_dataloader = DataLoader(
                dataset=dataset,
                sampler=train_sampler,
                **dl_kwargs
            )
        else:
            self.train_dataloader = DataLoader(dataset=dataset, **dl_kwargs)
        
        val_dataset = dataset.get_validation_dataset()
        self.val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # set normalizer
        self.model.module.set_normalizer(normalizer) if self.is_distributed else self.model.set_normalizer(normalizer)
        
        # ------------------------------
        # Optimizer, LR scheduler, EMA
        # ------------------------------
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        optimizer_to(self.optimizer, self.device)

        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step-1
        )

        self.ema_model = None
        self.ema: EMAModel = None
        if cfg.training.use_ema:
            model_ref = self.model.module if self.is_distributed else self.model
            self.ema_model = copy.deepcopy(model_ref)
            self.ema_model.to(self.device)
            self.ema_model.set_normalizer(normalizer)
            self.ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)

        if self.local_rank == 0:
            # configure env runner
            env_runner = None
            if cfg.training.rollout_every <= cfg.training.num_epochs:
                env_runner: BaseRunner = hydra.utils.instantiate(
                    cfg.task.env_runner,
                    output_dir=self.output_dir
                )
            
            # configure logging
            wandb_run = None
            if cfg.training.use_wandb:
                cfg.logging.name = str(cfg.logging.name)
                cprint("-----------------------------", "yellow")
                cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
                cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
                cprint("-----------------------------", "yellow")
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
                wandb.config.update({'output_dir': self.output_dir})

            # configure checkpoint manager
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        training_start_time = time.perf_counter()
        for local_epoch_idx in range(cfg.training.num_epochs):
            if self.is_distributed:
                try:
                    self.train_dataloader.sampler.set_epoch(local_epoch_idx)
                except Exception:
                    logger.warning("Unable to set epochs for dataloader samplers.")

            step_log = dict()
            train_losses = list()
            with tqdm.tqdm(
                self.train_dataloader,
                desc=f"Training epoch {self.epoch}", 
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec
            ) as tepoch:
                model_train = self.model.module if self.is_distributed else self.model
                model_train.train()

                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                
                    # compute loss
                    raw_loss, loss_dict = model_train.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        self.ema.step(model_train)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': self.lr_scheduler.get_last_lr()[0]
                    }
                    step_log.update(loss_dict)

                    is_last_batch = (batch_idx == (len(self.train_dataloader)-1))
                    if not is_last_batch:
                        if self.local_rank == 0 and cfg.training.use_wandb:
                            wandb_run.log(step_log, step=self.global_step)
                        self.global_step += 1

            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            # ========= eval start for this epoch ==========
            if self.local_rank == 0:
                model_eval = self.ema_model if cfg.training.use_ema else model_train
                model_eval.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and env_runner is not None:
                    runner_log = env_runner.run(model_eval)
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            self.val_dataloader,
                            desc=f"Validation epoch {self.epoch}", 
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                                loss, loss_dict = self.model.compute_loss(batch)
                                val_losses.append(loss)
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(self.device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        result = model_eval.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # save checkpoints
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

            # ========= eval end for this epoch ==========

                # log of last step is combined with validation and rollout
                if cfg.training.use_wandb:
                    wandb_run.log(step_log, step=self.global_step)

            self.global_step += 1
            self.epoch += 1
            del step_log
        
        # end of training
        logger.info(f"Overall training time: {(time.perf_counter() - training_start_time) / 3600}h")
        if self.is_distributed:    
            dist.destroy_process_group()
 
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def save_checkpoint(
        self,
        path: Optional[str]=None,
        tag: Optional[str]='latest', 
        exclude_keys: List[str]=None,
        include_keys: List[str]=None,
    ) -> str:
        if path is None:
            path = Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if isinstance(value, DDP):
                        value = value.module
                    payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        
        torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        return str(path.absolute())
    
    def save_snapshot(self, tag: Optional[str]='latest') -> str:
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag: Optional[str]='latest') -> Path:
        if tag=='latest':
            return Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")

    def load_payload(
        self,
        payload: Dict,
        exclude_keys: List[str]=None,
        include_keys: List[str]=None,
        **kwargs
    ) -> None:
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        # Load state_dicts
        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                # if the target object is DDP-wrapped, load into .module
                target = self.__dict__.get(key, None)
                if isinstance(target, DDP):
                    target = target.module
                if target is None:
                    logger.warning(f"Target {key} not found in workspace; skipping")
                    continue
                try:
                    target.load_state_dict(value, **kwargs)
                except Exception as e:
                    logger.error(f"Error loading {key}: {e}; skipping.")
                    continue

        # Load pickles
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(
        self,
        path: Optional[str]=None,
        tag: Optional[str]='latest', 
        exclude_keys: List[str]=None,
        include_keys: List[str]=None
    ) -> Dict:
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(
            payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys
        )
        return payload


@hydra.main(
    config_path=str(Path(__file__).resolve().parent.parent.joinpath('fgo', 'config')),
    version_base=None
)
def main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
