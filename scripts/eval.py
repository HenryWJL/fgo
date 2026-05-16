import os
import copy
import click
import hydra
import torch
import dill
import logging
from pathlib import Path
from omegaconf import OmegaConf
from termcolor import cprint
from typing import Optional, Dict, List
from hydra.core.hydra_config import HydraConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from fgo.policy.base_policy import BasePolicy
from fgo.env_runner.base_runner import BaseRunner

OmegaConf.register_new_resolver("eval", eval, replace=True)
logger = logging.getLogger(__name__)

class EvalWorkspace:

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self.global_step = 0
        self.epoch = 1
        
        # configure model
        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: BasePolicy = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except:
                self.ema_model = hydra.utils.instantiate(cfg.policy)

    def run(self, checkpoint: Optional[str]=None, device: Optional[str]='cuda') -> None:
        cfg = copy.deepcopy(self.cfg)

        # configure env runner
        env_runner: BaseRunner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir
        )

        # load checkpoints
        self.load_checkpoint(path=checkpoint, exclude_keys=('optimizer', 'lr_scheduler'))
        if cfg.training.use_ema:
            policy = copy.deepcopy(self.ema_model)
        else:
            policy = copy.deepcopy(self.model)
        policy.to(torch.device(device))
        policy.eval()
        
        # run evaluations
        cprint(f"Evaluate checkpoint {checkpoint}", 'yellow')
        runner_log = env_runner.run(policy)
        cprint(f"---------------- Eval Results --------------", 'magenta')
        for key, value in runner_log.items():
            if isinstance(value, float):
                cprint(f"{key}: {value:.4f}", 'magenta')
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
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


@click.command(help="Evaluate policies in simulation.")
@click.option("-t", "--task", type=str, required=True, help="See 'fgo/config/task/' for details.")
@click.option("-p", "--policy", type=str, required=True, help="See 'fgo/config/' for details.")
@click.option("-c", "--checkpoint", type=str, default=None, help="Pretrained checkpoint path.")
@click.option("-d", "--device", type=str, default="cuda", help="Device type.")
def main(task, policy, checkpoint, device):
    with hydra.initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent.joinpath('fgo', 'config')),
        version_base=None
    ):
        cfg = hydra.compose(config_name=str(policy), overrides=[f'task={task}'])
        workspace = EvalWorkspace(cfg, "data/outputs")
        workspace.run(checkpoint, device)


if __name__ == "__main__":
    main()
