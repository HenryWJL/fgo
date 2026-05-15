from typing import Dict
from fgo.policy.base_policy import BasePolicy


class BaseRunner:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
