"""Artificial typed-buffer contract fixture; never a supported model family."""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from ram_vram_finetune.adapters.protocol import BlockPlan
from ram_vram_finetune.adapters.registry import register
from ram_vram_finetune.buffers import RawBufferContract
from ram_vram_finetune.finetune.lora import create_lora_for_block

NAME='contract_fixture_only'
TARGET='mixers.unique_projection'


def make_source(path):
    path=Path(path);path.mkdir()
    generator=torch.Generator().manual_seed(5)
    save_file({'embedding':torch.randn(13,6,generator=generator),'head':torch.randn(13,6,generator=generator),
               'bytes':torch.arange(36,dtype=torch.uint8).reshape(6,6),'scale':torch.tensor([.01])},str(path/'source.safetensors'))
    (path/'contract.json').write_text(json.dumps({'identity':hashlib.sha256(b'fixture-decode-v1').hexdigest(),'workspace':4096}))
    (path/'tokens.jsonl').write_text(json.dumps({'input_ids':[1,2,3,4],'labels':[-100,-100,3,4]})+'\n'+json.dumps({'input_ids':[5,6,7],'labels':[-100,6,7]})+'\n')
    return path


class TypedAdapter:
    def __init__(self,path,*,rank=2,alpha=4.,device='cpu',**_):
        path=Path(path);self.device=torch.device(device);state=load_file(str(path/'source.safetensors'));config=json.loads((path/'contract.json').read_text())
        self.embedding=state['embedding'];self.head=state['head'].to(self.device)
        self.identity=hashlib.sha256((path/'source.safetensors').read_bytes()).hexdigest()
        adapters=create_lora_for_block(0,'generic_attention_kind', {TARGET+'.weight':(6,6)},rank,alpha,self.device,targets=[TARGET])
        def functional(hidden_states,position_embeddings,attention_mask,packed,scale,projection):
            # Real uint8 data, no dtype-punning or model-specific common branch.
            weight=(packed.reshape(6,6).float()-18)*scale
            return hidden_states+torch.tanh(projection.forward(hidden_states,weight))
        contract=RawBufferContract('fixture-subtract-scale-v1',config['identity'],config['workspace'])
        self.blocks=[BlockPlan(0,'generic_attention_kind',functional,{'projection':adapters[TARGET]},
            {'packed':state['bytes'].reshape(config.get('packed_shape',[6,6])),'scale':state['scale']},adapters,buffer_contract=contract)]

    def terminal_norm(self,hidden):return hidden
    def positions_and_masks(self,valid):return (),{'generic_attention_kind':None}
    def artifact_key(self,layer,target,parameter):return f'fixture.{layer}.{target}.{parameter}'
    def runtime_metadata(self):return {'architecture':'artificial_contract_only'}
    def close(self):self.blocks.clear();self.embedding=self.head=None


def register_fixture():
    from ram_vram_finetune.adapters.registry import _FACTORIES
    if NAME not in _FACTORIES:register(NAME,TypedAdapter)


if __name__=='__main__':
    import sys
    from ram_vram_finetune.config import TrainConfig
    from ram_vram_finetune.training.engine import run
    import ram_vram_finetune
    request=json.loads(Path(sys.argv[1]).read_text());register_fixture()
    result=run(TrainConfig(**request['config']),resume=request['resume'])
    Path(request['result']).write_text(json.dumps({'report':result,'package_file':ram_vram_finetune.__file__,'pid':__import__('os').getpid()}))
