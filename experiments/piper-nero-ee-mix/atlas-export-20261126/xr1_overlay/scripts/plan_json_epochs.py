import argparse,glob,json,math,pathlib
from hydra import compose,initialize_config_dir

def calculate(frames,world_size,batch_size,accumulation,epochs):
 if min(frames,world_size,batch_size,accumulation,epochs)<=0:raise ValueError('counts must be positive')
 per_rank=math.ceil(frames/world_size)
 batches=math.ceil(per_rank/batch_size)
 steps=math.ceil(batches/accumulation)
 return {'frames':frames,'world_size':world_size,'batch_per_device':batch_size,'accumulate_grad_batches':accumulation,'epochs':epochs,'samples_per_rank':per_rank,'batches_per_epoch':batches,'steps_per_epoch':steps,'max_steps':epochs*steps,'sampler_padding_per_epoch':per_rank*world_size-frames}

def main():
 p=argparse.ArgumentParser(description='Plan finite JSON epochs with DistributedSampler and DataLoader drop_last=False.')
 p.add_argument('--data-config',required=True);p.add_argument('--epochs',type=int,required=True);p.add_argument('--batch-size',type=int,required=True);p.add_argument('--world-size',type=int,required=True);p.add_argument('--accumulation',type=int,default=1);p.add_argument('--counts',action='store_true');a=p.parse_args()
 with initialize_config_dir(config_dir=str(pathlib.Path(__file__).resolve().parents[1]/'configs'),version_base=None):
  cfg=compose(config_name='config',overrides=['data='+a.data_config,'model=posttrain_piper_joint_full','trainer=deepspeed_npu_bf16_zero2'])
 files=set()
 paths=list(cfg.data.params.train_datasets.paths)
 if "replay_dataset" in cfg.data.params:
  paths.extend(cfg.data.params.replay_dataset.paths)
 for raw in paths:
  path=pathlib.Path(raw)
  if path.is_file() and path.suffix=='.json':files.add(str(path))
  elif path.is_dir():files.update(glob.glob(str(path/'**/*.json'),recursive=True))
  else:raise ValueError('invalid training path: '+str(path))
 if not files:raise ValueError('training paths contain no JSON files')
 frames=0
 for name in sorted(files):
  with open(name) as f:n=int(json.load(f)['num_frames'])
  frames+=max(0,n)
 result=calculate(frames,a.world_size,a.batch_size,a.accumulation,a.epochs);result['episodes']=len(files)
 if a.counts:print(result['frames'],result['steps_per_epoch'],result['max_steps'],result['episodes'])
 else:print(json.dumps(result,indent=2))
if __name__=='__main__':main()
