import argparse
from enum import Enum
from ControlVAECore.Env.vclode_track_env import VCLODETrackEnv
from ControlVAECore.Model.controlvae import ControlVAE
from ControlVAECore.Utils.misc import *
import psutil
import ControlVAECore.Utils.pytorch_utils as ptu

from mpi4py import MPI
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

import cProfile
def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator

def flatten_dict(dd, separator='_', prefix=''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def build_args(parser):
    # add args for each content 
    parser = VCLODETrackEnv.add_specific_args(parser)
    parser = ControlVAE.add_specific_args(parser)
    args = vars(parser.parse_args())
    # yaml
    config = load_yaml(args['config_file'])
    config = flatten_dict(config)
    args.update(config)
    
    if args['load'] and mpi_rank ==0:
        import tkinter.filedialog as fd
        config_file = fd.askopenfilename(filetypes=[('YAML','*.yml')])
        data_file = fd.askopenfilename(filetypes=[('DATA','*.data')])
        config = load_yaml(config_file)
        config = flatten_dict(config)
        args.update(config)
        args['load'] = True
        args['data_file'] = data_file
        
    #! important!
    seed = args['seed'] + mpi_rank
    args['seed'] = seed
    VCLODETrackEnv.seed(seed)
    ControlVAE.set_seed(seed)
    return args
   
    return agent, args

# @profile(filename='profile')
def train(agent):
    agent.train_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./Data/ControlVAE.yml', help= 'a yaml file contains the training information')
    parser.add_argument('--seed', type = int, default=0, help='seed for root process')
    parser.add_argument('--experiment_name', type = str, default="debug", help="")
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--gpu', type = int, default=0, help='gpu id')
    parser.add_argument('--cpu_b', type = int, default=0, help='cpu begin idx')
    parser.add_argument('--cpu_e', type = int, default=-1, help='cpu end idx')
    
    
    args = build_args(parser)
    ptu.init_gpu(True, gpu_id=args['gpu'])
    if args['cpu_e'] !=-1:
        p = psutil.Process()
        cpu_lst = p.cpu_affinity()
        try:
            p.cpu_affinity(range(args['cpu_b'],args['cpu_e']))   
        except:
            pass 
    
    #build each content
    env = VCLODETrackEnv(**args)
    agent = ControlVAE(323, 57, 120,env, **args)
    
    if args['load'] and mpi_rank ==0:
        agent.try_load(args['data_file'])
        
    agent.save_before_train(args)
    
    train(agent)
    pass
    