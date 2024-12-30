import yaml
import argparse
import pickle
try:
    from VclSimuBackend.ODESim.Loader.JsonSceneLoader import JsonSceneLoader
except ImportError:
    import VclSimuBackend
    JsonSceneLoader = VclSimuBackend.ODESim.JsonSceneLoader
from ControlVAECore.Utils.motion_dataset import MotionDataSet
from ControlVAECore.Utils.misc import *


if __name__ == '__main__':
    '''
    convert mocap bvh into binary file, the bvh will be downsampled and some 
    important data(such as state and observation of each frame) 
    '''    
    parser = argparse.ArgumentParser()
    parser.add_argument("--using_yaml",  default=True, help="if true, configuration will be specified with a yaml file", action='store_true')
    parser.add_argument("--bvh_folder", type=str, default="Data/ReferenceData/runwalkjumpgetup", help="name of reference bvh folder")
    parser.add_argument("--env_fps", type=int, default=20, help="target FPS of downsampled reference motion")
    parser.add_argument("--env_scene_fname", type = str, default = "odecharacter_scene.pickle", help="pickle file for scene")
    parser.add_argument("--config", type=str, default="Data/ControlVAE.yml", help="name of configuration file")
    args = vars(parser.parse_args())
    
    if args['using_yaml']:
        config = load_yaml(path=args['config'],initialdir='Data/Parameters/')
        config.update(args)
    args = config
        
    
    scene_loader = JsonSceneLoader()
    scene = scene_loader.file_load(args['env_scene_fname'])
    motion = MotionDataSet(args['env_fps'])
    
    assert args['bvh_folder'] is not None
    motion.add_folder_bvh(args['bvh_folder'], scene.character0)
    f = open(args['motion_dataset'], 'wb')
    pickle.dump(motion, f) 