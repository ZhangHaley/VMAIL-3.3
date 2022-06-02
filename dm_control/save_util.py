import os
import sys
import yaml
import pathlib
sys.path.append(os.getcwd())

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prRed2(prt): print("\033[38;5;198m {}\033[00m" .format(prt))
def prRed3(prt): print("\033[38;5;168m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prGreen2(prt): print("\033[38;5;36m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prYellow2(prt): print("\033[33m {}\033[00m" .format(prt))
def prYellow3(prt): print("\033[38;5;214m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))
def prBlue(prt): print("\033[0;34m {}\033[00m" .format(prt))
def prBlue2(prt): print("\033[38;5;33m {}\033[00m" .format(prt))
def prLightBlue(prt): print("\033[1;34m {}\033[00m" .format(prt))

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            #获取第几次 run
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir

def get_load_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            #获取第几次 run
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir
  
def prepare_logdir(log_writer_path,env_name):
  log_writer_path = os.path.join(os.getcwd(),log_writer_path)
  experiments_id = env_name+"_"+str(len(os.listdir(log_writer_path))+1)
  log_writer_path = os.path.join(log_writer_path,experiments_id)
  return log_writer_path

def get_training_num(path,stay_current = True):
  num = 0
  for i in path.iterdir():
    num = num+1
  if stay_current:
    num -= 1
  return f'run-test-{num}'

def save_yaml_file(config,save_path):
  save_path = str(save_path)
  save_file = os.path.join(save_path,'config.yaml')
  with open(save_file,'w') as f:
    yaml.dump(config,f)
