import torch

import os
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import sys
import argparse
import numpy as np
import yaml
import dill

sys.path.append('/home/src/experiments/utils')
sys.path.append('/home/src/experiments')
from utils.probing_utils import AccuracyProbe
from utils.data_utils import custom_pad_pos

def main(config):
  home = os.environ['LEARNING_DYNAMICS_HOME']
  if "device" not in config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
  else:
    device = config["device"]
    
  dataset_config = config["dataset"]
  dataset_dir = dataset_config["dir"]
  task_name = dataset_config["task_name"]
  
  layer_idx = config["layer_idx"]
  attention_head = config["attention_head"]
  # if layer_idx > 12:
  #   layer_idx = int(np.ceil((layer_idx - 12)/12))
  
  resid = config["resid"]
  model_name = config["model_name"].split('/')[-1]
  if "pythia" in model_name:
    model_name += "-step" + str(config["model_step"])
  
  probe_config = config["probe"]
  finetune_model = probe_config["finetune-model"]
  num_epochs = probe_config["epochs"]
  batch_size = probe_config["batch_size"]
  num_labels = probe_config["num_labels"]
  input_size = probe_config["input_size"]
  output_dir = os.path.join(home, probe_config["output_dir"])
  
  print(num_labels, input_size, task_name)
  lr = float(probe_config["lr"])

  train_dataset_path = os.path.join(
    home,
    dataset_dir,
    model_name,
    f"train-layer-{layer_idx}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
  )
  dev_dataset_path = os.path.join(
    home,
    dataset_dir,
    model_name,
    f"dev-layer-{layer_idx}{'-'+str(attention_head) if attention_head is not None else ''}.pt",
  )

  train_dataset = torch.load(train_dataset_path, pickle_module=dill)
  dev_dataset = torch.load(dev_dataset_path, pickle_module=dill)

  train_data_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      collate_fn=custom_pad_pos,
  )
  dev_data_loader = DataLoader(
      dev_dataset,
      batch_size=batch_size,
      shuffle=False,
      collate_fn=custom_pad_pos,
  )

  probe = AccuracyProbe(input_size, num_labels, finetune_model).to(device)
  ## cpos = 17, fpos = 34
  trainer = Trainer(max_epochs=num_epochs)
  trainer.fit(probe, train_data_loader, dev_data_loader)
    
  val_logs = trainer.validate(probe, dev_data_loader)
  layer_str = "layer-" + str(layer_idx)
  os.makedirs(os.path.join(output_dir, model_name, layer_str), exist_ok=True)
  if resid:
    with open(os.path.join(output_dir, model_name, layer_str, "val_acc.txt"), "w") as f:
      f.write(str(val_logs))
  else:
    with open(os.path.join(output_dir, model_name, layer_str, f"val_acc_out{'_head_' + str(attention_head) if attention_head is not None else ''}.txt"), "w") as f:
      f.write(str(val_logs))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True, type=str, help="path to config file")
  args = parser.parse_args()

  config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
  main(config)