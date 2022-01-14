import os
import torch
import sys

def save_model(model, episode_num, file_path='checkpoint.pth'):
    if model is not None:
        torch.save(model.state_dict(), file_path)
        sys.stdout.write(f'   model saved: episode {episode_num} checkpoint file: {file_path}')

def load_model(model, file_path):
    if os.path.exists('file_path.pth') and model is not None:
        torch.save(model.state_dict(), 'file_path.pth')