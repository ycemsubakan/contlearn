import torch


def print_and_log_scalar(writer, name, value, epoch, end_token='\n'):
    if isinstance(value, list):
        if len(value) == 0: return 
        value = torch.mean(torch.stack(value))
    #zeros = 40 - len(name) 
    #name += ' ' * zeros
    print('{} @ epoch {} = {:.4f}{}'.format(name, epoch, value, end_token))
    writer.add_scalar(name, value, epoch)
