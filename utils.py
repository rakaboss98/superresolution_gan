import torch

def save_checkpoint(model, optimizer, filename):
    print('==> Saving checkpoint')
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, device):
    print('==> Loading checkpoint')
    ckpt = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    optimizer.load_state_dict(ckpt["optimizer"])
