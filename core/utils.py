import json
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def tensor2image(var):
    invTrans = transforms.Compose([
        transforms.Normalize(mean = [ 0., 0., 0. ],
                            std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                            std = [ 1., 1., 1. ]),
    ])
    var = invTrans(var)
    return Image.fromarray((var.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def parse_images(img, y_trg, gen_img, domains, display_count=2):
    im_data = []

    display_count = min(display_count, len(img))
    for i in range(display_count):
        cur_im_data = {'input_image': tensor2image(img[i]), f'{domains[y_trg[i].item()]}': tensor2image(gen_img[i])}
        im_data.append(cur_im_data)
    return im_data
    
    
def vis_faces(log_hooks):
    display_count = len(log_hooks)
    fig = plt.figure(figsize=(6 * 2, 4 * display_count))
    gs = fig.add_gridspec(display_count, 2)
    for i in range(display_count):
        hooks_dict = log_hooks[i]

        fig.add_subplot(gs[i, 0])
        plt.imshow(hooks_dict['input_image'])
        plt.title('input_image')

        fig.add_subplot(gs[i, 1])
        domain = list(hooks_dict.keys())[1]
        plt.imshow(hooks_dict[domain])
        plt.title(domain)

    plt.tight_layout()
    return fig
