import torch
import torch.nn.functional as F
from models.utils import get_model, get_imagenet_model


class Base_alg:
    def __init__(self, val_loader, device, args, base_model=None):
        super(Base_alg, self).__init__()
        # load the pre-trained model.
        self.get_path(args)
        if base_model is not None:
            self.base_model = base_model
        elif ('imagenet' in args['dataname']) & (args['num_classes'] == 1000):
            self.base_model = get_imagenet_model(args['arch'], args['num_classes'], args['seed']).to(device)
        else:
            self.base_model = get_model(args['arch'], args['num_classes'], args['seed']).to(device)
            self.base_model.load_state_dict(torch.load('{}/base_model.pt'.format(self.save_dir_path), map_location=device, weights_only=False))

        self.val_loader = val_loader
        self.device = device
        self.args = args

    def get_path(self, args):
        dataname = args['dataname']
        if args['dataname'] in ['pacs', 'office_home', 'domainnet']:
            self.save_dir_path = './checkpoints/{}'.format(args['source'] + '_' + args['arch'])
        else:
            self.save_dir_path = './checkpoints/{}'.format(dataname + '_' + args['arch'])

    def _gpu_resize(self, inputs):
        h = inputs.shape[-1]
        if h == 224:
            return inputs
        if h == 64:  # TinyImageNet: Resize(256) + CenterCrop(224)
            inputs = F.interpolate(inputs, size=256, mode='bilinear', align_corners=False)
            return inputs[:, :, 16:240, 16:240]
        # CIFAR (32) or other: direct resize to 224
        return F.interpolate(inputs, size=224, mode='bilinear', align_corners=False)

    def evaluation(self):
        score = 0
        return score

    def test(self):
        self.base_model.eval()
        correct = 0
        total = 0
        val_loader = self.val_loader
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                inputs, targets = batch_data[0], batch_data[1]
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = self._gpu_resize(inputs)
                outputs = self.base_model(inputs)
                _, predicted = outputs.max(1)
                total += len(predicted)
                correct += predicted.eq(targets).sum().item()
        return 100 * correct/total
