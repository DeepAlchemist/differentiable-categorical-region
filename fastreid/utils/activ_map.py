import os
import cv2
import errno
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torchvision import utils
import torch.nn.functional as F

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _binary_conversion(score_map, block_size=3):
    """
    generate the binary mask
    """
    use_gpu = score_map.is_cuda
    b, h, w = score_map.size()
    soft_mask = score_map
    score_map = torch.mean(score_map, -1)  # (b h)

    weight = torch.ones(1, 1, block_size, 1)
    if use_gpu:
        weight = weight.to(score_map.device)
    score_map = F.conv2d(input=score_map[:, None, :, None],
                         weight=weight,
                         padding=(block_size // 2, 0))  # (b 1 h 1)

    if block_size % 2 == 0:
        score_map = score_map[:, :, :-1]

    max_index = torch.argmax(score_map.view(b, h), dim=1)  # (b,)

    # generate the stripe mask
    stripe_masks = torch.zeros(b, h)
    if use_gpu:
        stripe_masks = stripe_masks.to(score_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1

    stripe_masks = F.max_pool2d(input=stripe_masks[:, None, :, None],
                                kernel_size=(block_size, 1),
                                stride=(1, 1),
                                padding=(block_size // 2, 0))  # (b 1 h 1)
    if block_size % 2 == 0:
        stripe_masks = stripe_masks[:, :, 1:]

    stripe_masks = stripe_masks.view(b, h, 1).expand_as(soft_mask)  # (b h w)

    # output score map
    weight = torch.ones(1, 1, block_size, block_size)
    if use_gpu:
        weight = weight.to(score_map.device)
    soft_mask = F.conv2d(input=soft_mask[:, None, :, :],
                         weight=weight,
                         padding=block_size // 2)  # (b 1 h w)
    soft_mask = soft_mask.squeeze(1)
    return stripe_masks, soft_mask

def binary_conversion(activ_map,
                      s_filter):
    """ generate the binary mask
    activ_map(4D Tensor): (b 1 h w)
    """
    activ_map = F.conv2d(input=activ_map,
                         weight=s_filter,
                         stride=(5, 1),
                         padding=0)  # (b 1 h w)
    # import pdb; pdb.set_trace()
    b, ch, h, w = activ_map.size()
    max_index = torch.argmax(activ_map.view(b, -1), dim=1)  # (b h*w)

    # generate the stripe mask
    stripe_masks = torch.zeros([b, h * w], device=activ_map.device)
    batch_index = torch.arange(0, b, dtype=torch.long)
    stripe_masks[batch_index, max_index] = 1
    stripe_masks = stripe_masks.view(b, 1, h, w)  # (b 1 h w)

    # process soft mask
    activ_map = activ_map * 0.1
    activ_map = torch.softmax(activ_map.view(b, -1), dim=1).view(b, 1, h, w)

    return stripe_masks, activ_map

class ClassActivMap(object):
    def __init__(self,
                 model,
                 train_items,
                 pid_to_lbl,
                 in_size,
                 vis_pids,
                 save_dir='',
                 selected_module=['heads.bottleneck'],
                 selected_filter=0,
                 out_size=[128, 64]):
        super().__init__()
        '''
        Args:
            model: as one of the models in fastreid/modeling/meta_arch/
            selected_module (str): e.g., 'backbone.layer4'
            selected_filter (int): target filter of the selected_module to hook,
                                   hook all the filters if given a negative number.
        '''
        self.model = model
        self.model.eval()

        children_name = [name for name, _ in self.model.named_children()]
        self.selected_module = [sm.split('.') for sm in selected_module]
        assert self.selected_module[0][0] in children_name
        self.selected_filter = selected_filter

        self.in_size = in_size
        self.out_size = out_size
        self.save_dir = os.path.join(save_dir, 'CAMA-CAM')
        mkdir(self.save_dir)

        self.vis_pids = vis_pids  # person ids to visualize
        self.pid_to_lbl = pid_to_lbl
        self.lbl_to_pid = dict([(l, p) for p, l in self.pid_to_lbl.items()])

        # initialize
        assert hasattr(self.model.heads, 'classifier'), \
            'class activation map requires model to have a classifier'

        # self.weights = [self.model.heads.classifier.weight.detach()]  # (num_classes, dim_feat)
        self.weights = [cls.weight.detach() for cls in self.model.heads.classifier]  # (num_classes, dim_feat)

        # collect image paths of vis_pids
        self.pid_to_fpath = defaultdict(list)
        for i, item in enumerate(train_items):
            # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
            pid = item[1]
            if pid in self.vis_pids:
                self.pid_to_fpath[pid].append(item[0])

        self.module_output = None
        self.module_input = None

    def generate(self):
        # forward for getting feature maps
        self.model.eval()
        # register forward hook for the opt.selected_module to obtain the output of selected_module
        # self.module_forward_hook()
        # Traversing person ids
        for ii, (pid, fpath) in enumerate(self.pid_to_fpath.items()):
            print('class activation map of {} ({}/{}) ...'.format(pid, ii + 1, len(self.pid_to_fpath)))
            # list of cv2 images of a pid
            original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
            # resize, to_tensor, normalize cv2im and concatenate as an input batch
            batch_images = torch.cat(
                [self.cv2im_transformer(x, im_size=self.in_size, norm=False, div=False) for x in original_ims],
                dim=0
            )
            batch_images = batch_images.cuda()
            # forward at eval mode
            images = self.model.preprocess_image(batch_images)
            branches_feat_maps = self.model.local_branch(self.model.backbone(images))  # List(Tensor(b c h w)) len=num_branch
            branches_feat_maps = [neck(f) for neck, f in zip(self.model.heads.bottleneck, branches_feat_maps)]

            branches_ws = [w[self.pid_to_lbl[pid]] for w in self.weights]  # List[Tensor(d)]
            # weight = self.weights(torch.argmax(self.model.logit, dim=1))  # bs x dim_feat
            assert len(branches_ws) == len(branches_feat_maps)

            # Traversing branches
            branches_cam_on_ims = []  # len=num_branch*batch_size
            for feat_maps, w in zip(branches_feat_maps, branches_ws):
                # Traversing images of a pid and record c
                cam_on_ims = []
                for original_im, feat_map in zip(original_ims, feat_maps):
                    # (dim_feat h w) to (h w dim_feat) im
                    feat_map = feat_map.permute(1, 2, 0).contiguous()
                    # (h w dim_feat)*(dim_feat 1) to (h w 1) to (h w)
                    score_map = feat_map.matmul(w.unsqueeze(dim=1)).squeeze(-1)

                    # opt1
                    # score_map = feat_map.norm(p=2, dim=2) # (h w)
                    # opt2
                    # stripe_mask, score_map = _binary_conversion(score_map[None, :])
                    # score_map = stripe_mask * score_map
                    # score_map = score_map.squeeze(0)
                    # opt3
                    # stripe_mask, score_map = binary_conversion(score_map[None, None, :, :], self.model.smooth_filter)
                    # score_map = stripe_mask * score_map
                    # score_map = score_map[0, 0, ...]

                    score_map = score_map.detach().cpu().numpy()
                    score_map = cv2.resize(score_map, (self.out_size[1], self.out_size[0]))
                    # score_map = np.maximum(score_map, 0) # for positive cam
                    # score_map = np.minimum(score_map, 0) * (-1)  # for negative cam

                    # Normalize between 0-1
                    score_map = (score_map - np.min(score_map)) / (np.max(score_map) - np.min(score_map))

                    score_map = np.uint8(score_map * 255)  # Scale between 0-255 to visualize
                    cam_on_im = self.class_activation_map_on_im(original_im, score_map,
                                                                out_size=self.out_size)  # numpy array of shape HWC 0-255
                    # torch.from_numpy keeps value range and dtype, thus cam_on_im.dtype=torch.uint8, i.e. torch.ByteTensor
                    cam_on_ims.append(torch.from_numpy(cam_on_im.transpose(2, 0, 1)))  # HWC to CHW
                    # cam_on_ims.append(torch.from_numpy(original_im[..., ::-1].transpose(2, 0, 1))) # will save original images of pid
                branches_cam_on_ims += cam_on_ims

            # the value range of make_grid input tensor can be 0-1 or 0-255, output grid has the same range as input
            grid = utils.make_grid(branches_cam_on_ims, nrow=batch_images.size(0), padding=2, normalize=False,
                                   range=None, scale_each=False, pad_value=255)
            ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            filename = os.path.join(self.save_dir, 'raw_{}.jpg'.format(pid))

            im.save(filename)

    def module_forward_hook(self, hook_output=True):
        def hook_in(module, input, output):
            self.module_input = input

        def hook_out(module, input, output):
            self.module_output = output

        # get the selected module
        model = self.model
        for name in self.selected_module:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        # register a forward hook for the selected module
        assert isinstance(model, torch.nn.Module), \
            'register_forward_hook is an attribute of nn.Module'
        model.register_forward_hook(hook_out if hook_output else hook_in)

    def module_backward_hook(self, hook_output=True):
        def hook_in(module, grad_in, grad_out):
            # Gets the grad of input of the selected filter (from selected module)
            self.grad_in = grad_in[0]

        def hook_out(module, grad_in, grad_out):
            # Gets the grad output of the selected filter (from selected module)
            if self.selected_filter > 0:
                self.grad_out = grad_out[0, self.selected_filter]
            else:
                self.grad_out = grad_out

        # Hook the selected module
        model = self.model
        for name in self.selected_module:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        assert isinstance(model, torch.nn.Module), \
            'register_backward_hook is an attr of nn.Module'
        model.register_backward_hook(hook_out if hook_output else hook_in)

    def update_relu(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in list(self.model.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)

    def cv2im_transformer(self, cv2im, im_size=(256, 128),
                          requires_grad=False, norm=True, div=True):
        """ Resize, to_tensor and ImageNet normalized

        Args:
            cv2im: Image to process, 0-255, BGR, (H,W,C)
            im_size: (height, width) image size after resizing
        returns:
            im_as_ten(tensor): float tensor of shape 1CHW, 0-1
        """
        # mean and std for channels (ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # resize image
        cv2im = cv2.resize(cv2im, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)  # bilinear
        im_as_arr = np.array(cv2im)  # HWC
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])  # BGR to RGB
        # normalizing 0-1
        if div:
            im_as_arr = (im_as_arr / 255)  # HWC

        # normalize
        if norm:
            mean = np.array(mean)
            std = np.array(std)
            im_as_arr = (im_as_arr - mean) / std  # HWC

        im_as_ten = torch.tensor(im_as_arr, dtype=torch.float, requires_grad=requires_grad).contiguous()
        im_as_ten = im_as_ten.permute(2, 0, 1).contiguous()  # HWC to CHW
        im_as_ten.unsqueeze_(0)

        return im_as_ten

    def class_activation_map_on_im(self, org_im, activation_map, out_size=(128, 64)):
        '''
        Args:
            org_im: cv2 image, BGR, 0-255
            activation_map: target class activation map (grayscale) 0-255
            out_size: desired output image size (height width)

        Returns:
            numpy array of shape HWC, 0-255, uint8
        '''

        activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        # Heatmap on picture
        org_im = cv2.resize(org_im, (out_size[1], out_size[0]))
        im_with_heatmap = np.float32(activation_heatmap) + np.float32(org_im)
        im_with_heatmap = im_with_heatmap / np.max(im_with_heatmap)
        im_with_heatmap = im_with_heatmap[..., ::-1]  # BGR to RGB
        return np.uint8(255 * im_with_heatmap)

    def generate_bsl(self):
        # forward for getting feature maps
        self.model.eval()
        # register forward hook for the opt.selected_module to obtain the output of selected_module
        self.module_forward_hook()
        # Traversing person ids
        for ii, (pid, fpath) in enumerate(self.pid_to_fpath.items()):
            print('class activation map of {} ({}/{}) ...'.format(pid, ii + 1, len(self.pid_to_fpath)))
            # list of cv2 images of a pid
            original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
            # resize, to_tensor, normalize cv2im and concatenate as an input batch
            batch_images = torch.cat(
                [self.cv2im_transformer(x, im_size=self.in_size, norm=False, div=False) for x in original_ims],
                dim=0
            )
            batch_images = batch_images.cuda()
            # forward at eval mode
            _ = self.model(batch_images)

            w = self.weights[self.pid_to_lbl[pid]]  # dim_feat
            # weight = self.weights(torch.argmax(self.model.logit, dim=1))  # bs x dim_feat
            feat_maps = self.module_output  # NCHW
            # import pdb; pdb.set_trace()
            # Traversing images of a pid and record c
            cam_on_ims = []
            for original_im, feat_map in zip(original_ims, feat_maps):
                # (dim_feat h w) to (h w dim_feat) im
                feat_map = feat_map.permute(1, 2, 0).contiguous()
                # (h w dim_feat)*(dim_feat 1) to (h w 1) to (h w)
                score_map = feat_map.matmul(w.unsqueeze(dim=1)).squeeze(-1)
                # opt1
                # score_map = score_map * 0.05
                # score_map = torch.softmax(score_map.view(1, -1), dim=1).view(feat_map.size(0), -1)
                # opt2
                # stripe_mask, score_map = binary_conversion(score_map[None, :])
                # score_map = stripe_mask * score_map
                # score_map = score_map.squeeze(0)

                score_map = score_map.detach().cpu().numpy()
                score_map = cv2.resize(score_map, (self.out_size[1], self.out_size[0]))
                # score_map = np.maximum(score_map, 0) # for positive cam
                # score_map = np.minimum(score_map, 0) * (-1)  # for negative cam

                # Normalize between 0-1
                # opt3
                score_map = (score_map - np.min(score_map)) / (np.max(score_map) - np.min(score_map))

                score_map = np.uint8(score_map * 255)  # Scale between 0-255 to visualize
                cam_on_im = self.class_activation_map_on_im(original_im, score_map,
                                                            out_size=self.out_size)  # numpy array of shape HWC 0-255
                # torch.from_numpy keeps value range and dtype, thus cam_on_im.dtype=torch.uint8, i.e. torch.ByteTensor
                cam_on_ims.append(torch.from_numpy(cam_on_im.transpose(2, 0, 1)))  # HWC to CHW
                # cam_on_ims.append(torch.from_numpy(original_im[..., ::-1].transpose(2, 0, 1))) # will save original images of pid

            # the value range of make_grid input tensor can be 0-1 or 0-255, output grid has the same range as input
            grid = utils.make_grid(cam_on_ims, nrow=len(cam_on_ims), padding=2, normalize=False,
                                   range=None, scale_each=False, pad_value=255)
            ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            filename = os.path.join(self.save_dir, 'raw_{}.jpg'.format(pid))
            im.save(filename)

class GradClassActivMap(object):
    def __init__(self,
                 model,
                 train_items,
                 pid_to_lbl,
                 in_size,
                 vis_pids,
                 save_dir='',
                 out_size=[128, 64]):
        super().__init__()
        '''
        Args:
            model: as one of the models in fastreid/modeling/meta_arch/
            selected_module (str): e.g., 'backbone.layer4'
            selected_filter (int): target filter of the selected_module to hook,
                                   hook all the filters if given a negative number.
        '''
        self.model = model
        self.model.eval()

        self.in_size = in_size
        self.out_size = out_size
        self.save_dir = os.path.join(save_dir, 'CAMA-GradCAM')
        mkdir(self.save_dir)

        self.vis_pids = vis_pids  # person ids to visualize
        self.pid_to_lbl = pid_to_lbl
        self.lbl_to_pid = dict([(l, p) for p, l in self.pid_to_lbl.items()])

        # initialize
        assert hasattr(self.model.heads, 'classifier'), \
            'class activation map requires model to have a classifier'

        self.weights = [cls.weight for cls in self.model.heads.classifier]  # (num_classes, dim_feat)

        # collect image paths of vis_pids
        self.pid_to_fpath = defaultdict(list)
        for i, item in enumerate(train_items):
            # item(tuple): (img_path, pid, camid), pid = dataset_name + "_" + str(pid)
            pid = item[1]
            if pid in self.vis_pids:
                self.pid_to_fpath[pid].append(item[0])

        self.module_output = None
        self.module_input = None

    def hook_tensor(self, grad):
        self.gradients = grad

    def generate(self):
        # forward for getting feature maps
        self.model.eval()
        # Traversing person ids
        for ii, (pid, fpath) in enumerate(self.pid_to_fpath.items()):
            print('class activation map of {} ({}/{}) ...'.format(pid, ii + 1, len(self.pid_to_fpath)))
            # list of cv2 images of a pid
            original_ims = [cv2.imread(x, cv2.IMREAD_COLOR) for x in fpath]
            # resize, to_tensor, normalize cv2im and concatenate as an input batch
            batch_images = torch.cat(
                [self.cv2im_transformer(x, im_size=self.in_size, norm=False, div=False) for x in original_ims],
                dim=0
            )
            batch_images = batch_images.cuda()
            # forward at eval mode
            images = self.model.preprocess_image(batch_images)
            branches_feat_maps = self.model.local_branch(self.model.backbone(images))  # List(Tensor(b c h w)) len=num_branch
            global_feat = [self.model.heads.pool_layer(f) for f in branches_feat_maps]
            global_feat = [neck(f) for neck, f in zip(self.model.heads.bottleneck, global_feat)]
            global_feat = [f[..., 0, 0] for f in global_feat]
            logits = [F.linear(f, cls) for f, cls in zip(global_feat, self.weights)]  # (b num_classes)

            branches_ws = [w[self.pid_to_lbl[pid]] for w in self.weights]  # List[Tensor(d)]
            # weight = self.weights(torch.argmax(self.model.logit, dim=1))  # bs x dim_feat
            assert len(branches_ws) == len(branches_feat_maps)

            # Traversing branches
            branches_cam_on_ims = []  # len=num_branch*batch_size
            for feat_maps, w in zip(branches_feat_maps, branches_ws):
                # Traversing images of a pid and record c
                cam_on_ims = []
                for original_im, feat_map in zip(original_ims, feat_maps):
                    # (dim_feat h w) to (h w dim_feat) im
                    feat_map = feat_map.permute(1, 2, 0).contiguous()
                    # (h w dim_feat)*(dim_feat 1) to (h w 1) to (h w)
                    score_map = feat_map.matmul(w.unsqueeze(dim=1)).squeeze(-1)

                    # opt1
                    # score_map = feat_map.norm(p=2, dim=2) # (h w)
                    # opt2
                    # stripe_mask, score_map = _binary_conversion(score_map[None, :])
                    # score_map = stripe_mask * score_map
                    # score_map = score_map.squeeze(0)
                    # opt3
                    # stripe_mask, score_map = binary_conversion(score_map[None, None, :, :], self.model.smooth_filter)
                    # score_map = stripe_mask * score_map
                    # score_map = score_map[0, 0, ...]

                    score_map = score_map.detach().cpu().numpy()
                    score_map = cv2.resize(score_map, (self.out_size[1], self.out_size[0]))
                    # score_map = np.maximum(score_map, 0) # for positive cam
                    # score_map = np.minimum(score_map, 0) * (-1)  # for negative cam

                    # Normalize between 0-1
                    score_map = (score_map - np.min(score_map)) / (np.max(score_map) - np.min(score_map))

                    score_map = np.uint8(score_map * 255)  # Scale between 0-255 to visualize
                    cam_on_im = self.class_activation_map_on_im(original_im, score_map,
                                                                out_size=self.out_size)  # numpy array of shape HWC 0-255
                    # torch.from_numpy keeps value range and dtype, thus cam_on_im.dtype=torch.uint8, i.e. torch.ByteTensor
                    cam_on_ims.append(torch.from_numpy(cam_on_im.transpose(2, 0, 1)))  # HWC to CHW
                    # cam_on_ims.append(torch.from_numpy(original_im[..., ::-1].transpose(2, 0, 1))) # will save original images of pid
                branches_cam_on_ims += cam_on_ims

            # the value range of make_grid input tensor can be 0-1 or 0-255, output grid has the same range as input
            grid = utils.make_grid(branches_cam_on_ims, nrow=batch_images.size(0), padding=2, normalize=False,
                                   range=None, scale_each=False, pad_value=255)
            ndarr = grid.mul(1).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            im = Image.fromarray(ndarr)
            filename = os.path.join(self.save_dir, 'raw_{}.jpg'.format(pid))

            im.save(filename)

    def module_forward_hook(self, hook_output=True):
        def hook_in(module, input, output):
            self.module_input = input

        def hook_out(module, input, output):
            self.module_output = output

        # get the selected module
        model = self.model
        for name in self.selected_module:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        # register a forward hook for the selected module
        assert isinstance(model, torch.nn.Module), \
            'register_forward_hook is an attribute of nn.Module'
        model.register_forward_hook(hook_out if hook_output else hook_in)

    def module_backward_hook(self, hook_output=True):
        def hook_in(module, grad_in, grad_out):
            # Gets the grad of input of the selected filter (from selected module)
            self.grad_in = grad_in[0]

        def hook_out(module, grad_in, grad_out):
            # Gets the grad output of the selected filter (from selected module)
            if self.selected_filter > 0:
                self.grad_out = grad_out[0, self.selected_filter]
            else:
                self.grad_out = grad_out

        # Hook the selected module
        model = self.model
        for name in self.selected_module:
            assert hasattr(model, name), \
                '{} has no attr {}'.format(model.__class__.__name__, name)
            model = getattr(model, name)
            if isinstance(model, torch.nn.DataParallel):
                model = model.module

        assert isinstance(model, torch.nn.Module), \
            'register_backward_hook is an attr of nn.Module'
        model.register_backward_hook(hook_out if hook_output else hook_in)

    def update_relu(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in list(self.model.modules()):
            if isinstance(module, torch.nn.ReLU):
                module.register_backward_hook(relu_hook_function)

    def cv2im_transformer(self, cv2im, im_size=(256, 128),
                          requires_grad=False, norm=True, div=True):
        """ Resize, to_tensor and ImageNet normalized

        Args:
            cv2im: Image to process, 0-255, BGR, (H,W,C)
            im_size: (height, width) image size after resizing
        returns:
            im_as_ten(tensor): float tensor of shape 1CHW, 0-1
        """
        # mean and std for channels (ImageNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # resize image
        cv2im = cv2.resize(cv2im, (im_size[1], im_size[0]), interpolation=cv2.INTER_LINEAR)  # bilinear
        im_as_arr = np.array(cv2im)  # HWC
        im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])  # BGR to RGB
        # normalizing 0-1
        if div:
            im_as_arr = (im_as_arr / 255)  # HWC

        # normalize
        if norm:
            mean = np.array(mean)
            std = np.array(std)
            im_as_arr = (im_as_arr - mean) / std  # HWC

        im_as_ten = torch.tensor(im_as_arr, dtype=torch.float, requires_grad=requires_grad).contiguous()
        im_as_ten = im_as_ten.permute(2, 0, 1).contiguous()  # HWC to CHW
        im_as_ten.unsqueeze_(0)

        return im_as_ten

    def class_activation_map_on_im(self, org_im, activation_map, out_size=(128, 64)):
        '''
        Args:
            org_im: cv2 image, BGR, 0-255
            activation_map: target class activation map (grayscale) 0-255
            out_size: desired output image size (height width)

        Returns:
            numpy array of shape HWC, 0-255, uint8
        '''

        activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        # Heatmap on picture
        org_im = cv2.resize(org_im, (out_size[1], out_size[0]))
        im_with_heatmap = np.float32(activation_heatmap) + np.float32(org_im)
        im_with_heatmap = im_with_heatmap / np.max(im_with_heatmap)
        im_with_heatmap = im_with_heatmap[..., ::-1]  # BGR to RGB
        return np.uint8(255 * im_with_heatmap)
