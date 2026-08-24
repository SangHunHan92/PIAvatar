import os, sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
sys.path.append("AnimatableGaussians")
sys.path.append("gaussian-splatting")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trimesh.geometry
import yaml
import shutil
import collections
import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import glob
import datetime
import trimesh
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import importlib
import pytorch3d
from plyfile import PlyData, PlyElement
import faiss
import math

from AnimatableGaussians import config
from network.lpips import LPIPS
from dataset.dataset_pose import PoseDataset
import utils.net_util as net_util
import utils.visualize_util as visualize_util
from utils.renderer import Renderer
from utils.net_util import to_cuda
from utils.obj_io import save_mesh_as_ply
from gaussians.obj_io import save_gaussians_as_ply

os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"

def safe_exists(path):
    if path is None:
        return False
    return os.path.exists(path)


class AvatarTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.patch_size = 512
        self.iter_idx = 0
        self.iter_num = 800000
        self.lr_init = float(self.opt['train'].get('lr_init', 5e-4))
        self.live_pos_map = None

        avatar_module = self.opt['model'].get('module', 'network.avatar')
        print('Import AvatarNet from %s' % avatar_module)
        AvatarNet = importlib.import_module(avatar_module).AvatarNet
        self.avatar_net = AvatarNet(self.opt['model']).to(config.device)
        self.optm = torch.optim.Adam(
            self.avatar_net.parameters(), lr = self.lr_init
        )

        self.random_bg_color = self.opt['train'].get('random_bg_color', True)
        self.bg_color = (1., 1., 1.)
        self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
        self.loss_weight = self.opt['train']['loss_weight']
        self.finetune_color = self.opt['train']['finetune_color']

        print('# Parameter number of AvatarNet is %d' % (sum([p.numel() for p in self.avatar_net.parameters()])))

    def update_lr(self):
        alpha = 0.05
        progress = self.iter_idx / self.iter_num
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
        lr = self.lr_init * learning_factor
        for param_group in self.optm.param_groups:
            param_group['lr'] = lr
        return lr

    @staticmethod
    def requires_net_grad(net: torch.nn.Module, flag = True):
        for p in net.parameters():
            p.requires_grad = flag

    def crop_image(self, gt_mask, patch_size, randomly, *args):
        """
        :param gt_mask: (H, W)
        :param patch_size: resize the cropped patch to the given patch_size
        :param randomly: whether to randomly sample the patch
        :param args: input images with shape of (C, H, W)
        """
        mask_uv = torch.argwhere(gt_mask > 0.)
        min_v, min_u = mask_uv.min(0)[0]
        max_v, max_u = mask_uv.max(0)[0]
        len_v = max_v - min_v
        len_u = max_u - min_u
        max_size = max(len_v, len_u)

        cropped_images = []
        if randomly and max_size > patch_size:
            random_v = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
            random_u = torch.randint(0, max_size - patch_size + 1, (1,)).to(max_size)
        for image in args:
            cropped_image = self.bg_color_cuda[:, None, None] * torch.ones((3, max_size, max_size), dtype = image.dtype, device = image.device)
            if len_v > len_u:
                start_u = (max_size - len_u) // 2
                cropped_image[:, :, start_u: start_u + len_u] = image[:, min_v: max_v, min_u: max_u]
            else:
                start_v = (max_size - len_v) // 2
                cropped_image[:, start_v: start_v + len_v, :] = image[:, min_v: max_v, min_u: max_u]

            if randomly and max_size > patch_size:
                cropped_image = cropped_image[:, random_v: random_v + patch_size, random_u: random_u + patch_size]
            else:
                cropped_image = F.interpolate(cropped_image[None], size = (patch_size, patch_size), mode = 'bilinear')[0]
            cropped_images.append(cropped_image)

        # cv.imshow('cropped_image', cropped_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.imshow('cropped_gt_image', cropped_gt_image.detach().cpu().numpy().transpose(1, 2, 0))
        # cv.waitKey(0)

        if len(cropped_images) > 1:
            return cropped_images
        else:
            return cropped_images[0]

    def compute_lpips_loss(self, image, gt_image):
        assert image.shape[1] == image.shape[2] and gt_image.shape[1] == gt_image.shape[2]
        lpips_loss = self.lpips.forward(
            image[None, [2, 1, 0]],
            gt_image[None, [2, 1, 0]],
            normalize = True
        ).mean()
        return lpips_loss

    def forward_one_pass_pretrain(self, items):
        total_loss = 0
        batch_losses = {}
        l1_loss = torch.nn.L1Loss()

        items = net_util.delete_batch_idx(items)
        pose_map = items['smpl_pos_map'][:3]

        position_loss = l1_loss(self.avatar_net.get_positions(pose_map), self.avatar_net.cano_gaussian_model.get_xyz)
        total_loss += position_loss
        batch_losses.update({
            'position': position_loss.item()
        })

        opacity, scales, rotations = self.avatar_net.get_others(pose_map)
        opacity_loss = l1_loss(opacity, self.avatar_net.cano_gaussian_model.get_opacity)
        total_loss += opacity_loss
        batch_losses.update({
            'opacity': opacity_loss.item()
        })

        scale_loss = l1_loss(scales, self.avatar_net.cano_gaussian_model.get_scaling)
        total_loss += scale_loss
        batch_losses.update({
            'scale': scale_loss.item()
        })

        rotation_loss = l1_loss(rotations, self.avatar_net.cano_gaussian_model.get_rotation)
        total_loss += rotation_loss
        batch_losses.update({
            'rotation': rotation_loss.item()
        })

        total_loss.backward()

        self.optm.step()
        self.optm.zero_grad()

        return total_loss, batch_losses

    def forward_one_pass(self, items):
        # forward_start = torch.cuda.Event(enable_timing = True)
        # forward_end = torch.cuda.Event(enable_timing = True)
        # backward_start = torch.cuda.Event(enable_timing = True)
        # backward_end = torch.cuda.Event(enable_timing = True)
        # step_start = torch.cuda.Event(enable_timing = True)
        # step_end = torch.cuda.Event(enable_timing = True)

        if self.random_bg_color:
            self.bg_color = np.random.rand(3)
            self.bg_color_cuda = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)

        total_loss = 0
        batch_losses = {}

        items = net_util.delete_batch_idx(items)

        """ Optimize generator """
        if self.finetune_color:
            self.requires_net_grad(self.avatar_net.color_net, True)
            self.requires_net_grad(self.avatar_net.position_net, False)
            self.requires_net_grad(self.avatar_net.other_net, True)
        else:
            self.requires_net_grad(self.avatar_net, True)

        # forward_start.record()
        render_output = self.avatar_net.render(items, self.bg_color)
        image = render_output['rgb_map'].permute(2, 0, 1)
        offset = render_output['offset']

        # mask image & set bg color
        items['color_img'][~items['mask_img']] = self.bg_color_cuda
        gt_image = items['color_img'].permute(2, 0, 1)
        mask_img = items['mask_img'].to(torch.float32)
        boundary_mask_img = 1. - items['boundary_mask_img'].to(torch.float32)
        image = image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * self.bg_color_cuda[:, None, None]
        gt_image = gt_image * boundary_mask_img[None] + (1. - boundary_mask_img[None]) * self.bg_color_cuda[:, None, None]
        # cv.imshow('image', image.detach().permute(1, 2, 0).cpu().numpy())
        # cv.imshow('gt_image', gt_image.permute(1, 2, 0).cpu().numpy())
        # cv.waitKey(0)

        if self.loss_weight['l1'] > 0.:
            l1_loss = torch.abs(image - gt_image).mean()
            total_loss += self.loss_weight['l1'] * l1_loss
            batch_losses.update({
                'l1_loss': l1_loss.item()
            })

        if self.loss_weight.get('mask', 0.) and 'mask_map' in render_output:
            rendered_mask = render_output['mask_map'].squeeze(-1) * boundary_mask_img
            gt_mask = mask_img * boundary_mask_img
            # cv.imshow('rendered_mask', rendered_mask.detach().cpu().numpy())
            # cv.imshow('gt_mask', gt_mask.detach().cpu().numpy())
            # cv.waitKey(0)
            mask_loss = torch.abs(rendered_mask - gt_mask).mean()
            # mask_loss = torch.nn.BCELoss()(rendered_mask, gt_mask)
            total_loss += self.loss_weight.get('mask', 0.) * mask_loss
            batch_losses.update({
                'mask_loss': mask_loss.item()
            })

        if self.loss_weight['lpips'] > 0.:
            # crop images
            random_patch_flag = False if self.iter_idx < 300000 else True
            image, gt_image = self.crop_image(mask_img, self.patch_size, random_patch_flag, image, gt_image)
            # cv.imshow('image', image.detach().permute(1, 2, 0).cpu().numpy())
            # cv.imshow('gt_image', gt_image.permute(1, 2, 0).cpu().numpy())
            # cv.waitKey(0)
            lpips_loss = self.compute_lpips_loss(image, gt_image)
            total_loss += self.loss_weight['lpips'] * lpips_loss
            batch_losses.update({
                'lpips_loss': lpips_loss.item()
            })

        # if self.loss_weight['offset'] > 0.:
        if True:
            offset_loss = torch.linalg.norm(offset, dim = -1).mean()
            total_loss += self.loss_weight['offset'] * offset_loss
            batch_losses.update({
                'offset_loss': offset_loss.item()
            })

        # forward_end.record()

        # backward_start.record()
        total_loss.backward()
        # backward_end.record()

        # step_start.record()
        self.optm.step()
        self.optm.zero_grad()
        # step_end.record()

        # torch.cuda.synchronize()
        # print(f'Forward costs: {forward_start.elapsed_time(forward_end) / 1000.}, ',
        #       f'Backward costs: {backward_start.elapsed_time(backward_end) / 1000.}, ',
        #       f'Step costs: {step_start.elapsed_time(step_end) / 1000.}')

        return total_loss, batch_losses

    def pretrain(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        # tb writer
        log_dir = self.opt['train']['net_ckpt_dir'] + '/' + datetime.datetime.now().strftime('pretrain_%Y_%m_%d_%H_%M_%S')
        writer = SummaryWriter(log_dir)
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}

        for epoch_idx in range(0, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                self.iter_idx = batch_idx + epoch_idx * batch_num
                items = to_cuda(items)

                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass_pretrain(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, ' % (epoch_idx, batch_idx, self.iter_idx)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        writer.add_scalar('%s/Iter' % key, smooth_losses[key], self.iter_idx)
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                        fp.write(log_info + '\n')

                if self.iter_idx % 200 == 0 and self.iter_idx != 0:
                    self.mini_test(pretraining = True)

                if self.iter_idx == 5000:
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/pretrained'
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)
                    self.iter_idx = 0
                    return

    def train(self):
        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        self.dataset = MvRgbDataset(**self.opt['train']['data'])
        batch_size = self.opt['train']['batch_size']
        num_workers = self.opt['train']['num_workers']
        batch_num = len(self.dataset) // batch_size
        dataloader = torch.utils.data.DataLoader(self.dataset,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers = num_workers,
                                                 drop_last = True)

        if 'lpips' in self.opt['train']['loss_weight']:
            self.lpips = LPIPS(net = 'vgg').to(config.device)
            for p in self.lpips.parameters():
                p.requires_grad = False

        if self.opt['train']['prev_ckpt'] is not None:
            start_epoch, self.iter_idx = self.load_ckpt(self.opt['train']['prev_ckpt'], load_optm = True)
            start_epoch += 1
            self.iter_idx += 1
        else:
            prev_ckpt_path = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
            if safe_exists(prev_ckpt_path):
                start_epoch, self.iter_idx = self.load_ckpt(prev_ckpt_path, load_optm = True)
                start_epoch += 1
                self.iter_idx += 1
            else:
                if safe_exists(self.opt['train']['pretrained_dir']):
                    self.load_ckpt(self.opt['train']['pretrained_dir'], load_optm = False)
                elif safe_exists(self.opt['train']['net_ckpt_dir'] + '/pretrained'):
                    self.load_ckpt(self.opt['train']['net_ckpt_dir'] + '/pretrained', load_optm = False)
                else:
                    raise FileNotFoundError('Cannot find pretrained checkpoint!')

                self.optm.state = collections.defaultdict(dict)
                start_epoch = 0
                self.iter_idx = 0

        # one_step_start = torch.cuda.Event(enable_timing = True)
        # one_step_end = torch.cuda.Event(enable_timing = True)

        # tb writer
        log_dir = self.opt['train']['net_ckpt_dir'] + '/' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        writer = SummaryWriter(log_dir)
        yaml.dump(self.opt, open(log_dir + '/config_bk.yaml', 'w'), sort_keys = False)
        smooth_interval = 10
        smooth_count = 0
        smooth_losses = {}

        for epoch_idx in range(start_epoch, 9999999):
            self.epoch_idx = epoch_idx
            for batch_idx, items in enumerate(dataloader):
                lr = self.update_lr()

                items = to_cuda(items)

                # one_step_start.record()
                total_loss, batch_losses = self.forward_one_pass(items)
                # one_step_end.record()
                # torch.cuda.synchronize()
                # print('One step costs %f secs' % (one_step_start.elapsed_time(one_step_end) / 1000.))

                # record batch loss
                for key, loss in batch_losses.items():
                    if key in smooth_losses:
                        smooth_losses[key] += loss
                    else:
                        smooth_losses[key] = loss
                smooth_count += 1

                if self.iter_idx % smooth_interval == 0:
                    log_info = 'epoch %d, batch %d, iter %d, lr %e, ' % (epoch_idx, batch_idx, self.iter_idx, lr)
                    for key in smooth_losses.keys():
                        smooth_losses[key] /= smooth_count
                        writer.add_scalar('%s/Iter' % key, smooth_losses[key], self.iter_idx)
                        log_info = log_info + ('%s: %f, ' % (key, smooth_losses[key]))
                        smooth_losses[key] = 0.
                    smooth_count = 0
                    print(log_info)
                    with open(os.path.join(log_dir, 'loss.txt'), 'a') as fp:
                        fp.write(log_info + '\n')
                    torch.cuda.empty_cache()

                if self.iter_idx % self.opt['train']['eval_interval'] == 0 and self.iter_idx != 0:
                    if self.iter_idx % (10 * self.opt['train']['eval_interval']) == 0:
                        eval_cano_pts = True
                    else:
                        eval_cano_pts = False
                    self.mini_test(eval_cano_pts = eval_cano_pts)

                if self.iter_idx % self.opt['train']['ckpt_interval']['batch'] == 0 and self.iter_idx != 0:
                    for folder in glob.glob(self.opt['train']['net_ckpt_dir'] + '/batch_*'):
                        shutil.rmtree(folder)
                    model_folder = self.opt['train']['net_ckpt_dir'] + '/batch_%d' % self.iter_idx
                    os.makedirs(model_folder, exist_ok = True)
                    self.save_ckpt(model_folder, save_optm = True)

                if self.iter_idx == self.iter_num:
                    print('# Training is done.')
                    return

                self.iter_idx += 1

            """ End of epoch """
            if epoch_idx % self.opt['train']['ckpt_interval']['epoch'] == 0 and epoch_idx != 0:
                model_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_%d' % epoch_idx
                os.makedirs(model_folder, exist_ok = True)
                self.save_ckpt(model_folder)

            if batch_num > 50:
                latest_folder = self.opt['train']['net_ckpt_dir'] + '/epoch_latest'
                os.makedirs(latest_folder, exist_ok = True)
                self.save_ckpt(latest_folder)

    @torch.no_grad()
    def mini_test(self, pretraining = False, eval_cano_pts = False):
        self.avatar_net.eval()

        img_factor = self.opt['train'].get('eval_img_factor', 1.0)
        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_training_ids', (310, 19))
        intr = self.dataset.intr_mats[view_idx].copy()
        intr[:2] *= img_factor
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)

        gs_render = self.avatar_net.render(items, self.bg_color)
        # gs_render = self.avatar_net.render_debug(items)
        rgb_map = gs_render['rgb_map']
        rgb_map.clip_(0., 1.)
        rgb_map = (rgb_map.cpu().numpy() * 255).astype(np.uint8)
        # cv.imshow('rgb_map', rgb_map.cpu().numpy())
        # cv.waitKey(0)
        if not pretraining:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/training'
        else:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval_pretrain/training'
        gt_image, _ = self.dataset.load_color_mask_images(pose_idx, view_idx)
        if gt_image is not None:
            gt_image = cv.resize(gt_image, (0, 0), fx = img_factor, fy = img_factor)
            rgb_map = np.concatenate([rgb_map, gt_image], 1)
        os.makedirs(output_dir, exist_ok = True)
        cv.imwrite(output_dir + '/iter_%d.jpg' % self.iter_idx, rgb_map)
        if eval_cano_pts:
            os.makedirs(output_dir + '/cano_pts', exist_ok = True)
            save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % self.iter_idx, (self.avatar_net.init_points + gs_render['offset']).cpu().numpy())

        # training data
        pose_idx, view_idx = self.opt['train'].get('eval_testing_ids', (310, 19))
        intr = self.dataset.intr_mats[view_idx].copy()
        intr[:2] *= img_factor
        item = self.dataset.getitem(0,
                                    pose_idx = pose_idx,
                                    view_idx = view_idx,
                                    training = False,
                                    eval = True,
                                    img_h = int(self.dataset.img_heights[view_idx] * img_factor),
                                    img_w = int(self.dataset.img_widths[view_idx] * img_factor),
                                    extr = self.dataset.extr_mats[view_idx],
                                    intr = intr,
                                    exact_hand_pose = True)
        items = net_util.to_cuda(item, add_batch = False)

        gs_render = self.avatar_net.render(items, bg_color = self.bg_color)
        # gs_render = self.avatar_net.render_debug(items)
        rgb_map = gs_render['rgb_map']
        rgb_map.clip_(0., 1.)
        rgb_map = (rgb_map.cpu().numpy() * 255).astype(np.uint8)
        # cv.imshow('rgb_map', rgb_map.cpu().numpy())
        # cv.waitKey(0)
        if not pretraining:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval/testing'
        else:
            output_dir = self.opt['train']['net_ckpt_dir'] + '/eval_pretrain/testing'
        gt_image, _ = self.dataset.load_color_mask_images(pose_idx, view_idx)
        if gt_image is not None:
            gt_image = cv.resize(gt_image, (0, 0), fx = img_factor, fy = img_factor)
            rgb_map = np.concatenate([rgb_map, gt_image], 1)
        os.makedirs(output_dir, exist_ok = True)
        cv.imwrite(output_dir + '/iter_%d.jpg' % self.iter_idx, rgb_map)
        if eval_cano_pts:
            os.makedirs(output_dir + '/cano_pts', exist_ok = True)
            save_mesh_as_ply(output_dir + '/cano_pts/iter_%d.ply' % self.iter_idx, (self.avatar_net.init_points + gs_render['offset']).cpu().numpy())

        self.avatar_net.train()

    def construct_list_of_attributes(self, gs):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(gs['f_dc'].shape[1]):
            l.append('f_dc_{}'.format(i))
        for i in range(gs['f_rest'].shape[1]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(gs['scale'].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(gs['rotation'].shape[1]):
            l.append('rot_{}'.format(i))
        return l
            
    def save_ply(self, gaussian_vals, path):
        gs = dict()
        gs['xyz'] = gaussian_vals['positions'].cpu().numpy()
        gs['normals'] = np.zeros_like(gs['xyz'])
        gs['f_dc'] = gaussian_vals['colors'].detach().unsqueeze(0).permute(1, 0, 2).flatten(start_dim=1).contiguous().cpu().numpy() # (N, 1, 3)
        gs['f_rest'] = np.zeros([gs['xyz'].shape[0], 45]) # (N, 15, 3)
        gs['opacities'] = gaussian_vals['opacity'].detach().cpu().numpy()
        gs['scale'] = gaussian_vals['scales'].detach().cpu().numpy()
        gs['rotation'] = gaussian_vals['rotations'].detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(gs)]
        elements = np.empty(gs['xyz'].shape[0], dtype=dtype_full)
        attributes = np.concatenate((gs['xyz'], gs['normals'], gs['f_dc'], gs['f_rest'], gs['opacities'], gs['scale'], gs['rotation']), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    @torch.no_grad()
    def test(self):
        self.avatar_net.eval()

        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
        if self.opt['test'].get('n_pca', -1) >= 1:
            training_dataset.compute_pca(n_components = self.opt['test']['n_pca'])
        if 'pose_data' in self.opt['test']:
            testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
            dataset_name = testing_dataset.dataset_name
            seq_name = testing_dataset.seq_name
        else:
            testing_dataset = MvRgbDataset(**self.opt['test']['data'], training = False)
            dataset_name = 'training'
            seq_name = ''

        self.dataset = testing_dataset
        iter_idx = self.load_ckpt(self.opt['test']['prev_ckpt'], False)[1]

        output_dir = self.opt['test'].get('output_dir', None)
        if output_dir is None:
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'camera':
                view_folder = 'cam_%03d' % config.opt['test']['render_view_idx']
            else:
                view_folder = view_setting + '_view'
            exp_name = os.path.basename(os.path.dirname(self.opt['test']['prev_ckpt']))
            output_dir = f'./test_results/{training_dataset.subject_name}/{exp_name}/{dataset_name}_{seq_name}_{view_folder}' + '/batch_%06d' % iter_idx

        use_pca = self.opt['test'].get('n_pca', -1) >= 1
        if use_pca:
            output_dir += '/pca_%d_sigma_%.2f' % (self.opt['test'].get('n_pca', -1), float(self.opt['test'].get('sigma_pca', 1.)))
        else:
            output_dir += '/vanilla'
        print('# Output dir: \033[1;31m%s\033[0m' % output_dir)

        os.makedirs(output_dir + '/live_skeleton', exist_ok = True)
        os.makedirs(output_dir + '/rgb_map', exist_ok = True)
        os.makedirs(output_dir + '/mask_map', exist_ok = True)

        geo_renderer = None
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]
        # print('object_center: ', object_center.tolist())
        # print('global_orient: ', global_orient.tolist())
        # # exit(1)

        time_start = torch.cuda.Event(enable_timing = True)
        time_start_all = torch.cuda.Event(enable_timing = True)
        time_end = torch.cuda.Event(enable_timing = True)

        data_num = len(self.dataset)
        if self.opt['test'].get('fix_hand', False):
            self.avatar_net.generate_mean_hands()
        log_time = False

        for idx in tqdm(range(data_num), desc = 'Rendering avatars...'):
            if log_time:
                time_start.record()
                time_start_all.record()

            img_scale = self.opt['test'].get('img_scale', 1.0)
            view_setting = config.opt['test'].get('view_setting', 'free')
            if view_setting == 'camera':
                # training view setting
                cam_id = config.opt['test']['render_view_idx']
                intr = self.dataset.intr_mats[cam_id].copy()
                intr[:2] *= img_scale
                extr = self.dataset.extr_mats[cam_id].copy()
                img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)
            elif view_setting.startswith('free'):
                # free view setting
                # frame_num_per_circle = 360
                frame_num_per_circle = 216
                rot_Y = (idx % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = rot_Y,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('front'):
                # front view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = 0.,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('back'):
                # back view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = np.pi,
                                                   rot_X = 0.5 * np.pi / 4. if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('moving'):
                # moving camera setting
                extr = visualize_util.calc_free_mv(object_center,
                                                   # tar_pos = np.array([0, 0, 3.0]),
                                                   # rot_Y = -0.3,
                                                   tar_pos = np.array([0, 0, 2.5]),
                                                   rot_Y = 0.,
                                                   rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                   global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('cano'):
                cano_center = self.dataset.cano_bounds.mean(0)
                extr = np.identity(4, np.float32)
                extr[:3, 3] = -cano_center
                rot_x = np.identity(4, np.float32)
                rot_x[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
                extr = rot_x @ extr
                f_len = 5000
                extr[2, 3] += f_len / 512
                intr = np.array([[f_len, 0, 512], [0, f_len, 512], [0, 0, 1]], np.float32)
                # item = self.dataset.getitem(idx,
                #                             training = False,
                #                             extr = extr,
                #                             intr = intr,
                #                             img_w = 1024,
                #                             img_h = 1024)
                img_w, img_h = 1024, 1024
                # item['live_smpl_v'] = item['cano_smpl_v']
                # item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
                # item['live_bounds'] = item['cano_bounds']
            else:
                raise ValueError('Invalid view setting for animation!')

            #######################################################################################
            
            # getitem_func = self.dataset.getitem_fast
            
            # 1
            import time
            start = time.time()
            pose_idx = self.dataset.pose_list[idx]
            live_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                            global_orient = self.dataset.body_poses[pose_idx, :3][None], ##
                                            transl = self.dataset.transl[pose_idx][None],                ##
                                            body_pose = self.dataset.body_poses[pose_idx, 3: 66][None],
                                            left_hand_pose = self.dataset.left_hand_pose[pose_idx][None].to(config.device), ##
                                            right_hand_pose = self.dataset.right_hand_pose[pose_idx][None].to(config.device) ##
                                            )
            
            live_smpl_woRoot = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                            body_pose = self.dataset.body_poses[pose_idx, 3: 66][None],
                                            )
            
            data_item = dict()
            inv_cano_jnt_mats = torch.linalg.inv(self.dataset.cano_smpl['A']) # [55, 4, 4]
            data_item['extr'] = torch.from_numpy(extr).to(config.device)
            data_item['intr'] = torch.from_numpy(intr).to(config.device) # for render
            data_item['img_w'] = img_w                                   # for render
            data_item['img_h'] = img_h                                   # for render
            data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4]
            data_item['cano2live_jnt_mats_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
            
            # 2
            if 'smpl_pos_map' not in data_item:
                # data_item = self.avatar_net.get_pose_map(data_item)
                pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_woRoot'])
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # [N, 3]
                live_pos_map = torch.zeros_like(self.avatar_net.cano_smpl_map) # [1024, 2048, 3]
                live_pos_map[self.avatar_net.cano_smpl_mask] = live_pts # self.avatar_net.cano_smpl_mask.shape : [1024, 2048], boolean, sum = N
                live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0] # [3, 512, 1024]
                live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0) # [6, 512, 512]
                data_item.update({
                    'smpl_pos_map': live_pos_map # [6, 512, 512]
                })
               
            # 2.1 
            if use_pca:
                mask = training_dataset.pos_map_mask # same
                live_pos_map = data_item['smpl_pos_map'].permute(1, 2, 0).cpu().numpy() # diff
                front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
                pose_conds = front_live_pos_map[mask] # same
                new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(self.opt['test'].get('sigma_pca', 2.)))
                front_live_pos_map[mask] = new_pose_conds
                live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
                data_item.update({
                    'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
                })
                
            # 3
            # output = self.avatar_net.render(items, bg_color = self.bg_color, use_pca = use_pca)
            bg_color = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
            pose_map = data_item['smpl_pos_map'][:3]
            if use_pca:
                pose_map = data_item['smpl_pos_map_pca'][:3]
            
            # cano_pts, pos_map = self.get_positions(pose_map, return_map = True)
            # input : pose_map [3, 512, 512], output : [1, 6, 1024, 1024]
            pos_map, _ = self.avatar_net.position_net([self.avatar_net.position_style], pose_map[None], randomize_noise = False) # net
            front_position_map, back_position_map = torch.split(pos_map, [3, 3], 1)
            pos_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
            
            if pose_idx == 0:
                delta_position = 0.05 * pos_map[self.avatar_net.cano_smpl_mask] # [373056, 3]
                cano_pts = delta_position + self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3]
            # cano_pts = self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3], delta_position이 필요하긴하다
            
            # self.avatar_net.cano_gaussian_model.save_ply(f'./test/cano.ply')
                        
            # cano_pts, pos_map
            if pose_idx == 0:
                opacity, scales, rotations = self.avatar_net.get_others(pose_map) # net
            
            if self.avatar_net.with_viewdirs:
                # front_viewdirs, back_viewdirs = self.avatar_net.get_viewdir_feat(data_item)
                with torch.no_grad():
                    pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats']) # (N, 4, 4)
                    live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # (N, 3)
                    live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.cano_nmls) # (N, 3)
                    cam_pos = -torch.matmul(torch.linalg.inv(data_item['extr'][:3, :3]), data_item['extr'][:3, 3]) # [3]
                    viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3) # (N, 3)
                    # if self.training:
                    #     viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
                    viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3) # (N, 3)
                    viewdirs = (live_nmls * viewdirs).sum(-1) # (N)

                    viewdirs_map = torch.zeros(*self.avatar_net.cano_nml_map.shape[:2]).to(viewdirs) # [1024, 2048]
                    viewdirs_map[self.avatar_net.cano_smpl_mask] = viewdirs # [1024, 2048]

                    viewdirs_map = viewdirs_map[None, None] # [1, 1, 1024, 2048]
                    viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest') # [1, 1, 512, 1024]
                    front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

                front_viewdirs = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(front_viewdirs)
                back_viewdirs  = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(back_viewdirs)
            
            if pose_idx == 0:
                colors, color_map = self.avatar_net.get_colors(pose_map, front_viewdirs, back_viewdirs)
            
            import copy
            if pose_idx == 0:
                gaussian_vals_main = {
                    'positions': cano_pts,
                    'opacity': opacity,
                    'scales': scales,
                    'rotations': rotations,
                    'colors': colors,
                    'max_sh_degree': self.avatar_net.max_sh_degree
                }
                gaussian_vals = copy.deepcopy(gaussian_vals_main)
            else:
                gaussian_vals = copy.deepcopy(gaussian_vals_main)
            
            # gaussian_vals = self.avatar_net.transform_cano2live(gaussian_vals, data_item)            
            pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats'])
            gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions']) + pt_mats[..., :3, 3]
            rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations'])
            rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
            gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats)
            
            # print(cano_pts.shape) # [N, 3], N은 일정하다
            # print(time.time() - start)   
                
            # if idx==337:
            #     save_path = f'./test/{idx:02d}.ply'
            #     self.save_ply(gaussian_vals, save_path)
            
            from gaussians.gaussian_renderer import render3
            render_ret = render3(
                gaussian_vals,
                bg_color,
                data_item['extr'],
                data_item['intr'],
                data_item['img_w'],
                data_item['img_h']
            )
            rgb_map = render_ret['render'].permute(1, 2, 0)
            mask_map = render_ret['mask'].permute(1, 2, 0)
            # print(time.time() - start)
            
            rgb_map.clip_(0., 1.)
            rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
            
            # A-pose나 T-pose 가까운 pose로 먼저 만들고, 그 다음에 그 pose를 변형시키는 것이 좋을 듯
            os.makedirs(f'./test/thuman4_pose_00', exist_ok = True)
            cv.imwrite(f'./test/thuman4_pose_00/{idx:03d}.jpg', rgb_map)            
            
            continue
            
            #######################################################################################

            # 1. getitem_fast
            getitem_func = self.dataset.getitem_fast if hasattr(self.dataset, 'getitem_fast') else self.dataset.getitem
            item = getitem_func(
                idx,
                training = False,
                extr = extr,
                intr = intr,
                img_w = img_w,
                img_h = img_h
            )
            # item(dict) : idx, global_orient, joints, kin_parent, live_smpl_v, cano_joint, cano_smpl_v, cano2live_jnt_mats
            # live_bounds, cano_bounds, extr, intr, img_w, img_h
            items = to_cuda(item, add_batch = False)

            if view_setting.startswith('moving') or view_setting == 'free_moving':
                current_center = items['live_bounds'].cpu().numpy().mean(0)
                delta = current_center - object_center

                object_center[0] += delta[0]
                # object_center[1] += delta[1]
                # object_center[2] += delta[2]

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Loading data costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            if self.opt['test'].get('render_skeleton', False):
                from utils.visualize_skeletons import construct_skeletons
                skel_vertices, skel_faces = construct_skeletons(item['joints'].cpu().numpy(), item['kin_parent'].cpu().numpy())
                skel_mesh = trimesh.Trimesh(skel_vertices, skel_faces, process = False)

                if geo_renderer is None:
                    geo_renderer = Renderer(item['img_w'], item['img_h'], shader_name = 'phong_geometry', bg_color = (1, 1, 1))
                extr, intr = item['extr'], item['intr']
                geo_renderer.set_camera(extr, intr)
                geo_renderer.set_model(skel_vertices[skel_faces.reshape(-1)], skel_mesh.vertex_normals.astype(np.float32)[skel_faces.reshape(-1)])
                skel_img = geo_renderer.render()[:, :, :3]
                skel_img = (skel_img * 255).astype(np.uint8)
                cv.imwrite(output_dir + '/live_skeleton/%08d.jpg' % item['data_idx'], skel_img)

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering skeletons costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            # 2. items['smpl_pos_map']
            if 'smpl_pos_map' not in items:
                self.avatar_net.get_pose_map(items)

            # 2.1 smpl_pos_map_pca
            if use_pca:
                mask = training_dataset.pos_map_mask # same
                live_pos_map = items['smpl_pos_map'].permute(1, 2, 0).cpu().numpy() # diff
                front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
                pose_conds = front_live_pos_map[mask] # same
                new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(self.opt['test'].get('sigma_pca', 2.)))
                front_live_pos_map[mask] = new_pose_conds
                live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
                items.update({
                    'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
                })
            
            self.mask = mask
            self.live_pos_map = live_pos_map

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering pose conditions costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            # 3. render & make pose avatar
            output = self.avatar_net.render(items, bg_color = self.bg_color, use_pca = use_pca)
            
            #################################################################################################################
            
            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Rendering avatar costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                time_start.record()

            rgb_map = output['rgb_map']
            rgb_map.clip_(0., 1.)
            rgb_map = (rgb_map * 255).to(torch.uint8).cpu().numpy()
            cv.imwrite(output_dir + '/rgb_map/%08d.jpg' % item['data_idx'], rgb_map)

            if 'mask_map' in output:
                os.makedirs(output_dir + '/mask_map', exist_ok = True)
                mask_map = output['mask_map'][:, :, 0]
                mask_map.clip_(0., 1.)
                mask_map = (mask_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + '/mask_map/%08d.png' % item['data_idx'], mask_map.cpu().numpy())

            if self.opt['test'].get('save_tex_map', False):
                os.makedirs(output_dir + '/cano_tex_map', exist_ok = True)
                cano_tex_map = output['cano_tex_map']
                cano_tex_map.clip_(0., 1.)
                cano_tex_map = (cano_tex_map * 255).to(torch.uint8)
                cv.imwrite(output_dir + '/cano_tex_map/%08d.jpg' % item['data_idx'], cano_tex_map.cpu().numpy())

            if self.opt['test'].get('save_ply', False):
                save_gaussians_as_ply(output_dir + '/posed_gaussians/%08d.ply' % item['data_idx'], output['posed_gaussians'])

            if log_time:
                time_end.record()
                torch.cuda.synchronize()
                print('Saving images costs %.4f secs' % (time_start.elapsed_time(time_end) / 1000.))
                print('Animating one frame costs %.4f secs' % (time_start_all.elapsed_time(time_end) / 1000.))

            torch.cuda.empty_cache()

    def save_ckpt(self, path, save_optm = True):
        os.makedirs(path, exist_ok = True)
        net_dict = {
            'epoch_idx': self.epoch_idx,
            'iter_idx': self.iter_idx,
            'avatar_net': self.avatar_net.state_dict(),
        }
        print('Saving networks to ', path + '/net.pt')
        torch.save(net_dict, path + '/net.pt')

        if save_optm:
            optm_dict = {
                'avatar_net': self.optm.state_dict(),
            }
            print('Saving optimizers to ', path + '/optm.pt')
            torch.save(optm_dict, path + '/optm.pt')

    def load_ckpt(self, path, load_optm = True):
        print('Loading networks from ', path + '/net.pt')
        net_dict = torch.load(path + '/net.pt')
        if 'avatar_net' in net_dict:
            self.avatar_net.load_state_dict(net_dict['avatar_net'])
        else:
            print('[WARNING] Cannot find "avatar_net" from the network checkpoint!')
        epoch_idx = net_dict['epoch_idx']
        iter_idx = net_dict['iter_idx']

        if load_optm and os.path.exists(path + '/optm.pt'):
            print('Loading optimizers from ', path + '/optm.pt')
            optm_dict = torch.load(path + '/optm.pt')
            if 'avatar_net' in optm_dict:
                self.optm.load_state_dict(optm_dict['avatar_net'])
            else:
                print('[WARNING] Cannot find "avatar_net" from the optimizer checkpoint!')

        return epoch_idx, iter_idx
    
    def find_knn_faiss(self, k=3):
        """
        FAISS를 사용하여 KNN을 찾는 함수 (N x 3 크기의 점들에 대해 O(N logN) 복잡도로 처리 가능)
        
        Args:
            points (torch.Tensor): (N, 3) 크기의 3D 포인트 클라우드
            k (int): 찾을 이웃 개수 (기본값 3)
            
        Returns:
            knn_indices (torch.Tensor): (N, k) 형태의 KNN 인덱스
            knn_distances (torch.Tensor): (N, k) 형태의 KNN 거리
        """
        points = self.avatar_net.cano_gaussian_model.get_xyz
        
        N, D = points.shape  # N=점 개수, D=3 (x, y, z 좌표)

        # FAISS 인덱스 생성 (L2 거리 사용)
        index = faiss.IndexFlatL2(D)
        index.add(points.cpu().numpy())  # FAISS는 numpy 배열을 사용

        # KNN 검색 수행 (자기 자신 포함됨)
        distances, indices = index.search(points.cpu().numpy(), k + 1)  # 자기 자신 포함되므로 k+1

        # 자기 자신(인덱스 0) 제거
        knn_indices = torch.tensor(indices[:, 1:], dtype=torch.long, device=points.device)  # (N, k)
        knn_distances = torch.tensor(distances[:, 1:], dtype=torch.float32, device=points.device)  # (N, k)

        return knn_indices, knn_distances
    
    @torch.no_grad()
    def get_avatar_frame(self, idx):
        # for AG gt avatar rendering
        
        import time
        end = time.time()
        self.avatar_net.eval()

        dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
        MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
        training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
        if self.opt['test'].get('n_pca', -1) >= 1:
            training_dataset.compute_pca(n_components = self.opt['test']['n_pca'])
        if 'pose_data' in self.opt['test']:
            testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0])
            dataset_name = testing_dataset.dataset_name
            seq_name = testing_dataset.seq_name
        else:
            testing_dataset = MvRgbDataset(**self.opt['test']['data'], training = False)
            dataset_name = 'training'
            seq_name = ''
        
        self.dataset = testing_dataset
        iter_idx = self.load_ckpt(self.opt['test']['prev_ckpt'], False)[1]
        use_pca = self.opt['test'].get('n_pca', -1) >= 1
        
        item_0 = self.dataset.getitem(0, training = False)
        object_center = item_0['live_bounds'].mean(0)
        global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
        global_orient = cv.Rodrigues(global_orient)[0]

        img_scale = self.opt['test'].get('img_scale', 1.0)
        view_setting = config.opt['test'].get('view_setting', 'free')
        if view_setting == 'camera':
            # training view setting
            cam_id = config.opt['test']['render_view_idx']
            intr = self.dataset.intr_mats[cam_id].copy()
            intr[:2] *= img_scale
            extr = self.dataset.extr_mats[cam_id].copy()
            img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)
        elif view_setting.startswith('free'):
            # free view setting
            # frame_num_per_circle = 360
            frame_num_per_circle = 216
            rot_Y = (0 % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = rot_Y,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('front'):
            # front view setting
            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = 0.,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('back'):
            # back view setting
            extr = visualize_util.calc_free_mv(object_center,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = np.pi,
                                                rot_X = 0.5 * np.pi / 4. if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('moving'):
            # moving camera setting
            extr = visualize_util.calc_free_mv(object_center,
                                                # tar_pos = np.array([0, 0, 3.0]),
                                                # rot_Y = -0.3,
                                                tar_pos = np.array([0, 0, 2.5]),
                                                rot_Y = 0.,
                                                rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
            intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
            intr[:2] *= img_scale
            img_h = int(1024 * img_scale)
            img_w = int(1024 * img_scale)
        elif view_setting.startswith('cano'):
            cano_center = self.dataset.cano_bounds.mean(0)
            extr = np.identity(4, np.float32)
            extr[:3, 3] = -cano_center
            rot_x = np.identity(4, np.float32)
            rot_x[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
            extr = rot_x @ extr
            f_len = 5000
            extr[2, 3] += f_len / 512
            intr = np.array([[f_len, 0, 512], [0, f_len, 512], [0, 0, 1]], np.float32)
            # item = self.dataset.getitem(idx,
            #                             training = False,
            #                             extr = extr,
            #                             intr = intr,
            #                             img_w = 1024,
            #                             img_h = 1024)
            img_w, img_h = 1024, 1024
            # item['live_smpl_v'] = item['cano_smpl_v']
            # item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
            # item['live_bounds'] = item['cano_bounds']
        else:
            raise ValueError('Invalid view setting for animation!')
        
        # live_smpl of first frame        
        # start_frame = config.opt['test']['pose_data']['frame_range'][0]
        #############################################################################
        # 1
        # first_idx = self.dataset.pose_list[0] # -> cano pose?
        live_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                        global_orient = self.dataset.body_poses[idx, :3][None], # [1, 3]
                                        transl = self.dataset.transl[idx][None], # [1, 3]   
                                        body_pose = self.dataset.body_poses[idx, 3: 66][None], # [1, 63]
                                        left_hand_pose = self.dataset.left_hand_pose[idx][None].to(config.device), # [1, 45]
                                        right_hand_pose = self.dataset.right_hand_pose[idx][None].to(config.device) # [1, 45]
                                        # global_orient = torch.zeros([1, 3]).to(config.device), # [1, 3]
                                        # transl = torch.zeros([1, 3]).to(config.device), # [1, 3]
                                        # body_pose = torch.zeros([1, 63]).to(config.device),
                                        # left_hand_pose = torch.zeros([1, 45]).to(config.device), ##
                                        # right_hand_pose = torch.zeros([1, 45]).to(config.device), ##
                                        )
        
        live_smpl_woRoot = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                        body_pose = self.dataset.body_poses[idx, 3: 66][None],
                                        left_hand_pose = self.dataset.left_hand_pose[idx][None].to(config.device), ##
                                        right_hand_pose = self.dataset.right_hand_pose[idx][None].to(config.device), ##
                                        # body_pose = torch.zeros([1, 63]).to(config.device),
                                        # left_hand_pose = torch.zeros([1, 45]).to(config.device), ##
                                        # right_hand_pose = torch.zeros([1, 45]).to(config.device), ##
                                        )
            
        # 2
        data_item = dict()
        inv_cano_jnt_mats = torch.linalg.inv(self.dataset.cano_smpl['A']) # [55, 4, 4]
        data_item['extr'] = torch.from_numpy(extr).to(config.device)
        # data_item['intr'] = torch.from_numpy(intr).to(config.device) # for render
        # data_item['img_w'] = img_w                                   # for render
        # data_item['img_h'] = img_h                                   # for render
        data_item['cano2live_jnt_mats_1st'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4] * [55, 4, 4]
        data_item['cano2live_jnt_mats_1st_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
        
        # data_item = self.avatar_net.get_pose_map(data_item)
        pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st_woRoot'])
        live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # [N, 3]
        live_pos_map = torch.zeros_like(self.avatar_net.cano_smpl_map) # [1024, 2048, 3]
        live_pos_map[self.avatar_net.cano_smpl_mask] = live_pts # self.avatar_net.cano_smpl_mask.shape : [1024, 2048], boolean, sum = N
        live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0] # [3, 512, 1024]
        live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0) # [6, 512, 512]
        data_item.update({
            'smpl_pos_map': live_pos_map # live_pose_map_woRoot [6, 512, 512]
        })
        
        # 2.1
        if use_pca:
            mask = training_dataset.pos_map_mask # same
            live_pos_map = data_item['smpl_pos_map'].permute(1, 2, 0).cpu().numpy() # diff
            front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
            pose_conds = front_live_pos_map[mask] # same
            new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(self.opt['test'].get('sigma_pca', 2.)))
            front_live_pos_map[mask] = new_pose_conds
            live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
            data_item.update({
                'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
            })
        
        # 3
        bg_color = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
        pose_map = data_item['smpl_pos_map'][:3] # [3, 512, 512]
        if use_pca:
            pose_map = data_item['smpl_pos_map_pca'][:3]
        
        # get_positions, # net
        # pose_map <- live_pos_map <- live_pts
        # end2 = time.time()
        pos_map, _ = self.avatar_net.position_net([self.avatar_net.position_style], pose_map[None], randomize_noise = False) # net
        front_position_map, back_position_map = torch.split(pos_map, [3, 3], 1)
        pos_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
        
        delta_position = 0.05 * pos_map[self.avatar_net.cano_smpl_mask] # [373056, 3]
        cano_pts = delta_position + self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3]
        opacity, scales, rotations = self.avatar_net.get_others(pose_map) # net
        # print(time.time() - end2)
                
        if self.avatar_net.with_viewdirs:
            # front_viewdirs, back_viewdirs = self.avatar_net.get_viewdir_feat(data_item)
            with torch.no_grad():
                pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st']) # (N, 4, 4)
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # (N, 3)
                live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.cano_nmls) # (N, 3)
                cam_pos = -torch.matmul(torch.linalg.inv(data_item['extr'][:3, :3]), data_item['extr'][:3, 3]) # [3]
                viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3) # (N, 3)
                # if self.training:
                #     viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
                viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3) # (N, 3)
                viewdirs = (live_nmls * viewdirs).sum(-1) # (N)

                viewdirs_map = torch.zeros(*self.avatar_net.cano_nml_map.shape[:2]).to(viewdirs) # [1024, 2048]
                viewdirs_map[self.avatar_net.cano_smpl_mask] = viewdirs # [1024, 2048]

                viewdirs_map = viewdirs_map[None, None] # [1, 1, 1024, 2048]
                viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest') # [1, 1, 512, 1024]
                front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

            front_viewdirs = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(front_viewdirs)
            back_viewdirs  = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(back_viewdirs)
        
        colors, color_map = self.avatar_net.get_colors(pose_map, front_viewdirs, back_viewdirs) # net? no
        
        # import trimesh
        # mesh = trimesh.Trimesh(vertices = cano_pts.detach().cpu().numpy(), colors=(colors.detach().cpu().numpy()*255).astype(int))
        # mesh.export(f'./test/cano_pts.ply')
        # mesh = trimesh.Trimesh(vertices = self.avatar_net.cano_gaussian_model.get_xyz.detach().cpu().numpy(), colors=(colors.detach().cpu().numpy()*255).astype(int))
        # mesh.export(f'./test/cano_get_xyz.ply')
        
        # canonical human model of 1st frame        
        gaussian_vals = {
            'positions_ori': cano_pts,
            'opacity': opacity,
            'scales': scales,
            'rotations_ori': rotations,
            'colors': torch.flip(colors, dims=(1,)), # RGBtoBGR
            'max_sh_degree': 3 # self.avatar_net.max_sh_degree
        }
        
        gaussian_vals['positions'] = gaussian_vals['positions_ori']
        gaussian_vals['rotations'] = gaussian_vals['rotations_ori']
        
        from scene.gaussian_model import GaussianModel 
        cano_gaussians = GaussianModel(sh_degree=gaussian_vals['max_sh_degree'], device='cuda')
        cano_gaussians.create_from_values(gaussian_vals)
        
        # cano_gaussians of 1st frame
        pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st'])
        # pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st_woRoot'])
        gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], gaussian_vals['positions_ori']) + pt_mats[..., :3, 3]
        rot_mats = pytorch3d.transforms.quaternion_to_matrix(gaussian_vals['rotations_ori'])
        rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
        gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats) # [N, 4]
                
        # mesh = trimesh.Trimesh(vertices = gaussian_vals['positions'].detach().cpu().numpy())
        # mesh.export(f'./test/pose_position.ply')
        
        posed_gaussians = GaussianModel(sh_degree=gaussian_vals['max_sh_degree'], device='cuda')
        posed_gaussians.create_from_values(gaussian_vals)
        # posed_gaussians.get_rotation
        # self._rotation = self.rotation_activation(nn.Parameter(values['rotations'].requires_grad_(True)))
        # human_gaussians.save_ply(f'./test/human_first.ply')
        
        # lbs, joints        
        lbs = self.avatar_net.lbs
        joint_mats = []
        assert self.dataset.pose_list[-1] < self.dataset.body_poses.shape[0] # frame out of range
        data_num = len(self.dataset)
        first_idx = self.dataset.pose_list[0]
        print(time.time()-end)
        
        # live_smpl of sequence
        for idx in range(data_num):
            pose_idx = self.dataset.pose_list[idx]
            live_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                        global_orient = self.dataset.body_poses[pose_idx, :3][None] , ##
                                        transl = self.dataset.transl[pose_idx][None],
                                        body_pose = self.dataset.body_poses[pose_idx, 3: 66][None],
                                        left_hand_pose = self.dataset.left_hand_pose[pose_idx][None].to(config.device), ##
                                        right_hand_pose = self.dataset.right_hand_pose[pose_idx][None].to(config.device) ##
                                        )
            
            data_item['cano2live_jnt_mats'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4] * [55, 4, 4]            
            joint_mats.append(data_item['cano2live_jnt_mats'].cpu().detach().numpy())
            
        joint_mats = torch.as_tensor(np.array(joint_mats), device='cuda')
        
        # print("Save ply files")
        # cano_xyz = self.avatar_net.cano_gaussian_model.get_xyz
        # pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, joint_mats[0])
        # pose_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_xyz) + pt_mats[..., :3, 3]
        # mesh = trimesh.Trimesh(vertices = pose_pts.detach().cpu().numpy(),)
        # mesh.export(f'./test/cano_xyz_00.ply')
        # pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, joint_mats[99])
        # pose_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_xyz) + pt_mats[..., :3, 3]
        # mesh = trimesh.Trimesh(vertices = pose_pts.detach().cpu().numpy(),)
        # mesh.export(f'./test/cano_xyz_99.ply') 
        
        # pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, joint_mats[0])
        # pose_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        # mesh = trimesh.Trimesh(vertices = pose_pts.detach().cpu().numpy(),)
        # mesh.export(f'./test/cano_pts_00.ply')
        # pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, joint_mats[99])
        # pose_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
        # mesh = trimesh.Trimesh(vertices = pose_pts.detach().cpu().numpy(),)
        # mesh.export(f'./test/cano_pts_99.ply')
        
        return posed_gaussians, colors, cano_gaussians, lbs, joint_mats 

    @torch.no_grad()
    def get_avatars(self, subject_param):
        self.avatar_net.eval()
        with torch.no_grad():
            dataset_module = self.opt['train'].get('dataset', 'MvRgbDatasetAvatarReX')
            MvRgbDataset = importlib.import_module('dataset.dataset_mv_rgb').__getattribute__(dataset_module)
            training_dataset = MvRgbDataset(**self.opt['train']['data'], training = False)
            if self.opt['test'].get('n_pca', -1) >= 1:
                training_dataset.compute_pca(n_components = self.opt['test']['n_pca'])
            if 'pose_data' in self.opt['test']: # this
                testing_dataset = PoseDataset(**self.opt['test']['pose_data'], smpl_shape = training_dataset.smpl_data['betas'][0], device=config.device)
                dataset_name = testing_dataset.dataset_name
                seq_name = testing_dataset.seq_name
            else:
                testing_dataset = MvRgbDataset(**self.opt['test']['data'], training = False)
                dataset_name = 'training'
                seq_name = ''
            
            self.dataset = testing_dataset
            iter_idx = self.load_ckpt(self.opt['test']['prev_ckpt'], False)[1]
            use_pca = self.opt['test'].get('n_pca', -1) >= 1
            
            item_0 = self.dataset.getitem(0, training = False)
            object_center = item_0['live_bounds'].mean(0)
            global_orient = item_0['global_orient'].cpu().numpy() if isinstance(item_0['global_orient'], torch.Tensor) else item_0['global_orient']
            global_orient = cv.Rodrigues(global_orient)[0]

            img_scale = self.opt['test'].get('img_scale', 1.0)
            view_setting = config.opt['test'].get('view_setting', 'free') # front
            if view_setting == 'camera':
                # training view setting
                cam_id = config.opt['test']['render_view_idx']
                intr = self.dataset.intr_mats[cam_id].copy()
                intr[:2] *= img_scale
                extr = self.dataset.extr_mats[cam_id].copy()
                img_h, img_w = int(self.dataset.img_heights[cam_id] * img_scale), int(self.dataset.img_widths[cam_id] * img_scale)
            elif view_setting.startswith('free'):
                # free view setting
                # frame_num_per_circle = 360
                frame_num_per_circle = 216
                rot_Y = (0 % frame_num_per_circle) / float(frame_num_per_circle) * 2 * np.pi

                extr = visualize_util.calc_free_mv(object_center,
                                                    tar_pos = np.array([0, 0, 2.5]),
                                                    rot_Y = rot_Y,
                                                    rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                    global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('front'):
                # front view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                    tar_pos = np.array([0, 0, 2.5]),
                                                    rot_Y = 0.,
                                                    rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                    global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('back'):
                # back view setting
                extr = visualize_util.calc_free_mv(object_center,
                                                    tar_pos = np.array([0, 0, 2.5]),
                                                    rot_Y = np.pi,
                                                    rot_X = 0.5 * np.pi / 4. if view_setting.endswith('bird') else 0.,
                                                    global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('moving'):
                # moving camera setting
                extr = visualize_util.calc_free_mv(object_center,
                                                    # tar_pos = np.array([0, 0, 3.0]),
                                                    # rot_Y = -0.3,
                                                    tar_pos = np.array([0, 0, 2.5]),
                                                    rot_Y = 0.,
                                                    rot_X = 0.3 if view_setting.endswith('bird') else 0.,
                                                    global_orient = global_orient if self.opt['test'].get('global_orient', False) else None)
                intr = np.array([[1100, 0, 512], [0, 1100, 512], [0, 0, 1]], np.float32)
                intr[:2] *= img_scale
                img_h = int(1024 * img_scale)
                img_w = int(1024 * img_scale)
            elif view_setting.startswith('cano'):
                cano_center = self.dataset.cano_bounds.mean(0)
                extr = np.identity(4, np.float32)
                extr[:3, 3] = -cano_center
                rot_x = np.identity(4, np.float32)
                rot_x[:3, :3] = cv.Rodrigues(np.array([np.pi, 0, 0], np.float32))[0]
                extr = rot_x @ extr
                f_len = 5000
                extr[2, 3] += f_len / 512
                intr = np.array([[f_len, 0, 512], [0, f_len, 512], [0, 0, 1]], np.float32)
                # item = self.dataset.getitem(idx,
                #                             training = False,
                #                             extr = extr,
                #                             intr = intr,
                #                             img_w = 1024,
                #                             img_h = 1024)
                img_w, img_h = 1024, 1024
                # item['live_smpl_v'] = item['cano_smpl_v']
                # item['cano2live_jnt_mats'] = torch.eye(4, dtype = torch.float32)[None].expand(item['cano2live_jnt_mats'].shape[0], -1, -1)
                # item['live_bounds'] = item['cano_bounds']
            else:
                raise ValueError('Invalid view setting for animation!')

            # 1. Gaussian Color & 1st frame gaussian
            first_idx = self.dataset.pose_list[0]
            angle_z_arm = subject_param['wide_z_arm'] # degree
            self.dataset.body_poses[:, 13*3+2] += math.radians(angle_z_arm)
            self.dataset.body_poses[:, 14*3+2] -= math.radians(angle_z_arm)
            angle_z_leg = subject_param['wide_z_leg'] # degree
            self.dataset.body_poses[:, 1*3+2] += math.radians(angle_z_leg)
            self.dataset.body_poses[:, 2*3+2] -= math.radians(angle_z_leg)
            angle_y_arm = subject_param['wide_y_arm'] # degree
            self.dataset.body_poses[:, 13*3+1] += math.radians(angle_y_arm)
            self.dataset.body_poses[:, 14*3+1] -= math.radians(angle_y_arm)
            live_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                            global_orient = self.dataset.body_poses[first_idx, :3][None], # [1, 3]
                                            transl = self.dataset.transl[first_idx][None], # [1, 3]   
                                            body_pose = self.dataset.body_poses[first_idx, 3: 66][None], # [1, 63]
                                            left_hand_pose = self.dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                            right_hand_pose = self.dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
                                            )            
            live_smpl_woRoot = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                            body_pose = self.dataset.body_poses[first_idx, 3: 66][None],
                                            left_hand_pose = self.dataset.left_hand_pose[first_idx][None].to(config.device), ##
                                            right_hand_pose = self.dataset.right_hand_pose[first_idx][None].to(config.device), ##
                                            )
                    
            # 2
            data_item = dict()
            inv_cano_jnt_mats = torch.linalg.inv(self.dataset.cano_smpl['A']) # [55, 4, 4]
            extr = torch.from_numpy(extr).to(config.device)
            data_item['extr'] = extr
            # data_item['extr'] = torch.from_numpy(extr).to(config.device)
            data_item['cano2live_jnt_mats_1st'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4] * [55, 4, 4]
            data_item['cano2live_jnt_mats_1st_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)        
            joint_mat = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
            A_mat = live_smpl.A[0]
            cano_J = live_smpl.J[0, :22] # [55, 3]
            cano_J = F.pad(cano_J, (0, 1), mode='constant', value=0)
            # cano_J = np.concatenate([cano_J, np.zeros((22, 1))], axis=1)
            
            pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st_woRoot'])
            live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # [N, 3]
            live_pos_map = torch.zeros_like(self.avatar_net.cano_smpl_map) # [1024, 2048, 3]
            live_pos_map[self.avatar_net.cano_smpl_mask] = live_pts # self.avatar_net.cano_smpl_mask.shape : [1024, 2048], boolean, sum = N
            live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0] # [3, 512, 1024]
            live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0) # [6, 512, 512]
            data_item.update({'smpl_pos_map': live_pos_map})  # live_pose_map_woRoot [6, 512, 512]
            
            
            if 0:
                ########################################################################################
                # /workspace/physics/PhysGaussian/test_code/09_smplx_to_osso.py
                # from AnimatableGaussians import smplx
                import smplx as smplx_original
                smplx_model_ori = smplx_original.create(config.PROJ_DIR + '/../smpl_models/', model_type='smplx').to(config.device)
                smplx_faces = smplx_model_ori.faces
                self.dataset.smpl_model.faces
                smplx_faces != self.dataset.smpl_model.faces
                # smplx live
                mesh = trimesh.Trimesh(vertices = live_smpl.vertices[0].detach().cpu().numpy(), faces = self.dataset.smpl_model.faces, process=False)
                mesh.export('./test_data/osso/pose/smplx_pose.ply') ###
                mesh = trimesh.Trimesh(vertices = live_smpl.joints[0, :55].detach().cpu().numpy())
                mesh.export('./test_data/osso/pose/smplx_pose_joints.ply') ###
                import json
                smplx_json = {'vertices': live_smpl.vertices[0].detach().cpu().numpy().tolist(), 
                            'faces': smplx_faces.tolist(),
                            'joints': live_smpl.joints[0].detach().cpu().numpy().tolist(),
                            'betas': self.dataset.smpl_shape[None].detach().cpu().numpy().tolist(),
                            'global_orient': self.dataset.body_poses[first_idx, :3][None].detach().cpu().numpy().tolist(),
                            'transl': self.dataset.transl[first_idx][None].detach().cpu().numpy().tolist(),
                            'body_pose': self.dataset.body_poses[first_idx, 3: 66][None].detach().cpu().numpy().tolist(),
                            'left_hand_pose': self.dataset.left_hand_pose[first_idx][None].detach().cpu().numpy().tolist(),
                            'right_hand_pose': self.dataset.right_hand_pose[first_idx][None].detach().cpu().numpy().tolist()}
                with open('./test_data/osso/pose/smplx_pose.json', 'w') as f:
                    json.dump(smplx_json, f)
                
                # random seed
                keypoint_parents = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19])
                keypoint_line = [[int(keypoint_parents[i]), i] for i, p in enumerate(keypoint_parents)][1:]
                
                import open3d as o3d
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(live_smpl.joints[0, :22].detach().cpu().numpy()),
                    lines=o3d.utility.Vector2iVector(keypoint_line)
                )
                from test_code.line_ply_09 import write_ply_lineset
                # write_ply_lineset("./test_data/osso/pose/smplx_pose_line.ply", line_set)
                
                # ag live, ag face 없음
                mesh = trimesh.Trimesh(vertices = live_pts.detach().cpu().numpy())
                mesh.export('./test_data/osso/pose/ag_pose.ply') ###
                            
                # smplx cano
                cano_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                                global_orient = self.dataset.body_poses[first_idx, :3][None], # [1, 3]
                                                transl = self.dataset.transl[first_idx][None], # [1, 3]   
                                                body_pose = torch.zeros_like(self.dataset.body_poses[first_idx, 3: 66][None]), # [1, 63]
                                                left_hand_pose = torch.zeros_like(self.dataset.left_hand_pose[first_idx][None].to(config.device)), # [1, 45]
                                                right_hand_pose = torch.zeros_like(self.dataset.right_hand_pose[first_idx][None].to(config.device)) # [1, 45]
                                                )
                cano_smpl_woRoot = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                                # global_orient = self.dataset.body_poses[first_idx, :3][None], # [1, 3]
                                                # transl = self.dataset.transl[first_idx][None], # [1, 3]   
                                                body_pose = torch.zeros_like(self.dataset.body_poses[first_idx, 3: 66][None]), # [1, 63]
                                                left_hand_pose = torch.zeros_like(self.dataset.left_hand_pose[first_idx][None].to(config.device)), # [1, 45]
                                                right_hand_pose = torch.zeros_like(self.dataset.right_hand_pose[first_idx][None].to(config.device)) # [1, 45]
                                                )
                mesh = trimesh.Trimesh(vertices = cano_smpl.vertices[0].detach().cpu().numpy(), faces = self.dataset.smpl_model.faces, process=False)
                # mesh.export('./test_data/osso/cano/smplx_cano.ply') ###
                mesh = trimesh.Trimesh(vertices = cano_smpl.joints[0, :55].detach().cpu().numpy()) # :55
                # mesh.export('./test_data/osso/cano/smplx_cano_joints.ply') ###
                line_set = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(cano_smpl.joints[0, :22].detach().cpu().numpy()),
                    lines=o3d.utility.Vector2iVector(keypoint_line)
                )
                # write_ply_lineset("./test_data/osso/cano/smplx_cano_line.ply", line_set)
                
                import json
                smplx_json = {'vertices': cano_smpl.vertices[0].detach().cpu().numpy().tolist(), 
                            'faces': smplx_faces.tolist(),
                            'joints': cano_smpl.joints[0].detach().cpu().numpy().tolist(),
                            'betas': self.dataset.smpl_shape[None].detach().cpu().numpy().tolist(),
                            'global_orient': self.dataset.body_poses[first_idx, :3][None].detach().cpu().numpy().tolist(),
                            'transl': self.dataset.transl[first_idx][None].detach().cpu().numpy().tolist(),
                            'body_pose': torch.zeros_like(self.dataset.body_poses[first_idx, 3: 66][None]).detach().cpu().numpy().tolist(),
                            'left_hand_pose': torch.zeros_like(self.dataset.left_hand_pose[first_idx][None]).detach().cpu().numpy().tolist(),
                            'right_hand_pose': torch.zeros_like(self.dataset.right_hand_pose[first_idx][None]).detach().cpu().numpy().tolist()}
                # with open('./test_data/osso/cano/smplx_cano.json', 'w') as f:
                #     json.dump(smplx_json, f)
                            
                pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, torch.matmul(cano_smpl_woRoot.A[0], inv_cano_jnt_mats)) # cano without root
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # [N, 3]
                mesh = trimesh.Trimesh(vertices = live_pts.detach().cpu().numpy())
                mesh.export('./test_data/osso/cano/ag_cano.ply')    # T-pose, same as smplx
                
                cano_J_ = cano_smpl_woRoot.joints[:, :22].cpu().detach().numpy()  # posed_joints
                cano_J_mesh = trimesh.Trimesh(vertices=cano_J_[0], process=False)
                cano_J_mesh.export('./test_results/cano_J.ply')
                
                # lbs 비율
                # np.max(self.avatar_net.lbs.cpu().numpy(), axis=0), 0.5~0.9 까지 다양
                # (self.avatar_net.lbs > 0.8).float().mean(dim=0), 최대가 5% 수준, 쓰기 힘들다
                
            # 2.1
            if use_pca: # False
                mask = training_dataset.pos_map_mask # same
                live_pos_map = data_item['smpl_pos_map'].permute(1, 2, 0).cpu().numpy() # diff
                front_live_pos_map, back_live_pos_map = np.split(live_pos_map, [3], 2)
                pose_conds = front_live_pos_map[mask] # same
                new_pose_conds = training_dataset.transform_pca(pose_conds, sigma_pca = float(self.opt['test'].get('sigma_pca', 2.)))
                front_live_pos_map[mask] = new_pose_conds
                live_pos_map = np.concatenate([front_live_pos_map, back_live_pos_map], 2)
                data_item.update({
                    'smpl_pos_map_pca': torch.from_numpy(live_pos_map).to(config.device).permute(2, 0, 1)
                })
            
            # 3
            bg_color = torch.from_numpy(np.asarray(self.bg_color)).to(torch.float32).to(config.device)
            pose_map = data_item['smpl_pos_map'][:3] # [3, 512, 512]
            if use_pca: # False
                pose_map = data_item['smpl_pos_map_pca'][:3]
            
            # get_positions, # net
            pos_map, _ = self.avatar_net.position_net([self.avatar_net.position_style], pose_map[None], randomize_noise = False) # net
            front_position_map, back_position_map = torch.split(pos_map, [3, 3], 1)
            pos_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)
            
            delta_position = 0.05 * pos_map[self.avatar_net.cano_smpl_mask] # [373056, 3]
            cano_pts = delta_position + self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3]
            # cano_pts = self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3], 이거 금지, cano_pose일때의 delta_position이 필요하다
            opacity, scales, rotations = self.avatar_net.get_others(pose_map) # net
            # cano_mesh = trimesh.Trimesh(vertices = cano_pts.detach().cpu().numpy())
            # cano_mesh.export('./log/ag_new/cano_pts.ply')
            
            scales.data.clamp_(max=0.01)
            # scales = torch.clip(scales, 0., 0.013)
            # scales = torch.ones_like(scales) * 0.0001
                    
            if self.avatar_net.with_viewdirs: # True
                # front_viewdirs, back_viewdirs = self.avatar_net.get_viewdir_feat(data_item)
                with torch.no_grad():
                    pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st']) # (N, 4, 4)
                    live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # (N, 3)
                    live_nmls = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.cano_nmls) # (N, 3)
                    cam_pos = -torch.matmul(torch.linalg.inv(data_item['extr'][:3, :3]), data_item['extr'][:3, 3]) # [3]
                    viewdirs = F.normalize(cam_pos[None] - live_pts, dim = -1, eps = 1e-3) # (N, 3)
                    # if self.training:
                    #     viewdirs += torch.randn(*viewdirs.shape).to(viewdirs) * 0.1
                    viewdirs = F.normalize(viewdirs, dim = -1, eps = 1e-3) # (N, 3)
                    viewdirs = (live_nmls * viewdirs).sum(-1) # (N)

                    viewdirs_map = torch.zeros(*self.avatar_net.cano_nml_map.shape[:2]).to(viewdirs) # [1024, 2048]
                    viewdirs_map[self.avatar_net.cano_smpl_mask] = viewdirs # [1024, 2048]

                    viewdirs_map = viewdirs_map[None, None] # [1, 1, 1024, 2048]
                    viewdirs_map = F.interpolate(viewdirs_map, None, 0.5, 'nearest') # [1, 1, 512, 1024]
                    front_viewdirs, back_viewdirs = torch.split(viewdirs_map, [512, 512], -1)

                front_viewdirs = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(front_viewdirs)
                back_viewdirs  = self.avatar_net.opt.get('weight_viewdirs', 1.) * self.avatar_net.viewdir_net(back_viewdirs)
            
            colors, color_map = self.avatar_net.get_colors(pose_map, front_viewdirs, back_viewdirs) # no net, color_map [1024, 2048, 3]
            
            # cloth, hair
            if 0:
                import cv2
                front_color_map, back_color_map = color_map[:, :1024, :], color_map[:, 1024:, :]
                colors = color_map[self.avatar_net.cano_smpl_mask] # cano_smpl_mask [1024, 2048]
                front_color_map = np.clip(front_color_map.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                back_color_map = np.clip(back_color_map.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                front_color_map = front_color_map * self.avatar_net.cano_smpl_mask[:, :1024].unsqueeze(-1).detach().cpu().numpy()
                back_color_map = back_color_map * self.avatar_net.cano_smpl_mask[:, 1024:].unsqueeze(-1).detach().cpu().numpy()
                cv2.imwrite("./test_data/front_color_map.png", front_color_map)
                cv2.imwrite("./test_data/back_color_map.png", back_color_map)
                
                cano_smpl_mask = self.avatar_net.cano_smpl_mask.detach().cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite("./test_data/cano_smpl_mask.png", cano_smpl_mask)
                
            
            #########################################################################################################################################################
            # 2. posed_gaussians with bone
            
            bone_cano   = torch.empty(0, 3).to(config.device)
            bone_scales = torch.empty(0, 3).to(config.device)
            bone_colors = torch.empty(0, 3).to(config.device)
            bone_index = [0]
            bone_path = os.path.join(subject_param["osso_path"], 'osso_per_parts', 'part_split_meshes.glb')
            bone = trimesh.load(bone_path)
            bone_faces = []
            for i, (key, val) in enumerate(bone.geometry.items()):
                if i == 7:
                    continue
                else:
                    # print(val.vertices.shape) # num particle of bone
                    val.vertices = (val.vertices - val.centroid) * 0.82 + val.centroid # bone scale
                    # val.vertices = (val.vertices - val.centroid) * 0.0001 + val.centroid # bone scale
                    bone_cano = torch.cat([bone_cano, torch.from_numpy(val.vertices).float().to(config.device)], 0)
                    
                    edge_lengths = np.linalg.norm(val.vertices[val.edges[:,0]] - val.vertices[val.edges[:,1]], axis=1)
                    sigma = 0.05 * np.mean(edge_lengths)
                    bone_scales = torch.cat([bone_scales, torch.ones(val.vertices.shape[0], 3).to(config.device) * sigma])
                    
                    bone_colors = torch.cat([bone_colors, torch.from_numpy(val.visual.vertex_colors[:, :3] / 255).to(config.device)], 0)
                                    
                    bone_index.append(bone_index[-1] + val.vertices.shape[0])
                    bone_faces.append(val.faces)
                    # print(val.vertices.shape)
                    
                    # scales.sort(axis=1).values.mean(axis=0)
                    # 뼈대 cano랑 rotation 따로 만들어야함
            
            bone_colors    = torch.ones(bone_cano.shape[0], 3).to(config.device)   # 흰색 뼈
            bone_opacity   = torch.zeros(bone_cano.shape[0], 1).to(config.device)  # 투명 뼈           
            # bone_opacity   = torch.ones(bone_cano.shape[0], 1).to(config.device) # 색갈 뼈
            # opacity = torch.zeros_like(opacity) # Avatar 투명
            
            bone_rotations = torch.zeros((bone_cano.shape[0], 4), device=config.device)
            bone_rotations[:, 0] = 1
            
            # rotations
            # val.vertex_normals # 일단은 대충 해보자 
            # gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats) # [N, 4]
            
            from scene.gaussian_model import GaussianModel 
            
            # with bone
            gaussian_vals = {
                'positions_ori' : torch.cat([bone_cano,      cano_pts ]),        # [N, 3]
                'opacity'       : torch.cat([bone_opacity,   opacity  ]),    # [N, 1]
                'scales'        : torch.cat([bone_scales,    scales   ]),     # [N, 3]
                'rotations_ori' : torch.cat([bone_rotations, rotations]), # [N, 4]            
                'colors'        : torch.cat([bone_colors, torch.flip(colors, dims=(1,))]), # RGBtoBGR, [N, 3]
                # 'colors'        : torch.flip(torch.cat([bone_colors, colors]), dims=(1,)), # RGBtoBGR, [N, 3]
                'max_sh_degree' : 3 # self.avatar_net.max_sh_degree
            }
            gaussian_vals['opacity'] = torch.clamp(gaussian_vals['opacity'], min=1e-4, max=1.0 - 1e-4)
            # gaussian_vals = {
            #     'positions_ori' : cano_pts,        # [N, 3]
            #     'opacity'       : opacity ,    # [N, 1]
            #     'scales'        : scales  ,     # [N, 3]
            #     'rotations_ori' : rotations, # [N, 4]
            #     'colors'        : torch.flip(colors, dims=(1,)), # RGBtoBGR, [N, 3]
            #     'max_sh_degree' : 3 # self.avatar_net.max_sh_degree
            # }
            
            pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st']) # posed human xyz
            gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]        
            rot_mats = pytorch3d.transforms.quaternion_to_matrix(rotations)                                 # posed human rotation
            rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], rot_mats)
            gaussian_vals['rotations'] = pytorch3d.transforms.matrix_to_quaternion(rot_mats) # [N, 4]        
            # mesh = trimesh.Trimesh(vertices = gaussian_vals['positions'].detach().cpu().numpy())
            # mesh.export(f'./log/ag_new/pose_pts.ply')
            
            bone_pose  = torch.empty(0, 3).to(config.device)
            bone_rot   = torch.empty(0, 4).to(config.device)
            smpl_index = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 1, 4, 2, 5, 7, 8]       # 20
            for i in range(len(bone_index)-1):
                bone_pose_i = bone_cano[bone_index[i] : bone_index[i+1]] @ live_smpl.A[0, smpl_index[i], :3, :3].T + live_smpl.A[0, smpl_index[i], :3, 3]
                bone_pose = torch.cat([bone_pose, bone_pose_i])
                bone_rot_i = pytorch3d.transforms.matrix_to_quaternion(live_smpl.A[0, smpl_index[i], :3, :3]).unsqueeze(0).repeat(bone_pose_i.shape[0], 1)
                bone_rot  = torch.cat([bone_rot, bone_rot_i])
                
                # test_mesh = trimesh.Trimesh(vertices=pose_bone.detach().cpu().numpy() , faces=bone_faces[i])
                # test_mesh.export(f'./log/ag_new/pose_bone_{i}.ply')
            
            # test noise
            # gaussian_vals['positions'] += 0.1 * torch.rand(gaussian_vals['positions'].shape, device=config.device)
            
            gaussian_vals['positions'] = torch.cat([bone_pose, gaussian_vals['positions']])
            gaussian_vals['rotations'] = torch.cat([bone_rot,  gaussian_vals['rotations']])

            posed_gaussians = GaussianModel(sh_degree=gaussian_vals['max_sh_degree'], device=config.device)
            posed_gaussians.create_from_values(gaussian_vals)
            # posed_gaussians.save_ply('./AnimatableGaussians/avatarrex_zzr/point_cloud/iteration_0000/point_cloud.ply')
            
            # cano_bone이 필요하긴하다
            
            #########################################################################################################################################################
            # 3. test for posed gaussians, per frame
            # not used
            
            if 0:
                live_smpl = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                                global_orient = self.dataset.body_poses[first_idx, :3][None], # [1, 3]
                                                transl = self.dataset.transl[first_idx][None], # [1, 3]   
                                                body_pose = self.dataset.body_poses[first_idx, 3: 66][None], # [1, 63]
                                                left_hand_pose = self.dataset.left_hand_pose[first_idx][None].to(config.device), # [1, 45]
                                                right_hand_pose = self.dataset.right_hand_pose[first_idx][None].to(config.device) # [1, 45]
                )
                
                live_smpl_woRoot = self.dataset.smpl_model.forward(betas = self.dataset.smpl_shape[None],
                                                body_pose = self.dataset.body_poses[first_idx, 3: 66][None],
                                                left_hand_pose = self.dataset.left_hand_pose[first_idx][None].to(config.device), ##
                                                right_hand_pose = self.dataset.right_hand_pose[first_idx][None].to(config.device), ##
                                                )
                
                # inv_cano_jnt_mats
                # lbs : self.avatar_net.lbs
                # self.avatar_net.init_points, self.avatar_net.cano_smpl_map, self.avatar_net.cano_smpl_mask
                
                inv_cano_jnt_mats = torch.linalg.inv(self.dataset.cano_smpl['A'])
                data_item['cano2live_jnt_mats_1st_woRoot'] = torch.matmul(live_smpl_woRoot.A[0], inv_cano_jnt_mats)
                pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st_woRoot'])
                live_pts = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], self.avatar_net.init_points) + pt_mats[..., :3, 3] # [N, 3]
                live_pos_map = torch.zeros_like(self.avatar_net.cano_smpl_map) # [1024, 2048, 3]
                live_pos_map[self.avatar_net.cano_smpl_mask] = live_pts # self.avatar_net.cano_smpl_mask.shape : [1024, 2048], boolean, sum = N
                live_pos_map = F.interpolate(live_pos_map.permute(2, 0, 1)[None], None, [0.5, 0.5], mode = 'nearest')[0] # [3, 512, 1024]
                live_pos_map = torch.cat(torch.split(live_pos_map, [512, 512], 2), 0) # [6, 512, 512]
                pose_map = live_pos_map[:3]
                
                # self.avatar_net.position_net,  self.avatar_net.get_others
                # self.avatar_net.position_style
                pos_map, _ = self.avatar_net.position_net([self.avatar_net.position_style], pose_map[None], randomize_noise = False) # net
                front_position_map, back_position_map = torch.split(pos_map, [3, 3], 1)
                pos_map = torch.cat([front_position_map, back_position_map], 3)[0].permute(1, 2, 0)        
                delta_position = 0.05 * pos_map[self.avatar_net.cano_smpl_mask] # [373056, 3]
                
                cano_pts = delta_position + self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3]
                opacity, scales, rotations = self.avatar_net.get_others(pose_map) # net / opacity, scales는 랜더링에만 사용
                
                #
                joint_mat = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats)
                pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, joint_mat)
                positions = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]
                rot_mats = torch.einsum('nxy,nyz->nxz', pt_mats[..., :3, :3], pytorch3d.transforms.quaternion_to_matrix(rotations)) # [human_N, 3, 3]
            
            #################################################################################################################
            
            # cano_pts : avatar
            # bone_cano : bone
            cano_pts = delta_position + self.avatar_net.cano_gaussian_model.get_xyz # [373056, 3]
            pt_mats = torch.einsum('nj,jxy->nxy', self.avatar_net.lbs, data_item['cano2live_jnt_mats_1st']) # posed human xyz
            gaussian_vals['positions'] = torch.einsum('nxy,ny->nx', pt_mats[..., :3, :3], cano_pts) + pt_mats[..., :3, 3]        
            data_item['cano2live_jnt_mats_1st'] = torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # [55, 4, 4] * [55, 4, 4]
            inv_cano_jnt_mats = torch.linalg.inv(self.dataset.cano_smpl['A']) # [55, 4, 4]
            joint_mat; torch.matmul(live_smpl.A[0], inv_cano_jnt_mats) # live_smpl.A[0] 랑 joint_mat은 다름, 이거 감안해서 compute_human_particle_velocity 수정해야함
        
            # knn
            knn_indices = None
            # knn_save_path = os.path.join(self.opt['train']['data']['data_dir'], 'avatar_knn.pth')
            # if os.path.exists(knn_save_path):
            #     print("Loading KNN Neighbors from {}".format(knn_save_path))
            #     knn_save = torch.load(knn_save_path)
            #     knn_indices = knn_save['knn_indices']
            # else:
            #     print("Calculating KNN Neighbors for the first time... Takes about 1 minute.")
            #     knn_indices, knn_distances = self.find_knn_faiss(k=3) # self.avatar_net.cano_gaussian_model.get_xyz
            #     torch.save({'knn_indices': knn_indices, 'knn_distances': knn_distances}, knn_save_path)
            #     print("Saving KNN Neighbors to {}".format(knn_save_path))
            
            lbs_test = False
            if lbs_test:
                # lbs color
                lbs = self.avatar_net.lbs # [345918, 55], 20
                cano_pts
                eps = 1e-8
                w = lbs / (lbs.sum(dim=1, keepdim=True) + eps)
                row_max = w.max(dim=1, keepdim=True).values          # [N,1]
                w_bin = (w == row_max).to(w.dtype)
                
                seed = 7778
                g = torch.Generator(device=lbs.device)
                g.manual_seed(seed)
                joint_colors = torch.rand(55, 3, generator=g, device=lbs.device)
                colors = w @ joint_colors
                colors = colors.clamp(0.0, 1.0) * 255.0
                mesh = trimesh.Trimesh(vertices = cano_pts.detach().cpu().numpy(),
                                    vertex_colors = colors.detach().cpu().numpy().astype(np.uint8),
                                    process=False)
                mesh.export('./test_data/cano_pts_lbs2.ply')
                
                # lbs influencing
                eps = 1e-3
                mask_3plus = (lbs >= eps).sum(dim=1) >= 3
                indices_3plus = torch.nonzero(mask_3plus, as_tuple=False).squeeze(1)
                
                V = cano_pts.detach().cpu().numpy()          # [N,3] float
                idx_np = (indices_3plus.detach().cpu().numpy()
                        if isinstance(indices_3plus, torch.Tensor) else indices_3plus)
                N = V.shape[0]
                colors = np.tile(np.array([180, 180, 180, 255], dtype=np.uint8), (N, 1))  # [N,4]
                highlight = np.array([255, 100, 0, 255], dtype=np.uint8)
                colors[idx_np] = highlight
                pc = trimesh.points.PointCloud(vertices=V, colors=colors)  # colors: (N,4) uint8
                pc.export('./test_data/points_highlighted.ply')
        
        return posed_gaussians, self.avatar_net, self.dataset, cano_pts, rotations, cano_J, joint_mat, A_mat, extr, knn_indices, bone_cano, bone_index, bone_faces # cano_rot
           
          
           
if __name__ == '__main__':
    torch.manual_seed(31359)
    np.random.seed(31359)
    # torch.autograd.set_detect_anomaly(True)
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()
    arg_parser.add_argument('-c', '--config_path', type = str, help = 'Configuration file path.')
    arg_parser.add_argument('-m', '--mode', type = str, help = 'Running mode.', default = 'train')
    args = arg_parser.parse_args()

    config.load_global_opt(args.config_path)
    if args.mode is not None:
        config.opt['mode'] = args.mode

    trainer = AvatarTrainer(config.opt)
    if config.opt['mode'] == 'train':
        if not safe_exists(config.opt['train']['net_ckpt_dir'] + '/pretrained') \
                and not safe_exists(config.opt['train']['pretrained_dir'])\
                and not safe_exists(config.opt['train']['prev_ckpt']):
            trainer.pretrain()
        trainer.train()
    elif config.opt['mode'] == 'test':
        trainer.test()
    else:
        raise NotImplementedError('Invalid running mode!')
