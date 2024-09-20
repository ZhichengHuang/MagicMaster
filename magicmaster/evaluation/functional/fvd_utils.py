# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import os
import os.path as osp
import pickle
import sys
from contextlib import contextmanager
from copy import deepcopy
from typing import Optional, Tuple

import mmengine
import numpy as np
import torch
import torch.nn as nn
from mmengine import is_filepath, print_log
from mmengine.dataset import BaseDataset, Compose, pseudo_collate
from mmengine.dist import (all_gather, get_dist_info, get_world_size,
                           is_main_process)
from mmengine.evaluator import BaseMetric
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.models.inception import inception_v3

from magicmaster.utils import magicmaster_CACHE_DIR, download_from_url
from . import InceptionV3

ALLOWED_INCEPTION = ['StyleGAN', 'PyTorch']
TERO_INCEPTION_URL = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt'  # noqa



def load_i3d(i3d_args, metric):
    """Load Inception Model from given ``inception_args`` and ``metric``.

    This function would try to load Inception under the guidance of 'type'
    given in `inception_args`, if not given, we would try best to load Tero's
    ones. In detail, we would first try to load the model from disk with
    the given 'inception_path', and then try to download the checkpoint from
    'inception_url'. If both method are failed, pytorch version of Inception
    would be loaded.

    Args:
        inception_args (dict): Keyword args for inception net.
        metric (string): Metric to use the Inception. This argument would
            influence the pytorch's Inception loading.

    Returns:
        model (torch.nn.Module): Loaded Inception model.
        style (string): The version of the loaded Inception.
    """

    if not isinstance(i3d_args, dict):
        raise TypeError('Receive invalid \'i3d_args\': '
                        f'\'{i3d_args}\'')

    _i3d_args = deepcopy(i3d_args)
    inception_type = _i3d_args.pop('type', None)

    if torch.__version__ < '1.6.0':
        print_log(
            'Current Pytorch Version not support script module, load '
            'Inception Model from torch model zoo. If you want to use '
            'Tero\' script model, please update your Pytorch higher '
            f'than \'1.6\' (now is {torch.__version__})', 'current')
        return _load_i3d_torch(_i3d_args, metric), 'pytorch'

    # load pytorch version is specific
    if inception_type != 'StyleGAN':
        return _load_i3d_torch(_i3d_args, metric), 'pytorch'

    # try to load Tero's version
    path = _i3d_args.get('inception_path', TERO_INCEPTION_URL)
    if path is None:
        path = TERO_INCEPTION_URL

    # try to parse `path` as web url and download
    if 'http' not in path:
        model = _load_i3d_from_path(path)
        if isinstance(model, torch.nn.Module):
            return model, 'StyleGAN'

    # try to parse `path` as path on disk
    model = _load_i3d_from_url(path)
    if isinstance(model, torch.nn.Module):
        return model, 'StyleGAN'

    raise RuntimeError('Cannot Load Inception Model, please check the input '
                       f'`inception_args`: {i3d_args}')


def _load_i3d_from_path(inception_path):
    """Load inception from passed path.

    Args:
        inception_path (str): The path of inception.

    Returns:
        nn.Module: The loaded inception.
    """
    print_log(
        'Try to load Tero\'s Inception Model from '
        f'\'{inception_path}\'.', 'current')
    try:
        model = torch.jit.load(inception_path)
        print_log('Load Tero\'s Inception Model successfully.', 'current')
    except Exception as e:
        model = None
        print_log('Load Tero\'s Inception Model failed. '
                  f'\'{e}\' occurs.', 'current')
    return model


def _load_i3d_from_url(inception_url: str) -> nn.Module:
    """Load Inception network from the give `inception_url`"""
    inception_url = inception_url if inception_url else TERO_INCEPTION_URL
    print_log(f'Try to download Inception Model from {inception_url}...',
              'current')
    try:
        path = download_from_url(inception_url, dest_dir=magicmaster_CACHE_DIR)
        print_log('Download Finished.', 'current')
        return _load_i3d_from_path(path)
    except Exception as e:
        print_log(f'Download Failed. {e} occurs.', 'current')
        return None


def _load_i3d_torch(inception_args, metric) -> nn.Module:
    """Load Inception network from PyTorch's model zoo."""
    assert metric in ['FVD', 'IS']
    # if metric == 'FID':
    #     inception_model = InceptionV3([3], **inception_args)
    # elif metric == 'IS':
    #     inception_model = inception_v3(pretrained=True, transform_input=False)
    print_log(
        'Load Inception V3 Network from Pytorch Model Zoo '
        'for IS calculation. The results can only used '
        'for monitoring purposes. To get more accuracy IS, '
        'please use Tero\'s Inception V3 checkpoints '
        'and use Bicubic Interpolation with Pillow backend '
        'for image resizing. More details may refer to '
        'https://github.com/open-mmlab/MMEditing/blob/master/docs/en/quick_run.md#is.',  # noqa
        'current')
    return None


def get_i3d_feat_cache_name_and_args(dataloader: DataLoader,
                                           metric: BaseMetric, real_nums: int,
                                           capture_mean_cov: bool,
                                           capture_all: bool
                                           ) -> Tuple[str, dict]:
    """Get the name and meta info of the inception feature cache file
    corresponding to the input dataloader and metric.

    The meta info includes
    'data_root', 'data_prefix', 'meta_info' and 'pipeline' of the dataset, and
    'inception_style' and 'inception_args' of the metric. Then we calculate the
    hash value of the meta info dict with md5, and the name of the inception
    feature cache will be 'inception_feat_{HASH}.pkl'.
    Args:
        dataloader (Dataloader): The dataloader of real images.
        metric (BaseMetric): The metric which needs inception features.
        real_nums (int): Number of images used to extract inception feature.
        capture_mean_cov (bool): Whether save the mean and covariance of
            inception feature. Defaults to False.
        capture_all (bool): Whether save the raw inception feature. Defaults
            to False.
    Returns:
        Tuple[str, dict]: Filename and meta info dict of the inception feature
            cache.
    """

    dataset: BaseDataset = dataloader.dataset
    assert isinstance(dataset, Dataset), (
        f'Only support normal dataset, but receive {type(dataset)}.')

    # get dataset info
    data_root = deepcopy(dataset.data_root)
    # data_prefix = deepcopy(dataset.data_prefix)
    # metainfo = dataset.metainfo
    pipeline = repr(dataset.pipeline)

    # get metric info
    inception_style = metric.inception_style
    inception_args = getattr(metric, 'inception_args', None)

    real_key = 'gt_img' if metric.real_key is None else metric.real_key
    args = dict(
        data_root=data_root,
        # data_prefix=data_prefix,
        # metainfo=metainfo,
        pipeline=pipeline,
        inception_style=inception_style,
        inception_args=inception_args,
        # save `num_gpus` because this may influence the data loading order
        num_gpus=get_world_size(),
        capture_mean_cov=capture_mean_cov,
        capture_all=capture_all,
        real_keys=real_key,
        real_nums=real_nums)

    real_nums_str = 'full' if real_nums == -1 else str(real_nums)
    md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
    if capture_all:
        prefix = 'i3d_state-capture_all'
    elif capture_mean_cov:
        prefix = 'i3d_state-capture_mean_cov'
    else:
        prefix = 'i3d_state-capture_all_mean_cov'
    cache_tag = f'{prefix}-{real_nums_str}-{md5.hexdigest()}.pkl'
    return cache_tag, args


def prepare_i3d_feat(dataloader: DataLoader,
                           metric: BaseMetric,
                           data_preprocessor: Optional[nn.Module] = None,
                           capture_mean_cov: bool = False,
                           capture_all: bool = False) -> dict:
    """Prepare inception feature for the input metric.

    - If `metric.inception_pkl` is an online path, try to download and load
      it. If cannot download or load, corresponding error will be raised.
    - If `metric.inception_pkl` is local path and file exists, try to load the
      file. If cannot load, corresponding error will be raised.
    - If `metric.inception_pkl` is local path and file not exists, we will
      extract the inception feature manually and save to 'inception_pkl'.
    - If `metric.inception_pkl` is not defined, we will extract the inception
      feature and save it to default cache dir with default name.

    Args:
        dataloader (Dataloader): The dataloader of real images.
        metric (BaseMetric): The metric which needs inception features.
        data_preprocessor (Optional[nn.Module]): Data preprocessor of the
            module. Used to preprocess the real images. If not passed, real
            images will automatically normalized to [-1, 1]. Defaults to None.
        capture_mean_cov (bool): Whether save the mean and covariance of
            inception feature. Defaults to False.
        capture_all (bool): Whether save the raw inception feature. If true,
            it will take a lot of time to save the inception feature. Defaults
            to False.

    Returns:
        dict: Dict contains inception feature.
    """
    assert capture_mean_cov or capture_all, (
        'At least one of \'capture_mean_cov\' and \'capture_all\' is True.')
    if not hasattr(metric, 'inception_pkl'):
        return
    inception_pkl: Optional[str] = metric.inception_pkl

    if isinstance(inception_pkl, str):
        if is_filepath(inception_pkl) and osp.exists(inception_pkl):
            with open(inception_pkl, 'rb') as file:
                inception_state = pickle.load(file)
            print_log(
                f'\'{metric.prefix}\' successful load inception feature '
                f'from \'{inception_pkl}\'', 'current')
            return inception_state
        elif inception_pkl.startswith('s3'):
            try:
                raise NotImplementedError(
                    'Not support download from Ceph currently')
            except Exception as exp:
                raise exp('Not support download from Ceph currently')
        elif inception_pkl.startswith('http'):
            try:
                raise NotImplementedError(
                    'Not support download from url currently')
            except Exception as exp:
                # cannot download, raise error
                raise exp('Not support download from url currently')

    # cannot load or download from file, extract manually
    assert hasattr(metric, 'real_nums'), (
        f'Metric \'{metric.name}\' must have attribute \'real_nums\'.')
    real_nums = metric.real_nums
    if inception_pkl is None:
        inception_pkl, args = get_i3d_feat_cache_name_and_args(
            dataloader, metric, real_nums, capture_mean_cov, capture_all)
        inception_pkl = osp.join(magicmaster_CACHE_DIR, inception_pkl)
    else:
        args = dict()
    if osp.exists(inception_pkl):
        with open(inception_pkl, 'rb') as file:
            real_feat = pickle.load(file)
        print_log(f'load preprocessed feat from {inception_pkl}', 'current')
        return real_feat

    assert hasattr(metric, 'i3d'), (
        'Metric must have a inception network to extract inception features.')

    real_feat = []

    print_log(
        f'Inception pkl \'{inception_pkl}\' is not found, extract '
        'manually.', 'current')

    import rich.progress

    dataset, batch_size = dataloader.dataset, dataloader.batch_size
    if real_nums == -1:
        num_items = len(dataset)
    else:
        num_items = min(len(dataset), real_nums)

    rank, num_gpus = get_dist_info()
    item_subset = [(i * num_gpus + rank) % num_items
                   for i in range((num_items - 1) // num_gpus + 1)]
    inception_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=item_subset,
        collate_fn=pseudo_collate,
        shuffle=False,
        drop_last=False)
    # init rich pbar for the main process
    if is_main_process():
        # check the launcher
        slurm_env_name = ['SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NODELIST']
        if all([n in os.environ for n in slurm_env_name]):
            is_slurm = True
            pbar = mmengine.ProgressBar(len(inception_dataloader))
        else:
            is_slurm = False
            columns = [
                rich.progress.TextColumn('[bold blue]{task.description}'),
                rich.progress.BarColumn(bar_width=40),
                rich.progress.TaskProgressColumn(),
                rich.progress.TimeRemainingColumn(),
            ]
            pbar = rich.progress.Progress(*columns)
            pbar.start()
            task = pbar.add_task(
                'Calculate Inception Feature.',
                total=len(inception_dataloader),
                visible=True)

    for data in inception_dataloader:
        # need to update
        # set training = False to avoid norm + convert to BGR
        data_samples = data_preprocessor(data, False)['data_samples']

        # real_key = 'gt_img' if metric.real_key is None else metric.real_key
        real_key = 'gt_img'

        img = getattr(data_samples, real_key) # b,t,c,w,h
        # we need to formulate to [b,c,t,w,h]
        img = img.permute(0,2,1,3,4)

        real_feat_ = metric.forward_i3d(img).cpu()
        real_feat.append(real_feat_)

        if is_main_process():
            if is_slurm:
                pbar.update(1)
            else:
                pbar.update(task, advance=1)

    # stop the pbar
    if is_main_process():
        if is_slurm:
            sys.stdout.write('\n')
        else:
            pbar.stop()

    # collect results
    real_feat = torch.cat(real_feat)
    # use `all_gather` here, gather tensor is much quicker than gather object.
    real_feat = all_gather(real_feat)

    # only cat on the main process
    if is_main_process():
        inception_state = dict(**args)
        if capture_mean_cov:
            real_feat = torch.cat(real_feat, dim=0)[:num_items].numpy()
            real_mean = np.mean(real_feat, 0)
            real_cov = np.cov(real_feat, rowvar=False)
            inception_state['real_mean'] = real_mean
            inception_state['real_cov'] = real_cov
        if capture_all:
            inception_state['raw_feature'] = real_feat
        dir_name = osp.dirname(inception_pkl)
        os.makedirs(dir_name, exist_ok=True)
        print_log(
            f'Saving i3d pkl to {inception_pkl}. Please be patient.',
            'current')
        with open(inception_pkl, 'wb') as file:
            pickle.dump(inception_state, file)
        print_log('I3D pkl Finished.', 'current')
        return inception_state
