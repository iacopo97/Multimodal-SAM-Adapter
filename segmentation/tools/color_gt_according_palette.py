import mmcv
import warnings
import numpy as np
import mmseg_custom
from mmseg.datasets import build_dataloader, build_dataset
import os
import argparse
import glob
import os.path as osp



def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('cfg', help='test config file path')
    parser.add_argument('out_dir', help='output directory')
    args = parser.parse_args()

    return args


def show_result(
                img,
                result,
                palette=None,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None,
                opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        seg = result
        palette = np.array(palette)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            return img
        
def main():
    args = parse_args()
    cfg=mmcv.Config.fromfile(args.cfg)
    dataset = build_dataset(cfg.data.val)
    files=sorted(glob.glob(os.path.join(*[cfg.data.val.data_root,cfg.data.val.img_dir,'*.png'])))
    for file in files:
        img = file
        index=files.index(file)
        result = img.replace('images', 'annotations').replace('rgb_front.png', 'semantic_front.png')
        show_result(
            img,
            mmcv.imread(result,'unchanged'),
            palette=dataset.PALETTE,
            out_file=osp.join(args.out_dir, dataset.img_infos[index]['filename']),
            opacity=0.5)

    
if __name__ == '__main__':
    main()