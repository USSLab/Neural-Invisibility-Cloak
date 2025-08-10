from omegaconf import DictConfig
from utils.io import read_rgb_img
from .base import BaseBackdoor, Paster


class BasicBackdoor(BaseBackdoor):
    def __init__(self, paster: Paster, trigger: DictConfig, target: DictConfig):
        super().__init__(paster)
        self.x_t = read_rgb_img(trigger.path, trigger.size, only_upsample=False, square_crop=trigger.size).cuda()
        self.y_t = read_rgb_img(target.path, target.size, only_upsample=False, square_crop=target.size).cuda()
        self.mx_t = None
        self.my_t = None
        if trigger.get("mask", None) is not None:
            self.mx_t = read_rgb_img(trigger.mask, trigger.size, only_upsample=False, square_crop=trigger.size).cuda()
        if target.get("mask", None) is not None:
            self.my_t = read_rgb_img(target.mask, target.size, only_upsample=False, square_crop=target.size).cuda()

    def get_pasted(self):
        return self.x_t, self.y_t, self.mx_t, self.my_t
