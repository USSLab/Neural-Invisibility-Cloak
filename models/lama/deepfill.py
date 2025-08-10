import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple, Union
from mmcv.cnn import ConvModule
from mmcv.cnn import build_activation_layer
from mmengine.model.weight_init import kaiming_init


class LinearModule(nn.Module):
    """A linear block that contains linear/norm/activation layers.

    For low level vision, we add spectral norm and padding layer.

    Args:
        in_features (int): Same as nn.Linear.
        out_features (int): Same as nn.Linear.
        bias (bool): Same as nn.Linear. Default: True.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in linear module.
            Default: False.
        order (tuple[str]): The order of linear/activation layers. It is a
            sequence of "linear", "norm" and "act". Examples are
            ("linear", "act") and ("act", "linear").
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_cfg: Optional[dict] = dict(type="ReLU"),
        inplace: bool = True,
        with_spectral_norm: bool = False,
        order: Tuple[str, str] = ("linear", "act"),
    ):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 2
        assert set(order) == set(["linear", "act"])

        self.with_activation = act_cfg is not None
        self.with_bias = bias

        # build linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # export the attributes of self.linear to a higher level for
        # convenience
        self.in_features = self.linear.in_features
        self.out_features = self.linear.out_features

        if self.with_spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            act_cfg_.setdefault("inplace", inplace)
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default
        self.init_weights()

    def init_weights(self) -> None:
        """Init weights for the model."""
        if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
            nonlinearity = "leaky_relu"
            a = self.act_cfg.get("negative_slope", 0.01)
        else:
            nonlinearity = "relu"
            a = 0

        kaiming_init(self.linear, a=a, nonlinearity=nonlinearity)

    def forward(self, x: Tensor, activate: Optional[bool] = True) -> Tensor:
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of :math:`(n, *, c)`.
                Same as ``torch.nn.Linear``.
            activate (bool, optional): Whether to use activation layer.
                Defaults to True.

        Returns:
            torch.Tensor: Same as ``torch.nn.Linear``.
        """
        for layer in self.order:
            if layer == "linear":
                x = self.linear(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class MultiLayerDiscriminator(nn.Module):
    """Multilayer Discriminator.

    This is a commonly used structure with stacked multiply convolution layers.

    Args:
        in_channels (int): Input channel of the first input convolution.
        max_channels (int): The maximum channel number in this structure.
        num_conv (int): Number of stacked intermediate convs (including input
            conv but excluding output conv). Default to 5.
        fc_in_channels (int | None): Input dimension of the fully connected
            layer. If `fc_in_channels` is None, the fully connected layer will
            be removed. Default to None.
        fc_out_channels (int): Output dimension of the fully connected layer.
            Default to 1024.
        kernel_size (int): Kernel size of the conv modules. Default to 5.
        conv_cfg (dict): Config dict to build conv layer.
        norm_cfg (dict): Config dict to build norm layer.
        act_cfg (dict): Config dict for activation layer, "relu" by default.
        out_act_cfg (dict): Config dict for output activation, "relu" by
            default.
        with_input_norm (bool): Whether add normalization after the input conv.
            Default to True.
        with_out_convs (bool): Whether add output convs to the discriminator.
            The output convs contain two convs. The first out conv has the same
            setting as the intermediate convs but a stride of 1 instead of 2.
            The second out conv is a conv similar to the first out conv but
            reduces the number of channels to 1 and has no activation layer.
            Default to False.
        with_spectral_norm (bool): Whether use spectral norm after the conv
            layers. Default to False.
        kwargs (keyword arguments).
    """

    def __init__(
        self,
        in_channels: int = 4,
        max_channels: int = 256,
        num_convs: int = 6,
        kernel_size: int = 5,
        act_cfg: Optional[dict] = dict(type='LeakyReLU', negative_slope=0.2),
        with_spectral_norm: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.max_channels = max_channels
        self.num_convs = num_convs

        cur_channels = in_channels
        for i in range(num_convs):
            out_ch = min(64 * 2**i, max_channels)
            act_cfg_ = act_cfg
            self.add_module(
                f"conv{i + 1}",
                ConvModule(
                    cur_channels,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    norm_cfg=None,
                    act_cfg=act_cfg_,
                    with_spectral_norm=with_spectral_norm,
                    **kwargs,
                ),
            )
            cur_channels = out_ch

    def forward(self, x: Tensor) -> Tensor:
        """Forward Function.

        Args:
            x (torch.Tensor): Input tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w') or (n, c).
        """
        # out_convs has two additional ConvModules
        num_convs = self.num_convs
        for i in range(num_convs):
            x = getattr(self, f"conv{i + 1}")(x)
        return x


class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge', 'l1'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(
        self,
        gan_type: str,
        real_label_val: float = 1.0,
        fake_label_val: float = 0.0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.gan_type = gan_type
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss_weight = loss_weight
        if self.gan_type == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan":
            self.loss = self._wgan_loss
        elif self.gan_type == "hinge":
            self.loss = nn.ReLU()
        elif self.gan_type == "l1":
            self.loss = nn.L1Loss()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} is not implemented.")

    def _wgan_loss(self, input: torch.Tensor, target: bool) -> torch.Tensor:
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return -input.mean() if target else input.mean()

    def get_target_label(
        self, input: torch.Tensor, target_is_real: bool
    ) -> Union[bool, torch.Tensor]:
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type == "wgan":
            return target_is_real
        target_val = self.real_label_val if target_is_real else self.fake_label_val
        return input.new_ones(input.size()) * target_val

    def forward(
        self,
        input: torch.Tensor,
        target_is_real: bool,
        is_disc: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the target is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.
            mask (Tensor): The mask tensor. Default: None.

        Returns:
            Tensor: GAN loss value.
        """

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == "hinge":
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
