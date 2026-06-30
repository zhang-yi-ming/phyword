import torch
import torch.nn as nn

try:
    from .wan21_vae_encoder import (
        CausalConv3d,
        Decoder3d,
        DEFAULT_WAN21_VAE_CKPT,
        WAN21_LATENT_MEAN,
        WAN21_LATENT_STD,
        _load_wan21_vae_state_dict,
        _prefixed_state_dict,
        count_conv3d,
    )
except ImportError:
    from wan21_vae_encoder import (
        CausalConv3d,
        Decoder3d,
        DEFAULT_WAN21_VAE_CKPT,
        WAN21_LATENT_MEAN,
        WAN21_LATENT_STD,
        _load_wan21_vae_state_dict,
        _prefixed_state_dict,
        count_conv3d,
    )


class Wan21VAEDecoder(nn.Module):
    """Wan2.1 VAE decoder wrapper.

    Input:
        Normalized latent tensor shaped [B, 16, T_lat, H_lat, W_lat].

    Output:
        Reconstructed pixel tensor shaped [B, 3, T, H_lat*8, W_lat*8],
        with values in [-1, 1] by default.
    """

    latent_channels = 16
    spatial_compression = 8
    temporal_compression = 4

    def __init__(
        self,
        vae_pth=DEFAULT_WAN21_VAE_CKPT,
        dtype=torch.float32,
        device=None,
        freeze=True,
        normalized_latents=True,
        clamp_output=True,
    ):
        super().__init__()
        self.normalized_latents = normalized_latents
        self.clamp_output = clamp_output

        self.post_quant_conv = CausalConv3d(self.latent_channels, self.latent_channels, 1)
        self.decoder = Decoder3d(
            dim=96,
            z_dim=self.latent_channels,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_upsample=[True, True, False],
            dropout=0.0,
        )

        state_dict = _load_wan21_vae_state_dict(vae_pth)
        self.post_quant_conv.load_state_dict(_prefixed_state_dict(state_dict, 'conv2'))
        self.decoder.load_state_dict(_prefixed_state_dict(state_dict, 'decoder'))

        latent_mean = torch.tensor(WAN21_LATENT_MEAN).view(
            1, self.latent_channels, 1, 1, 1
        )
        latent_inv_std = (1.0 / torch.tensor(WAN21_LATENT_STD)).view(
            1, self.latent_channels, 1, 1, 1
        )
        self.register_buffer('latent_mean', latent_mean, persistent=False)
        self.register_buffer('latent_inv_std', latent_inv_std, persistent=False)

        if dtype is not None:
            self.to(dtype=dtype)
        if device is not None:
            self.to(device=device)
        if freeze:
            self.requires_grad_(False)
        self.eval()

    @staticmethod
    def get_pixel_num_frames(latent_frames):
        return 1 + (int(latent_frames) - 1) * Wan21VAEDecoder.temporal_compression

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_map = [None] * count_conv3d(self.decoder)

    def forward(self, z):
        squeeze_batch = False
        if z.ndim == 4:
            z = z.unsqueeze(0)
            squeeze_batch = True
        if z.ndim != 5:
            raise ValueError(f'Expected [B, 16, T_lat, H_lat, W_lat], got {tuple(z.shape)}')

        b, c, t, h, w = z.shape
        if c != self.latent_channels:
            raise ValueError(f'Expected latent channel size 16, got {c}')

        param = next(self.parameters())
        z = z.to(device=param.device, dtype=param.dtype)
        if self.normalized_latents:
            z = z / self.latent_inv_std + self.latent_mean

        self.clear_cache()
        x = self.post_quant_conv(z)
        out = None
        for i in range(t):
            self._conv_idx = [0]
            chunk_out = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            out = chunk_out if out is None else torch.cat([out, chunk_out], dim=2)

        self.clear_cache()
        if self.clamp_output:
            out = out.clamp(-1, 1)
        return out.squeeze(0) if squeeze_batch else out

    def decode(self, z):
        return self.forward(z)
