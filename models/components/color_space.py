import torch


_M_RGB2YUV = torch.tensor([
    [0.299,   0.587,   0.114 ],
    [-0.169, -0.331,   0.500 ],
    [0.500,  -0.419,  -0.081 ],
])

_M_YUV2RGB = torch.linalg.inv(_M_RGB2YUV)

_M_RGB2LMS = torch.tensor([
    [0.359283,  0.697605, -0.035891],
    [-0.192072,  1.100477,  0.075363],
    [0.007079,  0.074839,  0.843326],
])

_M_LMS2ICTCP = torch.tensor([
    [0.5,         0.5,         0.0],
    [1.613760,   -3.323486,    1.709727],
    [4.378089,   -4.878698,    0.500608],
])

_M_ICTCP2LMS = torch.linalg.inv(_M_LMS2ICTCP)
_M_LMS2RGB = torch.linalg.inv(_M_RGB2LMS)

_M_YUV2LMS = _M_RGB2LMS @ _M_YUV2RGB
_M_LMS2YUV = _M_RGB2YUV @ _M_LMS2RGB

_device_cache: dict = {}


def _get_matrices(device, dtype):
    key = (device, dtype)
    if key not in _device_cache:
        _device_cache[key] = (
            _M_YUV2LMS.to(device=device, dtype=dtype),
            _M_LMS2ICTCP.to(device=device, dtype=dtype),
            _M_ICTCP2LMS.to(device=device, dtype=dtype),
            _M_LMS2YUV.to(device=device, dtype=dtype),
        )
    return _device_cache[key]


@torch.jit.script
def _yuv2ictcp_jit(x, m1, m2):
    yuv = (x + 1) * 127.5
    y = yuv[:, 0]; u = yuv[:, 1] - 128.0; v = yuv[:, 2] - 128.0
    l = m1[0, 0]*y + m1[0, 1]*u + m1[0, 2]*v
    m = m1[1, 0]*y + m1[1, 1]*u + m1[1, 2]*v
    s = m1[2, 0]*y + m1[2, 1]*u + m1[2, 2]*v
    l, m, s = l / 255, m / 255, s / 255
    l = l.sign() * l.abs().pow(1 / 2.4)
    m = m.sign() * m.abs().pow(1 / 2.4)
    s = s.sign() * s.abs().pow(1 / 2.4)
    i = m2[0, 0]*l + m2[0, 1]*m + m2[0, 2]*s
    ct = m2[1, 0]*l + m2[1, 1]*m + m2[1, 2]*s
    cp = m2[2, 0]*l + m2[2, 1]*m + m2[2, 2]*s
    return torch.stack([i, ct, cp], dim=1)


@torch.jit.script
def _ictcp2yuv_jit(x, m3, m4):
    i = x[:, 0]; ct = x[:, 1]; cp = x[:, 2]
    pq_l = m3[0, 0]*i + m3[0, 1]*ct + m3[0, 2]*cp
    pq_m = m3[1, 0]*i + m3[1, 1]*ct + m3[1, 2]*cp
    pq_s = m3[2, 0]*i + m3[2, 1]*ct + m3[2, 2]*cp
    l = pq_l.sign() * pq_l.abs().pow(2.4)
    m = pq_m.sign() * pq_m.abs().pow(2.4)
    s = pq_s.sign() * pq_s.abs().pow(2.4)
    l, m, s = l * 255, m * 255, s * 255
    y = m4[0, 0]*l + m4[0, 1]*m + m4[0, 2]*s
    u = m4[1, 0]*l + m4[1, 1]*m + m4[1, 2]*s + 128.0
    v = m4[2, 0]*l + m4[2, 1]*m + m4[2, 2]*s + 128.0
    return torch.stack([y, u, v], dim=1) / 127.5 - 1.0


def yuv_to_ictcp(x: torch.Tensor) -> torch.Tensor:
    m1, m2, _, _ = _get_matrices(x.device, x.dtype)
    return _yuv2ictcp_jit(x, m1, m2)


def ictcp_to_yuv(x: torch.Tensor) -> torch.Tensor:
    _, _, m3, m4 = _get_matrices(x.device, x.dtype)
    return _ictcp2yuv_jit(x, m3, m4)


def _pq(x):
    return x.sign() * x.abs().pow(1 / 2.4)


def _pq_inv(x):
    return x.sign() * x.abs().pow(2.4)


def yuv_to_rgb(x: torch.Tensor) -> torch.Tensor:
    yuv = (x + 1) * 127.5
    y_ch = yuv[:, 0:1]
    u_ch = yuv[:, 1:2] - 128.0
    v_ch = yuv[:, 2:3] - 128.0
    m = _M_YUV2RGB.to(device=x.device, dtype=x.dtype)
    r = m[0, 0] * y_ch + m[0, 1] * u_ch + m[0, 2] * v_ch
    g = m[1, 0] * y_ch + m[1, 1] * u_ch + m[1, 2] * v_ch
    b = m[2, 0] * y_ch + m[2, 1] * u_ch + m[2, 2] * v_ch
    return torch.cat([r, g, b], dim=1) / 127.5 - 1.0


def rgb_to_yuv(x: torch.Tensor) -> torch.Tensor:
    rgb = (x + 1) * 127.5
    r_ch, g_ch, b_ch = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    m = _M_RGB2YUV.to(device=x.device, dtype=x.dtype)
    y = m[0, 0] * r_ch + m[0, 1] * g_ch + m[0, 2] * b_ch
    u = m[1, 0] * r_ch + m[1, 1] * g_ch + m[1, 2] * b_ch + 128.0
    v = m[2, 0] * r_ch + m[2, 1] * g_ch + m[2, 2] * b_ch + 128.0
    return torch.cat([y, u, v], dim=1) / 127.5 - 1.0


def rgb_to_ictcp(rgb: torch.Tensor) -> torch.Tensor:
    x = (rgb + 1) / 2
    m1 = _M_RGB2LMS.to(device=x.device, dtype=x.dtype)
    r_ch, g_ch, b_ch = x[:, 0], x[:, 1], x[:, 2]
    lms_l = m1[0, 0] * r_ch + m1[0, 1] * g_ch + m1[0, 2] * b_ch
    lms_m = m1[1, 0] * r_ch + m1[1, 1] * g_ch + m1[1, 2] * b_ch
    lms_s = m1[2, 0] * r_ch + m1[2, 1] * g_ch + m1[2, 2] * b_ch
    pq_l = _pq(lms_l)
    pq_m = _pq(lms_m)
    pq_s = _pq(lms_s)
    m2 = _M_LMS2ICTCP.to(device=x.device, dtype=x.dtype)
    i_ch = m2[0, 0] * pq_l + m2[0, 1] * pq_m + m2[0, 2] * pq_s
    ct_ch = m2[1, 0] * pq_l + m2[1, 1] * pq_m + m2[1, 2] * pq_s
    cp_ch = m2[2, 0] * pq_l + m2[2, 1] * pq_m + m2[2, 2] * pq_s
    return torch.stack([i_ch, ct_ch, cp_ch], dim=1)


def ictcp_to_rgb(ictcp: torch.Tensor) -> torch.Tensor:
    x = ictcp
    m_inv = _M_ICTCP2LMS.to(device=x.device, dtype=x.dtype)
    i_ch, ct_ch, cp_ch = x[:, 0], x[:, 1], x[:, 2]
    pq_l = m_inv[0, 0] * i_ch + m_inv[0, 1] * ct_ch + m_inv[0, 2] * cp_ch
    pq_m = m_inv[1, 0] * i_ch + m_inv[1, 1] * ct_ch + m_inv[1, 2] * cp_ch
    pq_s = m_inv[2, 0] * i_ch + m_inv[2, 1] * ct_ch + m_inv[2, 2] * cp_ch
    lms_l = _pq_inv(pq_l)
    lms_m = _pq_inv(pq_m)
    lms_s = _pq_inv(pq_s)
    m_rgb = _M_LMS2RGB.to(device=x.device, dtype=x.dtype)
    r_ch = m_rgb[0, 0] * lms_l + m_rgb[0, 1] * lms_m + m_rgb[0, 2] * lms_s
    g_ch = m_rgb[1, 0] * lms_l + m_rgb[1, 1] * lms_m + m_rgb[1, 2] * lms_s
    b_ch = m_rgb[2, 0] * lms_l + m_rgb[2, 1] * lms_m + m_rgb[2, 2] * lms_s
    return torch.stack([r_ch, g_ch, b_ch], dim=1) * 2 - 1
