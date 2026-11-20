"""Equivalent MATH-only SDPA context for PyTorch 2.0 and newer releases."""
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def math_attention():
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:
    from torch.backends.cuda import sdp_kernel

    def math_attention():
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
