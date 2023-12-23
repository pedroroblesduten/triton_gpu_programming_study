import triton

MAX_FUSED_SIZE = 65536  # 2**16
next_power_of_2 = triton.next_power_of_2

def calculate_settings_a(n):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

def calculate_settings(n):
    MAX_BLOCK_SIZE = 1024  # Maximum threads per block for NVIDIA GPUs
    WARP_SIZE = 32  # Warp size for NVIDIA GPUs

    # Adjust the block size to be a multiple of the warp size, not exceeding MAX_BLOCK_SIZE
    BLOCK_SIZE = min(MAX_BLOCK_SIZE, next_power_of_2(n))
    while BLOCK_SIZE % WARP_SIZE != 0:
        BLOCK_SIZE -= WARP_SIZE

    # Adjusting num_warps based on the block size
    num_warps = BLOCK_SIZE // WARP_SIZE
    num_warps = min(max(num_warps, 4), 32)  # Ensuring num_warps is between 4 and 32

    return BLOCK_SIZE, num_warps

