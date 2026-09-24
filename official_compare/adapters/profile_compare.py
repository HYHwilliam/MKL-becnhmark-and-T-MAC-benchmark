from t_mac.ops import GeMMCodegen, GeMMCLCodegen, QGeMMLUTBitsCodegen, QGeMMLUTBitsPreprocessorCodegen
import t_mac.utils
import logging
import os
from typing import Tuple
import pandas as pd
import argparse

from typing import Optional
import time
import gc

logger = logging.getLogger("profile")


def profile_codegen(
    MKN: Tuple[int, int, int],
    bits: int,
    num_threads: int,
    target: str,
    remote_kwargs: Optional[dict] = None,
    dtype: str = "int8",
    cc: Optional[str] = None,
    cc_opts: Optional[list] = None,
    eval_kwargs: Optional[dict] = None,
    out_dtype: str = "float16",
    aggregation_dtype: str = "int32",
):
    M, K, N = MKN
    m_groups = FLAGS.m_groups
    if target == "opencl" or target == "vulkan":
        target_host = "llvm -mtriple=arm64-linux-android -mattr=+neon"
    else:
        target_host = None

    codegen_keys = [
        FLAGS.kernel,
    ]

    codegen_kwargs = {
        "dtype": dtype,
        "target": target,
        "save_dir": FLAGS.out_path,
        "verify": True,
        "target_host": target_host,
        "tune": FLAGS.tune,
        "reuse_tuned": FLAGS.reuse_tuned,
        "remote_kwargs": remote_kwargs,
        "bits": bits,
        "cc": cc,
        "cc_opts": cc_opts,
        "out_dtype": out_dtype,
        "act_group_size": FLAGS.act_group_size if FLAGS.act_group_size != -1 else K,
        "num_threads": num_threads,
    }

    if target == "opencl":
        codegen_keys = [k + "_cl" for k in codegen_keys]
    elif target == "vulkan":
        codegen_keys = [k + "_vulkan" for k in codegen_keys]

    codegen_cls = {
        "gemm": GeMMCodegen,
        "gemm_cl": GeMMCLCodegen,
        "qgemm_lut": QGeMMLUTBitsCodegen,
        "preprocessor": QGeMMLUTBitsPreprocessorCodegen,
    }

    args = {
        "qgemm_lut": (M, N, K),
        "preprocessor": (N, K),
    }

    extra_kwargs = {
        "qgemm_lut": {
            "group_size": FLAGS.group_size,
            "fast_aggregation": FLAGS.fast_aggregation,
            "m_groups": m_groups,
            "aggregation_dtype": aggregation_dtype,
            "zero_point": False,
        },
        "preprocessor": {
            "M": M,
        },
    }

    def _eval(codegen_key):
        codegen = codegen_cls[codegen_key](name=codegen_key, **codegen_kwargs, **extra_kwargs[codegen_key])
        return 1000 * codegen.evaluate(
            *args[codegen_key],
            thread_affinity=FLAGS.thread_affinity,
            **eval_kwargs,
        )

    return {
        k: _eval(k)
        for k in codegen_keys
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="out")
    parser.add_argument("-d", "--device", type=str, choices=t_mac.utils.get_devices(), default="")
    parser.add_argument("-tgt", "--target", type=str, choices=["llvm", "opencl", "vulkan"], default="llvm")
    parser.add_argument("-ta", "--thread_affinity", type=int, default=1)
    parser.add_argument("-k", "--kernel", type=str, choices=["qgemm_lut", "preprocessor"], default="qgemm_lut")
    parser.add_argument("-t", "--tune", action="store_true")
    parser.add_argument("-r", "--reuse_tuned", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-mg", "--m_groups", type=int, default=-1)

    parser.add_argument("-gs", "--group_size", type=int, default=128)
    parser.add_argument("-ags", "--act_group_size", type=int, default=64, help="-1 for BitNet-like unified scale")
    parser.add_argument("-fa", "--fast_aggregation", action="store_true")
    parser.add_argument("--suite", choices=["smoke", "nscale1024", "vla_compare", "scaling", "hackmd_4096", "hackmd_safe", "hackmd"], default="smoke")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=2)
    parser.add_argument("--threads", type=str, default="1")
    return parser.parse_args()


def main():
    if FLAGS.suite == "smoke":
        MKNs = [
            [1024, 1024, 1],
        ]
    elif FLAGS.suite == "nscale1024":
        MKNs = [
            [1024, 1024, 1],
            [1024, 1024, 8],
            [1024, 1024, 32],
            [1024, 1024, 128],
            [1024, 1024, 512],
        ]
    elif FLAGS.suite == "vla_compare":
        MKNs = [
            # BitVLA exact shapes
            [2560, 2560, 8],      # Action Head residual FC
            [2560, 17920, 8],     # Action Head FC1
            [1152, 1152, 256],    # Vision Q/K/V/O

            # pi0 Action Expert proxies
            # Real suffix N=33 for chunk=32, but official T-MAC
            # requires N divisible by a legal BN, so N=32 is used.
            [2048, 1024, 32],     # Q
            [256,  1024, 32],     # K/V
            [1024, 2048, 32],     # O
            [4096, 1024, 32],     # Gate/Up
            [1024, 4096, 32],     # Down
        ]
    elif FLAGS.suite == "hackmd_4096":
        MKNs = [
            [4096, 4096, 4096],
        ]
    elif FLAGS.suite == "hackmd_safe":
        MKNs = [
            [256, 256, 256],
            [1024, 1024, 1024],
            [4096, 2048, 1024],
            [1024, 512, 1024],
        ]
    elif FLAGS.suite == "hackmd":
        MKNs = [
            [256, 256, 256],
            [1024, 1024, 1024],
            [4096, 4096, 4096],
            [4096, 2048, 1024],
            [1024, 512, 1024],
        ]
    else:
        batch_sizes = [1, 8, 32, 128, 512]
        groups = [
            [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)],
            [(1024, 4096), (2048, 8192), (4096, 11008), (4096, 14336)],
            [(4096, 1024), (8192, 2048), (11008, 4096), (14336, 4096)],
            [(1024, 2048), (2048, 1024), (2048, 4096), (4096, 2048)],
        ]

        MKNs = [
            [M, K, N]
            for group in groups
            for N in batch_sizes
            for M, K in group
        ]

    threads = [
        int(x.strip())
        for x in FLAGS.threads.split(",")
        if x.strip()
    ]

    dtypes = [
        "int8",
        # "float16",
    ]
    header = True
    bitss = [FLAGS.bits]

    device_kwargs = t_mac.utils.get_default_device_kwargs(FLAGS.device)

    for MKN in MKNs:
        for dtype in dtypes:
            for bits in bitss:
                for num_threads in threads:
                    results = {
                        "suite": FLAGS.suite,
                        "shape_mnk": f"{MKN[0]}x{MKN[2]}x{MKN[1]}",
                        "M": MKN[0],
                        "K": MKN[1],
                        "N": MKN[2],
                        "bits": bits,
                        "num_threads": num_threads,
                        "dtype": dtype,
                    }
                    _MKN = [MKN[0] * bits, MKN[1], MKN[2]]
                    results.update(
                        profile_codegen(
                            _MKN, bits, num_threads,
                            dtype=dtype,
                            **device_kwargs,
                        )
                    )
                    logger.info(results)

                    pd.DataFrame([results]).to_csv(
                        os.path.join(FLAGS.out_path, "results.csv"),
                        mode="a",
                        header=header,
                        index=False,
                    )
                    header = False

                    gc.collect()


if __name__ == "__main__":
    FLAGS = parse_args()

    if FLAGS.verbose:
        logging.basicConfig()
        logging.getLogger().setLevel(logging.INFO)

    main()
