"""Launcher policy checks without a GPU or device servers."""
from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from demo import buffering_options, device_command, launch_environment, parse_args, reconstruction_options
from host_buffering import memory_plan


class LaunchPolicyTests(unittest.TestCase):
    def test_output_blocks_freeze_capacity_and_check_host_consumer_budgets(self):
        args = parse_args(["--output-mode", "blocks", "--slices-per-block", "3"])
        buffers = buffering_options(args)
        self.assertEqual(buffers["output_mode"], "blocks")
        self.assertEqual(buffers["output_block_rows"], 3)
        args.slices_per_block = 32
        self.assertEqual(buffering_options(args, {"rows": 128})["output_block_rows"], 32)
        self.assertEqual(buffering_options(args, {"rows": 12})["output_block_rows"], 12)
        for flags in (["--output-mode", "blocks"], ["--output-host-mib", "0"],
                      ["--output-mode", "blocks", "--slices-per-block", "1", "--live",
                       "--output-host-mib", "1", "--slices", "24", "--pixels", "128"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_host_buffer_options_reach_scan_configuration_and_reject_oversized_layouts(self):
        self.assertEqual(buffering_options(parse_args([]))["sinogram_memory"], "gpu")
        args = parse_args(["--sinogram-memory", "host", "--host-buffer-mib", "32",
                           "--pinned-buffer-mib", "4", "--slices-per-block", "3"])
        buffers = buffering_options(args)
        self.assertEqual(buffers, dict(sinogram_memory="host", host_budget_bytes=32 * 1024**2,
                                       pinned_budget_bytes=4 * 1024**2))
        plan = memory_plan(dict(rows=args.slices, angles=args.angles, columns=args.pixels,
                                buffering=buffers), reconstruction_options(args))
        self.assertEqual(plan["gpu_sinogram_bytes"], 0)
        for flags in (["--sinogram-memory", "host", "--host-buffer-mib", "1", "--gpu-stress"],
                      ["--sinogram-memory", "host", "--pinned-buffer-mib", "1", "--gpu-stress"],
                      ["--host-buffer-mib", "0"], ["--pinned-buffer-mib", "-1"]):
            with self.subTest(flags=flags), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(flags)

    def test_slice_block_setting_reaches_scan_configuration(self):
        self.assertEqual(reconstruction_options(parse_args([]))["slices_per_block"], 0)
        for method in ("sirt", "fbp", "gridrec"):
            args = parse_args(["--algorithm", method, "--slices-per-block", "3", "--slices", "8"])
            self.assertEqual(reconstruction_options(args)["slices_per_block"], 3)
        for value in ("-1", "2.5"):
            with self.subTest(value=value), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args(["--slices-per-block", value])

    def test_profiles_and_interface_selection_reach_both_ucx_namespaces(self):
        for profile, tls, default_device in (
                ("tcp", "tcp,cuda_copy,self", "lo"),
                ("auto", "all", None), ("rdma", "rc,cuda", None)):
            with self.subTest(profile=profile), patch.dict(os.environ, {}, clear=True):
                args = parse_args(["--network", profile])
                env = launch_environment(args)
                self.assertEqual(env["UCX_TLS"], tls)
                self.assertEqual(env["TANGO_UCX_UCX_TLS"], tls)
                self.assertEqual(env.get("UCX_NET_DEVICES"), default_device)
                self.assertEqual(env.get("TANGO_UCX_UCX_NET_DEVICES"), default_device)
                args.net_devices = "mlx5_0:1"
                env = launch_environment(args)
                self.assertEqual(env["UCX_NET_DEVICES"], "mlx5_0:1")
                self.assertEqual(env["TANGO_UCX_UCX_NET_DEVICES"], "mlx5_0:1")

    def test_existing_interface_and_protocol_tuning_survive_launcher(self):
        inherited = dict(UCX_NET_DEVICES="eth0", TANGO_UCX_UCX_NET_DEVICES="eth1",
                         UCX_PROTO_INFO="y", TANGO_UCX_UCX_RNDV_THRESH="0")
        with patch.dict(os.environ, inherited, clear=True):
            env = launch_environment(parse_args([]))
        for key, value in inherited.items():
            self.assertEqual(env[key], value)

    def test_device_commands_select_independent_gpus_and_tcp_policy(self):
        upstream = dict(source="source-address", decompress="decode-address", correct="correct-address")
        for profile in ("tcp", "auto", "rdma"):
            args = parse_args(["--network", profile, "--gpu", "3", "--correct-gpu", "1",
                               "--reconstruct-gpu", "2"])
            for role, gpu in (("source", 3), ("decompress", 3), ("correct", 1), ("reconstruct", 2)):
                with self.subTest(profile=profile, role=role):
                    command = device_command(args, role, Path("/tmp/run"), 1234, "device-name", upstream)
                    def value(flag):
                        return command[command.index(flag) + 1]
                    self.assertEqual(value("--gpu"), str(gpu))
                    self.assertEqual(value("--allow-gpu-over-tcp"), "0" if profile == "rdma" else "1")
                    if role != "source":
                        self.assertIn(value("--upstream"), upstream.values())
                    else:
                        self.assertNotIn("--upstream", command)

    def test_gpu_indices_are_logical_and_negative_values_are_rejected(self):
        args = parse_args(["--gpu", "4", "--decompress-gpu", "0", "--profile", "auto"])
        self.assertEqual((args.decompress_gpu, args.correct_gpu, args.reconstruct_gpu), (0, 4, 4))
        self.assertEqual(args.network, "auto")
        for flag in ("--gpu", "--decompress-gpu", "--correct-gpu", "--reconstruct-gpu"):
            with self.subTest(flag=flag), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "-1"])


if __name__ == "__main__":
    unittest.main()
