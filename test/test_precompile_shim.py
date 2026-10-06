from __future__ import annotations

from collections import OrderedDict
import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import ANY
from unittest.mock import Mock
from unittest.mock import patch

from packaging.version import Version
import torch

from helion._argument_device import _find_argument_device
from helion._utils import triton_is_available
from helion.runtime.config import Config
from helion.runtime.precompile_shim import _device_probe_values
from helion.runtime.precompile_shim import make_precompiler


@unittest.skipUnless(triton_is_available(), "requires Triton")
class TestPrecompileShim(unittest.TestCase):
    def test_original_device_survives_tensor_descriptor_lowering(self) -> None:
        from triton.tools.tensor_descriptor import TensorDescriptor

        device = torch.device("cpu")
        tensor = torch.empty((16, 16), device=device)
        descriptor = TensorDescriptor(tensor, [16, 16], [16, 1], [16, 16])
        cached = SimpleNamespace(_helion_compilation_success=True)
        binder = Mock(return_value=({}, [], {}))
        fn = Mock(debug=False)
        fn.device_caches = {device: ({"cached": cached}, {}, None, None, binder)}
        with patch("triton.runtime.jit.compute_cache_key", return_value="cached"):
            for original in ([tensor], [{"input": [tensor]}], [device]):
                with self.subTest(original_type=type(original[0])):
                    finish = make_precompiler(
                        fn, Config(), Mock(), device_args=original
                    )(descriptor)
                    self.assertTrue(finish())
                    self.assertIs(binder.call_args.args[0], descriptor)
            # Calls without lowered descriptors retain normal discovery, and
            # original scalar-only calls can still find a lowered tensor.
            for original in ((), (17,)):
                self.assertTrue(
                    make_precompiler(fn, Config(), Mock(), device_args=original)(
                        tensor
                    )()
                )

    def test_precompile_paths_forward_original_device_arguments(self) -> None:
        from triton.tools.tensor_descriptor import TensorDescriptor

        from helion.autotuner.benchmark_provider import _triton_compile
        from helion.autotuner.precompile_future import _prepare_precompiler_for_fork

        tensor = torch.empty((16, 16))
        descriptor = TensorDescriptor(tensor, [16, 16], [16, 1], [16, 16])
        cached = SimpleNamespace(_helion_compilation_success=True)
        binder = Mock(return_value=({}, [], {}))
        fn = Mock(debug=False)
        fn.device_caches = {tensor.device: ({"cached": cached}, {}, None, None, binder)}

        def lowered(x, *, _launcher):
            _launcher(fn, (1,), descriptor)

        with patch("triton.runtime.jit.compute_cache_key", return_value="cached"):
            self.assertIsNone(
                _prepare_precompiler_for_fork(
                    lowered, [tensor], Config(), Mock(), "test", Mock()
                )
            )
            self.assertTrue(_triton_compile(lowered, [tensor], Config(), Mock()))
        self.assertEqual(binder.call_count, 2)
        self.assertIs(binder.call_args.args[0], descriptor)

    def test_indexless_original_device_uses_current_device_cache(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensor
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode()
        fn = Mock(debug=False)
        binders = [Mock(return_value=({}, [], {})) for _ in range(2)]
        cached = SimpleNamespace(_helion_compilation_success=True)
        fn.device_caches = {
            torch.device("cuda", index): ({"cached": cached}, {}, None, None, binder)
            for index, binder in zip((2, 3), binders, strict=True)
        }
        with (
            patch("triton.runtime.jit.compute_cache_key", return_value="cached"),
            patch("helion._argument_device._current_device_index", side_effect=(2, 3)),
        ):
            for index in (2, 3):
                tensor = FakeTensor(
                    mode,
                    torch.empty((16, 16), device="meta"),
                    torch.device("cuda", index),
                )
                self.assertTrue(
                    make_precompiler(
                        fn, Config(), Mock(), device_args=[torch.device("cuda")]
                    )(tensor)()
                )
        for binder in binders:
            binder.assert_called_once()

    def test_matches_triton_runtime_cache_key_and_argument_packing(self) -> None:
        kernel_cache: dict[object, object] = {}
        kernel_key_cache = object()
        target = object()
        backend = Mock()
        specialization = [("*fp32", "D")]
        bound_args = OrderedDict((("x", object()),))
        binder_options = SimpleNamespace(binder=True)
        packed_options = SimpleNamespace(packed=True)
        binder = Mock(return_value=(bound_args, specialization, binder_options))
        compiled_kernel = Mock()
        fn = Mock()
        fn.debug = False
        device = torch.device("cuda", 0)
        fn.device_caches = {
            device: (kernel_cache, kernel_key_cache, target, backend, binder)
        }
        fn._pack_args.return_value = (
            packed_options,
            {"x": "*fp32"},
            {},
            {(0,): "tt.divisibility"},
        )
        fn.ASTSource.return_value = "source"
        fn.compile.return_value = compiled_kernel
        kernel_module = importlib.import_module("helion.runtime.kernel")

        with (
            patch.object(kernel_module, "_find_device", return_value=device),
            patch(
                "helion.runtime.precompile_shim.get_triton_version",
                return_value=Version("3.7.0"),
            ),
            patch(
                "triton.runtime.jit.compute_cache_key",
                return_value="runtime-cache-key",
            ) as compute_cache_key,
        ):
            precompile = make_precompiler(fn, Config(), Mock())(object())
            self.assertTrue(precompile(in_child_process=False))

        compute_cache_key.assert_called_once_with(
            kernel_key_cache, specialization, binder_options
        )
        fn._pack_args.assert_called_once_with(
            backend, ANY, bound_args, specialization, binder_options
        )
        fn.ASTSource.assert_called_once_with(
            fn,
            {"x": "*fp32"},
            {},
            {(0,): "tt.divisibility"},
        )
        fn.compile.assert_called_once_with(
            "source", target=target, options={"packed": True}
        )
        self.assertIs(kernel_cache["runtime-cache-key"], compiled_kernel)
        compiled_kernel._init_handles.assert_called_once_with()

    def test_device_discovery_unwraps_tensor_descriptors(self) -> None:
        # A launch whose tensors all travel as tensor descriptors (every
        # indexing choice ``tensor_descriptor``, no bare tensor argument) used
        # to abort the whole autotune with NoTensorArgs in the fork precompile.
        from triton.tools.tensor_descriptor import TensorDescriptor

        base = torch.zeros(32, 64)
        descriptor = TensorDescriptor.from_tensor(base, [16, 16])
        probed = _device_probe_values([descriptor, 148])
        self.assertIs(probed[0], base)
        self.assertEqual(probed[1], 148)
        self.assertEqual(_find_argument_device(probed), base.device)


if __name__ == "__main__":
    unittest.main()
