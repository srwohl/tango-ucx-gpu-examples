"""Detector dtype checks through the real processors with CPU library boundaries."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import lz4.block
import numpy as np


class DetectorProcessorTests(unittest.TestCase):
    def load_processor(self):
        class Stream:
            ptr = 1
            def __enter__(self): return self
            def __exit__(self, *_): pass

        class Codec:
            def __init__(self, **options):
                self.options = options

            def compression_config(self, size):
                self.frame_bytes = size
                return size

            def decompression_config(self, config):
                return config

            def decode(self, compressed, *, out, decompression_config):
                if isinstance(compressed, list):
                    for source, destination, size in zip(compressed, out, decompression_config):
                        self.decode(source, out=destination, decompression_config=size)
                    return
                # A byte-configured decoder asks to resize to a byte count.
                # A typed external array can report fewer elements despite
                # sufficient backing storage, reproducing the nvCOMP failure.
                if out.size < decompression_config:
                    raise ValueError("cannot grow external decoder output")
                if out.nbytes != decompression_config:
                    raise ValueError("decoder output size differs from its configuration")
                raw = lz4.block.decompress(compressed.tobytes(), uncompressed_size=decompression_config)
                np.copyto(out, np.frombuffer(raw, dtype=out.dtype).reshape(out.shape))

        def kernel(inputs, outputs, operation, name):
            self.kernel_input_types.append(inputs)
            def correct(raw, dark, flat, output):
                transmission = (raw.astype(np.float32) - dark) / (flat - dark)
                np.copyto(output, -np.log(np.clip(transmission, 1e-6, 1)))
            return correct

        self.kernel_input_types = []
        self.raw_calls, self.raw_buffers = [], {}
        def raw_kernel(source, name):
            def launch(grid, block, arguments):
                self.raw_calls.append(name)
                pointers = arguments[0]
                if name == "correct_frames":
                    _, dark, flat, pixels, count = arguments
                    for i in range(int(count)):
                        raw = self.raw_buffers[int(pointers["inputs"][i])]
                        output = self.raw_buffers[int(pointers["outputs"][i])]
                        transmission = (raw.astype(np.float32) - dark) / (flat - dark)
                        np.copyto(output, -np.log(np.clip(transmission, 1e-6, 1)))
                else:
                    _, sinogram, pixels, columns, angles, first, count = arguments
                    for i in range(int(count)):
                        sinogram[:, int(first) + i, :] = self.raw_buffers[int(pointers["inputs"][i])]
            return launch
        cupy = SimpleNamespace(dtype=np.dtype, float32=np.float32, uint8=np.uint8,
                               empty=np.empty, empty_like=np.empty_like, copyto=np.copyto,
                               ElementwiseKernel=kernel, RawKernel=raw_kernel,
                               cuda=SimpleNamespace(Device=lambda gpu: SimpleNamespace(use=lambda: None),
                                                    ExternalStream=lambda pointer: Stream()))
        nvcomp = SimpleNamespace(Codec=Codec, BitstreamKind=SimpleNamespace(RAW="raw"),
                                 as_array=lambda array, **options: array)
        spec = importlib.util.spec_from_file_location("detector_processor_test", Path(__file__).with_name("processors.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"cupy": cupy, "nvidia": SimpleNamespace(nvcomp=nvcomp)}):
            spec.loader.exec_module(module)
        return module

    def test_uint16_and_float32_frames_decode_and_correct_with_matching_sizes(self):
        module = self.load_processor()
        for element, dtype, values in (("u16", np.uint16, (10, 400, 100)),
                                       ("f32", np.float32, (10.25, 400.75, 100.5))):
            with self.subTest(element=element):
                shape = (2, 3)
                frames = [np.full(shape, value, dtype) for value in values]
                compressed = [np.frombuffer(lz4.block.compress(frame.tobytes(), store_size=False), np.uint8)
                              for frame in frames]
                scan = dict(rows=2, columns=3, angles=1, theta=[.5], element=element,
                            max_compressed_bytes=max(frame.nbytes for frame in compressed))
                decoder = module.Processor("decompress", scan, 0, 1, 40)
                corrector = module.Processor("correct", scan, 0, 1, 40)
                self.assertEqual(decoder.codec.frame_bytes, 6 * np.dtype(dtype).itemsize)
                self.assertEqual(self.kernel_input_types[-1],
                                 f"{np.dtype(dtype).name} raw, float32 dark, float32 flat")
                detector = np.empty(shape, dtype)
                attenuation = np.empty(shape, np.float32)
                pointers = {2: detector, 3: attenuation}

                def array_at(pointer, requested_shape, requested_dtype, gpu):
                    result = pointers[pointer]
                    if pointer == 2 and np.dtype(requested_dtype) == np.dtype(np.uint8):
                        result = result.view(np.uint8).reshape(-1)
                        self.assertTrue(np.shares_memory(result, detector))
                    self.assertEqual(result.shape, requested_shape)
                    self.assertEqual(result.dtype, np.dtype(requested_dtype))
                    return result

                with patch.object(module, "array_at", side_effect=array_at):
                    for kind, (payload, frame) in enumerate(zip(compressed, frames)):
                        pointers[1] = payload
                        decoder.consume(1, payload.nbytes, 2, kind, 0, .5 if kind == 2 else 0)
                        np.testing.assert_array_equal(detector, frame)
                        corrector.consume(2, detector.nbytes, 3 if kind == 2 else 0,
                                          kind, 0, .5 if kind == 2 else 0)
                    decoder.finish()
                    corrector.finish()
                    with self.assertRaisesRegex(ValueError, "upstream array length"):
                        corrector.consume(2, detector.nbytes // 2, 3, 2, 0, .5)
                expected = -np.log((float(values[2]) - float(values[0])) /
                                    (float(values[1]) - float(values[0])))
                np.testing.assert_allclose(attenuation, expected, rtol=1e-6)

    def test_unknown_detector_element_fails_before_allocating(self):
        module = self.load_processor()
        with self.assertRaisesRegex(ValueError, "detector element must be u16 or f32"):
            module.Processor("correct", dict(rows=2, columns=3, element="f64"), 0, 1, 40)

    def test_decoder_batches_calibration_projections_and_partial_tail(self):
        module = self.load_processor()
        scan = dict(rows=2, columns=3, angles=4, theta=[0., .1, .2, .3],
                    max_compressed_bytes=32, element="u16")
        processor = module.Processor("decompress", scan, 0, 1, 1)
        pointers, frames, expected = {}, [], []
        for index in range(6):
            raw = np.full((2, 3), 20 + index, np.uint16)
            compressed = np.frombuffer(lz4.block.compress(raw.tobytes(), store_size=False), np.uint8)
            pointers[index + 1] = compressed
            pointers[index + 101] = np.empty(raw.nbytes, np.uint8)
            frames.append((index + 1, compressed.nbytes, index + 101,
                           index if index < 2 else 2, max(0, index - 2), 0.))
            expected.append(raw)
        with patch.object(module, "array_at", side_effect=lambda p, shape, dtype, gpu: pointers[p]):
            processor.consume_many(frames[:4])
            processor.consume_many(frames[4:])
            processor.finish()
        self.assertEqual(processor.received, 6)
        self.assertEqual(processor.compressed_batch.shape, (16, 256))
        self.assertEqual(set(processor.batch_decoding), {2, 4})
        for index, raw in enumerate(expected):
            np.testing.assert_array_equal(pointers[index + 101].view(np.uint16).reshape(raw.shape), raw)

    def test_invalid_batch_is_rejected_before_reads_or_counter_changes(self):
        module = self.load_processor()
        scan = dict(rows=2, columns=3, angles=2, theta=[0., .1],
                    max_compressed_bytes=32, element="u16")
        processor = module.Processor("decompress", scan, 0, 1, 1)
        valid = (1, 12, 2, 0, 0, 0.)
        for bad in ((1, 12, 2, 2, 0, 0.), (1, 33, 2, 1, 0, 0.)):
            with self.subTest(frame=bad), patch.object(module, "array_at") as borrowed:
                with self.assertRaises(ValueError):
                    processor.consume_many([valid, bad])
                borrowed.assert_not_called()
                self.assertEqual(processor.received, 0)
        with self.assertRaisesRegex(ValueError, "capacity"):
            processor.consume_many([valid] * 17)

    def test_correction_batches_calibration_and_projections_into_independent_slots(self):
        for element, dtype in (("u16", np.uint16), ("f32", np.float32)):
            with self.subTest(element=element):
                module = self.load_processor()
                scan = dict(rows=2, columns=3, angles=3, theta=[0., .1, .2], element=element)
                processor = module.Processor("correct", scan, 0, 1, 1)
                for cycle in range(2):
                    processor.begin_scan()
                    raw = [np.full((2, 3), value + cycle, dtype)
                           for value in (10, 100, 30, 60, 90)]
                    frames = []
                    for i, image in enumerate(raw):
                        self.raw_buffers[i + 1] = image
                        self.raw_buffers[i + 101] = np.full((2, 3), np.nan, np.float32)
                        frames.append((i + 1, image.nbytes, i + 101 if i >= 2 else 0,
                                       min(i, 2), max(0, i - 2), scan["theta"][max(0, i - 2)]))
                    with patch.object(module, "array_at", side_effect=lambda p, *args: self.raw_buffers[p]):
                        processor.consume_many(frames[:3])
                        processor.consume_many(frames[3:])
                    processor.finish()
                    self.assertEqual(processor.projections, 3)
                    for i in range(2, 5):
                        expected = -np.log((raw[i].astype(np.float32) - raw[0]) / (raw[1] - raw[0]))
                        np.testing.assert_allclose(self.raw_buffers[i + 101], expected, rtol=2e-6)
                self.assertEqual(self.raw_calls, ["correct_frames"] * 4)

    def test_invalid_correction_group_does_not_mutate_calibration_or_queue_reads(self):
        module = self.load_processor()
        processor = module.Processor("correct", dict(rows=2, columns=3, angles=2,
                                     theta=[0., .1], element="u16"), 0, 1, 1)
        calibration = [(1, 12, 0, 0, 0, 0.), (2, 12, 0, 1, 0, 0.)]
        for bad in ((3, 12, 4, 2, 1, .1), (3, 6, 4, 2, 0, 0.),
                    (3, 12, 4, 2, 0, .5), (3, 12, 0, 2, 0, 0.)):
            with self.subTest(bad=bad), patch.object(module, "array_at") as borrowed:
                with self.assertRaises(ValueError):
                    processor.consume_many(calibration + [bad])
                borrowed.assert_not_called()
                self.assertFalse(processor.dark_seen or processor.flat_seen)
                self.assertEqual(processor.projections, 0)
                self.assertFalse(self.raw_calls)

    def test_reconstruction_batches_pack_distinct_frames_and_reconstruct_once(self):
        module = self.load_processor()
        scan = dict(rows=2, columns=3, angles=3, theta=[0., .1, .2])
        processor = module.Processor("reconstruct", scan, 0, 1, 1)
        output = np.empty((2, 3, 3), np.float32)
        self.raw_buffers[99] = output
        frames = []
        for projection, angle in enumerate(scan["theta"]):
            image = (np.arange(6).reshape(2, 3) + projection * 10).astype(np.float32)
            self.raw_buffers[projection + 1] = image
            frames.append((projection + 1, image.nbytes, 99 if projection == 2 else 0,
                           2, projection, angle))
        with patch.object(module, "array_at", side_effect=lambda p, *args: self.raw_buffers[p]), \
             patch.object(processor, "reconstruct") as reconstruct:
            processor.consume_many(frames[:2])
            reconstruct.assert_not_called()
            processor.consume_many(frames[2:])
            reconstruct.assert_called_once_with(output)
        processor.finish()
        for projection in range(3):
            np.testing.assert_array_equal(processor.sinogram[:, projection], self.raw_buffers[projection + 1])
        self.assertEqual(self.raw_calls, ["pack_frames"])


@unittest.skipUnless(os.environ.get("TOMOGRAPHY_TEST_GPU") == "1", "set TOMOGRAPHY_TEST_GPU=1")
class BatchGpuTests(unittest.TestCase):
    def test_correction_and_sinogram_packing_use_real_borrowed_gpu_allocations(self):
        import cupy as cp
        from processors import Processor
        stream = cp.cuda.Stream(non_blocking=True)
        scan = dict(rows=3, columns=5, angles=5, theta=[0., .1, .2, .3, .4])
        for element, dtype in (("u16", cp.uint16), ("f32", cp.float32)):
            with self.subTest(element=element), stream:
                correction = Processor("correct", dict(scan, element=element), 0, stream.ptr, 1)
                reconstruction = Processor("reconstruct", scan, 0, stream.ptr, 1)
                spatial = cp.arange(15, dtype=cp.float32).reshape(3, 5) % 4
                inputs = [(value + spatial * (1 if i == 0 else 2 if i == 1 else 3)).astype(dtype)
                          for i, value in enumerate((10, 100, 20, 30, 40, 50, 60))]
                outputs = [cp.empty((3, 5), cp.float32) for _ in range(5)]
                frames = [(raw.data.ptr, raw.nbytes, outputs[i - 2].data.ptr if i >= 2 else 0,
                           min(i, 2), max(0, i - 2), scan["theta"][max(0, i - 2)])
                          for i, raw in enumerate(inputs)]
                correction.consume_many(frames[:4])
                correction.consume_many(frames[4:])
                correction.finish()
                received = [(output.data.ptr, output.nbytes, 0, 2, i, scan["theta"][i])
                            for i, output in enumerate(outputs)]
                # Inspect the retained sinogram before a backend call, independently of ASTRA.
                reconstruction.buffer_plan["output_mode"] = "blocks"
                reconstruction.consume_many(received[:3])
                reconstruction.consume_many(received[3:])
                reconstruction.finish()
                stream.synchronize()
                for i, output in enumerate(outputs):
                    dark, flat, raw = (inputs[j].get().astype(np.float32) for j in (0, 1, i + 2))
                    expected = -np.log(np.clip((raw - dark) / (flat - dark), 1e-6, 1))
                    np.testing.assert_allclose(output.get(), expected, rtol=2e-6)
                    np.testing.assert_allclose(reconstruction.sinogram[:, i].get(), expected, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
