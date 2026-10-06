"""Detector dtype checks through the real processors with CPU library boundaries."""
import importlib.util
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
        cupy = SimpleNamespace(dtype=np.dtype, float32=np.float32, uint8=np.uint8,
                               empty=np.empty, empty_like=np.empty_like, copyto=np.copyto,
                               ElementwiseKernel=kernel,
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


if __name__ == "__main__":
    unittest.main()
