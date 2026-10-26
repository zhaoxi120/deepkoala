"""Measured CUDA batching with bounded windows and recoverable OOM failures.

Memory budgets are scheduling targets, not CUDA allocator configuration. In
particular, PyTorch 2.4 cannot read/restore a caller's allocator fraction, so this
module never changes that process-wide setting. A conservative runtime estimate
and OOM splitting also accommodate different cuDNN implementations.
"""

import logging
import math

import torch
import torch.nn.functional as F

from .batching import GIB, ProteinRecord, iter_windows, plan_memory
from .data import _ProteinRecords, collate_fn


LOG = logging.getLogger(__name__)
# Covers CUDA context and library allocations outside the PyTorch allocator.
# This is a conservative allowance, not a measurement of all driver allocations.
CUDA_OVERHEAD_BYTES = 3 * GIB // 4


def predict_tensors(model, seqs, lens, device, topk):
    """Keep GPU tensors local, returning only ordinary Python predictions."""
    seqs, lens = seqs.to(device), lens.to(device)
    probabilities = F.softmax(model(seqs, lens), dim=1)
    values, indices = torch.topk(probabilities, min(topk, probabilities.size(1)), dim=1)
    return indices.tolist(), values.tolist()


def predict_records(model, records, device, topk):
    dataset = _ProteinRecords()
    _, seqs, lens = collate_fn([dataset[(r.name, r.sequence)] for r in records])
    return predict_tensors(model, seqs, lens, device, topk)


def try_predict(model, records, device, topk):
    """Exit the failing frame/exception handler before the caller retries.

    Returning a message instead of an exception drops the traceback and its GPU
    tensors. Catch only CUDA OOM: unrelated runtime errors must still propagate.
    """
    try:
        return predict_records(model, records, device, topk), None
    except torch.cuda.OutOfMemoryError as error:
        message = str(error)
    return None, message


class AdaptivePredictor:
    def __init__(
        self, model, device, topk, *, memory_preset="auto", max_batch_size=512,
        max_padded_residues=None, oom_fallback="error", cpu_model_factory=None,
    ):
        self.model, self.device, self.topk = model, torch.device(device), topk
        # PyTorch 2.4 mem_get_info requires an explicit index for torch.device.
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_batch_size = max_batch_size
        self.oom_fallback, self.cpu_model_factory = oom_fallback, cpu_model_factory
        self.cpu_model = None
        self.oom_retries = self.cpu_fallbacks = 0
        self.max_successful_padded = self.max_successful_batch = 0
        self.max_peak_allocated = self.max_peak_reserved = 0
        # Clear only unused blocks; existing live allocations count in baseline.
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.device)
        self.plan = plan_memory(total, free, torch.cuda.memory_reserved(self.device), memory_preset)
        self.allocator_budget = max(0, self.plan.budget_bytes - CUDA_OVERHEAD_BYTES)
        self.bytes_per_residue = self._calibrate()
        baseline = torch.cuda.memory_allocated(self.device)
        # Short sequences can have a substantial per-sequence classifier cost.
        per_sequence_output = model.cls.out_features * 4 * 3
        self.max_batch_size = min(max_batch_size, max(1, int(
            (self.allocator_budget - baseline) // (2 * per_sequence_output)
        )))
        output_allowance = self.max_batch_size * per_sequence_output
        available = self.allocator_budget - baseline - output_allowance
        self.gpu_unavailable_reason = None
        if available <= 0:
            self.gpu_unavailable_reason = "insufficient GPU headroom for the model and classifier within the memory budget"
            if oom_fallback != "cpu":
                raise RuntimeError(self.gpu_unavailable_reason + "; free GPU memory or use --oom-fallback cpu")
            LOG.warning("%s; routing sequences to CPU.", self.gpu_unavailable_reason)
        self.residue_limit = max(1, int(available / self.bytes_per_residue))
        if max_padded_residues is not None:
            self.residue_limit = min(self.residue_limit, max_padded_residues)
        LOG.warning(
            "CUDA preset %s: total %.2f GiB, free %.2f GiB, budget %.2f GiB "
            "(includes %.2f GiB CUDA allowance); batch <= %d, padded residues <= %d. "
            "OOM splitting enabled; singleton fallback=%s.",
            self.plan.preset.name, total / GIB, free / GIB, self.plan.budget_bytes / GIB,
            CUDA_OVERHEAD_BYTES / GIB, self.max_batch_size, self.residue_limit, oom_fallback,
        )

    def _calibrate(self):
        """Measure complete inference on small shapes with the loaded model."""
        # A conservative floor also covers an unavailable/tiny calibration budget.
        coefficient = 12 * 1024.0
        if torch.cuda.memory_allocated(self.device) + 128 * 2**20 > self.allocator_budget:
            LOG.warning("GPU headroom is too small for calibration; using conservative batches.")
            return coefficient
        for count, length in ((4, 256), (4, 1024), (16, 256)):
            records = [ProteinRecord(i, "calibration", "A" * length) for i in range(count)]
            torch.cuda.synchronize(self.device)
            baseline = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            result, error = try_predict(self.model, records, self.device, self.topk)
            self._record_peak()
            if error is not None:
                self._empty_cache()
                LOG.warning("CUDA calibration OOM; retaining conservative estimate and OOM retries.")
                break
            torch.cuda.synchronize(self.device)
            increment = max(0, torch.cuda.max_memory_allocated(self.device) - baseline)
            coefficient = max(coefficient, 1.35 * increment / (count * length))
            del result
        self._empty_cache()
        return coefficient

    def _empty_cache(self):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def _record_peak(self):
        self.max_peak_allocated = max(self.max_peak_allocated,
                                      torch.cuda.max_memory_allocated(self.device))
        self.max_peak_reserved = max(self.max_peak_reserved,
                                     torch.cuda.max_memory_reserved(self.device))

    def _singleton_fallback(self, record, reason):
        if self.oom_fallback != "cpu":
            raise RuntimeError(
                f"Sequence {record.name!r} (input index {record.index}, {record.length} aa) "
                f"cannot fit the CUDA batch budget: {reason}. "
                "Use --oom-fallback cpu, or run this sequence separately on a larger GPU/CPU. "
                "The sequence was not truncated or skipped."
            )
        LOG.warning("CPU fallback: sequence %r (index %d, %d aa): %s",
                    record.name, record.index, record.length, reason)
        if self.cpu_model is None:
            self.cpu_model = self.cpu_model_factory()
        result = predict_records(self.cpu_model, [record], torch.device("cpu"), self.topk)
        self.cpu_fallbacks += 1
        return [(record, result[0][0], result[1][0])]

    def _predict_batch(self, records):
        if self.gpu_unavailable_reason:
            return [prediction for record in records
                    for prediction in self._singleton_fallback(record, self.gpu_unavailable_reason)]
        padded = len(records) * max(r.length for r in records)
        # A previous OOM may have reduced the limit since this batch was formed.
        if padded > self.residue_limit:
            if len(records) == 1:
                return self._singleton_fallback(records[0], "padded residue limit exceeded")
            middle = len(records) // 2
            return self._predict_batch(records[:middle]) + self._predict_batch(records[middle:])
        torch.cuda.reset_peak_memory_stats(self.device)
        result, error = try_predict(self.model, records, self.device, self.topk)
        self._record_peak()
        if error is not None:
            self._empty_cache()
            self.oom_retries += 1
            if len(records) == 1:
                return self._singleton_fallback(records[0], "CUDA out of memory")
            self.residue_limit = min(self.residue_limit, max(1, padded // 2))
            LOG.warning("CUDA OOM: splitting %d sequences; padded residue limit reduced to %d.",
                        len(records), self.residue_limit)
            middle = len(records) // 2
            return self._predict_batch(records[:middle]) + self._predict_batch(records[middle:])
        peak = torch.cuda.max_memory_allocated(self.device)
        self.max_successful_padded = max(self.max_successful_padded, padded)
        self.max_successful_batch = max(self.max_successful_batch, len(records))
        if peak > self.allocator_budget:
            # cuDNN algorithm/workspace choices are shape-dependent. Feed actual
            # observations back into later batches even when CUDA did not OOM.
            self.residue_limit = max(1, min(self.residue_limit, math.floor(
                padded * self.allocator_budget / peak * 0.8
            )))
            LOG.warning("Measured GPU usage exceeded the scheduling target; reducing padded residue limit to %d.",
                        self.residue_limit)
            self._empty_cache()
        return [(record, indices, probabilities)
                for record, indices, probabilities in zip(records, *result)]

    def predict_window(self, records):
        """Greedily batch one window; consult updated limits after every retry."""
        ordered = sorted(records, key=lambda record: record.length)
        completed = []
        start = 0
        while start < len(ordered):
            end = start + 1
            while (end < len(ordered) and end - start < self.max_batch_size
                   and (end - start + 1) * ordered[end].length <= self.residue_limit):
                end += 1
            completed.extend(self._predict_batch(ordered[start:end]))
            start = end
        completed.sort(key=lambda item: item[0].index)
        return completed

    def iter_predictions(self, input_path, window_size):
        for window in iter_windows(input_path, window_size):
            predictions = self.predict_window(window)
            yield predictions
            # Release sequence strings/results before requesting the next window.
            del predictions, window

    def summary(self):
        return {
            "preset": self.plan.preset.name, "budget_bytes": self.plan.budget_bytes,
            "allocator_target_bytes": self.allocator_budget,
            "cuda_allowance_bytes": CUDA_OVERHEAD_BYTES,
            "bytes_per_padded_residue": self.bytes_per_residue,
            "max_padded_residues": self.residue_limit,
            "max_batch_size": self.max_batch_size,
            "max_successful_padded": self.max_successful_padded,
            "max_successful_batch": self.max_successful_batch,
            "peak_allocated_bytes": self.max_peak_allocated,
            "peak_reserved_bytes": self.max_peak_reserved,
            "oom_retries": self.oom_retries, "cpu_fallbacks": self.cpu_fallbacks,
        }
