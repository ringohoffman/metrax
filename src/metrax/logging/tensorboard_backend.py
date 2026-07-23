# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrax LoggingBackend implementation for Tensorboard."""

import os
import time
from typing import Any

from absl import logging
import jax
import numpy as np
from tensorboard.compat.proto import event_pb2
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.hparams import summary_v2 as hp_summary
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

_DEFAULT_STEP = 0


def _get_step(kwargs: dict[str, Any]) -> int:
  """Returns the step from the kwargs, or 0 if not provided."""
  step = kwargs.get("step")
  return _DEFAULT_STEP if step is None else int(step)


def _preprocess_event_name(event_name: str) -> str:
  """Preprocesses the event name before logging."""
  return event_name.lstrip("/")  # Remove leading slashes


def _gcsfuse_to_gs(path: str) -> str:
  """Translates a local GCSFuse mount path into a direct ``gs://`` URI.

  Returns the original path unchanged if it is already a ``gs://`` URI
  or is not on a GCSFuse mount.
  """
  if path.startswith("gs://"):
    return path

  from pathlib import Path

  abs_path = Path(path).resolve()
  proc_mounts = Path("/proc/mounts")
  if proc_mounts.exists():
    try:
      with proc_mounts.open("r", encoding="utf-8") as f:
        for line in f:
          parts = line.split()
          if len(parts) >= 3:
            device, mount_point_str, fstype, *_ = parts
            if "gcsfuse" in fstype or "gcsfuse" in device or "fuse" in fstype:
              mount_point = Path(mount_point_str).resolve()
              if abs_path == mount_point or mount_point in abs_path.parents:
                bucket_name = device.split(":")[-1].strip("/")
                if (
                    not bucket_name
                    or "/" in bucket_name
                    or bucket_name in ("gcsfuse", "fuse", "/dev/fuse")
                ):
                  bucket_name = mount_point.name
                rel_path = abs_path.relative_to(mount_point)
                gs_path = f"gs://{bucket_name}/{rel_path}".rstrip("/")
                logging.info(
                    "Converted GCSFuse path to gs:// URI: %s -> %s",
                    path,
                    gs_path,
                )
                return gs_path
    except (OSError, UnicodeDecodeError) as e:
      logging.warning(
          "Could not read /proc/mounts for GCSFuse translation: %s", e
      )

  return path


class TensorboardBackend:
  """A logging backend for Tensorboard using native EventFileWriter."""

  def __init__(
      self,
      log_dir: str,
      flush_every_n_steps: int = 100,
      flush_interval_s: float = 30.0,
      max_queue_size: int = 10,
      flush_secs: int = 30,
  ):
    self._flush_every_n_steps = flush_every_n_steps
    self._flush_interval_s = flush_interval_s
    self._last_flush_time = time.time()

    if jax.process_index() == 0:
      gs_dir = _gcsfuse_to_gs(log_dir)
      self._writer = EventFileWriter(
          gs_dir,
          max_queue_size=max_queue_size,
          flush_secs=flush_secs,
      )
    else:
      self._writer = None

  def log_scalar(self, event: str, value: float | np.ndarray, **kwargs):
    if self._writer is None:
      return
    current_step = _get_step(kwargs)
    event_name = _preprocess_event_name(event)
    summary = summary_pb2.Summary(
        value=[
            summary_pb2.Summary.Value(tag=event_name, simple_value=float(value))
        ]
    )
    ev = event_pb2.Event(
        wall_time=time.time(), step=current_step, summary=summary
    )
    self._writer.add_event(ev)

    now = time.time()
    if (current_step % self._flush_every_n_steps == 0) and (
        (now - self._last_flush_time) >= self._flush_interval_s
    ):
      self._writer.flush()
      self._last_flush_time = now

  def add_hparams(
      self, hparam_dict: dict[str, bool | int | float | str]
  ) -> None:
    """Write hparams into the *same* event file (no subdirectory)."""
    if self._writer is None:
      return
    summary = hp_summary.hparams_pb(hparam_dict)
    ev = event_pb2.Event(wall_time=time.time(), summary=summary)
    self._writer.add_event(ev)
    self._writer.flush()

  def add_text(self, tag: str, text_string: str, *, step: int = 0) -> None:
    """Write a text summary into the same event file."""
    if self._writer is None:
      return
    from tensorboard.compat.proto import tensor_pb2
    from tensorboard.compat.proto import tensor_shape_pb2
    from tensorboard.compat.proto import types_pb2

    tensor = tensor_pb2.TensorProto(
        dtype=types_pb2.DT_STRING,
        string_val=[text_string.encode("utf-8")],
        tensor_shape=tensor_shape_pb2.TensorShapeProto(
            dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
        ),
    )
    meta = text_metadata.create_summary_metadata(
        display_name=tag, description=""
    )
    summary = summary_pb2.Summary(
        value=[summary_pb2.Summary.Value(tag=tag, metadata=meta, tensor=tensor)]
    )
    ev = event_pb2.Event(wall_time=time.time(), step=step, summary=summary)
    self._writer.add_event(ev)

  def flush(self):
    if self._writer:
      self._writer.flush()

  def close(self):
    if self._writer:
      self._writer.close()
