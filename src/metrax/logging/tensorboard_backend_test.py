# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import mock

from absl.testing import absltest
import metrax.logging as metrax_logging

TensorboardBackend = metrax_logging.TensorboardBackend


class TensorboardBackendTest(absltest.TestCase):

  @mock.patch("metrax.logging.tensorboard_backend.EventFileWriter")
  def test_init_and_log_success_main_process(self, mock_event_file_writer):
    """Tests successful init, logging, and closing on the main process."""
    mock_writer_instance = mock_event_file_writer.return_value

    with mock.patch("jax.process_index", return_value=0):
      backend = TensorboardBackend(
          log_dir="/fake/logs", flush_every_n_steps=2, flush_interval_s=0
      )

      mock_event_file_writer.assert_called_once_with(
          "/fake/logs", max_queue_size=10, flush_secs=30
      )
      mock_writer_instance.reset_mock()

      backend.log_scalar("/event1", 1.0, step=1)
      self.assertEqual(mock_writer_instance.add_event.call_count, 1)
      mock_writer_instance.flush.assert_not_called()

      backend.log_scalar("event2", 2.0, step=2)
      self.assertEqual(mock_writer_instance.add_event.call_count, 2)
      mock_writer_instance.flush.assert_called_once()

      backend.log_scalar("event_no_step", 3.0)
      self.assertEqual(mock_writer_instance.add_event.call_count, 3)

      backend.close()
      mock_writer_instance.close.assert_called_once()

  @mock.patch("metrax.logging.tensorboard_backend.EventFileWriter")
  def test_init_non_main_process_is_noop(self, mock_event_file_writer):
    """Tests that the backend does nothing on non-main processes."""
    mock_writer_instance = mock_event_file_writer.return_value

    with mock.patch("jax.process_index", return_value=1):
      backend = TensorboardBackend(log_dir="/fake/logs")

      mock_event_file_writer.assert_not_called()

      backend.log_scalar("myevent", 1.0, step=1)
      mock_writer_instance.add_event.assert_not_called()

      backend.close()
      mock_writer_instance.close.assert_not_called()

  @mock.patch("time.time")
  @mock.patch("metrax.logging.tensorboard_backend.EventFileWriter")
  def test_log_scalar_flush_rate_limited(
      self, mock_event_file_writer, mock_time
  ):
    """Tests that flush honors both step frequency and time interval."""
    mock_writer_instance = mock_event_file_writer.return_value
    mock_time.return_value = 1000.0

    with mock.patch("jax.process_index", return_value=0):
      # Configure to flush every 1 step, but strictly rate-limited to 30 seconds
      backend = TensorboardBackend(
          log_dir="/fake/logs", flush_every_n_steps=1, flush_interval_s=30.0
      )
      mock_writer_instance.reset_mock()

      backend.log_scalar("event1", 1.0, step=1)
      self.assertEqual(mock_writer_instance.add_event.call_count, 1)
      mock_writer_instance.flush.assert_not_called()

      mock_time.return_value = 1020.0
      backend.log_scalar("event2", 2.0, step=2)
      mock_writer_instance.flush.assert_not_called()

      mock_time.return_value = 1035.0
      backend.log_scalar("event3", 3.0, step=3)
      mock_writer_instance.flush.assert_called_once()

  @mock.patch("metrax.logging.tensorboard_backend.EventFileWriter")
  def test_add_pr_curve_main_process(self, mock_event_file_writer):
    """Tests add_pr_curve logging on the main process."""
    mock_writer_instance = mock_event_file_writer.return_value

    with mock.patch("jax.process_index", return_value=0):
      import numpy as np

      backend = TensorboardBackend(log_dir="/fake/logs")
      mock_writer_instance.reset_mock()

      labels = np.array([True, False, True, False])
      predictions = np.array([0.9, 0.8, 0.3, 0.1], dtype=np.float32)

      backend.add_pr_curve(
          "eval/pr_curve", labels, predictions, step=5, num_thresholds=11
      )
      self.assertEqual(mock_writer_instance.add_event.call_count, 1)
      event = mock_writer_instance.add_event.call_args[0][0]
      self.assertEqual(event.step, 5)
      self.assertEqual(len(event.summary.value), 1)
      summary_val = event.summary.value[0]
      self.assertEqual(summary_val.tag, "eval/pr_curve/pr_curves")
      self.assertEqual(
          summary_val.metadata.plugin_data.plugin_name, "pr_curves"
      )

  @mock.patch("metrax.logging.tensorboard_backend.EventFileWriter")
  def test_add_pr_curve_non_main_process_is_noop(self, mock_event_file_writer):
    """Tests add_pr_curve does nothing on non-main processes."""
    mock_writer_instance = mock_event_file_writer.return_value

    with mock.patch("jax.process_index", return_value=1):
      import numpy as np

      backend = TensorboardBackend(log_dir="/fake/logs")
      labels = np.array([True, False])
      predictions = np.array([0.9, 0.1], dtype=np.float32)

      backend.add_pr_curve("eval/pr_curve", labels, predictions, step=5)
      mock_writer_instance.add_event.assert_not_called()


if __name__ == "__main__":
  absltest.main()
