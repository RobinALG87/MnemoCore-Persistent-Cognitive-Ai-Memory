"""
Tests for Process Utilities
============================
Tests for src/mnemocore/utils/process.py covering lower_process_priority()
for both Unix and Windows platforms, and error handling when psutil is unavailable.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mnemocore.utils.process import lower_process_priority


class TestLowerProcessPriority:
    """Tests for lower_process_priority function."""

    @pytest.mark.skipif(os.name != 'nt', reason="Windows-specific test")
    def test_windows_priority_lowered(self):
        """Test that Windows process priority is lowered to BELOW_NORMAL."""
        psutil = pytest.importorskip("psutil")

        original_nice = psutil.Process(os.getpid()).nice()

        lower_process_priority()

        new_nice = psutil.Process(os.getpid()).nice()

        # On Windows, BELOW_NORMAL_PRIORITY_CLASS should be set
        # The exact value may vary, but it should be different from normal
        # Note: This test may not show change if already at below normal
        assert new_nice == psutil.BELOW_NORMAL_PRIORITY_CLASS or new_nice != original_nice

    @patch('os.name', 'nt')
    @patch('mnemocore.utils.process.psutil')
    def test_windows_calls_psutil_nice(self, mock_psutil):
        """Test that psutil.Process.nice() is called on Windows."""
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        mock_psutil.BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        lower_process_priority()

        mock_process.nice.assert_called_once_with(mock_psutil.BELOW_NORMAL_PRIORITY_CLASS)

    @patch('os.name', 'posix')
    @patch('os.nice', create=True)
    def test_unix_nice_increased(self, mock_nice):
        """Test that os.nice() is called on Unix-like systems."""
        lower_process_priority()

        # Should call os.nice(10) to lower priority
        mock_nice.assert_called_once_with(10)

    @patch('os.name', 'posix')
    @patch('os.nice', side_effect=OSError("Permission denied"), create=True)
    def test_unix_nice_error_handling(self, mock_nice):
        """Test that errors from os.nice() are caught and logged."""
        # Should not raise an exception
        lower_process_priority()

        # Verify os.nice was called
        mock_nice.assert_called_once()

    @patch('os.name', 'nt')
    def test_psutil_import_error_handling(self):
        """Test graceful handling when psutil is not installed on Windows."""
        # Simulate ImportError when trying to import psutil
        with patch.dict('sys.modules', {'psutil': None}):
            # Need to patch at the function level since import happens inside
            with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
                       MagicMock() if name != 'psutil' else (_ for _ in ()).throw(ImportError())):
                # Should not raise an exception
                lower_process_priority()

    @patch('os.name', 'nt')
    @patch('mnemocore.utils.process.psutil')
    def test_windows_psutil_error_handling(self, mock_psutil):
        """Test that psutil errors are caught on Windows."""
        mock_psutil.Process.side_effect = Exception("Process access denied")
        mock_psutil.BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        # Should not raise an exception
        lower_process_priority()

        mock_psutil.Process.assert_called_once()

    @patch('os.name', 'nt')
    @patch('mnemocore.utils.process.psutil')
    def test_windows_sets_below_normal_priority(self, mock_psutil):
        """Test that BELOW_NORMAL_PRIORITY_CLASS is used on Windows."""
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        mock_psutil.BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        lower_process_priority()

        mock_process.nice.assert_called_with(0x00004000)

    @patch('os.name', 'posix')
    @patch('os.nice', create=True)
    def test_unix_increases_niceness_by_10(self, mock_nice):
        """Test that niceness is increased by 10 on Unix systems."""
        lower_process_priority()

        mock_nice.assert_called_once_with(10)


class TestLowerProcessPriorityLogging:
    """Tests for logging behavior in lower_process_priority."""

    @patch('os.name', 'posix')
    @patch('os.nice', create=True)
    @patch('mnemocore.utils.process.logger')
    def test_unix_logs_debug_message(self, mock_logger, mock_nice):
        """Test that debug message is logged on Unix."""
        lower_process_priority()

        # Check that debug was called
        mock_logger.debug.assert_called()

    @patch('os.name', 'nt')
    @patch('mnemocore.utils.process.psutil')
    @patch('mnemocore.utils.process.logger')
    def test_windows_logs_debug_message(self, mock_logger, mock_psutil):
        """Test that debug message is logged on Windows."""
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        mock_psutil.BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        lower_process_priority()

        mock_logger.debug.assert_called()


class TestLowerProcessPriorityIntegration:
    """Integration tests for lower_process_priority."""

    def test_function_exists_and_callable(self):
        """Test that the function exists and is callable."""
        assert callable(lower_process_priority)

    def test_function_returns_none(self):
        """Test that the function returns None (or doesn't return a value)."""
        result = lower_process_priority()
        assert result is None

    @patch('os.name', 'nt')
    @patch('mnemocore.utils.process.psutil')
    def test_getpid_called(self, mock_psutil):
        """Test that os.getpid() is called to get current process ID."""
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        mock_psutil.BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        with patch('os.getpid') as mock_getpid:
            mock_getpid.return_value = 12345
            lower_process_priority()
            assert mock_getpid.called
