# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive deletion monitoring for tt-metal-cache directory.

This module provides a monitoring system that:
1. Monkey-patches ALL Python deletion functions to capture stack traces at deletion time
2. Uses watchdog as fallback to catch deletions from C++/external processes
3. Works generically - doesn't rely on specific function names or patterns
4. Tracks both Python and C++ deletions with full context

Key Features:
- Captures full Python traceback at the exact moment of deletion
- Identifies C++ processes performing deletions
- Generic approach - catches ANY deletion method (shutil.rmtree, os.remove, subprocess, etc.)
- Monitors directory-level operations (entire cache deletion)
- Minimal noise - only logs on creation/deletion, not modification
"""

import os
import sys
import shutil
import traceback
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime
from loguru import logger

# Watchdog for catching external process deletions
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object
    FileSystemEvent = object
    Observer = None
    logger.warning("watchdog library not available. External process deletion detection will be limited.")


class CacheDeletionMonitor:
    """
    Comprehensive monitoring system for tt-metal-cache deletions.
    
    This monitor uses a multi-layered approach:
    1. Monkey-patches Python deletion functions to capture immediate stack traces
    2. Uses watchdog to catch deletions from external processes (C++, etc.)
    3. Provides full traceback showing the complete call chain
    """
    
    def __init__(self, cache_paths: List[str], enable_creation_tracking: bool = True):
        """
        Initialize the deletion monitor.
        
        Args:
            cache_paths: List of cache directory paths to monitor
            enable_creation_tracking: Track cache creation events (default: True)
        """
        self.cache_paths = [Path(p).resolve() for p in cache_paths if p]
        self.enable_creation_tracking = enable_creation_tracking
        self.is_monitoring = False
        self.lock = threading.RLock()
        
        # Store original functions for restoration
        self._original_functions = {}
        
        # Watchdog observer for external processes
        self.observer = None
        self.watchdog_handler = None
        
        # Track cache state
        self._cache_exists_state = {}
        self._last_check_time = None
        self._active_cache_path = None  # Track which path cache actually exists in
        
        # Events log
        self.deletion_events = []
        self.creation_events = []
        
        # Deduplication: Track recent deletions to avoid logging duplicates
        self._recent_deletions = {}  # path -> timestamp
        self._dedup_window_seconds = 1.0  # Consider deletions within 1 second as duplicates
        
    def _is_cache_path(self, path: str) -> bool:
        """
        Check if a path is within any monitored cache directory.
        
        If we've detected where cache actually exists (_active_cache_path),
        only check against that path. Otherwise check all possible paths.
        """
        try:
            path_obj = Path(path).resolve()
            
            # If we know where cache exists, only check that path
            if self._active_cache_path:
                try:
                    path_obj.relative_to(self._active_cache_path)
                    return True
                except ValueError:
                    return False
            
            # Otherwise check all possible paths
            for cache_path in self.cache_paths:
                try:
                    path_obj.relative_to(cache_path)
                    return True
                except ValueError:
                    continue
            return False
        except Exception:
            return False
    
    def _get_current_test_name(self) -> str:
        """Get the current test name from pytest environment."""
        test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
        if test_name:
            test_name = test_name.replace(" (call)", "").replace(" (setup)", "").replace(" (teardown)", "")
            # Extract just the test function name for cleaner display
            if "::" in test_name:
                return test_name.split("::")[-1]
        
        # Fallback: check .pytest_current_test_executing file
        try:
            test_file = Path(".pytest_current_test_executing")
            if test_file.exists():
                test_name = test_file.read_text().strip()
                if test_name and "::" in test_name:
                    return test_name.split("::")[-1]
                return test_name
        except Exception:
            pass
        
        return "unknown_test"
    
    def _format_traceback_for_deletion(self, tb_frames: List) -> str:
        """
        Format traceback frames into a readable deletion report.
        
        Args:
            tb_frames: List of traceback frames (from traceback.extract_stack())
        
        Returns:
            Formatted traceback string
        """
        lines = []
        lines.append("\n" + "="*80)
        lines.append("[CACHE_DELETION_DETECTED] tt-metal-cache directory deletion detected!")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        lines.append(f"Test: {self._get_current_test_name()}")
        lines.append(f"Process: PID={os.getpid()}, Name={Path(sys.executable).name}")
        lines.append("")
        lines.append("Python Stack Trace (deletion call chain):")
        lines.append("-" * 80)
        
        # Filter out monitoring infrastructure frames
        skip_files = ['cache_deletion_monitor.py', 'cache_monitor.py', 'conftest.py', 'threading.py']
        
        # Build the traceback, showing most recent frames last (like Python's default)
        user_frames = []
        for frame in tb_frames:
            filename = frame.filename
            # Skip our own monitoring code
            if any(skip in filename for skip in skip_files):
                continue
            user_frames.append(frame)
        
        # Show frames in order (oldest to newest)
        for i, frame in enumerate(user_frames):
            filename = frame.filename
            
            # Make path relative to common roots for readability
            display_path = filename
            for root in ['forge/', 'test/', 'third_party/', '/proj_sw/', '/__w/']:
                if root in filename:
                    idx = filename.find(root)
                    display_path = filename[idx:]
                    break
            
            # Show each frame
            lines.append(f"  File \"{display_path}\", line {frame.lineno}, in {frame.name}")
            if frame.line:
                lines.append(f"    {frame.line.strip()}")
        
        # Highlight the deletion point (last frame)
        if user_frames:
            last_frame = user_frames[-1]
            lines.append("")
            lines.append("↑ DELETION CALLED HERE ↑")
            lines.append(f"Location: {last_frame.filename}:{last_frame.lineno}")
            lines.append(f"Function: {last_frame.name}()")
            if last_frame.line:
                lines.append(f"Line: {last_frame.line.strip()}")
        
        lines.append("="*80 + "\n")
        
        return "\n".join(lines)
    
    def _is_duplicate_deletion(self, path: str) -> bool:
        """
        Check if this deletion was already logged recently.
        
        When shutil.rmtree() is called, it internally calls os.remove(), os.rmdir(), etc.
        for each file/dir. We intercept all of these, causing duplicate logs.
        
        This function deduplicates by checking if we logged this path recently.
        """
        now = datetime.now()
        path_str = str(Path(path).resolve())
        
        # Check if we logged this path recently
        if path_str in self._recent_deletions:
            last_logged = self._recent_deletions[path_str]
            time_diff = (now - last_logged).total_seconds()
            
            if time_diff < self._dedup_window_seconds:
                # This is a duplicate - already logged within the dedup window
                return True
        
        # Not a duplicate - record this deletion
        self._recent_deletions[path_str] = now
        
        # Clean up old entries (keep only last 100)
        if len(self._recent_deletions) > 100:
            # Remove oldest entries
            sorted_entries = sorted(self._recent_deletions.items(), key=lambda x: x[1])
            self._recent_deletions = dict(sorted_entries[-100:])
        
        return False
    
    def _log_deletion(self, path: str, tb_frames: List, deletion_type: str = "Python"):
        """
        Log a deletion event with full traceback.
        
        Args:
            path: Path being deleted
            tb_frames: Traceback frames
            deletion_type: Type of deletion (Python, C++, external)
        """
        with self.lock:
            # Check for duplicates
            if self._is_duplicate_deletion(path):
                # This is a duplicate deletion event - skip logging
                return
            
            # Format and log the traceback
            traceback_str = self._format_traceback_for_deletion(tb_frames)
            
            # Log to logger only (not both logger AND print - reduces duplicate output)
            logger.critical(traceback_str)
            
            # Store event
            event = {
                'timestamp': datetime.now().isoformat(),
                'path': str(path),
                'deletion_type': deletion_type,
                'test': self._get_current_test_name(),
                'pid': os.getpid(),
                'traceback': traceback_str,
                'frames': [(f.filename, f.lineno, f.name, f.line) for f in tb_frames]
            }
            self.deletion_events.append(event)
    
    def _log_creation(self, path: str):
        """Log a cache creation event and set active cache path."""
        if not self.enable_creation_tracking:
            return
        
        with self.lock:
            # Detect and set the active cache path (where cache actually exists)
            path_obj = Path(path).resolve()
            
            # Find which base cache path this belongs to
            for cache_path in self.cache_paths:
                try:
                    path_obj.relative_to(cache_path)
                    # Found it - this is where cache actually exists
                    if not self._active_cache_path:
                        self._active_cache_path = cache_path
                        logger.info(f"[CACHE_MONITOR] Detected active cache location: {cache_path}")
                        logger.info(f"[CACHE_MONITOR] Will now monitor only this path (not all possible paths)")
                    break
                except ValueError:
                    continue
            
            msg = (
                f"\n{'='*80}\n"
                f"[CACHE_CREATION_DETECTED] tt-metal-cache directory created\n"
                f"Path: {path}\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"Test: {self._get_current_test_name()}\n"
                f"Process: PID={os.getpid()}\n"
                f"{'='*80}\n"
            )
            logger.info(msg)
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'path': str(path),
                'test': self._get_current_test_name(),
                'pid': os.getpid()
            }
            self.creation_events.append(event)
    
    # ==========================================================================
    # Monkey-patching: Intercept Python deletion functions
    # ==========================================================================
    
    def _wrap_rmtree(self, original_rmtree):
        """Wrap shutil.rmtree to capture stack trace on cache deletion."""
        def wrapped_rmtree(path, *args, **kwargs):
            # Check if this is a cache path
            if self._is_cache_path(str(path)):
                # Capture stack trace BEFORE deletion
                tb_frames = traceback.extract_stack()[:-1]  # Exclude this wrapper frame
                self._log_deletion(path, tb_frames, "Python:shutil.rmtree")
            
            # Proceed with actual deletion
            return original_rmtree(path, *args, **kwargs)
        
        return wrapped_rmtree
    
    def _wrap_remove(self, original_remove):
        """Wrap os.remove/os.unlink to capture stack trace on cache file deletion."""
        def wrapped_remove(path, *args, **kwargs):
            if self._is_cache_path(str(path)):
                tb_frames = traceback.extract_stack()[:-1]
                self._log_deletion(path, tb_frames, "Python:os.remove")
            return original_remove(path, *args, **kwargs)
        return wrapped_remove
    
    def _wrap_rmdir(self, original_rmdir):
        """Wrap os.rmdir to capture stack trace on cache directory deletion."""
        def wrapped_rmdir(path, *args, **kwargs):
            if self._is_cache_path(str(path)):
                tb_frames = traceback.extract_stack()[:-1]
                self._log_deletion(path, tb_frames, "Python:os.rmdir")
            return original_rmdir(path, *args, **kwargs)
        return wrapped_rmdir
    
    def _wrap_path_rmdir(self, original_rmdir):
        """Wrap pathlib.Path.rmdir to capture stack trace."""
        def wrapped_rmdir(self_path, *args, **kwargs):
            if self_path and CacheDeletionMonitor._instance._is_cache_path(str(self_path)):
                tb_frames = traceback.extract_stack()[:-1]
                CacheDeletionMonitor._instance._log_deletion(self_path, tb_frames, "Python:Path.rmdir")
            return original_rmdir(self_path, *args, **kwargs)
        return wrapped_rmdir
    
    def _wrap_path_unlink(self, original_unlink):
        """Wrap pathlib.Path.unlink to capture stack trace."""
        def wrapped_unlink(self_path, *args, **kwargs):
            if self_path and CacheDeletionMonitor._instance._is_cache_path(str(self_path)):
                tb_frames = traceback.extract_stack()[:-1]
                CacheDeletionMonitor._instance._log_deletion(self_path, tb_frames, "Python:Path.unlink")
            return original_unlink(self_path, *args, **kwargs)
        return wrapped_unlink
    
    def _wrap_subprocess_run(self, original_run):
        """Wrap subprocess.run to detect shell commands that delete cache."""
        def wrapped_run(args, *run_args, **run_kwargs):
            # Check if command is a deletion command targeting cache
            if isinstance(args, (list, tuple)) and len(args) > 0:
                cmd_str = ' '.join(str(a) for a in args)
                
                # Generic detection: look for 'rm' commands with cache path
                if 'rm' in cmd_str.lower() and any(str(cache_path) in cmd_str for cache_path in self.cache_paths):
                    tb_frames = traceback.extract_stack()[:-1]
                    self._log_deletion(f"subprocess: {cmd_str}", tb_frames, "Python:subprocess")
            
            return original_run(args, *run_args, **run_kwargs)
        return wrapped_run
    
    def _install_monkey_patches(self):
        """Install all monkey patches for Python deletion functions."""
        try:
            # Store singleton instance for pathlib wrappers
            CacheDeletionMonitor._instance = self
            
            # Wrap shutil.rmtree
            if hasattr(shutil, 'rmtree'):
                self._original_functions['shutil.rmtree'] = shutil.rmtree
                shutil.rmtree = self._wrap_rmtree(shutil.rmtree)
            
            # Wrap os.remove and os.unlink
            if hasattr(os, 'remove'):
                self._original_functions['os.remove'] = os.remove
                os.remove = self._wrap_remove(os.remove)
            
            if hasattr(os, 'unlink'):
                self._original_functions['os.unlink'] = os.unlink
                os.unlink = self._wrap_remove(os.unlink)
            
            # Wrap os.rmdir
            if hasattr(os, 'rmdir'):
                self._original_functions['os.rmdir'] = os.rmdir
                os.rmdir = self._wrap_rmdir(os.rmdir)
            
            # Wrap pathlib.Path methods
            if hasattr(Path, 'rmdir'):
                self._original_functions['Path.rmdir'] = Path.rmdir
                Path.rmdir = self._wrap_path_rmdir(Path.rmdir)
            
            if hasattr(Path, 'unlink'):
                self._original_functions['Path.unlink'] = Path.unlink
                Path.unlink = self._wrap_path_unlink(Path.unlink)
            
            # Wrap subprocess.run
            if hasattr(subprocess, 'run'):
                self._original_functions['subprocess.run'] = subprocess.run
                subprocess.run = self._wrap_subprocess_run(subprocess.run)
            
            logger.info("[CACHE_DELETION_MONITOR] Installed Python deletion monitoring hooks")
            
        except Exception as e:
            logger.error(f"[CACHE_DELETION_MONITOR] Error installing monkey patches: {e}")
    
    def _uninstall_monkey_patches(self):
        """Restore original functions."""
        try:
            for key, original_func in self._original_functions.items():
                if key == 'shutil.rmtree':
                    shutil.rmtree = original_func
                elif key == 'os.remove':
                    os.remove = original_func
                elif key == 'os.unlink':
                    os.unlink = original_func
                elif key == 'os.rmdir':
                    os.rmdir = original_func
                elif key == 'Path.rmdir':
                    Path.rmdir = original_func
                elif key == 'Path.unlink':
                    Path.unlink = original_func
                elif key == 'subprocess.run':
                    subprocess.run = original_func
            
            self._original_functions.clear()
            logger.info("[CACHE_DELETION_MONITOR] Uninstalled Python deletion monitoring hooks")
            
        except Exception as e:
            logger.error(f"[CACHE_DELETION_MONITOR] Error uninstalling monkey patches: {e}")
    
    # ==========================================================================
    # Watchdog: Monitor for external process deletions (C++, etc.)
    # ==========================================================================
    
    class _WatchdogHandler(FileSystemEventHandler):
        """Watchdog handler for detecting external process deletions."""
        
        def __init__(self, monitor: 'CacheDeletionMonitor'):
            super().__init__()
            self.monitor = monitor
        
        def on_deleted(self, event: FileSystemEvent):
            """Handle deletion events from external processes."""
            # Only log directory deletions or large-scale deletions
            if event.is_directory:
                self.monitor._log_external_deletion(event.src_path)
        
        def on_created(self, event: FileSystemEvent):
            """Handle creation events."""
            if event.is_directory and self.monitor.enable_creation_tracking:
                self.monitor._log_creation(event.src_path)
    
    def _log_external_deletion(self, path: str):
        """Log a deletion from an external process (C++, etc.)."""
        with self.lock:
            # Check for duplicates
            if self._is_duplicate_deletion(path):
                return
            
            msg = (
                f"\n{'='*80}\n"
                f"[CACHE_DELETION_DETECTED] External process deletion detected\n"
                f"Path: {path}\n"
                f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
                f"Test: {self._get_current_test_name()}\n"
                f"\nNOTE: This deletion was performed by a non-Python process (likely C++).\n"
                f"Python stack trace is not available for external process deletions.\n"
                f"This may be from tt-metal, tt-mlir, or other C++ components.\n"
                f"{'='*80}\n"
            )
            logger.critical(msg)
            
            # Try to identify the process
            self._try_identify_external_process(path)
            
            event = {
                'timestamp': datetime.now().isoformat(),
                'path': str(path),
                'deletion_type': 'External/C++',
                'test': self._get_current_test_name(),
                'pid': os.getpid()
            }
            self.deletion_events.append(event)
    
    def _try_identify_external_process(self, path: str):
        """Try to identify which external process deleted the cache."""
        try:
            import psutil
            
            # Look for processes that might be tt-metal related
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe']):
                try:
                    name = proc.info['name'] or ''
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    
                    # Look for tt-metal, riscv, or kernel-related processes
                    if any(keyword in name.lower() or keyword in cmdline.lower() 
                           for keyword in ['tt-metal', 'riscv', 'kernel', 'g++', 'gcc', 'ld']):
                        logger.info(f"  Possible process: PID={proc.info['pid']}, Name={name}, Cmd={cmdline[:100]}")
                        print(f"  Possible process: PID={proc.info['pid']}, Name={name}, Cmd={cmdline[:100]}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error identifying external process: {e}")
    
    def _start_watchdog(self):
        """Start watchdog observer for external process monitoring."""
        if not WATCHDOG_AVAILABLE or not self.cache_paths:
            return False
        
        try:
            self.observer = Observer()
            self.watchdog_handler = self._WatchdogHandler(self)
            
            # If we know where cache exists, only monitor that path
            paths_to_monitor = []
            if self._active_cache_path and self._active_cache_path.exists():
                paths_to_monitor = [self._active_cache_path]
                logger.info(f"[CACHE_DELETION_MONITOR] Watchdog monitoring active cache: {self._active_cache_path}")
            else:
                # Monitor all existing paths
                paths_to_monitor = [p for p in self.cache_paths if p.exists()]
                if paths_to_monitor:
                    logger.info(f"[CACHE_DELETION_MONITOR] Watchdog monitoring all existing paths: {len(paths_to_monitor)} paths")
            
            for cache_path in paths_to_monitor:
                self.observer.schedule(self.watchdog_handler, str(cache_path), recursive=True)
            
            if paths_to_monitor:
                self.observer.start()
                return True
            else:
                logger.debug("[CACHE_DELETION_MONITOR] No existing cache paths to monitor with watchdog")
                return False
        except Exception as e:
            logger.error(f"[CACHE_DELETION_MONITOR] Failed to start watchdog: {e}")
            return False
    
    def _stop_watchdog(self):
        """Stop watchdog observer."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2)
            except Exception as e:
                logger.error(f"[CACHE_DELETION_MONITOR] Error stopping watchdog: {e}")
    
    # ==========================================================================
    # Public API
    # ==========================================================================
    
    def start(self):
        """Start comprehensive cache deletion monitoring."""
        if self.is_monitoring:
            logger.warning("[CACHE_DELETION_MONITOR] Already monitoring")
            return False
        
        logger.info("[CACHE_DELETION_MONITOR] Starting comprehensive deletion monitoring")
        logger.info(f"[CACHE_DELETION_MONITOR] Possible cache paths: {[str(p) for p in self.cache_paths]}")
        
        # Check which paths actually exist
        existing_paths = [p for p in self.cache_paths if p.exists()]
        if existing_paths:
            # Cache already exists - set active path
            self._active_cache_path = existing_paths[0]
            logger.info(f"[CACHE_DELETION_MONITOR] Cache found at: {self._active_cache_path}")
            logger.info(f"[CACHE_DELETION_MONITOR] Will monitor only this path")
        else:
            logger.info(f"[CACHE_DELETION_MONITOR] Cache not yet created, will detect location on first creation")
        
        # Install Python hooks
        self._install_monkey_patches()
        
        # Start watchdog for external processes
        self._start_watchdog()
        
        self.is_monitoring = True
        logger.info("[CACHE_DELETION_MONITOR] Monitoring started successfully")
        return True
    
    def stop(self):
        """Stop monitoring and restore original functions."""
        if not self.is_monitoring:
            return
        
        logger.info("[CACHE_DELETION_MONITOR] Stopping monitoring")
        
        # Stop watchdog
        self._stop_watchdog()
        
        # Restore original functions
        self._uninstall_monkey_patches()
        
        self.is_monitoring = False
        
        # Log summary
        self._log_summary()
    
    def _log_summary(self):
        """Log summary of all detected deletions."""
        with self.lock:
            if self.deletion_events:
                logger.info(f"\n{'='*80}")
                logger.info(f"[CACHE_DELETION_MONITOR] Session Summary")
                logger.info(f"Total deletions detected: {len(self.deletion_events)}")
                for i, event in enumerate(self.deletion_events, 1):
                    logger.info(f"  {i}. {event['deletion_type']} at {event['timestamp']}")
                    logger.info(f"     Path: {event['path']}")
                    logger.info(f"     Test: {event['test']}")
                logger.info(f"{'='*80}\n")
    
    def get_deletion_events(self) -> List[Dict]:
        """Get all deletion events captured during monitoring."""
        with self.lock:
            return self.deletion_events.copy()
    
    def get_creation_events(self) -> List[Dict]:
        """Get all creation events captured during monitoring."""
        with self.lock:
            return self.creation_events.copy()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Singleton instance holder (for pathlib wrappers)
CacheDeletionMonitor._instance = None


def get_cache_paths() -> List[str]:
    """
    Get tt-metal-cache directory paths.
    
    Returns all possible cache paths, regardless of whether they exist.
    This allows monitoring to be ready when the cache is created.
    """
    cache_paths = []
    
    # Environment variable (highest priority)
    env_cache = os.environ.get("TT_METAL_CACHE_DIR")
    if env_cache:
        cache_paths.append(env_cache)
    
    # CI environment
    cache_paths.append("/github/home/.cache/tt-metal-cache")
    
    # Local environment
    cache_paths.append(os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache"))
    
    # Fallback
    cache_paths.append("/tmp/tt-metal-cache")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in cache_paths:
        try:
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(resolved)
        except Exception:
            # If path resolution fails, use as-is
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
    
    return unique_paths
