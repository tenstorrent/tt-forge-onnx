# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
File system monitoring utility for tracking operations on tt-metal-cache directory.

This module provides monitoring capabilities to track all file system operations
(create, modify, delete) on the tt-metal-cache directory and identify which
processes/functions are performing these operations.
"""

import os
import sys
import threading
import traceback
import inspect
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
from collections import defaultdict
from loguru import logger

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Create dummy classes when watchdog is not available
    class FileSystemEventHandler:
        """Dummy base class when watchdog is not available."""
        pass
    
    class FileSystemEvent:
        """Dummy event class when watchdog is not available."""
        def __init__(self, src_path="", dest_path=""):
            self.src_path = src_path
            self.dest_path = dest_path
            self.is_directory = False
    
    Observer = None
    logger.warning("watchdog library not available. Cache monitoring will be limited.")


class CacheMonitorHandler(FileSystemEventHandler):
    """Handler for file system events on the cache directory."""
    
    def __init__(self, cache_path: str, log_callback: Optional[Callable] = None):
        """
        Initialize the cache monitor handler.
        
        Args:
            cache_path: Path to the cache directory being monitored
            log_callback: Optional callback function to log events
        """
        super().__init__()
        self.cache_path = Path(cache_path).resolve()
        self.log_callback = log_callback or self._default_log
        self.events: List[Dict] = []
        self.lock = threading.Lock()
        self.operation_counts = defaultdict(int)
        
    def _is_user_code(self, filepath: str) -> bool:
        """
        Generic check if a file path belongs to user codebase vs third-party libraries or standard library.
        Works for any file type and any programming language.
        
        Args:
            filepath: Full path to the file (can be any file type: .py, .cpp, .rs, .go, .java, etc.)
            
        Returns:
            True if the file is in user codebase, False otherwise
        """
        filepath_lower = filepath.lower()
        
        # Patterns that indicate user codebase (generic - works for any language/file type)
        user_code_patterns = [
            '/forge/',
            '/test/',
            'third_party/tt-mlir',
            'third_party/tt-metal',
            'third_party/tt_forge_models',
            '/proj_sw/',
            '/github/workspace/',
            '/__w/',
            '/workspace/',
            '/home/',
            '/user_dev/',
        ]
        
        # Patterns that indicate third-party or standard library code (generic)
        skip_patterns = [
            'site-packages',
            'dist-packages',
            'lib/python',
            'lib64/python',
            'threading.py',
            'concurrent/',
            'multiprocessing/',
            'asyncio/',
            'watchdog/',
            'pytest/',
            'cache_monitor.py',  # Skip our own monitoring code
            '/usr/lib',
            '/usr/local/lib',
            '/opt/',
            '/snap/',
            '/var/lib',
        ]
        
        # Check if it's in skip patterns first
        for pattern in skip_patterns:
            if pattern in filepath_lower:
                return False
        
        # Check if it's in user code patterns
        for pattern in user_code_patterns:
            if pattern in filepath_lower:
                return True
        
        # Generic heuristic: if path doesn't look like system/third-party library,
        # and it's an absolute path in a user-like directory, consider it user code
        if os.path.isabs(filepath):
            # Skip system directories
            if any(skip in filepath_lower for skip in ['/usr/', '/opt/', '/var/', '/snap/', '/lib/', '/bin/']):
                # But allow if it's clearly in a user/project directory
                if any(user_dir in filepath_lower for user_dir in ['/home/', '/proj_', '/workspace/', '/github/']):
                    return True
                return False
            
            # If it's an absolute path not in system directories, might be user code
            # Be conservative - only return True if we're confident
            if any(user_dir in filepath_lower for user_dir in ['/home/', '/proj_', '/workspace/', '/github/', '/user_dev/']):
                return True
        
        return False
    
    def _is_system_or_third_party_code(self, filepath: str) -> bool:
        """
        Generic check if a file path belongs to system/third-party code vs user codebase.
        Works for any file type and any programming language.
        
        Args:
            filepath: Full path to the file (can be any file type: .py, .cpp, .rs, .go, .java, etc.)
            
        Returns:
            True if the file is system/third-party code, False if it's user code
        """
        filepath_lower = filepath.lower()
        
        # Patterns that indicate system/third-party code (generic - works for any language/file type)
        system_patterns = [
            'site-packages',
            'dist-packages',
            'lib/python',
            'lib64/python',
            'threading.py',
            'concurrent/',
            'multiprocessing/',
            'asyncio/',
            'watchdog/',
            'pytest/',
            'cache_monitor.py',  # Skip our own monitoring code
            '/usr/lib',
            '/usr/local/lib',
            '/opt/',
            '/snap/',
            '/var/lib',
            '/lib/',
            '/bin/',
            'shutil.py',
            'os.py',
            'pathlib',
        ]
        
        # Check if it matches any system pattern
        for pattern in system_patterns:
            if pattern in filepath_lower:
                return True
        
        return False
    
    def _get_caller_info(self) -> Dict[str, str]:
        """
        Extract caller information from the stack trace using completely generic approach.
        
        This method:
        - Works for any file type (.py, .cpp, .rs, .go, .java, etc.)
        - Works for any function name (doesn't rely on specific function names)
        - Uses path-based detection to identify user code vs system/third-party code
        - Handles standard library wrappers by finding the actual caller
        
        Returns:
            Dictionary with 'file', 'function', 'line', and 'full_path' keys
        """
        try:
            stack = inspect.stack()
        except Exception:
            return {
                'file': 'unknown',
                'function': 'unknown',
                'line': 0,
                'full_path': 'unknown'
            }
        
        # Skip internal frames (this method, event handler, watchdog internals)
        # Start from frame 3 to skip: _get_caller_info, _record_event, watchdog handler
        
        # Generic approach: Walk through the stack and find the first frame that:
        # 1. Is in user codebase (forge/, test/, etc.) - identified by path patterns
        # 2. Is not in system/third-party code (site-packages, lib/python, etc.)
        # 3. Is not a wrapper/infrastructure frame (threading, watchdog, etc.)
        
        for i, frame_info in enumerate(stack[3:], start=3):
            filename = frame_info.filename
            function = frame_info.function
            lineno = frame_info.lineno
            
            # Skip if it's system/third-party code
            if self._is_system_or_third_party_code(filename):
                # If we're in a standard library file operation (shutil, os, etc.),
                # look at the next frame to find the actual caller
                if any(op in filename.lower() for op in ['shutil', 'os.py', 'pathlib']) or 'lib/python' in filename.lower():
                    # Found a standard library file operation - find what called it
                    if i + 1 < len(stack):
                        caller_frame = stack[i + 1]
                        caller_filename = caller_frame.filename
                        caller_function = caller_frame.function
                        caller_lineno = caller_frame.lineno
                        
                        # Skip if still in system code
                        if not self._is_system_or_third_party_code(caller_filename):
                            # Check if it's user code
                            if self._is_user_code(caller_filename):
                                return {
                                    'file': os.path.basename(caller_filename),
                                    'function': caller_function,
                                    'line': caller_lineno,
                                    'full_path': caller_filename
                                }
                            # Even if not clearly user code, return it if it's not system code
                            return {
                                'file': os.path.basename(caller_filename),
                                'function': caller_function,
                                'line': caller_lineno,
                                'full_path': caller_filename
                            }
                continue
            
            # Check if this is user code (generic - works for any file type)
            if self._is_user_code(filename):
                return {
                    'file': os.path.basename(filename),
                    'function': function,
                    'line': lineno,
                    'full_path': filename
                }
        
        # Second pass: Handle threading/async contexts
        # Look deeper in the stack when we encounter threading wrappers
        for i, frame_info in enumerate(stack[3:]):
            filename = frame_info.filename
            function = frame_info.function
            
            # Check if we're in threading/async infrastructure
            if 'threading.py' in filename.lower() or 'asyncio' in filename.lower() or \
               function in ['_bootstrap_inner', '_bootstrap', 'run', '_run']:
                # Look ahead in the stack for the actual caller (check up to 25 frames ahead)
                for j, next_frame in enumerate(stack[3+i+1:3+i+26]):  # Check next 25 frames
                    next_filename = next_frame.filename
                    next_function = next_frame.function
                    next_lineno = next_frame.lineno
                    
                    # Skip if still in threading/async or watchdog infrastructure
                    if 'threading.py' in next_filename.lower() or 'asyncio' in next_filename.lower() or \
                       'watchdog' in next_filename.lower():
                        continue
                    
                    # Skip if it's system/third-party code
                    if self._is_system_or_third_party_code(next_filename):
                        # If it's a standard library file operation, find what called it
                        if any(op in next_filename.lower() for op in ['shutil', 'os.py', 'pathlib']):
                            if 3+i+1+j+1 < len(stack):
                                caller_frame = stack[3+i+1+j+1]
                                caller_filename = caller_frame.filename
                                caller_function = caller_frame.function
                                caller_lineno = caller_frame.lineno
                                
                                # Skip if still in system code
                                if not self._is_system_or_third_party_code(caller_filename):
                                    if self._is_user_code(caller_filename):
                                        return {
                                            'file': os.path.basename(caller_filename),
                                            'function': caller_function,
                                            'line': caller_lineno,
                                            'full_path': caller_filename
                                        }
                                    return {
                                        'file': os.path.basename(caller_filename),
                                        'function': caller_function,
                                        'line': caller_lineno,
                                        'full_path': caller_filename
                                    }
                        continue
                    
                    # Found a potential caller - check if it's user code
                    if self._is_user_code(next_filename):
                        return {
                            'file': os.path.basename(next_filename),
                            'function': next_function,
                            'line': next_lineno,
                            'full_path': next_filename
                        }
                    
                    # If not system code, return it (generic approach - don't filter by function name)
                    if not self._is_system_or_third_party_code(next_filename):
                        return {
                            'file': os.path.basename(next_filename),
                            'function': next_function,
                            'line': next_lineno,
                            'full_path': next_filename
                        }
        
        # Third pass: Return the first non-watchdog, non-system frame (completely generic)
        for frame_info in stack[3:]:
            filename = frame_info.filename
            function = frame_info.function
            lineno = frame_info.lineno
            
            # Skip system/third-party code
            if self._is_system_or_third_party_code(filename):
                continue
            
            # Return the first non-system frame (generic - any function name is fine)
            return {
                'file': os.path.basename(filename),
                'function': function,
                'line': lineno,
                'full_path': filename
            }
        
        return {
            'file': 'unknown',
            'function': 'unknown',
            'line': 0,
            'full_path': 'unknown'
        }
    
    def _find_process_with_file(self, file_path: str) -> Optional[Dict[str, str]]:
        """
        Find the process that has the file open or recently modified it.
        This works for C++ processes and other non-Python processes.
        """
        try:
            import psutil
            import subprocess
            
            # Try to find process using lsof (works on Linux/macOS)
            try:
                result = subprocess.run(
                    ['lsof', file_path],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout:
                    # Parse lsof output: COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:  # Skip header
                        parts = lines[1].split()
                        if len(parts) >= 2:
                            pid = int(parts[1])
                            try:
                                process = psutil.Process(pid)
                                exe_path = process.exe() if hasattr(process, 'exe') else 'unknown'
                                cmdline = process.cmdline()
                                
                                # Extract executable name (generic - works for any executable: C++, Rust, Go, etc.)
                                exe_name = os.path.basename(exe_path) if exe_path != 'unknown' else process.name()
                                
                                # Build informative command line (completely generic approach)
                                # Extract meaningful parts: executable + files or key args (works for any language)
                                cmdline_parts = [exe_name]  # Start with executable name
                                
                                # Generic approach: look for any files or meaningful arguments
                                # Works for any language: C++, Rust, Go, Java, Python, etc.
                                for arg in cmdline[1:8]:  # Check next 7 args after executable
                                    # Include any file arguments (source files, object files, etc.)
                                    # Generic check: has a file extension or looks like a file path
                                    if '.' in arg and os.path.basename(arg) != arg:
                                        # Looks like a file path with extension
                                        cmdline_parts.append(os.path.basename(arg))
                                    # Include important flags/options (any language)
                                    elif arg.startswith('-') and len(arg) > 2:
                                        cmdline_parts.append(arg)
                                    # Include short non-path arguments that might be meaningful
                                    elif not arg.startswith('/') and len(arg) < 100 and ' ' not in arg:
                                        # Avoid very long paths and multi-word arguments
                                        cmdline_parts.append(os.path.basename(arg) if '/' in arg else arg)
                                    
                                    if len(cmdline_parts) >= 4:  # Limit to 4 parts total for readability
                                        break
                                
                                cmdline_str = ' '.join(cmdline_parts) if len(cmdline_parts) > 1 else (cmdline[0] if cmdline else 'unknown')
                                
                                return {
                                    'pid': pid,
                                    'name': process.name(),
                                    'cmdline': cmdline_str,
                                    'exe': exe_path,
                                    'exe_name': exe_name  # Add executable name for better identification
                                }
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Fallback: Check all processes for files in the cache directory
            # This is slower but works when lsof is not available
            # Generic approach - works for any executable type
            try:
                cache_dir = str(self.cache_path)
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'open_files']):
                    try:
                        if proc.info['open_files']:
                            for file_info in proc.info['open_files']:
                                if cache_dir in file_info.path:
                                    cmdline = proc.info['cmdline']
                                    # Generic command line extraction (same as above)
                                    exe_name = proc.info['name']
                                    cmdline_parts = [exe_name]
                                    
                                    # Extract meaningful parts (generic - works for any language)
                                    for arg in (cmdline[1:8] if cmdline else []):
                                        if '.' in arg and os.path.basename(arg) != arg:
                                            cmdline_parts.append(os.path.basename(arg))
                                        elif arg.startswith('-') and len(arg) > 2:
                                            cmdline_parts.append(arg)
                                        elif not arg.startswith('/') and len(arg) < 100 and ' ' not in arg:
                                            cmdline_parts.append(os.path.basename(arg) if '/' in arg else arg)
                                        
                                        if len(cmdline_parts) >= 4:
                                            break
                                    
                                    cmdline_str = ' '.join(cmdline_parts) if len(cmdline_parts) > 1 else (cmdline[0] if cmdline else 'unknown')
                                    
                                    return {
                                        'pid': proc.info['pid'],
                                        'name': proc.info['name'],
                                        'cmdline': cmdline_str,
                                        'exe': 'unknown',
                                        'exe_name': exe_name
                                    }
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
            except Exception:
                pass
                
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
    
    def _get_process_info(self, file_path: Optional[str] = None) -> Dict[str, str]:
        """
        Get information about the process performing the operation.
        For C++ processes, tries to find the actual process that has the file open.
        """
        # First try to find the process with the file (for C++ processes)
        if file_path:
            proc_info = self._find_process_with_file(file_path)
            if proc_info:
                return proc_info
        
        # Fallback to current Python process
        try:
            import psutil
            process = psutil.Process()
            return {
                'pid': process.pid,
                'name': process.name(),
                'cmdline': ' '.join(process.cmdline()[:5]) if len(process.cmdline()) > 0 else 'unknown',
                'exe': process.exe() if hasattr(process, 'exe') else 'unknown',
                'exe_name': os.path.basename(process.exe()) if hasattr(process, 'exe') else process.name()
            }
        except ImportError:
            # psutil not available, use basic info
            return {
                'pid': os.getpid(),
                'name': 'python',
                'cmdline': ' '.join(sys.argv[:5]) if len(sys.argv) > 1 else 'unknown',
                'exe': sys.executable,
                'exe_name': os.path.basename(sys.executable)
            }
        except Exception:
            return {
                'pid': os.getpid(),
                'name': 'unknown',
                'cmdline': 'unknown',
                'exe': 'unknown',
                'exe_name': 'unknown'
            }
    
    def _get_current_test_name(self) -> str:
        """Get the current test name from pytest environment or global variable."""
        # Try PYTEST_CURRENT_TEST environment variable first (set by pytest)
        test_name = os.environ.get("PYTEST_CURRENT_TEST", "")
        if test_name:
            # Format: "test_file.py::test_function (call)" -> "test_file.py::test_function"
            test_name = test_name.replace(" (call)", "").replace(" (setup)", "").replace(" (teardown)", "")
            return test_name
        
        # Try reading from .pytest_current_test_executing file (set by conftest.py)
        try:
            test_file = Path(".pytest_current_test_executing")
            if test_file.exists():
                test_name = test_file.read_text().strip()
                if test_name:
                    return test_name
        except Exception:
            pass
        
        return "unknown_test"
    
    def _default_log(self, event_type: str, src_path: str, details: Dict):
        """Default logging callback - logs events immediately as they occur."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        caller = details.get('caller', {})
        process = details.get('process', {})
        relative_path = details.get('relative_path', src_path)
        
        # Get current test name
        test_name = self._get_current_test_name()
        # Extract just the test function name for cleaner display
        if "::" in test_name:
            test_display = test_name.split("::")[-1]  # Just the function name
        else:
            test_display = test_name
        
        # Build process info string
        proc_name = process.get('exe_name') or process.get('name', '?')
        proc_exe = process.get('exe', '')
        if proc_exe and proc_exe != 'unknown' and not proc_name:
            # Extract executable name from path
            proc_name = os.path.basename(proc_exe)
        
        # Build caller info string with file, function, and line number
        caller_file = caller.get('file', '?')
        caller_func = caller.get('function', '?')
        caller_line = caller.get('line', 0)
        caller_full_path = caller.get('full_path', '')
        
        # For non-Python processes (C++, Rust, Go, Java, etc.), we won't have Python caller info
        # Show command line instead (generic - works for any executable)
        if caller_file == 'unknown' or '?' in caller_file:
            # This is likely a non-Python process, show command line info
            cmdline = process.get('cmdline', 'unknown')
            if cmdline and cmdline != 'unknown':
                # Extract meaningful parts from command line (executable and key args)
                # Generic approach: show first few parts (works for any language)
                cmd_parts = cmdline.split()[:4]  # First 4 parts for better context
                # Try to identify language/type from executable name, but be generic
                if any(lang in cmdline.lower() for lang in ['g++', 'gcc', 'clang', 'rustc', 'go', 'javac', 'node']):
                    lang_hint = os.path.basename(cmd_parts[0]) if cmd_parts else 'executable'
                    caller_info = f"[{lang_hint}:{' '.join(cmd_parts[1:4])}]" if len(cmd_parts) > 1 else f"[{lang_hint}]"
                else:
                    caller_info = f"[exec:{' '.join(cmd_parts)}]"
            else:
                caller_info = f"[exec:{proc_name}]"
        else:
            # Python process - show full path, function, and line number
            # Format: path/to/file.py:function():line
            if caller_full_path and caller_full_path != 'unknown':
                # Use full path, but make it relative to workspace root if possible
                # Try to extract a meaningful path (relative to common project roots)
                display_path = caller_full_path
                workspace_roots = ['forge/', 'test/', 'third_party/']
                for root in workspace_roots:
                    if root in caller_full_path:
                        idx = caller_full_path.find(root)
                        display_path = caller_full_path[idx:]
                        break
                
                if caller_line > 0:
                    caller_info = f"{display_path}:{caller_func}():{caller_line}"
                else:
                    caller_info = f"{display_path}:{caller_func}()"
            else:
                # Fallback to just filename if full path not available
                if caller_line > 0:
                    caller_info = f"{caller_file}:{caller_func}():{caller_line}"
                else:
                    caller_info = f"{caller_file}:{caller_func}()"
        
        # Log immediately with both logger and print for immediate visibility
        log_message = (
            f"[CACHE_MONITOR] {timestamp} | {event_type:12} | "
            f"Test:{test_display} | "
            f"PID:{process.get('pid', '?')} ({proc_name}) | "
            f"{caller_info} | "
            f"{relative_path}"
        )
        logger.info(log_message)
        print(log_message)  # Print immediately for real-time visibility
        
        # For DELETE events, also print the full stack trace
        if event_type == 'DELETED':
            stack_trace = details.get('stack_trace', [])
            caller_file = caller.get('file', '?')
            caller_func = caller.get('function', '?')
            caller_line = caller.get('line', 0)
            caller_full_path = caller.get('full_path', '')
            
            # Check if we found a valid caller (not unknown/exec fallback)
            has_valid_caller = (caller_file != 'unknown' and '?' not in caller_file and 
                              caller_file != 'exec' and caller_line > 0)
            
            if stack_trace:
                logger.info(f"[CACHE_MONITOR] Full stack trace for DELETE operation:")
                print(f"[CACHE_MONITOR] Full stack trace for DELETE operation:")
                for frame in stack_trace:
                    logger.info(f"[CACHE_MONITOR] {frame}")
                    print(f"[CACHE_MONITOR] {frame}")
                
                # Add helpful note about asynchronous detection
                if not has_valid_caller:
                    logger.warning(f"[CACHE_MONITOR] WARNING: Deletion detected asynchronously by watchdog thread.")
                    logger.warning(f"[CACHE_MONITOR] The stack trace above shows the watchdog detection path, not the original deletion path.")
                    logger.warning(f"[CACHE_MONITOR] To find the actual deletion location, search for 'shutil.rmtree' or 'os.remove' calls in the codebase.")
                    print(f"[CACHE_MONITOR] WARNING: Deletion detected asynchronously by watchdog thread.")
                    print(f"[CACHE_MONITOR] The stack trace above shows the watchdog detection path, not the original deletion path.")
                    print(f"[CACHE_MONITOR] To find the actual deletion location, search for 'shutil.rmtree' or 'os.remove' calls in the codebase.")
                else:
                    logger.info(f"[CACHE_MONITOR] Actual deletion caller: {caller_full_path}:{caller_func}():{caller_line}")
                    print(f"[CACHE_MONITOR] Actual deletion caller: {caller_full_path}:{caller_func}():{caller_line}")
            else:
                logger.info(f"[CACHE_MONITOR] Stack trace not available for DELETE operation")
                if has_valid_caller:
                    logger.info(f"[CACHE_MONITOR] Actual deletion caller: {caller_full_path}:{caller_func}():{caller_line}")
                    print(f"[CACHE_MONITOR] Actual deletion caller: {caller_full_path}:{caller_func}():{caller_line}")
                else:
                    logger.warning(f"[CACHE_MONITOR] WARNING: Could not determine deletion caller. Deletion detected asynchronously.")
                    print(f"[CACHE_MONITOR] WARNING: Could not determine deletion caller. Deletion detected asynchronously.")
    
    def _get_full_stack_trace(self) -> List[str]:
        """
        Get the full stack trace as a list of formatted strings.
        Generic approach - filters out watchdog/threading/system frames and shows user code frames.
        Works for any file type and function name.
        
        The challenge: When watchdog detects a deletion, it's called asynchronously from a thread,
        so we need to walk through the stack to find the actual caller that triggered the deletion.
        """
        try:
            stack = inspect.stack()
            stack_trace = []
            
            # Skip the first 3 frames: _get_full_stack_trace, _record_event, watchdog handler
            # We need to walk through the stack to find user code frames
            
            # First, look for standard library file operations (shutil.rmtree, os.remove, etc.)
            # and trace back to find the actual caller
            stdlib_file_ops = ['shutil', 'os.py', 'pathlib']
            
            for i, frame_info in enumerate(stack[3:], start=3):
                filename = frame_info.filename
                function = frame_info.function
                lineno = frame_info.lineno
                
                # Check if we're in a standard library file operation
                if any(op in filename.lower() for op in stdlib_file_ops) or 'lib/python' in filename.lower():
                    # Found shutil.rmtree, os.remove, etc. - find what called it
                    # Look at the next frame (the caller)
                    if i + 1 < len(stack):
                        caller_frame = stack[i + 1]
                        caller_filename = caller_frame.filename
                        caller_function = caller_frame.function
                        caller_lineno = caller_frame.lineno
                        
                        # Skip if still in system code
                        if not self._is_system_or_third_party_code(caller_filename):
                            # Format the stdlib function
                            stdlib_display = os.path.basename(filename)
                            workspace_roots = ['forge/', 'test/', 'third_party/']
                            for root in workspace_roots:
                                if root in filename:
                                    idx = filename.find(root)
                                    stdlib_display = filename[idx:]
                                    break
                            
                            # Format the caller
                            display_path = caller_filename
                            for root in workspace_roots:
                                if root in caller_filename:
                                    idx = caller_filename.find(root)
                                    display_path = caller_filename[idx:]
                                    break
                            
                            # Add stdlib frame and caller frame
                            stack_trace.append(f"  {stdlib_display}:{function}():{lineno} [STDLIB]")
                            stack_trace.append(f"  {display_path}:{caller_function}():{caller_lineno}")
                            
                            # Continue walking up the stack to find more user code frames
                            for j in range(i + 2, min(i + 30, len(stack))):  # Check up to 30 more frames
                                next_frame = stack[j]
                                next_filename = next_frame.filename
                                next_function = next_frame.function
                                next_lineno = next_frame.lineno
                                
                                # Skip system/third-party code
                                if self._is_system_or_third_party_code(next_filename):
                                    continue
                                
                                # Check if it's user code
                                if self._is_user_code(next_filename):
                                    next_display = next_filename
                                    for root in workspace_roots:
                                        if root in next_filename:
                                            idx = next_filename.find(root)
                                            next_display = next_filename[idx:]
                                            break
                                    stack_trace.append(f"  {next_display}:{next_function}():{next_lineno}")
                            
                            break
            
            # If we didn't find stdlib file ops, look for user code frames directly
            if not stack_trace:
                for frame_info in stack[3:]:
                    filename = frame_info.filename
                    function = frame_info.function
                    lineno = frame_info.lineno
                    
                    # Skip system/third-party code
                    if self._is_system_or_third_party_code(filename):
                        continue
                    
                    # Check if it's user code
                    if self._is_user_code(filename):
                        display_path = filename
                        workspace_roots = ['forge/', 'test/', 'third_party/']
                        for root in workspace_roots:
                            if root in filename:
                                idx = filename.find(root)
                                display_path = filename[idx:]
                                break
                        
                        stack_trace.append(f"  {display_path}:{function}():{lineno}")
            
            return stack_trace
        except Exception:
            return []
    
    def _record_event(self, event_type: str, src_path: str, dest_path: Optional[str] = None):
        """Record a file system event with full context."""
        caller_info = self._get_caller_info()
        # Try to find the actual process that performed the operation (works for C++ processes)
        process_info = self._get_process_info(src_path)
        
        # Capture full stack trace for DELETE events
        stack_trace = []
        if event_type == 'DELETED':
            stack_trace = self._get_full_stack_trace()
        
        event_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'src_path': src_path,
            'dest_path': dest_path,
            'caller': caller_info,
            'process': process_info,
            'relative_path': str(Path(src_path).relative_to(self.cache_path)) if str(self.cache_path) in src_path else src_path,
            'stack_trace': stack_trace  # Full stack trace for DELETE events
        }
        
        with self.lock:
            self.events.append(event_data)
            self.operation_counts[event_type] += 1
            
        # Call the log callback
        if self.log_callback:
            self.log_callback(event_type, src_path, event_data)
    
    def on_created(self, event: FileSystemEvent):
        """Handle file/directory creation events."""
        if not event.is_directory:
            self._record_event('CREATED', event.src_path)
    
    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if not event.is_directory:
            self._record_event('MODIFIED', event.src_path)
    
    def on_deleted(self, event: FileSystemEvent):
        """Handle file/directory deletion events."""
        self._record_event('DELETED', event.src_path)
    
    def on_moved(self, event: FileSystemEvent):
        """Handle file/directory move/rename events."""
        self._record_event('MOVED', event.src_path, event.dest_path)
    
    def get_events(self) -> List[Dict]:
        """Get all recorded events."""
        with self.lock:
            return self.events.copy()
    
    def get_summary(self) -> Dict:
        """Get a summary of all operations."""
        with self.lock:
            return {
                'total_events': len(self.events),
                'operation_counts': dict(self.operation_counts),
                'events': self.events[-50:]  # Last 50 events
            }
    
    def clear_events(self):
        """Clear all recorded events."""
        with self.lock:
            self.events.clear()
            self.operation_counts.clear()


class CacheMonitor:
    """Monitor for tracking file system operations on tt-metal-cache directory."""
    
    def __init__(self, cache_paths: List[str], log_callback: Optional[Callable] = None):
        """
        Initialize the cache monitor.
        
        Args:
            cache_paths: List of cache directory paths to monitor
            log_callback: Optional callback function for logging events
        """
        self.cache_paths = [Path(p).resolve() for p in cache_paths if p]
        self.log_callback = log_callback
        self.observer: Optional[Observer] = None
        self.handlers: List[CacheMonitorHandler] = []
        self.is_monitoring = False
        
    def start(self):
        """Start monitoring the cache directories."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog not available, cannot start file system monitoring")
            return False
        
        if self.is_monitoring:
            logger.warning("Monitor is already running")
            return False
        
        # Filter to only existing cache directories
        existing_paths = [p for p in self.cache_paths if p.exists()]
        
        if not existing_paths:
            logger.debug("No existing cache directories found to monitor")
            return False
        
        self.observer = Observer()
        
        for cache_path in existing_paths:
            handler = CacheMonitorHandler(str(cache_path), self.log_callback)
            self.handlers.append(handler)
            
            try:
                # Only monitor existing cache directories
                self.observer.schedule(handler, str(cache_path), recursive=True)
                logger.info(f"Started monitoring cache directory: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to start monitoring {cache_path}: {e}")
        
        if not self.handlers:
            logger.warning("No handlers created, cannot start monitoring")
            return False
        
        try:
            self.observer.start()
            self.is_monitoring = True
            logger.info("Cache monitor started successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to start cache monitor: {e}")
            return False
    
    def stop(self):
        """Stop monitoring the cache directories."""
        if not self.is_monitoring or not self.observer:
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            self.is_monitoring = False
            logger.info("Cache monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping cache monitor: {e}")
    
    def get_all_events(self) -> List[Dict]:
        """Get all events from all handlers."""
        all_events = []
        for handler in self.handlers:
            all_events.extend(handler.get_events())
        # Sort by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', ''))
        return all_events
    
    def get_summary(self) -> Dict:
        """Get summary from all handlers."""
        summaries = []
        for handler in self.handlers:
            summaries.append(handler.get_summary())
        
        # Combine summaries
        total_events = sum(s['total_events'] for s in summaries)
        combined_counts = defaultdict(int)
        for s in summaries:
            for op, count in s['operation_counts'].items():
                combined_counts[op] += count
        
        return {
            'total_events': total_events,
            'operation_counts': dict(combined_counts),
            'handlers': summaries
        }
    
    def clear_events(self):
        """Clear all events from all handlers."""
        for handler in self.handlers:
            handler.clear_events()
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def get_cache_paths() -> List[str]:
    """
    Get tt-metal-cache directory paths that actually exist.
    
    Only returns paths where the cache directory itself exists.
    The cache will only be present in one of the possible paths, not all of them.
    
    Returns:
        List of existing cache directory paths to monitor (empty if none exist)
    """
    cache_paths = []
    
    # CI environment
    ci_cache = "/github/home/.cache/tt-metal-cache"
    if os.path.exists(ci_cache):
        cache_paths.append(ci_cache)
    
    # Local environment
    local_cache = os.path.join(os.path.expanduser("~"), ".cache", "tt-metal-cache")
    if os.path.exists(local_cache):
        cache_paths.append(local_cache)
    
    # Fallback
    fallback_cache = "/tmp/tt-metal-cache"
    if os.path.exists(fallback_cache):
        cache_paths.append(fallback_cache)
    
    # Also check environment variable
    env_cache = os.environ.get("TT_METAL_CACHE_DIR")
    if env_cache and os.path.exists(env_cache) and env_cache not in cache_paths:
        cache_paths.append(env_cache)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in cache_paths:
        resolved = str(Path(path).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(resolved)
    
    return unique_paths
