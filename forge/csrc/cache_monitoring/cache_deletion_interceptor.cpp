// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cache_deletion_interceptor.cpp
 * @brief Generic C++ cache deletion interceptor using LD_PRELOAD
 * 
 * This library intercepts all file deletion system calls and captures stack traces
 * when tt-metal-cache directories are deleted. Works with ANY C++ code using ANY
 * deletion method.
 * 
 * Usage:
 *   1. Compile this into a shared library (.so)
 *   2. Use LD_PRELOAD to inject it before running tests
 *   3. All cache deletions will be automatically traced
 * 
 * Example:
 *   LD_PRELOAD=/path/to/libcache_interceptor.so pytest test.py
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // Required for RTLD_NEXT
#endif
#include <dlfcn.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <string>
#include <vector>
#include <sstream>

// Thread-local flag to prevent recursion
static __thread int in_interceptor = 0;

// Function pointer types for original functions
typedef int (*orig_unlink_t)(const char *pathname);
typedef int (*orig_rmdir_t)(const char *pathname);
typedef int (*orig_remove_t)(const char *pathname);
typedef int (*orig_unlinkat_t)(int dirfd, const char *pathname, int flags);

// Mutex for thread-safe logging
static pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
static time_t last_log_timestamp = 0;
static std::string last_log_path;
static std::string last_log_operation;

/**
 * @brief Check if path is the tt-metal-cache ROOT directory (not a subfile/subdir).
 * Only log when the root is deleted to avoid flooding logs during remove_all recursion.
 */
static bool is_cache_root(const char *path)
{
    if (!path || !path[0]) return false;
    
    std::string p(path);
    while (p.size() > 1 && p.back() == '/') p.pop_back();  // trim trailing slash
    
    size_t pos = p.rfind("tt-metal-cache");
    if (pos == std::string::npos) return false;
    
    if (pos + 14 != p.size()) return false;  // must end with "tt-metal-cache"
    if (pos == 0) return true;               // path is "tt-metal-cache"
    if (p[pos - 1] == '/') return true;      // path ends with "/tt-metal-cache"
    
    return false;
}

/**
 * @brief Demangle C++ symbol names
 */
static std::string demangle(const char* mangled)
{
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    
    if (status == 0 && demangled != nullptr)
    {
        std::string result(demangled);
        free(demangled);
        return result;
    }
    
    return std::string(mangled);
}

/**
 * @brief Parse backtrace symbol to extract binary path and address
 * Format: /path/to/binary(mangled+0x123) [0x7f682484c13b]
 */
static bool parse_backtrace_frame(const char* symbol, std::string& binary_out, void*& addr_out)
{
    std::string sym(symbol);
    
    size_t paren = sym.find('(');
    size_t bracket = sym.rfind('[');
    size_t bracket_end = sym.rfind(']');
    
    if (paren == std::string::npos || bracket == std::string::npos || bracket_end == std::string::npos)
        return false;
    
    binary_out = sym.substr(0, paren);
    std::string addr_str = sym.substr(bracket + 1, bracket_end - bracket - 1);
    
    uintptr_t addr_val = 0;
    if (sscanf(addr_str.c_str(), "%lx", (unsigned long*)&addr_val) != 1)
        return false;
    
    addr_out = (void*)addr_val;
    return true;
}

/**
 * @brief Resolve address to file:line using addr2line
 */
static std::string resolve_to_file_line(void* addr)
{
    Dl_info info;
    if (dladdr(addr, &info) == 0 || !info.dli_fname)
        return "";
    
    uintptr_t offset = (uintptr_t)addr - (uintptr_t)info.dli_fbase;
    
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "addr2line -e %s 0x%lx 2>/dev/null", info.dli_fname, (unsigned long)offset);
    
    FILE* fp = popen(cmd, "r");
    if (!fp) return "";
    
    char loc_buf[1024] = {0};
    if (fgets(loc_buf, sizeof(loc_buf), fp))
    {
        pclose(fp);
        
        std::string loc(loc_buf);
        while (!loc.empty() && (loc.back() == '\n' || loc.back() == '\r')) loc.pop_back();
        
        if (loc.find("??") != std::string::npos)
            return "";  // No debug info (addr2line returns ?? or ??:0)
        if (loc.find(':') == std::string::npos) return "";
        
        size_t colon = loc.rfind(':');
        std::string file = loc.substr(0, colon);
        std::string line = loc.substr(colon + 1);
        
        if (file.find("??") != std::string::npos || line == "?" || line == "0")
            return "";

        size_t last_slash = file.find_last_of('/');
        if (last_slash != std::string::npos)
            file = file.substr(last_slash + 1);

        return file + ":" + line;
    }
    
    pclose(fp);
    return "";
}

/**
 * @brief Parse backtrace to human-readable form (fallback when addr2line has no debug info)
 */
static std::string parse_backtrace_symbol(const char* symbol)
{
    std::string sym(symbol);
    size_t paren = sym.find('(');
    size_t plus = (paren != std::string::npos) ? sym.find('+', paren) : std::string::npos;
    size_t close = (plus != std::string::npos) ? sym.find(')', plus) : std::string::npos;
    if (paren == std::string::npos || plus == std::string::npos) return "  " + sym;
    std::string binary = sym.substr(0, paren);
    std::string mangled = sym.substr(paren + 1, plus - paren - 1);
    std::string offset = (close != std::string::npos) ? sym.substr(plus, close - plus) : "";
    size_t last_slash = binary.find_last_of('/');
    if (last_slash != std::string::npos) binary = binary.substr(last_slash + 1);
    if (mangled.empty()) return "  " + sym;
    return "  " + demangle(mangled.c_str()) + " " + offset + " [" + binary + "]";
}

/**
 * @brief Truncate long function names for log readability
 */
static std::string truncate_for_log(const std::string& s, size_t max_len = 140)
{
    if (s.size() <= max_len) return s;
    return s.substr(0, max_len) + "...";
}

/**
 * @brief Stop point for call-chain rendering (Python/runtime entry).
 * We show complete C/C++ chain from delete() upward until Python runtime.
 */
static bool is_chain_terminator(const char* symbol)
{
    std::string sym(symbol);
    return (sym.find("python") != std::string::npos ||
            sym.find("Python") != std::string::npos ||
            sym.find("_Py") != std::string::npos ||
            sym.find("PyEval") != std::string::npos ||
            sym.find("Py_") != std::string::npos ||
            sym.find("libc.so") != std::string::npos ||
            sym.find("__libc_") != std::string::npos ||
            sym.find("_start") != std::string::npos);
}

/**
 * @brief Format a single frame as: function at file:line (or fallback)
 */
static std::string format_frame(const char* symbol)
{
    std::string binary;
    void* addr = nullptr;
    
    if (!parse_backtrace_frame(symbol, binary, addr))
        return parse_backtrace_symbol(symbol);
    (void)binary;
    
    std::string resolved = resolve_to_file_line(addr);
    std::string fallback = parse_backtrace_symbol(symbol);
    std::string func = fallback.rfind("  ", 0) == 0 ? fallback.substr(2) : fallback;
    func = truncate_for_log(func);

    if (!resolved.empty())
        return "  " + func + " @ " + resolved;
    
    return "  " + func;
}

/**
 * @brief Capture and format C++ stack trace from deletion call to top-level C++ calls
 */
static std::string capture_stack_trace(const char* operation, const char* path)
{
    std::ostringstream oss;
    
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    char time_buf[32];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);
    
    oss << "\n[CPP_CACHE_DELETION_DETECTED] tt-metal-cache root deleted | " << time_buf << " | " << operation << "() | " << path << "\n";
    oss << "[CPP_CALL_CHAIN] delete() -> top-level C++ calls\n";
    
    void* buffer[64];
    int nframes = backtrace(buffer, 64);
    
    if (nframes > 0)
    {
        char** symbols = backtrace_symbols(buffer, nframes);
        
        if (symbols != nullptr)
        {
            int frames_shown = 0;
            const int max_frames = 24;  // Show full C++ chain while avoiding runaway logs
            
            for (int i = 3; i < nframes && frames_shown < max_frames; i++)
            {
                if (is_chain_terminator(symbols[i]))
                    break;  // Stop when we reach Python/runtime boundary
                
                std::string line = format_frame(symbols[i]);
                if (!line.empty())
                {
                    oss << "  #" << frames_shown << " " << line.substr(2) << "\n";
                    frames_shown++;
                }
            }
            
            free(symbols);
        }
    }
    
    return oss.str();
}

/**
 * @brief Log the deletion event
 */
static void log_deletion(const char* operation, const char* path)
{
    // Prevent recursive calls
    if (in_interceptor) return;
    in_interceptor = 1;
    
    // Thread-safe logging
    pthread_mutex_lock(&log_mutex);

    // Deduplicate bursts: same operation/path within 1 second
    time_t now = time(NULL);
    if (last_log_operation == operation && last_log_path == path && (now - last_log_timestamp) <= 1)
    {
        pthread_mutex_unlock(&log_mutex);
        in_interceptor = 0;
        return;
    }
    last_log_operation = operation;
    last_log_path = path;
    last_log_timestamp = now;
    
    std::string trace = capture_stack_trace(operation, path);
    
    // Write to stderr for immediate visibility
    fprintf(stderr, "%s\n", trace.c_str());
    fflush(stderr);
    
    // Also try to write to a log file
    const char* log_file = getenv("CPP_CACHE_MONITOR_LOG");
    if (log_file)
    {
        FILE* f = fopen(log_file, "a");
        if (f)
        {
            fprintf(f, "%s\n", trace.c_str());
            fclose(f);
        }
    }
    
    pthread_mutex_unlock(&log_mutex);
    
    in_interceptor = 0;
}

// ============================================================================
// INTERCEPTED FUNCTIONS
// ============================================================================

/**
 * @brief Intercept unlink() - removes a file
 * Only log when tt-metal-cache ROOT is deleted (not every file during remove_all)
 */
extern "C" int unlink(const char *pathname)
{
    bool should_log = is_cache_root(pathname);
    
    static orig_unlink_t orig_unlink = nullptr;
    if (!orig_unlink)
    {
        orig_unlink = (orig_unlink_t)dlsym(RTLD_NEXT, "unlink");
    }
    
    if (should_log && !in_interceptor)
    {
        log_deletion("unlink", pathname);
    }
    
    return orig_unlink(pathname);
}

/**
 * @brief Intercept rmdir() - removes a directory
 */
extern "C" int rmdir(const char *pathname)
{
    bool should_log = is_cache_root(pathname);
    
    static orig_rmdir_t orig_rmdir = nullptr;
    if (!orig_rmdir)
    {
        orig_rmdir = (orig_rmdir_t)dlsym(RTLD_NEXT, "rmdir");
    }
    
    if (should_log && !in_interceptor)
    {
        log_deletion("rmdir", pathname);
    }
    
    return orig_rmdir(pathname);
}

/**
 * @brief Intercept remove() - removes a file or directory
 */
extern "C" int remove(const char *pathname)
{
    bool should_log = is_cache_root(pathname);
    
    static orig_remove_t orig_remove = nullptr;
    if (!orig_remove)
    {
        orig_remove = (orig_remove_t)dlsym(RTLD_NEXT, "remove");
    }
    
    if (should_log && !in_interceptor)
    {
        log_deletion("remove", pathname);
    }
    
    return orig_remove(pathname);
}

/**
 * @brief Intercept unlinkat() - removes a file relative to directory fd
 */
extern "C" int unlinkat(int dirfd, const char *pathname, int flags)
{
    bool should_log = is_cache_root(pathname);
    
    static orig_unlinkat_t orig_unlinkat = nullptr;
    if (!orig_unlinkat)
    {
        orig_unlinkat = (orig_unlinkat_t)dlsym(RTLD_NEXT, "unlinkat");
    }
    
    if (should_log && !in_interceptor)
    {
        log_deletion("unlinkat", pathname);
    }
    
    return orig_unlinkat(dirfd, pathname, flags);
}

// ============================================================================
// LIBRARY INITIALIZATION
// ============================================================================

/**
 * @brief Constructor - called when library is loaded
 * Silenced by default to avoid log noise; set CPP_CACHE_MONITOR_VERBOSE=1 to enable
 */
__attribute__((constructor))
static void init_interceptor()
{
    if (getenv("CPP_CACHE_MONITOR_VERBOSE") != nullptr)
    {
        fprintf(stderr, "[CPP_CACHE_INTERCEPTOR] Cache deletion monitor loaded\n");
        fflush(stderr);
    }
}

/**
 * @brief Destructor - called when library is unloaded
 */
__attribute__((destructor))
static void cleanup_interceptor()
{
    if (getenv("CPP_CACHE_MONITOR_VERBOSE") != nullptr)
    {
        fprintf(stderr, "[CPP_CACHE_INTERCEPTOR] Cache deletion monitor unloaded\n");
        fflush(stderr);
    }
}
