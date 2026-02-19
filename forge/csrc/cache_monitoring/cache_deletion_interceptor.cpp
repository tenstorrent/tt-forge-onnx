// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file cache_deletion_interceptor.cpp
 * @brief LD_PRELOAD interceptor: catches tt-metal-cache deletions and emits
 *        the ENTIRE call chain from the delete syscall back to the origin.
 *
 * Handles both directions:
 *   • Python -> C++  (e.g. test -> pybind11 -> forge C++ -> remove_all)
 *   • C++ -> Python  (e.g. C++ calls Python, deletion happens in Python)
 *
 * Shows every frame — C++, pybind11, libc, and Python — with file:line and
 * actual source code. No pattern matching; same format for all frames.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
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

static __thread int in_interceptor = 0;

typedef int (*orig_unlink_t)  (const char *pathname);
typedef int (*orig_rmdir_t)   (const char *pathname);
typedef int (*orig_remove_t)  (const char *pathname);
typedef int (*orig_unlinkat_t)(int dirfd, const char *pathname, int flags);

static pthread_mutex_t log_mutex   = PTHREAD_MUTEX_INITIALIZER;
static time_t          last_log_ts = 0;
static std::string     last_log_path;
static std::string     last_log_op;

// ============================================================================
// PATH FILTER — only tt-metal-cache root (no sub-file noise)
// ============================================================================

static bool is_cache_root(const char *path)
{
    if (!path || !path[0]) return false;
    std::string p(path);
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    size_t pos = p.rfind("tt-metal-cache");
    if (pos == std::string::npos || pos + 14 != p.size()) return false;
    return (pos == 0 || p[pos - 1] == '/');
}

// ============================================================================
// UTILITIES
// ============================================================================

static std::string demangle(const char* m)
{
    int s = 0;
    char* d = abi::__cxa_demangle(m, nullptr, nullptr, &s);
    if (s == 0 && d) { std::string r(d); free(d); return r; }
    return std::string(m);
}

/**
 * Read one source line (1-based) from a file.
 * Returns "" on any failure — callers never need to check.
 * Trims indentation; no length cap (full line for complete local analysis).
 */
static std::string read_source_line(const std::string& path, int lineno)
{
    if (path.empty() || lineno <= 0 || path[0] != '/') return "";
    FILE* f = fopen(path.c_str(), "r");
    if (!f) return "";
    char buf[4096] = {0};
    for (int n = 1; n <= lineno; n++)
        if (!fgets(buf, sizeof(buf), f)) { fclose(f); return ""; }
    fclose(f);
    std::string s(buf);
    size_t st = s.find_first_not_of(" \t");
    if (st == std::string::npos) return "";
    s = s.substr(st);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
        s.pop_back();
    return s;
}

// ============================================================================
// C++ FRAME RESOLUTION
// ============================================================================

static bool parse_backtrace_frame(const char* sym, std::string& bin, void*& addr)
{
    std::string s(sym);
    size_t paren  = s.find('(');
    size_t bracket = s.rfind('[');
    size_t bend    = s.rfind(']');
    if (paren == std::string::npos || bracket == std::string::npos || bend == std::string::npos)
        return false;
    bin = s.substr(0, paren);
    uintptr_t v = 0;
    if (sscanf(s.substr(bracket + 1, bend - bracket - 1).c_str(), "%lx", (unsigned long*)&v) != 1)
        return false;
    addr = (void*)v;
    return true;
}

static std::string parse_backtrace_symbol(const char* sym)
{
    std::string s(sym);
    size_t paren = s.find('(');
    size_t plus  = (paren != std::string::npos) ? s.find('+', paren) : std::string::npos;
    size_t close = (plus  != std::string::npos) ? s.find(')', plus)  : std::string::npos;
    if (paren == std::string::npos || plus == std::string::npos) return s;
    std::string bin     = s.substr(0, paren);
    std::string mangled = s.substr(paren + 1, plus - paren - 1);
    std::string off     = (close != std::string::npos) ? s.substr(plus, close - plus) : "";
    size_t sl = bin.find_last_of('/');
    if (sl != std::string::npos) bin = bin.substr(sl + 1);
    if (mangled.empty()) return bin + "(" + off + ")";
    return demangle(mangled.c_str()) + " " + off + " [" + bin + "]";
}

/**
 * Full information for one C++ stack frame.
 * fullpath stored so we can read the actual source line.
 */
struct FrameInfo
{
    std::string func;      // demangled function name
    std::string loc;       // "dir/file.cpp:line" (last 2 path components)
    std::string fullpath;  // absolute source path (for read_source_line)
    int         lineno = 0;
};

/**
 * Resolve address via addr2line -C -f.
 * Returns function name + file:line + full path + line number.
 */
static FrameInfo resolve_frame_info(void* addr)
{
    FrameInfo r;
    Dl_info di;
    if (!dladdr(addr, &di) || !di.dli_fname) return r;
    uintptr_t off = (uintptr_t)addr - (uintptr_t)di.dli_fbase;

    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "addr2line -C -f -e %s 0x%lx 2>/dev/null",
             di.dli_fname, (unsigned long)off);
    FILE* fp = popen(cmd, "r");
    if (!fp) return r;

    char fbuf[1024] = {0}, lbuf[1024] = {0};
    bool gf = (fgets(fbuf, sizeof(fbuf), fp) != nullptr);
    bool gl = (fgets(lbuf, sizeof(lbuf), fp) != nullptr);
    pclose(fp);

    if (gf) {
        std::string fn(fbuf);
        while (!fn.empty() && (fn.back() == '\n' || fn.back() == '\r')) fn.pop_back();
        if (!fn.empty() && fn != "??" && fn.find("??") == std::string::npos)
            r.func = fn;
    }
    if (gl) {
        std::string loc(lbuf);
        while (!loc.empty() && (loc.back() == '\n' || loc.back() == '\r')) loc.pop_back();
        if (loc.find("??") == std::string::npos && loc.find(':') != std::string::npos) {
            size_t colon = loc.rfind(':');
            std::string file = loc.substr(0, colon);
            std::string line = loc.substr(colon + 1);
            if (file.find("??") == std::string::npos && line != "?" && line != "0") {
                r.fullpath = file;
                r.lineno   = atoi(line.c_str());
                size_t s2 = file.find_last_of('/');
                if (s2 != std::string::npos && s2 > 0) {
                    size_t s1 = file.find_last_of('/', s2 - 1);
                    if (s1 != std::string::npos) file = file.substr(s1 + 1);
                }
                r.loc = file + ":" + line;
            }
        }
    }
    return r;
}

/**
 * Format one C++ frame.
 * Returns:  "func @ dir/file.cpp:line\n         |  <source line>"
 * Falls back to parse_backtrace_symbol when addr2line has no debug info.
 * No truncation — full function name for complete local analysis.
 */
static std::string format_cpp_frame(const char* sym)
{
    std::string bin;
    void* addr = nullptr;
    if (!parse_backtrace_frame(sym, bin, addr))
        return parse_backtrace_symbol(sym);
    (void)bin;

    FrameInfo fi = resolve_frame_info(addr);

    std::string func;
    if (!fi.func.empty()) {
        func = fi.func;
    } else {
        func = parse_backtrace_symbol(sym);
    }

    std::string out;
    if (!fi.loc.empty())
        out = func + " @ " + fi.loc;
    else
        out = func;

    // Show the actual source code line beneath the frame
    if (!fi.fullpath.empty() && fi.lineno > 0) {
        std::string src = read_source_line(fi.fullpath, fi.lineno);
        if (!src.empty())
            out += "\n         |  " + src;
    }
    return out;
}

// ============================================================================
// FRAME CLASSIFICATION
// ============================================================================

enum class FrameKind { CPP, PYBIND, PYTHON, LIBC };

static FrameKind classify_frame(const char* sym)
{
    const std::string s(sym);
    if (s.find("libc.so")  != std::string::npos ||
        s.find("__libc_")  != std::string::npos ||
        s.find("_start")   != std::string::npos)
        return FrameKind::LIBC;
    if (s.find("pybind11") != std::string::npos)
        return FrameKind::PYBIND;
    if (s.find("python")   != std::string::npos ||
        s.find("Python")   != std::string::npos ||
        s.find("_Py")      != std::string::npos ||
        s.find("PyEval")   != std::string::npos ||
        s.find("Py_")      != std::string::npos)
        return FrameKind::PYTHON;
    return FrameKind::CPP;
}

// ============================================================================
// PYTHON SOURCE FRAME EXTRACTION
// ============================================================================

/**
 * Walk the Python frame-object stack via dlsym-resolved Python C API.
 *
 * backtrace() only gives C-level interpreter frames (_PyEval_EvalFrameDefault…).
 * This walks the PYTHON frame stack, giving the actual .py file + function + line.
 *
 * Shows the ENTIRE Python stack — every frame, no pattern matching, no special
 * markers. Same format for all frames (function @ path:line + source line).
 */
static std::string get_python_traceback()
{
    typedef void*       (*fn_ts_t)  (void);
    typedef void*       (*fn_gf_t)  (void*);
    typedef void*       (*fn_bk_t)  (void*);
    typedef void*       (*fn_cd_t)  (void*);
    typedef int         (*fn_ln_t)  (void*);
    typedef void*       (*fn_at_t)  (void*, const char*);
    typedef const char* (*fn_u8_t)  (void*);
    typedef void        (*fn_dr_t)  (void*);
    typedef int         (*fn_eo_t)  (void);
    typedef void        (*fn_ec_t)  (void);

    static fn_ts_t _ts=nullptr;
    static fn_gf_t _gf=nullptr;
    static fn_bk_t _bk=nullptr;
    static fn_cd_t _cd=nullptr;
    static fn_ln_t _ln=nullptr;
    static fn_at_t _at=nullptr;
    static fn_u8_t _u8=nullptr;
    static fn_dr_t _dr=nullptr;
    static fn_eo_t _eo=nullptr;
    static fn_ec_t _ec=nullptr;
    static bool    _done=false;

    if (!_done) {
        _ts = (fn_ts_t)dlsym(RTLD_DEFAULT, "PyGILState_GetThisThreadState");
        _gf = (fn_gf_t)dlsym(RTLD_DEFAULT, "PyThreadState_GetFrame");
        _bk = (fn_bk_t)dlsym(RTLD_DEFAULT, "PyFrame_GetBack");
        _cd = (fn_cd_t)dlsym(RTLD_DEFAULT, "PyFrame_GetCode");
        _ln = (fn_ln_t)dlsym(RTLD_DEFAULT, "PyFrame_GetLineNumber");
        _at = (fn_at_t)dlsym(RTLD_DEFAULT, "PyObject_GetAttrString");
        _u8 = (fn_u8_t)dlsym(RTLD_DEFAULT, "PyUnicode_AsUTF8");
        _dr = (fn_dr_t)dlsym(RTLD_DEFAULT, "Py_DecRef");
        _eo = (fn_eo_t)dlsym(RTLD_DEFAULT, "PyErr_Occurred");
        _ec = (fn_ec_t)dlsym(RTLD_DEFAULT, "PyErr_Clear");
        _done = true;
    }
    if (!_ts || !_gf || !_bk || !_cd || !_ln || !_at || !_u8 || !_dr) return "";

    void* tstate = _ts();
    if (!tstate) return "";
    void* frame = _gf(tstate);
    if (!frame)  return "";

    // Collect ALL Python frames — no filtering, no pattern matching
    struct PyFrame {
        std::string fullpath, funcname, disppath;
        int lineno = 0;
    };
    std::vector<PyFrame> pyframes;

    for (int depth = 0; frame && depth < 256; depth++)
    {
        void* code   = _cd(frame);
        int   lineno = _ln(frame);
        PyFrame pf;
        pf.lineno = lineno;

        if (code) {
            void* co_file  = _at(code, "co_filename");
            void* co_qname = _at(code, "co_qualname");
            if (!co_qname && _ec) _ec();
            void* co_name  = co_qname ? co_qname : _at(code, "co_name");
            if (co_file) { const char* s=_u8(co_file); if(s) pf.fullpath=s; _dr(co_file); }
            if (co_name) { const char* s=_u8(co_name); if(s) pf.funcname=s; _dr(co_name); }
            _dr(code);
        }
        if (_eo && _eo() && _ec) _ec();

        if (!pf.fullpath.empty() && !pf.funcname.empty()) {
            // Uniform path display: keep last 3 components for all frames
            std::string dp = pf.fullpath;
            size_t pos = dp.size();
            for (int c = 0; c < 3; c++) {
                if (pos == 0) break;
                size_t f2 = dp.find_last_of('/', pos - 1);
                if (f2 == std::string::npos) break;
                pos = f2;
            }
            if (pos != std::string::npos && pos + 1 < dp.size())
                dp = dp.substr(pos + 1);
            pf.disppath = dp;
            pyframes.push_back(pf);
        }

        void* back = _bk(frame);
        _dr(frame);
        frame = back;
    }
    if (frame) _dr(frame);
    if (pyframes.empty()) return "";

    std::ostringstream oss;
    oss << "[PYTHON_STACK] Complete Python traceback (innermost first):\n";

    for (int i = 0; i < (int)pyframes.size(); i++)
    {
        const PyFrame& pf = pyframes[i];
        oss << "  #" << i << "  " << pf.funcname << " @ " << pf.disppath << ":" << pf.lineno << "\n";

        std::string src = read_source_line(pf.fullpath, pf.lineno);
        if (!src.empty())
            oss << "         |  " << src << "\n";
    }

    return oss.str();
}

// ============================================================================
// MAIN STACK TRACE CAPTURE
// ============================================================================

/**
 * Capture the COMPLETE mixed Python/C++ call chain.
 *
 * Works for both directions:
 *   Python->C++: test -> compile.py -> pybind -> forge C++ -> remove_all
 *   C++->Python: C++ -> Python callback -> shutil.rmtree
 *
 * Every frame shown: C++, pybind11, libc, Python. No filtering.
 */
static std::string capture_stack_trace(const char* operation, const char* path)
{
    std::ostringstream oss;

    time_t now = time(NULL);
    struct tm* ti = localtime(&now);
    char tbuf[32];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%d %H:%M:%S", ti);

    const char* SEP = "================================================================\n";

    void* buffer[128];   // capture up to 128 C frames (no artificial 24-frame cap)
    int nframes = backtrace(buffer, 128);

    oss << "\n" << SEP;
    oss << "[CPP_CACHE_DELETION_DETECTED] tt-metal-cache deleted"
        << " | " << tbuf
        << " | " << operation << "()"
        << " | " << path << "\n";

    if (nframes <= 0) { oss << SEP; return oss.str(); }

    char** syms = backtrace_symbols(buffer, nframes);
    if (!syms)  { oss << "(backtrace_symbols failed)\n" << SEP; return oss.str(); }

    // Classify ALL C-level frames (skip only our own interceptor frames 0-2)
    struct CFrame { int idx; FrameKind kind; };
    std::vector<CFrame> frames;
    for (int i = 3; i < nframes; i++)
        frames.push_back({i, classify_frame(syms[i])});

    bool has_cpp    = false;
    bool has_python = false;
    for (auto& f : frames) {
        if (f.kind == FrameKind::CPP)                                    has_cpp    = true;
        if (f.kind == FrameKind::PYTHON || f.kind == FrameKind::PYBIND) has_python = true;
    }
    FrameKind origin = frames.empty() ? FrameKind::CPP : frames[0].kind;

    // Chain type: Python->C++ (e.g. test -> pybind -> forge C++ -> remove_all)
    // or C++->Python (e.g. C++ calls back into Python) or pure C++/Python
    if (has_python && has_cpp) {
        if (origin == FrameKind::CPP || origin == FrameKind::PYBIND)
            oss << "[MIXED_CHAIN] Python -> C++ (Python called C++ lib to delete)\n";
        else
            oss << "[MIXED_CHAIN] C++ -> Python (C++ called Python; deletion in Python context)\n";
    } else if (has_python) {
        oss << "[PYTHON_CHAIN] Python-initiated (shutil / os -> " << operation << ")\n";
    } else {
        oss << "[CPP_CHAIN] Pure C++ deletion chain\n";
    }

    oss << "---- C++ traceback (innermost first, entire chain) ----\n";
    int  shown  = 0;
    bool py_sep = false;

    for (auto& f : frames) {
        if (f.kind == FrameKind::PYTHON) {
            // Python interpreter C frames replaced by real Python stack below
            if (!py_sep) {
                if (shown > 0)
                    oss << "       -------- [C++ / Python boundary] --------\n";
                py_sep = true;
            }
            continue;
        }

        if (f.kind == FrameKind::PYBIND) {
            if (!py_sep) {
                oss << "       -------- [pybind11 bridge] --------\n";
                py_sep = true;
            }
            std::string body = format_cpp_frame(syms[f.idx]);
            if (!body.empty()) {
                oss << "  #" << shown++ << " [pybind] " << body;
                if (body.back() != '\n') oss << "\n";
            }
            continue;
        }

        if (f.kind == FrameKind::LIBC) {
            std::string body = format_cpp_frame(syms[f.idx]);
            if (!body.empty()) {
                oss << "  #" << shown++ << " [libc]   " << body;
                if (body.back() != '\n') oss << "\n";
            }
            continue;
        }

        // CPP frame
        std::string body = format_cpp_frame(syms[f.idx]);
        if (!body.empty()) {
            oss << "  #" << shown++ << " [cpp]    " << body;
            if (body.back() != '\n') oss << "\n";
        }
    }

    // ---- Append real Python source frames ----
    if (has_python) {
        std::string py = get_python_traceback();
        if (!py.empty())
            oss << py;
        else
            oss << "[PYTHON_STACK] (Python C API unavailable)\n";
    }

    oss << SEP;
    free(syms);
    return oss.str();
}

// ============================================================================
// LOGGING
// ============================================================================

static void log_deletion(const char* op, const char* path)
{
    if (in_interceptor) return;
    in_interceptor = 1;
    pthread_mutex_lock(&log_mutex);

    // Deduplicate: same op+path within 3 seconds (prevents double-log from
    // remove_all calling both rmdir and unlink on the same root)
    time_t now = time(NULL);
    if (last_log_op == op && last_log_path == path && (now - last_log_ts) <= 3) {
        pthread_mutex_unlock(&log_mutex);
        in_interceptor = 0;
        return;
    }
    last_log_op   = op;
    last_log_path = path;
    last_log_ts   = now;

    std::string trace = capture_stack_trace(op, path);

    // Always write to stderr
    fprintf(stderr, "%s", trace.c_str());
    fflush(stderr);

    // Write to log file when CPP_CACHE_MONITOR_LOG is set
    const char* lf = getenv("CPP_CACHE_MONITOR_LOG");
    if (lf) {
        FILE* f = fopen(lf, "a");
        if (f) { fprintf(f, "%s", trace.c_str()); fclose(f); }
    }

    pthread_mutex_unlock(&log_mutex);
    in_interceptor = 0;
}

// ============================================================================
// INTERCEPTED SYSCALLS
// ============================================================================

extern "C" int unlink(const char *pathname)
{
    static orig_unlink_t orig = nullptr;
    if (!orig) orig = (orig_unlink_t)dlsym(RTLD_NEXT, "unlink");
    if (is_cache_root(pathname) && !in_interceptor) log_deletion("unlink", pathname);
    return orig(pathname);
}

extern "C" int rmdir(const char *pathname)
{
    static orig_rmdir_t orig = nullptr;
    if (!orig) orig = (orig_rmdir_t)dlsym(RTLD_NEXT, "rmdir");
    if (is_cache_root(pathname) && !in_interceptor) log_deletion("rmdir", pathname);
    return orig(pathname);
}

extern "C" int remove(const char *pathname)
{
    static orig_remove_t orig = nullptr;
    if (!orig) orig = (orig_remove_t)dlsym(RTLD_NEXT, "remove");
    if (is_cache_root(pathname) && !in_interceptor) log_deletion("remove", pathname);
    return orig(pathname);
}

extern "C" int unlinkat(int dirfd, const char *pathname, int flags)
{
    static orig_unlinkat_t orig = nullptr;
    if (!orig) orig = (orig_unlinkat_t)dlsym(RTLD_NEXT, "unlinkat");
    if (is_cache_root(pathname) && !in_interceptor) log_deletion("unlinkat", pathname);
    return orig(dirfd, pathname, flags);
}

// ============================================================================
// INIT / FINI
// ============================================================================

__attribute__((constructor))
static void init_interceptor()
{
    if (getenv("CPP_CACHE_MONITOR_VERBOSE"))
        fprintf(stderr, "[CPP_CACHE_INTERCEPTOR] tt-metal-cache deletion monitor loaded\n");
}

__attribute__((destructor))
static void cleanup_interceptor()
{
    if (getenv("CPP_CACHE_MONITOR_VERBOSE"))
        fprintf(stderr, "[CPP_CACHE_INTERCEPTOR] tt-metal-cache deletion monitor unloaded\n");
}
