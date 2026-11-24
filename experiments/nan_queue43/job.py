"""Contained Windows child trees for the local DELAY39 queue."""
import _winapi
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import msvcrt
import os
import subprocess


CREATE_SUSPENDED = 0x00000004
BELOW_NORMAL = 0x00004000
BASIC_ACCOUNTING = 1
BASIC_PROCESS_IDS = 3
ERROR_MORE_DATA = 234


class Accounting(ctypes.Structure):
    _fields_ = [
        (key, ctypes.c_longlong) for key in
        ("total_user", "total_kernel", "period_user", "period_kernel")
    ] + [
        (key, wintypes.DWORD) for key in
        ("page_faults", "total_processes", "active_processes", "terminated_processes")
    ]


def kernel32():
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    api = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "QueryInformationJobObject": ([wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
        "GetProcessTimes": ([wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)], wintypes.BOOL),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
    }
    for name, (args, result) in api.items():
        method = getattr(dll, name)
        method.argtypes = args
        method.restype = result
    return dll


class ChildTree:
    def __init__(self, kernel, job, launcher, pid):
        self.kernel, self.job, self.launcher, self.pid = kernel, job, launcher, pid

    def accounting(self):
        value = Accounting()
        if not self.kernel.QueryInformationJobObject(self.job, BASIC_ACCOUNTING, ctypes.byref(value),
                                                     ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.active_processes, value.total_processes

    def active_pids(self):
        capacity = 16
        while capacity <= 256:
            class ProcessIDs(ctypes.Structure):
                _fields_ = [("assigned", wintypes.DWORD), ("written", wintypes.DWORD),
                            ("pids", ctypes.c_size_t * capacity)]

            ids = ProcessIDs()
            if self.kernel.QueryInformationJobObject(self.job, BASIC_PROCESS_IDS, ctypes.byref(ids),
                                                    ctypes.sizeof(ids), None):
                if ids.written != ids.assigned or ids.written > capacity:
                    raise ValueError("Windows returned an incomplete process job inventory")
                return tuple(ids.pids[:ids.written])
            error = ctypes.get_last_error()
            if error != ERROR_MORE_DATA:
                raise ctypes.WinError(error)
            capacity *= 2
        raise RuntimeError("The owned Python job exceeds the registered child limit")

    def launcher_creation(self):
        values = [wintypes.FILETIME() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.launcher, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        birth = values[0]
        return (birth.dwHighDateTime << 32) | birth.dwLowDateTime

    def launcher_code(self):
        if _winapi.WaitForSingleObject(self.launcher, 0) != _winapi.WAIT_OBJECT_0:
            raise RuntimeError("The launcher handle has not ended")
        return _winapi.GetExitCodeProcess(self.launcher)


@contextmanager
def child_tree(command, cwd, log_path):
    """Assign the suspended launcher before any redirector or model code can execute."""
    kernel = kernel32()
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    launcher = thread = None
    resumed = False
    try:
        with open(os.devnull, "rb") as stdin, log_path.open("xb") as log:
            parent = _winapi.GetCurrentProcess()
            handles = []
            try:
                for stream in (stdin, log):
                    handles.append(_winapi.DuplicateHandle(parent, msvcrt.get_osfhandle(stream.fileno()), parent,
                                                          0, True, _winapi.DUPLICATE_SAME_ACCESS))
                startup = subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESTDHANDLES
                startup.hStdInput, startup.hStdOutput = handles
                startup.hStdError = handles[1]
                startup.lpAttributeList = {"handle_list": handles}
                flags = subprocess.CREATE_NO_WINDOW | CREATE_SUSPENDED | BELOW_NORMAL
                launcher, thread, pid, _ = _winapi.CreateProcess(
                    command[0], subprocess.list2cmdline(command), None, None, True, flags, None, str(cwd), startup)
            finally:
                for handle in handles:
                    _winapi.CloseHandle(handle)
        if not kernel.AssignProcessToJobObject(job, launcher):
            raise ctypes.WinError(ctypes.get_last_error())
        tree = ChildTree(kernel, job, launcher, pid)
        if tree.active_pids() != (pid,):
            raise RuntimeError("The suspended launcher is not the sole initial member")
        result = kernel.ResumeThread(thread)
        if result == 0xffffffff:
            raise ctypes.WinError(ctypes.get_last_error())
        resumed = True
        if result != 1:
            raise RuntimeError("Unexpected suspension count on the freshly created launcher")
        yield tree
    finally:
        if launcher is not None and not resumed and _winapi.WaitForSingleObject(launcher, 0) != _winapi.WAIT_OBJECT_0:
            # The worker never reached user code. Release an owned suspended process on setup failure.
            _winapi.TerminateProcess(launcher, 4)
            _winapi.WaitForSingleObject(launcher, 5000)
        for handle in (thread, launcher, job):
            if handle is not None:
                _winapi.CloseHandle(handle)
