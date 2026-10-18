"""Platform capabilities without native extensions or dependency installation."""
import os
import sys

try:
    import resource as _resource
except ImportError:
    _resource = None


def rss_high_water_bytes(*, platform_name=None):
    """Return bytes where the resource units are known; None is unavailable."""
    name = sys.platform if platform_name is None else platform_name
    if _resource is None or not (name.startswith('linux') or name == 'darwin'):
        return None
    value = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    return int(value) * (1 if name == 'darwin' else 1024)


def sync_directory(path, *, platform_name=None):
    """POSIX directory durability; Windows has no supported directory fsync here.

    This does not suppress POSIX filesystem failures. File flush/close and
    same-parent rename remain required independently of this capability.
    """
    name = sys.platform if platform_name is None else platform_name
    if name == 'win32':
        return False
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def platform_capabilities():
    known_rss = _resource is not None and (sys.platform.startswith('linux') or sys.platform == 'darwin')
    return {'platform': sys.platform,
            'rss_high_water_source': 'resource.ru_maxrss' if known_rss else 'unavailable',
            'directory_fsync_supported': sys.platform != 'win32',
            'native_windows_validation': 'NOT_ESTABLISHED'}
