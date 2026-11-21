"""CI-only registration check; never changes the frozen Windows runtime check."""
import hashlib
import json
from pathlib import Path

REGISTRATION = 'b9a62f0eedaedc3eca4667412873d5c7151b6a4fb1f98b41bf1bc6978649043f'
WINDOWS_CHECKOUT_PATHS = {'mechinterp/dark_fiber_gyms/arg03_nanbeige.py',
                          'mechinterp/dark_fiber_gyms/q4_3050.py'}


def registered_mode(path, raw, expected, *,dependency=False):
    if hashlib.sha256(raw).hexdigest() == expected:
        return 'exact'
    if dependency and path in WINDOWS_CHECKOUT_PATHS and b'\r' not in raw and b'\n' in raw:
        if hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest() == expected:
            return 'Git LF blob / registered Windows CRLF checkout'
    return None


def main():
    root = Path('experiments/nan_comp14'); manifest = root / 'SOURCE_MANIFEST.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == REGISTRATION
    seal = json.loads(manifest.read_bytes()); checked = 0; converted = []
    for field in ('critical_files_sha256', 'host_files_sha256', 'dependencies_sha256'):
        for name, expected in seal[field].items():
            dependency = field == 'dependencies_sha256'
            path = Path(name) if dependency else root / name
            mode = registered_mode(name, path.read_bytes(), expected, dependency=dependency)
            assert mode, str(path)
            checked += 1
            if mode != 'exact':
                converted.append(name)
    print(dict(registered_files_checked=checked, explicit_checkout_conversion_paths=converted,
               frozen_runtime_source_changed=False, pretrained_model_loaded=False))


if __name__ == '__main__':
    main()
