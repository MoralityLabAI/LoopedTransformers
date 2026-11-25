#!/bin/bash
set -euo pipefail
printf '%s\n' 'parity struct_05' 'parity disrupt_05' 'mod4 struct_05' 'mod4 disrupt_05' 'mod4 struct_15' 'mod4 disrupt_15' 'mod4 random' | xargs -P4 -n2 bash -c 'python train11.py --task "$0" --arm "$1" > "$0-$1.log" 2>&1'
