#!/bin/bash
# symlink gns_utils.py and hook.py into examples/nanoGPT
ln -s $PWD/gns_utils.py examples/nanoGPT/gns_utils.py
ln -s $PWD/hook.py examples/nanoGPT/hook.py
