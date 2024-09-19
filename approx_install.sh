#!/bin/bash
#    Copyright 2023 Cerebras Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# symlink gns_utils.py and hook.py into examples/nanoGPT
ln -s $PWD/gns_utils.py examples/nanoGPT/gns_utils.py
ln -s $PWD/hook.py examples/nanoGPT/hook.py
patch -p1 examples/nanoGPT/train.py approx.train.patch
