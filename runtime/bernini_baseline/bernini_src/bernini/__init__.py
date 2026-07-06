# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bernini Renderer: unified video / image generation and editing inference."""

__all__ = ["BerniniRendererPipeline"]


def __getattr__(name):
    # Lazy so that importing a subpackage (e.g. bernini.parallel) does not pull
    # in the full model / diffusers stack.
    if name == "BerniniRendererPipeline":
        from .pipeline import BerniniRendererPipeline

        return BerniniRendererPipeline
    raise AttributeError(f"module 'bernini' has no attribute '{name}'")
