# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility helpers for running TriGS on non-CUDA backends (Apple MPS, CPU).

Added by the macOS port. CUDA code paths are kept unchanged; these helpers only
pick a backend and dispatch backend-specific calls.
"""
