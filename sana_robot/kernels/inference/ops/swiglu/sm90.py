# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import weakref

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


@dsl_user_op
def _tanh_approx(a, *, loc=None, ip=None):
    """Return the hardware tanh approximation of a float32 value."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class _HopperFusedSwiglu:
    """Persistent warp-specialized wgmma GEMM of x [M, K] with the permuted packed weight [2F, K] that returns silu(x @ Wg^T + bg) * (x @ Wu^T + bu) [M, F]: the permutation puts the gate and up accumulators of an output column in one thread's registers, where the epilogue adds the float32 bias and computes silu(g) = g / 2 + g / 2 * tanh(g / 2) on the pre-halved gate."""

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        fuse_variant: int = 0,
        k_pipe_mmas: int = 1,
        store_slots=(1, 3, 5, 7),
    ):
        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.fuse_variant = fuse_variant
        self.k_pipe_mmas_cfg = k_pipe_mmas
        self.store_slots = tuple(store_slots)
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        self.atom_layout_mnk = (2, 1, 1) if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128 else (1, 1, 1)
        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group
        self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.num_mma_threads)

    def _setup_attributes(self):
        """Derive the tiled MMA, the tile shapes, the multicast flags, the pipeline stages and the shared-memory layouts from the tensors of the call."""
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1] // 2),
        )
        self.tiled_mma_epi = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1] // 2),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], mma_inst_shape_k * mma_inst_tile_k)
        self.tile_shape_c_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1] // 2, self.tile_shape_mnk[2])
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_c_mnk, self.c_dtype, is_cooperative=is_cooperative
        )
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )
        self.a_smem_layout_staged, self.b_smem_layout_staged, self.epi_smem_layout_staged = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    def _shared_storage(self):
        """Return the shared-memory struct of one CTA: the pipeline barriers, the staged A and B tiles, the output tiles and the bias of one tile."""
        fields = {
            "mainloop_pipeline_array_ptr": cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2],
            "sA": cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes
            ],
            "sB": cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes
            ],
            "sC": cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.epi_smem_layout_staged)], self.buffer_align_bytes
            ],
            "sBias": cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.tile_shape_mnk[1]], 16],
        }
        return cute.struct(type("SharedStorage", (), {"__annotations__": fields}))

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda_driver.CUstream,
    ):
        """Set the kernel up for a [M, K], b [2F, K], c [M, F] and the float32 bias [2F], and launch it on ``stream`` over at most ``max_active_clusters`` clusters."""
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        self._setup_attributes()
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, self.a_smem_layout_staged, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]), self.cluster_shape_mn[1]
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b, self.b_smem_layout_staged, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), self.cluster_shape_mn[0]
        )
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(c, self.epi_smem_layout_staged, self.epi_tile)
        mBias_mnl = cute.make_tensor(bias.iterator, cute.make_layout((c.shape[0], b.shape[0], 1), stride=(0, 1, 0)))
        tile_sched_params, grid = self._compute_grid(
            c, self.tile_shape_c_mnk, self.cluster_shape_mn, self.swizzle_size, self.raster_along_m, max_active_clusters
        )
        self.shared_storage = self._shared_storage()
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            bias,
            mBias_mnl,
            self.tiled_mma,
            self.tiled_mma_epi,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mBias: cute.Tensor,
        mBias_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_mma_epi: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        """The device kernel: one warp loads the A and B tiles by TMA, two warp groups accumulate the two halves of a tile and fuse them, and the stores of a tile run between the first MMAs of the next."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
        if warp_idx == self.epi_store_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout
        )
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        consumer_arrive_cnt = mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, consumer_arrive_cnt)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)
        sBias = storage.sBias.get_tensor(cute.make_layout(self.tile_shape_mnk[1]))
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.tile_shape_mnk, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.tile_shape_mnk, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.tile_shape_c_mnk, (None, None, 0)), (None, None, None))
        gBias_mnl = cute.local_tile(mBias_mnl, cute.slice_(self.tile_shape_mnk, (None, None, 0)), (None, None, None))
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, a_cta_crd, a_cta_layout, cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2)
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, b_cta_crd, b_cta_layout, cute.group_modes(sB, 0, 2), cute.group_modes(gB_nkl, 0, 2)
        )
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
        thr_mma_epi = tiled_mma_epi.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
        tCsA = thr_mma.partition_A(sA)
        half_n = self.tile_shape_mnk[1] // 2
        sB0 = cute.local_tile(sB, (half_n, self.tile_shape_mnk[2]), (0, 0, None))
        sB1 = cute.local_tile(sB, (half_n, self.tile_shape_mnk[2]), (1, 0, None))
        tCsB0 = thr_mma.partition_B(sB0)
        tCsB1 = thr_mma.partition_B(sB1)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB0 = tiled_mma.make_fragment_B(tCsB0)
        tCrB1 = tiled_mma.make_fragment_B(tCsB1)
        tCgC = thr_mma_epi.partition_C(gC_mnl)
        fused_shape = tCgC.shape[:3]
        acc0 = cute.make_rmem_tensor(fused_shape, self.acc_dtype)
        acc1 = cute.make_rmem_tensor(fused_shape, self.acc_dtype)
        fused = cute.make_rmem_tensor(fused_shape, self.acc_dtype)
        fused_size = cute.size(fused)
        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mainloop_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.ab_stage)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[None, tile_coord_mnl[0], None, tile_coord_mnl[2]]
                tBgB_nkl = tBgB[None, tile_coord_mnl[1], None, tile_coord_mnl[2]]
                mainloop_producer_state.reset_count()
                for k_tile in range(k_tile_cnt):
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    tAgA_k = tAgA_mkl[None, mainloop_producer_state.count]
                    tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                    tBgB_k = tBgB_nkl[None, mainloop_producer_state.count]
                    tBsB_pipe = tBsB[None, mainloop_producer_state.index]
                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(mainloop_producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            mainloop_pipeline.producer_tail(mainloop_producer_state)
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            num_k_blocks = cute.size(tCrA, mode=[2])
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout, elem_ty_d=self.c_dtype, elem_ty_acc=self.acc_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4), self.c_dtype
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma_epi)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            tRS_rAcc = tiled_copy_r2s.retile(fused)
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)
            k_pipe_mmas = self.k_pipe_mmas_cfg
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)
            tma_store_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_threads)
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage, producer_group=tma_store_producer_group
            )

            def store_epi(
                epi_idx,
                pm,
                pn,
                pl,
                fused,
                tRS_rD,
                tRS_rD_out,
                tRS_rAcc,
                tRS_sD,
                gC_mnl,
                tma_atom_c,
                sC,
                tiled_copy_r2s,
                tma_store_pipeline,
                size_tRS_rD,
                epi_tile,
                c_dtype,
                sync_bar,
                store_warp_id,
                warp_idx,
            ):
                gC_mnl_slice = gC_mnl[None, None, pm, pn, pl]
                tCgC_for_tma_partition = cute.zipped_divide(gC_mnl_slice, epi_tile)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2), tCgC_for_tma_partition
                )
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
                for epi_v in cutlass.range_constexpr(size_tRS_rD):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]
                acc_vec = tRS_rD.load()
                tRS_rD_out.store(acc_vec.to(c_dtype))
                cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD[None, None, None, epi_idx])
                cute.arch.fence_proxy("async.shared", space="cta")
                sync_bar.arrive_and_wait()
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                if warp_idx == store_warp_id:
                    cute.copy(tma_atom_c, bSG_sD[None, epi_idx], bSG_gD[None, gmem_coord])
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                sync_bar.arrive_and_wait()

            INTERLEAVE_K = max(self.store_slots) + 2
            m_total = cutlass.Int32(cute.size(mC_mnl, mode=[0]))
            num_m_tiles = (m_total + (self.tile_shape_mnk[0] - 1)) // self.tile_shape_mnk[0]
            tail_rows = m_total - (num_m_tiles - 1) * self.tile_shape_mnk[0]
            upper_skippable = (tail_rows <= self.tile_shape_mnk[0] // 2) & (
                warp_group_idx == self.num_dma_warp_groups + 1
            )
            pend = cutlass.Boolean(False)
            pm = cutlass.Int32(0)
            pn = cutlass.Int32(0)
            pl = cutlass.Int32(0)
            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                cm = cutlass.Int32(tile_coord_mnl[0])
                cn = cutlass.Int32(tile_coord_mnl[1])
                cl = cutlass.Int32(tile_coord_mnl[2])
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                skip_mma = upper_skippable & (cm == num_m_tiles - 1)
                mma_tid = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
                bias_col = cn * self.tile_shape_mnk[1] + mma_tid
                sBias[mma_tid] = mBias[bias_col]
                acc0.fill(0.0)
                acc1.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()
                for kc in cutlass.range_constexpr(INTERLEAVE_K):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    if not skip_mma:
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                            cute.gemm(tiled_mma, acc0, tCrA[k_block_coord], tCrB0[k_block_coord], acc0)
                            cute.gemm(tiled_mma, acc1, tCrA[k_block_coord], tCrB1[k_block_coord], acc1)
                        cute.nvgpu.warpgroup.commit_group()
                    if cutlass.const_expr(kc >= 1):
                        cute.nvgpu.warpgroup.wait_group(1)
                        mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                        mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()
                    if cutlass.const_expr(kc in self.store_slots):
                        if pend:
                            store_epi(
                                self.store_slots.index(kc),
                                pm,
                                pn,
                                pl,
                                fused,
                                tRS_rD,
                                tRS_rD_out,
                                tRS_rAcc,
                                tRS_sD,
                                gC_mnl,
                                tma_atom_c,
                                sC,
                                tiled_copy_r2s,
                                tma_store_pipeline,
                                size_tRS_rD,
                                self.epi_tile,
                                self.c_dtype,
                                self.epilog_sync_barrier,
                                self.epi_store_warp_id,
                                warp_idx,
                            )
                for k_tile in range(INTERLEAVE_K, k_tile_cnt - 1):
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    if not skip_mma:
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                            cute.gemm(tiled_mma, acc0, tCrA[k_block_coord], tCrB0[k_block_coord], acc0)
                            cute.gemm(tiled_mma, acc1, tCrA[k_block_coord], tCrB1[k_block_coord], acc1)
                        cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(1)
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()
                mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                if not skip_mma:
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                        cute.gemm(tiled_mma, acc0, tCrA[k_block_coord], tCrB0[k_block_coord], acc0)
                    cute.nvgpu.warpgroup.commit_group()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
                        cute.gemm(tiled_mma, acc1, tCrA[k_block_coord], tCrB1[k_block_coord], acc1)
                    cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(2)
                mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                mainloop_consumer_release_state.advance()
                mainloop_consumer_read_state.advance()
                cute.nvgpu.warpgroup.wait_group(1)
                self.epilog_sync_barrier.arrive_and_wait()
                lane_col = 2 * (mma_tid % 4)
                if not skip_mma:
                    for i in cutlass.range_constexpr(fused_size // 2):
                        q = i // 4
                        r = i % 4
                        v0 = r % 2
                        hg = acc0[8 * q + r]
                        u = acc0[8 * q + r + 4]
                        hg = hg + sBias[16 * q + lane_col + v0]
                        u = u + sBias[16 * q + 8 + lane_col + v0]
                        t = _tanh_approx(hg)
                        s = hg * t + hg
                        fused[i] = s * u
                cute.nvgpu.warpgroup.wait_group(0)
                mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                mainloop_consumer_release_state.advance()
                if not skip_mma:
                    for i in cutlass.range_constexpr(fused_size // 2, fused_size):
                        q = i // 4
                        r = i % 4
                        v0 = r % 2
                        ql = q - fused_size // 8
                        hg = acc1[8 * ql + r]
                        u = acc1[8 * ql + r + 4]
                        hg = hg + sBias[16 * q + lane_col + v0]
                        u = u + sBias[16 * q + 8 + lane_col + v0]
                        t = _tanh_approx(hg)
                        s = hg * t + hg
                        fused[i] = s * u
                self.epilog_sync_barrier.arrive_and_wait()
                pend = cutlass.Boolean(True)
                pm = cm
                pn = cn
                pl = cl
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            if pend:
                for epi_idx in cutlass.range_constexpr(4):
                    store_epi(
                        epi_idx,
                        pm,
                        pn,
                        pl,
                        fused,
                        tRS_rD,
                        tRS_rD_out,
                        tRS_rAcc,
                        tRS_sD,
                        gC_mnl,
                        tma_atom_c,
                        sC,
                        tiled_copy_r2s,
                        tma_store_pipeline,
                        size_tRS_rD,
                        self.epi_tile,
                        self.c_dtype,
                        self.epilog_sync_barrier,
                        self.epi_store_warp_id,
                        warp_idx,
                    )
            tma_store_pipeline.producer_tail()

    @staticmethod
    def _compute_stages(tile_shape_mnk, a_dtype, b_dtype, epi_tile, c_dtype, smem_capacity, occupancy):
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage
        mbar_helpers_bytes = 1024
        ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)) // ab_bytes_per_stage
        return (ab_stage, epi_stage)

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk, element_type, is_cooperative=False, epi_tile_override=None
    ):
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk, epi_tile, a_dtype, a_layout, b_dtype, b_layout, ab_stage, c_dtype, c_layout, epi_stage
    ):
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size), a_dtype
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom, cute.append(a_smem_shape, ab_stage), order=(0, 1, 2) if a_is_k_major else (1, 0, 2)
        )
        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size), b_dtype
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom, cute.append(b_smem_shape, ab_stage), order=(0, 1, 2) if b_is_k_major else (1, 0, 2)
        )
        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(c_layout, c_dtype, c_major_mode_size), c_dtype
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )
        return (a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged)

    @staticmethod
    def _compute_grid(c, tile_shape_mnk, cluster_shape_mn, swizzle_size, raster_along_m, max_active_clusters):
        c_shape = cute.slice_(tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[0, (None, None, None)].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, swizzle_size, raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return (tile_sched_params, grid)

    @staticmethod
    def _make_tma_store_atoms_and_tensors(tensor_c, epi_smem_layout_staged, epi_tile):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), tensor_c, epi_smem_layout, epi_tile
        )
        return (tma_atom_c, tma_tensor_c)

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile, mcast_dim):
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim
        )
        return (tma_atom, tma_tensor)


_TILE_N = 256
_OUT_POOL = 4
_POOLS = {}
_COMPILED = {}


def _tier_config(m):
    """Return (cluster shape, swizzle size, raster along M) for ``m`` token rows, tuned on the H100."""
    if m > 6000:
        return ((1, 2), 4, True)
    return ((1, 2), 1, True)


def _perm_index(f2, tile_n=_TILE_N):
    """Return, for every accumulator column of the kernel, its source row in the packed [gate; up] weight."""
    ff = f2 // 2
    idx = torch.arange(f2)
    t = idx // tile_n
    cc = idx % tile_n
    grp = cc // 8
    r = cc % 8
    q = grp // 2
    o = tile_n // 2 * t + 8 * q + r
    return torch.where(grp % 2 == 0, o, ff + o)


class _CuteCtx:
    """What the kernel keeps for one packed weight and bias: their permuted copies with the gate rows halved, and the tensor wrappers of the launches."""

    def __init__(self, weight, bias):
        f2 = weight.shape[0]
        src = _perm_index(f2).to(weight.device)
        wp = weight.index_select(0, src).contiguous()
        bp = bias.index_select(0, src).float().contiguous()
        gate_mask = torch.arange(f2, device=weight.device) % _TILE_N // 8 % 2 == 0
        wp[gate_mask] *= 0.5
        bp[gate_mask] *= 0.5
        self.wp = wp
        self.bp = bp
        self.f2 = f2
        self.mB = from_dlpack(wp.unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        self.mBias = from_dlpack(bp, assumed_align=16).mark_layout_dynamic(leading_dim=0)
        self.sms = torch.cuda.get_device_properties(weight.device).multi_processor_count
        self.k = weight.shape[1]
        self.x_wrappers = {}

    def tier(self, x, m, k):
        """Return the compiled kernel and the rotating output buffers for ``m`` token rows, compiling the kernel and allocating the buffers at their first use."""
        cfg = _tier_config(m)
        ff = self.f2 // 2
        dev = x.device
        pool_key = (m, ff, dev)
        pool_state = _POOLS.get(pool_key)
        if pool_state is None:
            pool = [torch.empty(1, m, ff, device=dev, dtype=torch.bfloat16) for _ in range(_OUT_POOL)]
            wrappers = [
                from_dlpack(h.view(m, ff).unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
                for h in pool
            ]
            pool_state = [pool, wrappers, 0]
            _POOLS[pool_key] = pool_state
        compiled_key = (cfg, k, self.f2)
        compiled = _COMPILED.get(compiled_key)
        if compiled is None:
            cluster, swz, rm = cfg
            gemm = _HopperFusedSwiglu(cutlass.Float32, (128, _TILE_N), cluster, swz, rm)
            mac = self.sms // (cluster[0] * cluster[1])
            mA = from_dlpack(x.view(m, k).unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
            stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled = cute.compile(gemm, mA, self.mB, wrappers[0], self.mBias, mac, stream)
            _COMPILED[compiled_key] = compiled
        return (compiled, pool_state)

    def wrap_x(self, x, m, k):
        """Return the kernel's view of the input tensor, remembered while that tensor lives."""
        key = id(x)
        ptr = x.data_ptr()
        ent = self.x_wrappers.get(key)
        if ent is not None and ent[0] == ptr:
            return ent[1]
        mA = from_dlpack(x.view(m, k).unsqueeze(-1), assumed_align=16).mark_layout_dynamic(leading_dim=1)
        self.x_wrappers[key] = (ptr, mA)
        weakref.finalize(x, self.x_wrappers.pop, key, None)
        return mA


_PREPPED = {}


def _prepare_cute(weight, bias):
    """Return the ``_CuteCtx`` of a packed weight tensor, built at its first call and dropped with the tensor."""
    key = id(weight)
    guard = (weight.data_ptr(), bias.data_ptr())
    entry = _PREPPED.get(key)
    if entry is not None and entry[0] == guard:
        return entry[1]
    ctx = _CuteCtx(weight, bias)
    _PREPPED[key] = (guard, ctx)
    weakref.finalize(weight, _PREPPED.pop, key, None)
    return ctx


def _run_cute(x, weight, bias):
    """Launch the kernel on the current stream into the next of the rotating output buffers of this shape and return that buffer."""
    ctx = _prepare_cute(weight, bias)
    b, m, k = x.shape
    compiled, pool_state = ctx.tier(x, m, k)
    pool, wrappers, idx = pool_state
    mA = ctx.wrap_x(x, m, k)
    h = pool[idx]
    mC = wrappers[idx]
    pool_state[2] = (idx + 1) % _OUT_POOL
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(mA, ctx.mB, mC, ctx.mBias, stream)
    return h


def supports(in_features: int, hidden_features: int) -> bool:
    """Return whether the kernel serves a SwiGLU of these sizes: whole 256-column tiles of the packed [gate; up] output and more than 576 input features in multiples of 8."""
    return 2 * hidden_features % _TILE_N == 0 and in_features > 576 and in_features % 8 == 0


def swiglu_gate_up(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Return ``silu(x @ Wg^T + bg) * (x @ Wu^T + bu)`` [1, M, F] in bf16 for contiguous bf16 ``x`` [1, M, K], the packed weight [2F, K] (gate rows, then up rows) and bias [2F], within bf16 rounding of the separate layers. The result is one of ``_OUT_POOL`` buffers that the calls of this shape write in turn, so the caller consumes it before ``_OUT_POOL`` more calls."""
    return _run_cute(x, weight, bias)
