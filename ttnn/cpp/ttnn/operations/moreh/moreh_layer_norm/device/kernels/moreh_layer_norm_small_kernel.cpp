// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t num_inner = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(7) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(8) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(9) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(10) == 1;

    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_0, tt::CBIndex::c_16);

    constexpr auto cb_x = tt::CBIndex::c_0;       // input
    constexpr auto cb_scaler = tt::CBIndex::c_1;  // scaler
    constexpr auto cb_eps = tt::CBIndex::c_2;     // epsilon
    constexpr auto cb_gamma = tt::CBIndex::c_3;   // gamma
    constexpr auto cb_beta = tt::CBIndex::c_4;    // beta
    constexpr auto cb_mask_h = tt::CBIndex::c_5;  // mask_h
    constexpr auto cb_mask_w = tt::CBIndex::c_6;  // mask_w

    constexpr auto cb_out = tt::CBIndex::c_16;   // output
    constexpr auto cb_mean = tt::CBIndex::c_17;  // mean
    constexpr auto cb_rstd = tt::CBIndex::c_18;  // rstd

    constexpr auto cb_ex = tt::CBIndex::c_24;          // E[x]
    constexpr auto cb_xmm = tt::CBIndex::c_25;         // x - E[x]
    constexpr auto cb_xmm2 = tt::CBIndex::c_26;        // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CBIndex::c_27;     // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CBIndex::c_28;         // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_std = tt::CBIndex::c_29;   // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CBIndex::c_30;  // p * gamm + beta
    constexpr auto cb_xsum = tt::CBIndex::c_31;        // Sum[x]

    constexpr uint32_t onetile = 1;

    ckernel::cb_wait_front(cb_scaler, onetile);  // comes from the reader
    ckernel::cb_wait_front(cb_eps, onetile);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    if (do_mask_h) {
        ckernel::cb_wait_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        ckernel::cb_wait_front(cb_mask_w, onetile);
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        /*
         * Sum[x]
         * cb_xsum
         */
        ckernel::cb_wait_front(cb_x, num_inner);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                if (w_idx == 0) {
                    ckernel:: tile_regs_acquire();
                    ckernel::cb_reserve_back(cb_xsum, onetile);

                    copy_tile_init_with_dt(cb_x);
                    ckernel:: copy_tile(cb_x, first_tile, dst0);  // input

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(cb_mask_h);
                        ckernel:: copy_tile(cb_mask_h, first_tile, dst1);  // mask_h

                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(cb_mask_w);
                        ckernel:: copy_tile(cb_mask_w, first_tile, dst1);  // mask_w

                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    ckernel:: tile_regs_commit();

                    ckernel::tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_xsum);
                    ckernel::cb_push_back(cb_xsum, onetile);
                    ckernel::tile_regs_release();
                } else {
                    ckernel:: tile_regs_acquire();
                    // I use cb_ex temporarily.
                    constexpr auto cb_tmp = cb_ex;
                    ckernel::cb_reserve_back(cb_tmp, onetile);

                    copy_tile_init_with_dt(cb_x);
                    ckernel:: copy_tile(cb_x, inner_idx + j, dst0);  // input

                    const uint32_t mask_dst = dst0 < 15 ? dst0 + 1 : 0;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(cb_mask_h);
                        ckernel:: copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h

                        mask_tile_init();
                        mask_tile(dst0, mask_dst);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile_init_with_dt(cb_mask_w);
                        ckernel:: copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w

                        mask_tile_init();
                        mask_tile(dst0, mask_dst);
                    }
                    ckernel:: tile_regs_commit();

                    ckernel::tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_tmp);
                    ckernel::cb_push_back(cb_tmp, onetile);
                    ckernel::tile_regs_release();

                    ckernel:: tile_regs_acquire();
                    ckernel::cb_wait_front(cb_tmp, onetile);
                    ckernel::cb_wait_front(cb_xsum, onetile);
                    ckernel::cb_reserve_back(cb_xsum, onetile);

                    add_tiles_init_with_dt(cb_xsum, cb_tmp);
                    add_tiles(cb_xsum, cb_tmp, first_tile, first_tile, dst0);
                    ckernel:: tile_regs_commit();

                    ckernel::tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_xsum);

                    ckernel::cb_pop_front(cb_tmp, onetile);
                    ckernel::cb_pop_front(cb_xsum, onetile);
                    ckernel::cb_push_back(cb_xsum, onetile);
                    ckernel::tile_regs_release();
                }
            }  // block_size loop
        }  // num_inner loop
        // We don't pop cb_x until we compute xmm.

        /*
         * E[x]
         * cb_ex
         */
        ckernel:: tile_regs_acquire();
        ckernel::cb_wait_front(cb_xsum, onetile);
        ckernel::cb_reserve_back(cb_ex, onetile);

        reduce_init_delta_with_dt<false>(cb_ex, cb_xsum, cb_scaler);
        reduce_tile(cb_xsum, cb_scaler, first_tile, first_tile, dst0);
        ckernel::reduce_revert_delta(cb_ex);
        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_ex);

        ckernel::cb_pop_front(cb_xsum, onetile);
        ckernel::cb_push_back(cb_ex, onetile);
        ckernel::tile_regs_release();

        ckernel::cb_wait_front(cb_ex, onetile);
        if (mean_has_value) {
            // Write on cb_mean.
            ckernel:: tile_regs_acquire();
            ckernel::cb_reserve_back(cb_mean, onetile);

            copy_tile_init_with_dt(cb_ex, is_lastdim_layernorm);
            ckernel:: copy_tile(cb_ex, first_tile, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_mean);

            ckernel::cb_push_back(cb_mean, onetile);
            ckernel::tile_regs_release();
        }
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        ckernel::cb_reserve_back(cb_xmm, num_inner);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                ckernel:: tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_short_with_dt(cb_x, cb_ex);
                    sub_tiles_bcast_cols(cb_x, cb_ex, w_idx, first_tile, j);
                } else {
                    sub_tiles_bcast_scalar_init_short_with_dt(cb_x, cb_ex);
                    sub_tiles_bcast_scalar(cb_x, cb_ex, w_idx, first_tile, j);
                }
                // mask xmm
                if (do_mask_h || do_mask_w) {
                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;
                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(cb_mask_h);
                        ckernel:: copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h

                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }
                    if (do_mask_w && (w_idx + 1) % origin_Wt == 0) {
                        copy_tile_init_with_dt(cb_mask_w);
                        ckernel:: copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w

                        mask_tile_init();
                        mask_tile(j, mask_dst);
                    }
                }
                ckernel:: tile_regs_commit();

                ckernel::tile_regs_wait();
                pack_tile_with_dt(j, cb_xmm);
                ckernel::tile_regs_release();
            }  // block_size loop
            ckernel::cb_push_back(cb_xmm, block_size);
        }  // num_inner loop
        ckernel::cb_pop_front(cb_ex, onetile);
        ckernel::cb_pop_front(cb_x, num_inner);

        /*
         * Sum[(x - E[x])^2]
         * cb_xmm2sum
         */
        ckernel::cb_wait_front(cb_xmm, num_inner);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            ckernel:: tile_regs_acquire();
            ckernel::cb_reserve_back(cb_xmm2, onetile);

            mul_tiles_init_with_dt(cb_xmm, cb_xmm);
            mul_tiles(cb_xmm, cb_xmm, inner_idx, inner_idx, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_xmm2);

            ckernel::cb_push_back(cb_xmm2, onetile);
            ckernel::tile_regs_release();
            if (inner_idx == 0) {
                ckernel:: tile_regs_acquire();
                ckernel::cb_wait_front(cb_xmm2, onetile);
                ckernel::cb_reserve_back(cb_xmm2sum, onetile);

                copy_tile_init_with_dt(cb_xmm2);
                ckernel:: copy_tile(cb_xmm2, first_tile, dst0);
                ckernel:: tile_regs_commit();

                ckernel::tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xmm2sum);

                ckernel::cb_pop_front(cb_xmm2, onetile);
                ckernel::cb_push_back(cb_xmm2sum, onetile);
                ckernel::tile_regs_release();
            } else {
                ckernel:: tile_regs_acquire();
                ckernel::cb_wait_front(cb_xmm2sum, onetile);
                ckernel::cb_wait_front(cb_xmm2, onetile);
                ckernel::cb_reserve_back(cb_xmm2sum, onetile);

                add_tiles_init_with_dt(cb_xmm2sum, cb_xmm2);
                add_tiles(cb_xmm2sum, cb_xmm2, first_tile, first_tile, dst0);
                ckernel:: tile_regs_commit();

                ckernel::tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xmm2sum);

                ckernel::cb_pop_front(cb_xmm2sum, onetile);
                ckernel::cb_pop_front(cb_xmm2, onetile);
                ckernel::cb_push_back(cb_xmm2sum, onetile);
                ckernel::tile_regs_release();
            }
        }  // num_inner loop
        // We don't pop cb_xmm here.

        /*
         * E[(x-E[x])^2 = Var[x]
         * cb_var
         */
        ckernel:: tile_regs_acquire();
        ckernel::cb_wait_front(cb_xmm2sum, onetile);
        ckernel::cb_reserve_back(cb_var, onetile);

        reduce_init_delta_with_dt<false>(cb_var, cb_xmm2sum, cb_scaler);
        reduce_tile(cb_xmm2sum, cb_scaler, first_tile, first_tile, dst0);
        ckernel::reduce_revert_delta(cb_var);
        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_var);

        ckernel::cb_pop_front(cb_xmm2sum, onetile);
        ckernel::cb_push_back(cb_var, onetile);
        ckernel::tile_regs_release();

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_std
         */
        ckernel::cb_wait_front(cb_var, onetile);
        ckernel::cb_reserve_back(cb_recip_std, onetile);

        ckernel:: tile_regs_acquire();
        add_tiles_init_with_dt(cb_var, cb_eps);
        add_tiles(cb_var, cb_eps, first_tile, first_tile, dst0);

        rsqrt_tile_init();
        rsqrt_tile(dst0);
        ckernel:: tile_regs_commit();

        ckernel::tile_regs_wait();
        pack_tile_with_dt(dst0, cb_recip_std);

        ckernel::cb_pop_front(cb_var, onetile);
        ckernel::cb_push_back(cb_recip_std, onetile);
        ckernel::tile_regs_release();

        ckernel::cb_wait_front(cb_recip_std, onetile);
        if (rstd_has_value) {
            // Write on cb_rstd.
            ckernel:: tile_regs_acquire();
            ckernel::cb_reserve_back(cb_rstd, onetile);

            copy_tile_init_with_dt(cb_recip_std, is_lastdim_layernorm);
            ckernel:: copy_tile(cb_recip_std, first_tile, dst0);
            ckernel:: tile_regs_commit();

            ckernel::tile_regs_wait();
            pack_tile_with_dt(dst0, cb_rstd);

            ckernel::cb_push_back(cb_rstd, onetile);
            ckernel::tile_regs_release();
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            ckernel::cb_reserve_back(cb_gamma_beta_or_out, block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                ckernel:: tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    mul_bcast_cols_init_short_with_dt(cb_xmm, cb_recip_std);
                    mul_tiles_bcast_cols(cb_xmm, cb_recip_std, inner_idx + j, first_tile, j);
                } else {
                    mul_tiles_bcast_scalar_init_short_with_dt(cb_xmm, cb_recip_std);
                    mul_tiles_bcast_scalar(cb_xmm, cb_recip_std, inner_idx + j, first_tile, j);
                }
                ckernel:: tile_regs_commit();

                ckernel::tile_regs_wait();
                pack_tile_with_dt(j, cb_gamma_beta_or_out);
                ckernel::tile_regs_release();
            }  // block_size loop
            ckernel::cb_push_back(cb_gamma_beta_or_out, block_size);

            // * gamma
            if (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                ckernel::cb_wait_front(cb_gamma_beta_or_out, block_size);
                ckernel::cb_wait_front(cb_gamma, block_size);
                ckernel::cb_reserve_back(cb_outg, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    ckernel:: tile_regs_acquire();
                    if (is_groupnorm) {
                        mul_tiles_bcast_scalar_init_short_with_dt(cb_gamma_beta_or_out, cb_gamma);
                        mul_tiles_bcast_scalar(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            mul_bcast_rows_init_short_with_dt(cb_gamma_beta_or_out, cb_gamma);
                            mul_tiles_bcast_rows(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        } else {
                            mul_tiles_init_with_dt(cb_gamma_beta_or_out, cb_gamma);
                            mul_tiles(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        }
                    }
                    ckernel:: tile_regs_commit();

                    ckernel::tile_regs_wait();
                    pack_tile_with_dt(j, cb_outg);
                    ckernel::tile_regs_release();
                }  // block_size loop
                ckernel::cb_pop_front(cb_gamma_beta_or_out, block_size);
                ckernel::cb_pop_front(cb_gamma, block_size);
                ckernel::cb_push_back(cb_outg, block_size);
            }  // if (gamma_has_value)

            // + beta
            if (beta_has_value) {
                ckernel::cb_wait_front(cb_gamma_beta, block_size);
                ckernel::cb_wait_front(cb_beta, block_size);
                ckernel::cb_reserve_back(cb_out, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    ckernel:: tile_regs_acquire();
                    if (is_groupnorm) {
                        add_bcast_scalar_init_short_with_dt(cb_gamma_beta, cb_beta);
                        add_tiles_bcast_scalar(cb_gamma_beta, cb_beta, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            add_bcast_rows_init_short_with_dt(cb_gamma_beta, cb_beta);
                            add_tiles_bcast_rows(cb_gamma_beta, cb_beta, j, j, j);
                        } else {
                            add_tiles_init_with_dt(cb_gamma_beta, cb_beta);
                            add_tiles(cb_gamma_beta, cb_beta, j, j, j);
                        }
                    }
                    ckernel:: tile_regs_commit();

                    ckernel::tile_regs_wait();
                    pack_tile_with_dt(j, cb_out);
                    ckernel::tile_regs_release();
                }  // block_size loop
                ckernel::cb_pop_front(cb_gamma_beta, block_size);
                ckernel::cb_pop_front(cb_beta, block_size);
                ckernel::cb_push_back(cb_out, block_size);
            }  // if (beta_has_value)
        }  // num_inner loop
        ckernel::cb_pop_front(cb_recip_std, onetile);
        ckernel::cb_pop_front(cb_xmm, num_inner);
    }  // num_rows_per_core loop
    ckernel::cb_pop_front(cb_scaler, onetile);
    ckernel::cb_pop_front(cb_eps, onetile);

    if (do_mask_h) {
        ckernel::cb_pop_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        ckernel::cb_pop_front(cb_mask_w, onetile);
    }
}  // void MAIN
}  // namespace NAMESPACE
