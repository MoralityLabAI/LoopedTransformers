//! Sparse u64-bitset and continuous preference trellis kernels for LBF-18.
//! Zero external crate dependencies (`std` only) for reproducible <3s offline builds.

const CE: usize = 20;
const CB: usize = 4;
const A6: usize = 6;
const A6_SQ: usize = 36;

#[inline(always)]
fn erf_exact_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let z = x.abs();
    if z >= 6.0 {
        return sign;
    }
    // Exact Maclaurin series in exp(-z^2) form: erf(z) = (2/sqrt(pi)) * exp(-z^2) * z * sum_{n>=0} (2*z^2)^n / (2n+1)!!
    // For z < 6.0, 64 terms converge to < 1e-16 (0 ULP in f32).
    let two_z2 = 2.0 * z * z;
    let mut term = z;
    let mut sum = term;
    let mut n = 1usize;
    while n <= 64 {
        term *= two_z2 / ((2 * n + 1) as f64);
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
        n += 1;
    }
    let two_over_sqrt_pi = 1.128_379_167_095_512_6_f64;
    let val = (two_over_sqrt_pi * (-z * z).exp() * sum).min(1.0);
    sign * val
}

#[inline(always)]
fn gelu_f32(x: f32) -> f32 {
    let xd = x as f64;
    if xd <= -9.0 {
        return 0.0;
    }
    if xd >= 9.0 {
        return x;
    }
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    (0.5 * xd * (1.0 + erf_exact_f64(xd * inv_sqrt2))) as f32
}

#[inline(always)]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        (x as f64).exp() as f32
    } else {
        (1.0 + (x as f64).exp()).ln() as f32
    }
}

#[inline(always)]
fn sigmoid_f32(x: f32) -> f32 {
    (1.0 / (1.0 + (-(x as f64)).exp())) as f32
}

#[inline(always)]
fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Exact u64 forward-backward Viterbi trellis filter for arbitrary alphabet size `alpha <= 8`.
#[inline]
fn trellis_filter_u64(arcs: &mut [u64], alpha: usize) {
    let lm1 = arcs.len();
    if lm1 == 0 {
        return;
    }
    let row_mask: u64 = (1u64 << alpha) - 1;
    let mut fwd = [0u8; 65];
    let mut bwd = [0u8; 65];

    let mut f0 = 0u8;
    for x in 0..alpha {
        if ((arcs[0] >> (x * alpha)) & row_mask) != 0 {
            f0 |= 1u8 << x;
        }
    }
    fwd[0] = f0;
    for p in 0..lm1 {
        let cur = fwd[p];
        let ap = arcs[p];
        let mut nxt = 0u8;
        for x in 0..alpha {
            if ((cur >> x) & 1) != 0 {
                nxt |= ((ap >> (x * alpha)) & row_mask) as u8;
            }
        }
        fwd[p + 1] = nxt;
    }

    let mut bl = 0u8;
    let alast = arcs[lm1 - 1];
    for x in 0..alpha {
        bl |= ((alast >> (x * alpha)) & row_mask) as u8;
    }
    bwd[lm1] = bl;
    for p in (0..lm1).rev() {
        let nxt = bwd[p + 1] as u64;
        let ap = arcs[p];
        let mut cur = 0u8;
        for x in 0..alpha {
            if ((ap >> (x * alpha)) & nxt) != 0 {
                cur |= 1u8 << x;
            }
        }
        bwd[p] = cur;
    }

    for p in 0..lm1 {
        let fp_m = fwd[p];
        let bp_m = bwd[p + 1] as u64;
        let mut keep = 0u64;
        for x in 0..alpha {
            if ((fp_m >> x) & 1) != 0 {
                keep |= bp_m << (x * alpha);
            }
        }
        arcs[p] &= keep;
    }
}

/// Kernel 1: Batch u64-bitset symbolic `prov_{Fp} x adj2` abstract interpreter.
/// Supports `alpha <= 8` (`alpha^2 <= 64`), `word_len <= 64`, arbitrary rule widths `>= 2`,
/// `fp <= 32`, `k_max <= 16`.
/// `guard_mode`: `0 = legacy`, `1 = fable_len_guard` (`fp >= ell_max` & no longer Pass-1 redex at `p`),
///               `2 = pr348_mask_guard` (`D_f subseteq [p, p+ll)`), `3 = none`.
/// Writes `out_bitsets` of shape `(n_words, k_max + 1, fp + 1, word_len - 1)` where fiber slot `0`
/// is the root word `base_a2` and fiber slots `1..=fp` are the `fp` provenance fibers.
/// Returns total executed rule-window checks.
#[no_mangle]
pub unsafe extern "C" fn lbf18_sym_prov_adj2_batch(
    words_ptr: *const u8,      // (n_words, word_len)
    n_words: usize,
    word_len: usize,
    alpha: usize,              // <= 8
    rules_lhs_ptr: *const u8,  // (n_ors, max_rlen)
    rules_rhs_ptr: *const u8,  // (n_ors, max_rlen)
    rules_len_ptr: *const u8,  // (n_ors,)
    n_ors: usize,
    max_rlen: usize,
    fp: usize,
    k_max: usize,
    guard_mode: usize,
    out_bitsets: *mut u64,     // (n_words, k_max + 1, fp + 1, lm1)
) -> u64 {
    if word_len < 2 || alpha > 8 || word_len > 64 || fp == 0 || fp > 32 {
        return 0;
    }
    let lm1 = word_len - 1;
    let words = std::slice::from_raw_parts(words_ptr, n_words * word_len);
    let r_lhs = std::slice::from_raw_parts(rules_lhs_ptr, n_ors * max_rlen);
    let r_rhs = std::slice::from_raw_parts(rules_rhs_ptr, n_ors * max_rlen);
    let r_len = std::slice::from_raw_parts(rules_len_ptr, n_ors);
    let out_stride_k = (fp + 1) * lm1;
    let out_stride_w = (k_max + 1) * out_stride_k;
    let out = std::slice::from_raw_parts_mut(out_bitsets, n_words * out_stride_w);
    out.fill(0);

    let mut actual_max_rlen = 2usize;
    for ri in 0..n_ors {
        let ll = r_len[ri] as usize;
        if ll > actual_max_rlen {
            actual_max_rlen = ll;
        }
    }

    let mut total_checks = 0u64;

    for wi in 0..n_words {
        let w0 = &words[wi * word_len..(wi + 1) * word_len];
        let mut base = [0u64; 64];
        for p in 0..lm1 {
            let bit = (w0[p] as usize) * alpha + (w0[p + 1] as usize);
            base[p] = 1u64 << bit;
        }
        let w_off = wi * out_stride_w;
        for p in 0..lm1 {
            out[w_off + p] = base[p];
        }
        if k_max == 0 {
            continue;
        }

        let mut fibers = [[0u64; 64]; 32];
        let mut active = [false; 32];
        let mut w1_buf = [0u8; 64];
        let mut max_origin_len = [0usize; 64];
        let mut changed_mask = [0u64; 32];

        for ri in 0..n_ors {
            let ll = r_len[ri] as usize;
            if ll < 2 || ll > word_len {
                continue;
            }
            let lhs = &r_lhs[ri * max_rlen..ri * max_rlen + ll];
            let rhs = &r_rhs[ri * max_rlen..ri * max_rlen + ll];
            for p in 0..=(word_len - ll) {
                total_checks += 1;
                if &w0[p..p + ll] == lhs && lhs != rhs {
                    w1_buf[..word_len].copy_from_slice(w0);
                    w1_buf[p..p + ll].copy_from_slice(rhs);
                    let f = p % fp;
                    for q in 0..lm1 {
                        let bit = (w1_buf[q] as usize) * alpha + (w1_buf[q + 1] as usize);
                        fibers[f][q] |= 1u64 << bit;
                    }
                    active[f] = true;
                    if ll > max_origin_len[p] {
                        max_origin_len[p] = ll;
                    }
                    for q in 0..ll {
                        if rhs[q] != w0[p + q] {
                            changed_mask[f] |= 1u64 << (p + q);
                        }
                    }
                }
            }
        }

        for f in 0..fp {
            if active[f] {
                trellis_filter_u64(&mut fibers[f][..lm1], alpha);
            }
        }
        let k1_off = w_off + out_stride_k;
        for p in 0..lm1 {
            out[k1_off + p] = base[p];
        }
        for f in 0..fp {
            let f_off = k1_off + (f + 1) * lm1;
            for p in 0..lm1 {
                out[f_off + p] = fibers[f][p];
            }
        }

        for step_idx in 1..k_max {
            for f in 0..fp {
                if !active[f] {
                    continue;
                }
                let mut ta = [0u64; 64];
                ta[..lm1].copy_from_slice(&fibers[f][..lm1]);
                trellis_filter_u64(&mut ta[..lm1], alpha);
                let mut nxt = [0u64; 64];
                nxt[..lm1].copy_from_slice(&ta[..lm1]);

                for ri in 0..n_ors {
                    let ll = r_len[ri] as usize;
                    if ll < 2 || ll > word_len {
                        continue;
                    }
                    let lhs = &r_lhs[ri * max_rlen..ri * max_rlen + ll];
                    let rhs = &r_rhs[ri * max_rlen..ri * max_rlen + ll];
                    for p in 0..=(word_len - ll) {
                        total_checks += 1;
                        if step_idx == 1 && (p % fp == f) && &w0[p..p + ll] == rhs {
                            let suppress = match guard_mode {
                                0 => true,
                                1 => (fp >= actual_max_rlen) && (max_origin_len[p] <= ll),
                                2 => {
                                    let win_m = ((1u64 << ll) - 1) << p;
                                    (changed_mask[f] & !win_m) == 0
                                }
                                _ => false,
                            };
                            if suppress {
                                continue;
                            }
                        }
                        let mut matched = true;
                        for q in 0..(ll - 1) {
                            let bit = (lhs[q] as usize) * alpha + (lhs[q + 1] as usize);
                            if ((ta[p + q] >> bit) & 1) == 0 {
                                matched = false;
                                break;
                            }
                        }
                        if !matched {
                            continue;
                        }
                        for q in 0..(ll - 1) {
                            let ybit = (rhs[q] as usize) * alpha + (rhs[q + 1] as usize);
                            nxt[p + q] |= 1u64 << ybit;
                        }
                        if p > 0 {
                            let x0 = lhs[0] as usize;
                            let y0 = rhs[0] as usize;
                            let prev_a = ta[p - 1];
                            for s in 0..alpha {
                                if ((prev_a >> (s * alpha + x0)) & 1) != 0 {
                                    nxt[p - 1] |= 1u64 << (s * alpha + y0);
                                }
                            }
                        }
                        let re = p + ll - 1;
                        if re < lm1 {
                            let xlast = lhs[ll - 1] as usize;
                            let ylast = rhs[ll - 1] as usize;
                            let right_a = ta[re];
                            for s in 0..alpha {
                                if ((right_a >> (xlast * alpha + s)) & 1) != 0 {
                                    nxt[re] |= 1u64 << (ylast * alpha + s);
                                }
                            }
                        }
                    }
                }
                trellis_filter_u64(&mut nxt[..lm1], alpha);
                fibers[f][..lm1].copy_from_slice(&nxt[..lm1]);
            }

            let ks_off = w_off + (step_idx + 1) * out_stride_k;
            for p in 0..lm1 {
                out[ks_off + p] = base[p];
            }
            for f in 0..fp {
                let f_off = ks_off + (f + 1) * lm1;
                for p in 0..lm1 {
                    out[f_off + p] = fibers[f][p];
                }
            }
        }
    }

    total_checks
}

/// Weight bundle struct passed from Python (`ctypes.Structure`) for `RelationalProvenanceFrescoNet`
/// (`ce = 20, cb = 4, A = 6`).
#[repr(C)]
pub struct RelProvWeightsPtrs {
    pub env_init_w: *const f32,      // (20, 3)
    pub env_init_b: *const f32,      // (20,)
    pub env_gate_wpos: *const f32,   // (20, 20)  -- precomputed softplus(proj.w)/sqrt(20)
    pub env_gate_gw: *const f32,     // (20, 40)
    pub env_gate_gb: *const f32,     // (20,)
    pub pos_r2_w1: *const f32,       // (40, 60)
    pub pos_r2_b1: *const f32,       // (40,)
    pub pos_r2_w2: *const f32,       // (20, 40)
    pub pos_r2_b2: *const f32,       // (20,)
    pub pos_r3_w1: *const f32,       // (40, 80)
    pub pos_r3_b1: *const f32,       // (40,)
    pub pos_r3_w2: *const f32,       // (20, 40)
    pub pos_r3_b2: *const f32,       // (20,)
    pub pos_res_w1: *const f32,      // (40, 20)
    pub pos_res_b1: *const f32,      // (40,)
    pub pos_res_w2: *const f32,      // (20, 40)
    pub pos_res_b2: *const f32,      // (20,)
    pub bi_gate_wpos: *const f32,    // (4, 4)    -- precomputed softplus(proj.w)/sqrt(4)
    pub bi_gate_gw: *const f32,      // (4, 8)
    pub bi_gate_gb: *const f32,      // (4,)
    pub bi_r2_w1: *const f32,        // (8, 4)
    pub bi_r2_b1: *const f32,        // (8,)
    pub bi_r2_w2: *const f32,        // (4, 8)
    pub bi_r2_b2: *const f32,        // (4,)
    pub bi_r3_w1: *const f32,        // (8, 12)
    pub bi_r3_b1: *const f32,        // (8,)
    pub bi_r3_w2: *const f32,        // (4, 8)
    pub bi_r3_b2: *const f32,        // (4,)
}

#[inline]
fn linear_gelu_linear_softplus(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    in_d: usize,
    hid_d: usize,
    out_d: usize,
    out: &mut [f32],
) {
    let mut hid = [0.0f32; 48];
    for h in 0..hid_d {
        let row = &w1[h * in_d..(h + 1) * in_d];
        let mut acc = b1[h];
        for i in 0..in_d {
            acc += row[i] * x[i];
        }
        hid[h] = gelu_f32(acc);
    }
    for o in 0..out_d {
        let row = &w2[o * hid_d..(o + 1) * hid_d];
        let mut acc = b2[o];
        for h in 0..hid_d {
            acc += row[h] * hid[h];
        }
        out[o] = softplus_f32(acc);
    }
}

/// Continuous soft Viterbi trellis arc filter (`rho = e / (e + 1e-4)`) accelerated by `u64` bitset.
/// Zero heap allocations inside the inner loop (`rho`, `fwd`, `bwd` passed as reusable scratch slices).
#[inline]
fn trellis_arc_filter_soft_f32(
    bank: &mut [f32],     // (lm1 * 36 * CB)
    bits: &mut [u64],     // (lm1,)
    lm1: usize,
    rho: &mut [f32],      // scratch >= lm1 * 36 * CB
    fwd: &mut [f32],      // scratch >= lm1 * 6 * CB
    bwd: &mut [f32],      // scratch >= lm1 * 6 * CB
) {
    // First prune unreachable arcs with exact O(lm1) u64 bitwise trellis pass!
    trellis_filter_u64(&mut bits[..lm1], A6);

    rho[..lm1 * A6_SQ * CB].fill(0.0);
    for p in 0..lm1 {
        let mask = bits[p];
        if mask == 0 {
            let base = p * A6_SQ * CB;
            bank[base..base + A6_SQ * CB].fill(0.0);
            continue;
        }
        for ab in 0..A6_SQ {
            let idx = (p * A6_SQ + ab) * CB;
            if ((mask >> ab) & 1) == 0 {
                for c in 0..CB {
                    bank[idx + c] = 0.0;
                }
            } else {
                for c in 0..CB {
                    let v = bank[idx + c];
                    if v > 0.0 {
                        rho[idx + c] = v / (v + 1e-4);
                    }
                }
            }
        }
    }

    fwd[..lm1 * A6 * CB].fill(0.0);
    let mut cur_f = [0.0f32; A6 * CB];
    if bits[0] != 0 {
        for a in 0..A6 {
            for b in 0..A6 {
                if ((bits[0] >> (a * A6 + b)) & 1) != 0 {
                    let r_off = (a * A6 + b) * CB;
                    for c in 0..CB {
                        let v = rho[r_off + c];
                        if v > cur_f[a * CB + c] {
                            cur_f[a * CB + c] = v;
                        }
                    }
                }
            }
        }
    }
    for p in 0..lm1 {
        fwd[p * A6 * CB..(p + 1) * A6 * CB].copy_from_slice(&cur_f);
        if p + 1 < lm1 {
            let mut nxt_f = [0.0f32; A6 * CB];
            let mask = bits[p];
            if mask != 0 {
                for a in 0..A6 {
                    for b in 0..A6 {
                        if ((mask >> (a * A6 + b)) & 1) != 0 {
                            let r_off = (p * A6_SQ + a * A6 + b) * CB;
                            for c in 0..CB {
                                let cand = cur_f[a * CB + c] * rho[r_off + c];
                                if cand > nxt_f[b * CB + c] {
                                    nxt_f[b * CB + c] = cand;
                                }
                            }
                        }
                    }
                }
            }
            cur_f = nxt_f;
        }
    }

    bwd[..lm1 * A6 * CB].fill(0.0);
    let mut cur_b = [0.0f32; A6 * CB];
    if bits[lm1 - 1] != 0 {
        for a in 0..A6 {
            for b in 0..A6 {
                if ((bits[lm1 - 1] >> (a * A6 + b)) & 1) != 0 {
                    let r_off = ((lm1 - 1) * A6_SQ + a * A6 + b) * CB;
                    for c in 0..CB {
                        let v = rho[r_off + c];
                        if v > cur_b[b * CB + c] {
                            cur_b[b * CB + c] = v;
                        }
                    }
                }
            }
        }
    }
    for p in (0..lm1).rev() {
        bwd[p * A6 * CB..(p + 1) * A6 * CB].copy_from_slice(&cur_b);
        if p > 0 {
            let mut prev_b = [0.0f32; A6 * CB];
            let mask = bits[p];
            if mask != 0 {
                for a in 0..A6 {
                    for b in 0..A6 {
                        if ((mask >> (a * A6 + b)) & 1) != 0 {
                            let r_off = (p * A6_SQ + a * A6 + b) * CB;
                            for c in 0..CB {
                                let cand = rho[r_off + c] * cur_b[b * CB + c];
                                if cand > prev_b[a * CB + c] {
                                    prev_b[a * CB + c] = cand;
                                }
                            }
                        }
                    }
                }
            }
            cur_b = prev_b;
        }
    }

    for p in 0..lm1 {
        let mask = bits[p];
        if mask == 0 {
            continue;
        }
        for a in 0..A6 {
            for b in 0..A6 {
                if ((mask >> (a * A6 + b)) & 1) != 0 {
                    let idx = (p * A6_SQ + a * A6 + b) * CB;
                    for c in 0..CB {
                        let m = fwd[(p * A6 + a) * CB + c] * bwd[(p * A6 + b) * CB + c];
                        bank[idx + c] *= m;
                    }
                }
            }
        }
    }
}

/// Kernel 2: Fused sparse-bitset-pruned continuous multi-pass word-envelope evaluator
/// for `RelationalProvenanceFrescoNet` (`rel_prov_opened`, `rel_prov_standard`,
/// `adj2_noprov_opened`, `param_nonrecurrent`), with decoupled exact Boolean `u64` support!
///
/// `env_mode`: `0 = normal`, `1 = env_zero`, `2 = env_roll`
/// `nonrecurrent`: `1` if `param_nonrecurrent` (`g_all = 0`), `0` otherwise
/// `guard_mode`: `0 = legacy`, `1 = fable_len_guard`, `2 = pr348_mask_guard`, `3 = none`
///
/// Writes:
/// - `out_q`:         `(n_words, passes, L, 6, 20)`
/// - `out_b`:         `(n_words, passes, L, 6, 20)`
/// - `out_bi`:        `(n_words, passes, fp, L - 1, 36, 4)`
/// - `out_supp_bits`: `(n_words, passes, fp + 1, L - 1)` exact pure-Boolean `u64` support (slot 0 = root)
/// - `out_min_pos`:   `(n_words, passes)` minimum float32 channel-max across exact-Boolean-supported cells
#[no_mangle]
pub unsafe extern "C" fn lbf18_rel_prov_word_envelopes_f32(
    words_ptr: *const u8,       // (n_words, L)
    n_words: usize,
    word_len: usize,
    r2_ptr: *const u8,          // (n_r2, 4): [x0, x1, y0, y1]
    n_r2: usize,
    r3_ptr: *const u8,          // (n_r3, 6): [x0, x1, x2, y0, y1, y2]
    n_r3: usize,
    fp: usize,                  // 4 or 1
    passes: usize,              // 1..=16
    env_mode: usize,            // 0 = normal, 1 = env_zero, 2 = env_roll
    nonrecurrent: usize,        // 0 or 1
    guard_mode: usize,          // 0 = legacy, 1 = fable_len_guard, 2 = pr348_mask_guard, 3 = none
    weights_ptr: *const RelProvWeightsPtrs,
    out_q_ptr: *mut f32,        // (n_words, passes, L, 6, 20)
    out_b_ptr: *mut f32,        // (n_words, passes, L, 6, 20)
    out_bi_ptr: *mut f32,       // (n_words, passes, fp, L-1, 36, 4)
    out_supp_bits_ptr: *mut u64,// (n_words, passes, fp + 1, L-1)
    out_min_pos_ptr: *mut f32,  // (n_words, passes)
) -> u64 {
    if n_words == 0 || word_len < 2 || word_len > 64 || (fp != 1 && fp != 4) || passes == 0 {
        return 0;
    }
    let l = word_len;
    let lm1 = l - 1;
    let use_prov = fp > 1;
    let actual_max_rlen = if n_r3 > 0 { 3usize } else { 2usize };
    let wt = &*weights_ptr;

    let words = std::slice::from_raw_parts(words_ptr, n_words * l);
    let r2 = if n_r2 > 0 { std::slice::from_raw_parts(r2_ptr, n_r2 * 4) } else { &[][..] };
    let r3 = if n_r3 > 0 { std::slice::from_raw_parts(r3_ptr, n_r3 * 6) } else { &[][..] };

    let env_init_w = std::slice::from_raw_parts(wt.env_init_w, CE * 3);
    let env_init_b = std::slice::from_raw_parts(wt.env_init_b, CE);
    let pos_r2_w1 = std::slice::from_raw_parts(wt.pos_r2_w1, 40 * 60);
    let pos_r2_b1 = std::slice::from_raw_parts(wt.pos_r2_b1, 40);
    let pos_r2_w2 = std::slice::from_raw_parts(wt.pos_r2_w2, 20 * 40);
    let pos_r2_b2 = std::slice::from_raw_parts(wt.pos_r2_b2, 20);
    let pos_r3_w1 = std::slice::from_raw_parts(wt.pos_r3_w1, 40 * 80);
    let pos_r3_b1 = std::slice::from_raw_parts(wt.pos_r3_b1, 40);
    let pos_r3_w2 = std::slice::from_raw_parts(wt.pos_r3_w2, 20 * 40);
    let pos_r3_b2 = std::slice::from_raw_parts(wt.pos_r3_b2, 20);
    let pos_res_w1 = std::slice::from_raw_parts(wt.pos_res_w1, 40 * 20);
    let pos_res_b1 = std::slice::from_raw_parts(wt.pos_res_b1, 40);
    let pos_res_w2 = std::slice::from_raw_parts(wt.pos_res_w2, 20 * 40);
    let pos_res_b2 = std::slice::from_raw_parts(wt.pos_res_b2, 20);

    let bi_r2_w1 = std::slice::from_raw_parts(wt.bi_r2_w1, 8 * 4);
    let bi_r2_b1 = std::slice::from_raw_parts(wt.bi_r2_b1, 8);
    let bi_r2_w2 = std::slice::from_raw_parts(wt.bi_r2_w2, 4 * 8);
    let bi_r2_b2 = std::slice::from_raw_parts(wt.bi_r2_b2, 4);
    let bi_r3_w1 = std::slice::from_raw_parts(wt.bi_r3_w1, 8 * 12);
    let bi_r3_b1 = std::slice::from_raw_parts(wt.bi_r3_b1, 8);
    let bi_r3_w2 = std::slice::from_raw_parts(wt.bi_r3_w2, 4 * 8);
    let bi_r3_b2 = std::slice::from_raw_parts(wt.bi_r3_b2, 4);

    let (env_gate_wpos, env_gate_gw, env_gate_gb, bi_gate_wpos, bi_gate_gw, bi_gate_gb) =
        if nonrecurrent == 0 {
            (
                std::slice::from_raw_parts(wt.env_gate_wpos, CE * CE),
                std::slice::from_raw_parts(wt.env_gate_gw, CE * 40),
                std::slice::from_raw_parts(wt.env_gate_gb, CE),
                std::slice::from_raw_parts(wt.bi_gate_wpos, CB * CB),
                std::slice::from_raw_parts(wt.bi_gate_gw, CB * 8),
                std::slice::from_raw_parts(wt.bi_gate_gb, CB),
            )
        } else {
            (&[][..], &[][..], &[][..], &[][..], &[][..], &[][..])
        };

    let pos_pass_sz = l * A6 * CE;
    let bi_pass_sz = fp * lm1 * A6_SQ * CB;
    let supp_pass_sz = (fp + 1) * lm1;
    let out_q = std::slice::from_raw_parts_mut(out_q_ptr, n_words * passes * pos_pass_sz);
    let out_b = std::slice::from_raw_parts_mut(out_b_ptr, n_words * passes * pos_pass_sz);
    let out_bi = std::slice::from_raw_parts_mut(out_bi_ptr, n_words * passes * bi_pass_sz);
    let out_supp = std::slice::from_raw_parts_mut(out_supp_bits_ptr, n_words * passes * supp_pass_sz);
    let out_min_pos = std::slice::from_raw_parts_mut(out_min_pos_ptr, n_words * passes);
    out_supp.fill(0);
    out_min_pos.fill(1.0);

    // Precompute all 4 `h_res` trajectories across passes `k = 0..passes` ONCE per batch!
    // (Since `h_{k+1} = tanh(h_k + 0.25 * res_ff(h_k))` depends only on the 4 initial slot types!)
    let mut hk_tab = [[[0.0f32; CE]; 4]; 16]; // [pass_k][slot_kind: 0=zero, 1=mid, 2=left, 3=right]
    for c in 0..CE {
        let b0 = env_init_b[c];
        let w0 = env_init_w[c * 3];
        let w1 = env_init_w[c * 3 + 1];
        let w2 = env_init_w[c * 3 + 2];
        hk_tab[0][0][c] = b0.tanh();
        hk_tab[0][1][c] = (b0 + w0).tanh();
        hk_tab[0][2][c] = (b0 + w0 + w1).tanh();
        hk_tab[0][3][c] = (b0 + w0 + w2).tanh();
    }
    let max_k = passes.min(16);
    let mut res_hid_init = [0.0f32; 40];
    for k in 0..(max_k - 1) {
        for kind in 0..4 {
            let hx = hk_tab[k][kind];
            for h in 0..40 {
                let row = &pos_res_w1[h * CE..(h + 1) * CE];
                let mut acc = pos_res_b1[h];
                for i in 0..CE {
                    acc += row[i] * hx[i];
                }
                res_hid_init[h] = gelu_f32(acc);
            }
            for c in 0..CE {
                let row = &pos_res_w2[c * 40..(c + 1) * 40];
                let mut acc = pos_res_b2[c];
                for h in 0..40 {
                    acc += row[h] * res_hid_init[h];
                }
                hk_tab[k + 1][kind][c] = (hx[c] + 0.25 * acc).tanh();
            }
        }
    }

    let win_per_word = (n_r2 * lm1 + n_r3 * l.saturating_sub(2)) as u64;
    let mut rule_checks = 0u64;

    let mut b_cur = vec![0.0f32; pos_pass_sz];
    let mut m_pos = vec![0.0f32; pos_pass_sz];
    let mut q_pos = vec![0.0f32; pos_pass_sz];
    let mut rew_pos = vec![0.0f32; pos_pass_sz];

    let mut bi_cur = vec![0.0f32; bi_pass_sz];
    let mut m_bi = vec![0.0f32; bi_pass_sz];
    let mut bi_bits_cur = vec![0u64; fp * lm1];
    let mut m_bi_bits = vec![0u64; fp * lm1];
    let mut qw_bi = vec![0.0f32; bi_pass_sz];
    let mut qind_bi = vec![0.0f32; bi_pass_sz];
    let mut q_bi_bits = vec![0u64; fp * lm1];
    let mut rew_bi = vec![0.0f32; bi_pass_sz];
    let mut act2_sum = vec![0.0f32; fp * lm1 * CB];
    let mut act3_sum = vec![0.0f32; fp * lm1 * CB];
    let mut rho_buf = vec![0.0f32; lm1 * A6_SQ * CB];
    let mut fwd_buf = vec![0.0f32; lm1 * A6 * CB];
    let mut bwd_buf = vec![0.0f32; lm1 * A6 * CB];

    for wi in 0..n_words {
        let w = &words[wi * l..(wi + 1) * l];
        b_cur.fill(0.0);
        bi_cur.fill(0.0);
        bi_bits_cur.fill(0);

        // Precompute root `u64` bitsets and Pass-1 origin lengths / changed-coordinate masks
        // for pure-Boolean exact support (`Theorem B` independent of float32 brightness) & guarded transfer.
        let mut root_bits = [0u64; 64];
        for p in 0..lm1 {
            root_bits[p] = 1u64 << ((w[p] as usize) * A6 + (w[p + 1] as usize));
        }
        let mut max_origin_len = [0usize; 64];
        let mut changed_mask = [0u64; 4];
        let mut exact_fib_bits = [[0u64; 64]; 4];
        let mut exact_active = [false; 4];
        let mut w1_tmp = [0u8; 64];

        for e in 0..n_r2 {
            let x0 = r2[e * 4];
            let x1 = r2[e * 4 + 1];
            let y0 = r2[e * 4 + 2];
            let y1 = r2[e * 4 + 3];
            for p in 0..lm1 {
                if w[p] == x0 && w[p + 1] == x1 && (x0 != y0 || x1 != y1) {
                    let f = if use_prov { p % fp } else { 0 };
                    if 2 > max_origin_len[p] {
                        max_origin_len[p] = 2;
                    }
                    if y0 != w[p] {
                        changed_mask[f] |= 1u64 << p;
                    }
                    if y1 != w[p + 1] {
                        changed_mask[f] |= 1u64 << (p + 1);
                    }
                    w1_tmp[..l].copy_from_slice(w);
                    w1_tmp[p] = y0;
                    w1_tmp[p + 1] = y1;
                    for q in 0..lm1 {
                        exact_fib_bits[f][q] |=
                            1u64 << ((w1_tmp[q] as usize) * A6 + (w1_tmp[q + 1] as usize));
                    }
                    exact_active[f] = true;
                }
            }
        }
        if l >= 3 && n_r3 > 0 {
            for e in 0..(n_r3) {
                let x0 = r3[e * 6];
                let x1 = r3[e * 6 + 1];
                let x2 = r3[e * 6 + 2];
                let y0 = r3[e * 6 + 3];
                let y1 = r3[e * 6 + 4];
                let y2 = r3[e * 6 + 5];
                for p in 0..(l - 2) {
                    if w[p] == x0 && w[p + 1] == x1 && w[p + 2] == x2
                        && (x0 != y0 || x1 != y1 || x2 != y2)
                    {
                        let f = if use_prov { p % fp } else { 0 };
                        if 3 > max_origin_len[p] {
                            max_origin_len[p] = 3;
                        }
                        if y0 != w[p] {
                            changed_mask[f] |= 1u64 << p;
                        }
                        if y1 != w[p + 1] {
                            changed_mask[f] |= 1u64 << (p + 1);
                        }
                        if y2 != w[p + 2] {
                            changed_mask[f] |= 1u64 << (p + 2);
                        }
                        w1_tmp[..l].copy_from_slice(w);
                        w1_tmp[p] = y0;
                        w1_tmp[p + 1] = y1;
                        w1_tmp[p + 2] = y2;
                        for q in 0..lm1 {
                            exact_fib_bits[f][q] |=
                                1u64 << ((w1_tmp[q] as usize) * A6 + (w1_tmp[q + 1] as usize));
                        }
                        exact_active[f] = true;
                    }
                }
            }
        }
        let mut exact_pass1_bits = [[0u64; 64]; 4];
        for f in 0..fp {
            if exact_active[f] {
                trellis_filter_u64(&mut exact_fib_bits[f][..lm1], A6);
                exact_pass1_bits[f][..lm1].copy_from_slice(&exact_fib_bits[f][..lm1]);
            }
        }
        let mut exact_cur_bits = [[0u64; 64]; 4];

        for k in 0..passes {
            rule_checks += win_per_word * (1 + (if k == 0 { 1 } else { fp as u64 }));

            // ---------------- 1. Positional Cell (`SparseMultiplicativeRewriteCell`) ----------------
            let mut m_pos_bits = [0u8; 64];
            if nonrecurrent != 0 || k == 0 || env_mode == 1 {
                m_pos.fill(0.0);
            } else if env_mode == 2 {
                // roll(1, dims=1): m_pos[p] = b_cur[(p + L - 1) % L]
                for p in 0..l {
                    let src_p = (p + l - 1) % l;
                    m_pos[p * A6 * CE..(p + 1) * A6 * CE]
                        .copy_from_slice(&b_cur[src_p * A6 * CE..(src_p + 1) * A6 * CE]);
                }
                for p in 0..l {
                    let mut pb = 0u8;
                    for s in 0..A6 {
                        let sl = &m_pos[(p * A6 + s) * CE..(p * A6 + s + 1) * CE];
                        if sl.iter().any(|&v| v > 0.0) {
                            pb |= 1u8 << s;
                        }
                    }
                    m_pos_bits[p] = pb;
                }
            } else {
                m_pos.copy_from_slice(&b_cur);
                for p in 0..l {
                    let mut pb = 0u8;
                    for s in 0..A6 {
                        let sl = &m_pos[(p * A6 + s) * CE..(p * A6 + s + 1) * CE];
                        if sl.iter().any(|&v| v > 0.0) {
                            pb |= 1u8 << s;
                        }
                    }
                    m_pos_bits[p] = pb;
                }
            }

            let mut q_pos_bits = [0u8; 64];
            for p in 0..l {
                let sym = w[p] as usize;
                let mut pb = 0u8;
                for s in 0..A6 {
                    let base_v = if s == sym { 1.0f32 } else { 0.0f32 };
                    let off = (p * A6 + s) * CE;
                    if nonrecurrent == 0 && ((m_pos_bits[p] >> s) & 1) != 0 {
                        let kind = if s != sym {
                            0
                        } else if p == 0 {
                            2
                        } else if p + 1 == l {
                            3
                        } else {
                            1
                        };
                        let hx = &hk_tab[k.min(15)][kind];
                        let mx = &m_pos[off..off + CE];
                        let mut lin_msg = [0.0f32; CE];
                        for c in 0..CE {
                            let row = &env_gate_wpos[c * CE..(c + 1) * CE];
                            let mut acc = 0.0f32;
                            for i in 0..CE {
                                let mv = mx[i];
                                if mv > 0.0 {
                                    acc += row[i] * mv;
                                }
                            }
                            lin_msg[c] = acc;
                        }
                        let mut any_pos = false;
                        for c in 0..CE {
                            let grow = &env_gate_gw[c * 40..(c + 1) * 40];
                            let mut gacc = env_gate_gb[c];
                            for i in 0..CE {
                                gacc += grow[i] * hx[i] + grow[CE + i] * mx[i];
                            }
                            let gval = sigmoid_f32(gacc) * lin_msg[c];
                            let qv = clamp_f32(base_v + gval, 0.0, 1.5);
                            q_pos[off + c] = qv;
                            if qv > 0.0 {
                                any_pos = true;
                            }
                        }
                        if any_pos {
                            pb |= 1u8 << s;
                        }
                    } else {
                        for c in 0..CE {
                            q_pos[off + c] = base_v;
                        }
                        if base_v > 0.0 {
                            pb |= 1u8 << s;
                        }
                    }
                }
                q_pos_bits[p] = pb;
            }

            rew_pos.fill(0.0);
            let mut feat2 = [0.0f32; 60];
            let mut act2 = [0.0f32; CE];
            for e in 0..n_r2 {
                let x0 = r2[e * 4] as usize;
                let x1 = r2[e * 4 + 1] as usize;
                let y0 = r2[e * 4 + 2] as usize;
                let y1 = r2[e * 4 + 3] as usize;
                for p in 0..lm1 {
                    if ((q_pos_bits[p] >> x0) & 1) == 0 || ((q_pos_bits[p + 1] >> x1) & 1) == 0 {
                        continue;
                    }
                    let q0 = &q_pos[(p * A6 + x0) * CE..(p * A6 + x0 + 1) * CE];
                    let q1 = &q_pos[((p + 1) * A6 + x1) * CE..((p + 1) * A6 + x1 + 1) * CE];
                    let mut nz = false;
                    for c in 0..CE {
                        let pr = q0[c] * q1[c];
                        feat2[c] = pr;
                        feat2[CE + c] = q0[c];
                        feat2[2 * CE + c] = q1[c];
                        if pr > 0.0 {
                            nz = true;
                        }
                    }
                    if !nz {
                        continue;
                    }
                    linear_gelu_linear_softplus(
                        &feat2, pos_r2_w1, pos_r2_b1, pos_r2_w2, pos_r2_b2, 60, 40, CE, &mut act2,
                    );
                    let d0 = (p * A6 + y0) * CE;
                    let d1 = ((p + 1) * A6 + y1) * CE;
                    for c in 0..CE {
                        let v = act2[c] * feat2[c];
                        rew_pos[d0 + c] += v;
                        rew_pos[d1 + c] += v;
                    }
                }
            }

            if l >= 3 && n_r3 > 0 {
                let mut feat3 = [0.0f32; 80];
                let mut act3 = [0.0f32; CE];
                for e in 0..n_r3 {
                    let x0 = r3[e * 6] as usize;
                    let x1 = r3[e * 6 + 1] as usize;
                    let x2 = r3[e * 6 + 2] as usize;
                    let y0 = r3[e * 6 + 3] as usize;
                    let y1 = r3[e * 6 + 4] as usize;
                    let y2 = r3[e * 6 + 5] as usize;
                    for p in 0..(l - 2) {
                        if ((q_pos_bits[p] >> x0) & 1) == 0
                            || ((q_pos_bits[p + 1] >> x1) & 1) == 0
                            || ((q_pos_bits[p + 2] >> x2) & 1) == 0
                        {
                            continue;
                        }
                        let q0 = &q_pos[(p * A6 + x0) * CE..(p * A6 + x0 + 1) * CE];
                        let q1 = &q_pos[((p + 1) * A6 + x1) * CE..((p + 1) * A6 + x1 + 1) * CE];
                        let q2 = &q_pos[((p + 2) * A6 + x2) * CE..((p + 2) * A6 + x2 + 1) * CE];
                        let mut nz = false;
                        for c in 0..CE {
                            let pr = q0[c] * q1[c] * q2[c];
                            feat3[c] = pr;
                            feat3[CE + c] = q0[c];
                            feat3[2 * CE + c] = q1[c];
                            feat3[3 * CE + c] = q2[c];
                            if pr > 0.0 {
                                nz = true;
                            }
                        }
                        if !nz {
                            continue;
                        }
                        linear_gelu_linear_softplus(
                            &feat3, pos_r3_w1, pos_r3_b1, pos_r3_w2, pos_r3_b2, 80, 40, CE, &mut act3,
                        );
                        let d0 = (p * A6 + y0) * CE;
                        let d1 = ((p + 1) * A6 + y1) * CE;
                        let d2 = ((p + 2) * A6 + y2) * CE;
                        for c in 0..CE {
                            let v = act3[c] * feat3[c];
                            rew_pos[d0 + c] += v;
                            rew_pos[d1 + c] += v;
                            rew_pos[d2 + c] += v;
                        }
                    }
                }
            }

            // Update b_cur
            for ps in 0..(l * A6) {
                let off = ps * CE;
                for c in 0..CE {
                    let prev_p = if m_pos[off + c] > 0.0 { m_pos[off + c] } else { 0.0 };
                    b_cur[off + c] = (rew_pos[off + c] + 0.65 * prev_p).tanh();
                }
            }

            // ---------------- 2. Provenance Bigram Cell (`FastProvenanceBigramCell`) ----------------
            if nonrecurrent != 0 || k == 0 || env_mode == 1 {
                m_bi.fill(0.0);
                m_bi_bits.fill(0);
            } else if env_mode == 2 {
                // roll(1, dims=2): along Lm1
                for f in 0..fp {
                    for p in 0..lm1 {
                        let src_p = (p + lm1 - 1) % lm1;
                        let dst_o = (f * lm1 + p) * A6_SQ * CB;
                        let src_o = (f * lm1 + src_p) * A6_SQ * CB;
                        m_bi[dst_o..dst_o + A6_SQ * CB]
                            .copy_from_slice(&bi_cur[src_o..src_o + A6_SQ * CB]);
                        m_bi_bits[f * lm1 + p] = bi_bits_cur[f * lm1 + src_p];
                    }
                }
            } else {
                m_bi.copy_from_slice(&bi_cur);
                m_bi_bits.copy_from_slice(&bi_bits_cur);
            }

            let q_fp = if k == 0 { 1 } else { fp };
            qw_bi.fill(0.0);
            qind_bi.fill(0.0);
            q_bi_bits.fill(0);

            if k == 0 {
                let ind_val = 1.0f32 / (1.0f32 + 1e-4f32);
                for p in 0..lm1 {
                    let ab = (w[p] as usize) * A6 + (w[p + 1] as usize);
                    q_bi_bits[p] = 1u64 << ab;
                    let off = (p * A6_SQ + ab) * CB;
                    for c in 0..CB {
                        qw_bi[off + c] = 1.0;
                        qind_bi[off + c] = ind_val;
                    }
                }
            } else {
                for f in 0..fp {
                    for p in 0..lm1 {
                        let mask = m_bi_bits[f * lm1 + p];
                        if mask == 0 {
                            continue;
                        }
                        let root_ab = (w[p] as usize) * A6 + (w[p + 1] as usize);
                        let mut out_mask = 0u64;
                        for ab in 0..A6_SQ {
                            if ((mask >> ab) & 1) == 0 {
                                continue;
                            }
                            let x_val = if ab == root_ab { 1.0f32 } else { 0.0f32 };
                            let off = ((f * lm1 + p) * A6_SQ + ab) * CB;
                            let mx = &m_bi[off..off + CB];
                            let mut lin_m = [0.0f32; CB];
                            for c in 0..CB {
                                let row = &bi_gate_wpos[c * CB..(c + 1) * CB];
                                let mut acc = 0.0f32;
                                for i in 0..CB {
                                    if mx[i] > 0.0 {
                                        acc += row[i] * mx[i];
                                    }
                                }
                                lin_m[c] = acc;
                            }
                            let mut any_q = false;
                            for c in 0..CB {
                                let grow = &bi_gate_gw[c * 8..(c + 1) * 8];
                                let mut gacc = bi_gate_gb[c];
                                for i in 0..CB {
                                    gacc += grow[i] * x_val + grow[CB + i] * mx[i];
                                }
                                let gv = clamp_f32(sigmoid_f32(gacc) * lin_m[c], 0.0, 1.5);
                                qw_bi[off + c] = gv;
                                if gv > 0.0 {
                                    qind_bi[off + c] = gv / (gv + 1e-4);
                                    any_q = true;
                                }
                            }
                            if any_q {
                                out_mask |= 1u64 << ab;
                            }
                        }
                        q_bi_bits[f * lm1 + p] = out_mask;
                    }
                }
            }

            rew_bi.fill(0.0);
            act2_sum.fill(0.0);
            let mut bi_act = [0.0f32; CB];

            if n_r2 > 0 {
                for qf in 0..q_fp {
                    for e in 0..n_r2 {
                        let x0 = r2[e * 4] as usize;
                        let x1 = r2[e * 4 + 1] as usize;
                        let y0 = r2[e * 4 + 2] as usize;
                        let y1 = r2[e * 4 + 3] as usize;
                        let x_ab = x0 * A6 + x1;
                        let y_ab = y0 * A6 + y1;
                        for p in 0..lm1 {
                            if ((q_bi_bits[qf * lm1 + p] >> x_ab) & 1) == 0 {
                                continue;
                            }
                            let dst_f = if k == 0 && use_prov { p % fp } else { qf };
                            let ctx_f = if k == 0 && use_prov { 0 } else { dst_f };
                            if k == 1 && use_prov && (p % fp == dst_f)
                                && (w[p] as usize == y0) && (w[p + 1] as usize == y1)
                            {
                                let sup2 = match guard_mode {
                                    0 => true,
                                    1 => (fp >= actual_max_rlen) && (max_origin_len[p] <= 2),
                                    2 => {
                                        let win_m = 0b11u64 << p;
                                        (changed_mask[dst_f] & !win_m) == 0
                                    }
                                    _ => false,
                                };
                                if sup2 {
                                    continue;
                                }
                            }
                            let pi2 = &qw_bi[((qf * lm1 + p) * A6_SQ + x_ab) * CB
                                ..((qf * lm1 + p) * A6_SQ + x_ab + 1) * CB];
                            linear_gelu_linear_softplus(
                                pi2, bi_r2_w1, bi_r2_b1, bi_r2_w2, bi_r2_b2, CB, 8, CB, &mut bi_act,
                            );
                            let mut av = [0.0f32; CB];
                            let core_off = ((dst_f * lm1 + p) * A6_SQ + y_ab) * CB;
                            let sum_off = (dst_f * lm1 + p) * CB;
                            for c in 0..CB {
                                av[c] = bi_act[c] * pi2[c];
                                rew_bi[core_off + c] += av[c];
                                act2_sum[sum_off + c] += av[c];
                            }
                            if p > 0 {
                                let lmask = q_bi_bits[ctx_f * lm1 + (p - 1)];
                                for s in 0..A6 {
                                    let lab = s * A6 + x0;
                                    if ((lmask >> lab) & 1) != 0 {
                                        let qi_o = ((ctx_f * lm1 + (p - 1)) * A6_SQ + lab) * CB;
                                        let bd_o = ((dst_f * lm1 + (p - 1)) * A6_SQ + (s * A6 + y0)) * CB;
                                        for c in 0..CB {
                                            rew_bi[bd_o + c] += qind_bi[qi_o + c] * av[c];
                                        }
                                    }
                                }
                            }
                            if p + 1 < lm1 {
                                let rmask = q_bi_bits[ctx_f * lm1 + (p + 1)];
                                for s in 0..A6 {
                                    let rab = x1 * A6 + s;
                                    if ((rmask >> rab) & 1) != 0 {
                                        let qi_o = ((ctx_f * lm1 + (p + 1)) * A6_SQ + rab) * CB;
                                        let bd_o = ((dst_f * lm1 + (p + 1)) * A6_SQ + (y1 * A6 + s)) * CB;
                                        for c in 0..CB {
                                            rew_bi[bd_o + c] += qind_bi[qi_o + c] * av[c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Far-context identity preservation for r2 (`|q - p| > 1`)
                for dst_f in 0..fp {
                    let ctx_f = if k == 0 && use_prov { 0 } else { dst_f };
                    for q in 0..lm1 {
                        let qmask = q_bi_bits[ctx_f * lm1 + q];
                        if qmask == 0 {
                            continue;
                        }
                        let mut cw = [0.0f32; CB];
                        let mut any_cw = false;
                        for p in 0..lm1 {
                            if q + 1 < p || q > p + 1 {
                                let so = (dst_f * lm1 + p) * CB;
                                for c in 0..CB {
                                    cw[c] += act2_sum[so + c];
                                    if act2_sum[so + c] > 0.0 {
                                        any_cw = true;
                                    }
                                }
                            }
                        }
                        if !any_cw {
                            continue;
                        }
                        for ab in 0..A6_SQ {
                            if ((qmask >> ab) & 1) != 0 {
                                let qi_o = ((ctx_f * lm1 + q) * A6_SQ + ab) * CB;
                                let dst_o = ((dst_f * lm1 + q) * A6_SQ + ab) * CB;
                                for c in 0..CB {
                                    rew_bi[dst_o + c] += cw[c] * qind_bi[qi_o + c];
                                }
                            }
                        }
                    }
                }
            }

            if lm1 >= 2 && n_r3 > 0 {
                let lm2 = lm1 - 1;
                act3_sum[..fp * lm2 * CB].fill(0.0);
                let mut feat3_bi = [0.0f32; 12];
                for qf in 0..q_fp {
                    for e in 0..n_r3 {
                        let x0 = r3[e * 6] as usize;
                        let x1 = r3[e * 6 + 1] as usize;
                        let x2 = r3[e * 6 + 2] as usize;
                        let y0 = r3[e * 6 + 3] as usize;
                        let y1 = r3[e * 6 + 4] as usize;
                        let y2 = r3[e * 6 + 5] as usize;
                        let xb0 = x0 * A6 + x1;
                        let xb1 = x1 * A6 + x2;
                        let yb0 = y0 * A6 + y1;
                        let yb1 = y1 * A6 + y2;
                        for p in 0..lm2 {
                            if ((q_bi_bits[qf * lm1 + p] >> xb0) & 1) == 0
                                || ((q_bi_bits[qf * lm1 + p + 1] >> xb1) & 1) == 0
                            {
                                continue;
                            }
                            let dst_f = if k == 0 && use_prov { p % fp } else { qf };
                            let ctx_f = if k == 0 && use_prov { 0 } else { dst_f };
                            if k == 1 && use_prov && (p % fp == dst_f)
                                && (w[p] as usize == y0)
                                && (w[p + 1] as usize == y1)
                                && (w[p + 2] as usize == y2)
                            {
                                let sup3 = match guard_mode {
                                    0 => true,
                                    1 => (fp >= actual_max_rlen) && (max_origin_len[p] <= 3),
                                    2 => {
                                        let win_m = 0b111u64 << p;
                                        (changed_mask[dst_f] & !win_m) == 0
                                    }
                                    _ => false,
                                };
                                if sup3 {
                                    continue;
                                }
                            }
                            let m0 = &qw_bi[((qf * lm1 + p) * A6_SQ + xb0) * CB
                                ..((qf * lm1 + p) * A6_SQ + xb0 + 1) * CB];
                            let m1 = &qw_bi[((qf * lm1 + p + 1) * A6_SQ + xb1) * CB
                                ..((qf * lm1 + p + 1) * A6_SQ + xb1 + 1) * CB];
                            let mut nz = false;
                            for c in 0..CB {
                                let pr = m0[c] * m1[c];
                                feat3_bi[c] = pr;
                                feat3_bi[CB + c] = m0[c];
                                feat3_bi[2 * CB + c] = m1[c];
                                if pr > 0.0 {
                                    nz = true;
                                }
                            }
                            if !nz {
                                continue;
                            }
                            linear_gelu_linear_softplus(
                                &feat3_bi, bi_r3_w1, bi_r3_b1, bi_r3_w2, bi_r3_b2, 12, 8, CB, &mut bi_act,
                            );
                            let mut av = [0.0f32; CB];
                            let co0 = ((dst_f * lm1 + p) * A6_SQ + yb0) * CB;
                            let co1 = ((dst_f * lm1 + p + 1) * A6_SQ + yb1) * CB;
                            let so = (dst_f * lm2 + p) * CB;
                            for c in 0..CB {
                                av[c] = bi_act[c] * feat3_bi[c];
                                rew_bi[co0 + c] += av[c];
                                rew_bi[co1 + c] += av[c];
                                act3_sum[so + c] += av[c];
                            }
                            if p > 0 {
                                let lmask = q_bi_bits[ctx_f * lm1 + (p - 1)];
                                for s in 0..A6 {
                                    let lab = s * A6 + x0;
                                    if ((lmask >> lab) & 1) != 0 {
                                        let qi_o = ((ctx_f * lm1 + (p - 1)) * A6_SQ + lab) * CB;
                                        let bd_o = ((dst_f * lm1 + (p - 1)) * A6_SQ + (s * A6 + y0)) * CB;
                                        for c in 0..CB {
                                            rew_bi[bd_o + c] += qind_bi[qi_o + c] * av[c];
                                        }
                                    }
                                }
                            }
                            if p + 2 < lm1 {
                                let rmask = q_bi_bits[ctx_f * lm1 + (p + 2)];
                                for s in 0..A6 {
                                    let rab = x2 * A6 + s;
                                    if ((rmask >> rab) & 1) != 0 {
                                        let qi_o = ((ctx_f * lm1 + (p + 2)) * A6_SQ + rab) * CB;
                                        let bd_o = ((dst_f * lm1 + (p + 2)) * A6_SQ + (y2 * A6 + s)) * CB;
                                        for c in 0..CB {
                                            rew_bi[bd_o + c] += qind_bi[qi_o + c] * av[c];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Far-context identity preservation for r3 (`q < p - 1 || q > p + 2`)
                for dst_f in 0..fp {
                    let ctx_f = if k == 0 && use_prov { 0 } else { dst_f };
                    for q in 0..lm1 {
                        let qmask = q_bi_bits[ctx_f * lm1 + q];
                        if qmask == 0 {
                            continue;
                        }
                        let mut cw = [0.0f32; CB];
                        let mut any_cw = false;
                        for p in 0..lm2 {
                            if q + 1 < p || q > p + 2 {
                                let so = (dst_f * lm2 + p) * CB;
                                for c in 0..CB {
                                    cw[c] += act3_sum[so + c];
                                    if act3_sum[so + c] > 0.0 {
                                        any_cw = true;
                                    }
                                }
                            }
                        }
                        if !any_cw {
                            continue;
                        }
                        for ab in 0..A6_SQ {
                            if ((qmask >> ab) & 1) != 0 {
                                let qi_o = ((ctx_f * lm1 + q) * A6_SQ + ab) * CB;
                                let dst_o = ((dst_f * lm1 + q) * A6_SQ + ab) * CB;
                                for c in 0..CB {
                                    rew_bi[dst_o + c] += cw[c] * qind_bi[qi_o + c];
                                }
                            }
                        }
                    }
                }
            }

            // Pure-Boolean exact `u64` transition for pass `k` (100% independent of float32!)
            if k == 0 || nonrecurrent != 0 {
                for f in 0..fp {
                    exact_cur_bits[f][..lm1].copy_from_slice(&exact_pass1_bits[f][..lm1]);
                }
            } else if env_mode == 1 {
                for f in 0..fp {
                    exact_cur_bits[f][..lm1].fill(0);
                }
            } else {
                let mut ex_in = [[0u64; 64]; 4];
                for f in 0..fp {
                    if env_mode == 2 {
                        for p in 0..lm1 {
                            ex_in[f][p] = exact_cur_bits[f][(p + lm1 - 1) % lm1];
                        }
                        trellis_filter_u64(&mut ex_in[f][..lm1], A6);
                    } else {
                        ex_in[f][..lm1].copy_from_slice(&exact_cur_bits[f][..lm1]);
                        trellis_filter_u64(&mut ex_in[f][..lm1], A6);
                    }
                }
                for f in 0..fp {
                    let mut nxt_b = [0u64; 64];
                    nxt_b[..lm1].copy_from_slice(&ex_in[f][..lm1]);
                    if nxt_b[..lm1].iter().all(|&m| m == 0) {
                        exact_cur_bits[f][..lm1].fill(0);
                        continue;
                    }
                    for e in 0..n_r2 {
                        let x0 = r2[e * 4] as usize;
                        let x1 = r2[e * 4 + 1] as usize;
                        let y0 = r2[e * 4 + 2] as usize;
                        let y1 = r2[e * 4 + 3] as usize;
                        let x_ab = x0 * A6 + x1;
                        let y_ab = y0 * A6 + y1;
                        for p in 0..lm1 {
                            if ((ex_in[f][p] >> x_ab) & 1) == 0 {
                                continue;
                            }
                            if k == 1 && use_prov && (p % fp == f)
                                && (w[p] as usize == y0) && (w[p + 1] as usize == y1)
                            {
                                let sup2 = match guard_mode {
                                    0 => true,
                                    1 => (fp >= actual_max_rlen) && (max_origin_len[p] <= 2),
                                    2 => {
                                        let win_m = 0b11u64 << p;
                                        (changed_mask[f] & !win_m) == 0
                                    }
                                    _ => false,
                                };
                                if sup2 {
                                    continue;
                                }
                            }
                            nxt_b[p] |= 1u64 << y_ab;
                            if p > 0 {
                                let pm = ex_in[f][p - 1];
                                for s in 0..A6 {
                                    if ((pm >> (s * A6 + x0)) & 1) != 0 {
                                        nxt_b[p - 1] |= 1u64 << (s * A6 + y0);
                                    }
                                }
                            }
                            if p + 1 < lm1 {
                                let rm = ex_in[f][p + 1];
                                for s in 0..A6 {
                                    if ((rm >> (x1 * A6 + s)) & 1) != 0 {
                                        nxt_b[p + 1] |= 1u64 << (y1 * A6 + s);
                                    }
                                }
                            }
                        }
                    }
                    if lm1 >= 2 && n_r3 > 0 {
                        for e in 0..n_r3 {
                            let x0 = r3[e * 6] as usize;
                            let x1 = r3[e * 6 + 1] as usize;
                            let x2 = r3[e * 6 + 2] as usize;
                            let y0 = r3[e * 6 + 3] as usize;
                            let y1 = r3[e * 6 + 4] as usize;
                            let y2 = r3[e * 6 + 5] as usize;
                            let xb0 = x0 * A6 + x1;
                            let xb1 = x1 * A6 + x2;
                            let yb0 = y0 * A6 + y1;
                            let yb1 = y1 * A6 + y2;
                            for p in 0..(lm1 - 1) {
                                if ((ex_in[f][p] >> xb0) & 1) == 0
                                    || ((ex_in[f][p + 1] >> xb1) & 1) == 0
                                {
                                    continue;
                                }
                                if k == 1 && use_prov && (p % fp == f)
                                    && (w[p] as usize == y0)
                                    && (w[p + 1] as usize == y1)
                                    && (w[p + 2] as usize == y2)
                                {
                                    let sup3 = match guard_mode {
                                        0 => true,
                                        1 => (fp >= actual_max_rlen) && (max_origin_len[p] <= 3),
                                        2 => {
                                            let win_m = 0b111u64 << p;
                                            (changed_mask[f] & !win_m) == 0
                                        }
                                        _ => false,
                                    };
                                    if sup3 {
                                        continue;
                                    }
                                }
                                nxt_b[p] |= 1u64 << yb0;
                                nxt_b[p + 1] |= 1u64 << yb1;
                                if p > 0 {
                                    let pm = ex_in[f][p - 1];
                                    for s in 0..A6 {
                                        if ((pm >> (s * A6 + x0)) & 1) != 0 {
                                            nxt_b[p - 1] |= 1u64 << (s * A6 + y0);
                                        }
                                    }
                                }
                                if p + 2 < lm1 {
                                    let rm = ex_in[f][p + 2];
                                    for s in 0..A6 {
                                        if ((rm >> (x2 * A6 + s)) & 1) != 0 {
                                            nxt_b[p + 2] |= 1u64 << (y2 * A6 + s);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    trellis_filter_u64(&mut nxt_b[..lm1], A6);
                    exact_cur_bits[f][..lm1].copy_from_slice(&nxt_b[..lm1]);
                }
            }

            // Combine `rew_bi + 0.65 * relu(m_bi)`, build bitset, and run `trellis_arc_filter_soft_f32` per fiber!
            let mut min_pos_pass = f32::INFINITY;
            let mut any_exact_supp = false;
            for f in 0..fp {
                let f_off = f * lm1 * A6_SQ * CB;
                for p in 0..lm1 {
                    let mut pb = 0u64;
                    for ab in 0..A6_SQ {
                        let idx = f_off + (p * A6_SQ + ab) * CB;
                        let mut any_pos = false;
                        for c in 0..CB {
                            let r = rew_bi[idx + c];
                            let v = if k == 0 {
                                r.tanh()
                            } else {
                                let mp = if m_bi[idx + c] > 0.0 { m_bi[idx + c] } else { 0.0 };
                                (r + 0.65 * mp).tanh()
                            };
                            bi_cur[idx + c] = v;
                            if v > 0.0 {
                                any_pos = true;
                            }
                        }
                        if any_pos {
                            pb |= 1u64 << ab;
                        }
                    }
                    // In guarded modes (`guard_mode != 0`), drive the trellis mask from `exact_cur_bits`
                    // so float32 underflow can NEVER prematurely prune a Boolean-supported arc!
                    bi_bits_cur[f * lm1 + p] = if guard_mode == 0 { pb } else { exact_cur_bits[f][p] };
                }
                trellis_arc_filter_soft_f32(
                    &mut bi_cur[f_off..f_off + lm1 * A6_SQ * CB],
                    &mut bi_bits_cur[f * lm1..(f + 1) * lm1],
                    lm1,
                    &mut rho_buf,
                    &mut fwd_buf,
                    &mut bwd_buf,
                );
                for p in 0..lm1 {
                    let em = exact_cur_bits[f][p];
                    if em == 0 {
                        continue;
                    }
                    for ab in 0..A6_SQ {
                        if ((em >> ab) & 1) != 0 {
                            any_exact_supp = true;
                            let idx = f_off + (p * A6_SQ + ab) * CB;
                            let mut cmax = 0.0f32;
                            for c in 0..CB {
                                if bi_cur[idx + c] > cmax {
                                    cmax = bi_cur[idx + c];
                                }
                            }
                            if cmax < min_pos_pass {
                                min_pos_pass = cmax;
                            }
                        }
                    }
                }
            }

            // Write pass-k outputs + independent exact Boolean support + min_pos_f32 margin
            let q_dst_off = (wi * passes + k) * pos_pass_sz;
            out_q[q_dst_off..q_dst_off + pos_pass_sz].copy_from_slice(&q_pos);
            out_b[q_dst_off..q_dst_off + pos_pass_sz].copy_from_slice(&b_cur);
            let bi_dst_off = (wi * passes + k) * bi_pass_sz;
            out_bi[bi_dst_off..bi_dst_off + bi_pass_sz].copy_from_slice(&bi_cur);

            let supp_dst_off = (wi * passes + k) * supp_pass_sz;
            out_supp[supp_dst_off..supp_dst_off + lm1].copy_from_slice(&root_bits[..lm1]);
            for f in 0..fp {
                let sf_off = supp_dst_off + (f + 1) * lm1;
                out_supp[sf_off..sf_off + lm1].copy_from_slice(&exact_cur_bits[f][..lm1]);
            }
            out_min_pos[wi * passes + k] = if any_exact_supp { min_pos_pass } else { 1.0 };
        }
    }

    rule_checks
}
