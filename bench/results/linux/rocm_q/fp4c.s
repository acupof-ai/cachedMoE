	.amdgcn_target "amdgcn-amd-amdhsa--gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	fp4_block               ; -- Begin function fp4_block
	.globl	fp4_block
	.p2align	8
	.type	fp4_block,@function
fp4_block:                              ; @fp4_block
; %bb.0:
	s_clause 0x2
	s_load_b128 s[36:39], s[4:5], 0x0
	s_load_b64 s[34:35], s[4:5], 0x10
	s_load_b32 s33, s[4:5], 0x18
	v_dual_mov_b32 v31, v0 :: v_dual_mov_b32 v0, 0
	s_mov_b32 s12, s8
	s_add_u32 s8, s4, 32
	s_mov_b32 s13, s9
	s_addc_u32 s9, s5, 0
	s_mov_b32 s14, s10
	s_mov_b64 s[10:11], s[6:7]
	s_getpc_b64 s[16:17]
	s_add_u32 s16, s16, _Z13get_global_idj@rel32@lo+4
	s_addc_u32 s17, s17, _Z13get_global_idj@rel32@hi+12
	s_mov_b64 s[4:5], s[0:1]
	s_mov_b64 s[6:7], s[2:3]
	s_mov_b32 s32, 0
	s_swappc_b64 s[30:31], s[16:17]
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc1 .LBB0_131
; %bb.1:
	v_mul_lo_u32 v5, s33, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v6, 0 :: v_dual_lshlrev_b32 v9, 5, v5
	v_mov_b32_e32 v10, v6
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_lshlrev_b32_e32 v57, 28, v1
	v_lshrrev_b32_e32 v58, 4, v1
	v_lshrrev_b32_e32 v59, 8, v1
	v_add_nc_u32_e32 v9, 32, v9
	v_add_nc_u32_e32 v5, 1, v5
	v_and_or_b32 v11, 0x80000000, v57, v11
	v_lshlrev_b32_e32 v57, 28, v58
	v_lshrrev_b32_e32 v58, 12, v1
	v_lshlrev_b32_e32 v59, 28, v59
	s_add_i32 s33, s33, -1
	s_waitcnt vmcnt(15)
	v_fma_mix_f32 v10, v11, v12, v10 op_sel_hi:[0,1,0]
	v_and_or_b32 v11, 0x80000000, v57, v13
	v_mov_b16_e32 v13.h, 0
	v_mov_b16_e32 v13.l, v1.h
	v_lshlrev_b32_e32 v57, 28, v58
	v_and_or_b32 v14, 0x80000000, v59, v14
	v_fma_mix_f32 v10, v11, v12, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v1
	v_lshlrev_b32_e32 v12, 28, v13
	v_and_or_b32 v16, 0x80000000, v57, v16
	v_mov_b16_e32 v13.l, v4.h
	s_waitcnt vmcnt(14)
	v_fma_mix_f32 v10, v14, v15, v10 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v14, 24, v1
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v12, 0x80000000, v12, v17
	v_and_or_b32 v1, 0x80000000, v1, v22
	v_fma_mix_f32 v10, v16, v15, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v14, 28, v14
	v_and_or_b32 v11, 0x80000000, v11, v19
	s_cmp_lg_u32 s33, 0
	s_waitcnt vmcnt(13)
	v_fma_mix_f32 v10, v12, v18, v10 op_sel_hi:[0,1,0]
	v_and_or_b32 v12, 0x80000000, v14, v20
	v_lshlrev_b32_e32 v14, 28, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v10, v11, v18, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 4, v2
	v_and_or_b32 v14, 0x80000000, v14, v23
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v10, v12, v21, v10 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v12, 8, v2
	v_lshlrev_b32_e32 v11, 28, v11
	v_fma_mix_f32 v1, v1, v21, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 12, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v12, 28, v12
	v_and_or_b32 v11, 0x80000000, v11, v25
	s_waitcnt vmcnt(11)
	v_fma_mix_f32 v1, v14, v24, v1 op_sel_hi:[0,1,0]
	v_mov_b16_e32 v14.l, v2.h
	v_mov_b16_e32 v14.h, v13.h
	v_lshlrev_b32_e32 v10, 28, v10
	v_and_or_b32 v12, 0x80000000, v12, v26
	v_fma_mix_f32 v1, v11, v24, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v2
	v_lshlrev_b32_e32 v14, 28, v14
	v_and_or_b32 v10, 0x80000000, v10, v28
	s_waitcnt vmcnt(10)
	v_fma_mix_f32 v1, v12, v27, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v12, 24, v2
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v14, 0x80000000, v14, v29
	v_and_or_b32 v2, 0x80000000, v2, v34
	v_fma_mix_f32 v1, v10, v27, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v10, 28, v12
	v_and_or_b32 v11, 0x80000000, v11, v31
	v_lshlrev_b32_e32 v12, 28, v3
	s_waitcnt vmcnt(9)
	v_fma_mix_f32 v1, v14, v30, v1 op_sel_hi:[0,1,0]
	v_and_or_b32 v10, 0x80000000, v10, v32
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v12, 0x80000000, v12, v35
	v_fma_mix_f32 v1, v11, v30, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 4, v3
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v10, v33, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 8, v3
	v_lshlrev_b32_e32 v11, 28, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v2, v33, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 12, v3
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v11, 0x80000000, v11, v37
	s_waitcnt vmcnt(7)
	v_fma_mix_f32 v1, v12, v36, v1 op_sel_hi:[0,1,0]
	v_mov_b16_e32 v12.l, v3.h
	v_mov_b16_e32 v12.h, v13.h
	v_lshlrev_b32_e32 v2, 28, v2
	v_and_or_b32 v10, 0x80000000, v10, v38
	v_fma_mix_f32 v1, v11, v36, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v3
	v_lshlrev_b32_e32 v12, 28, v12
	v_and_or_b32 v2, 0x80000000, v2, v40
	s_waitcnt vmcnt(6)
	v_fma_mix_f32 v1, v10, v39, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 24, v3
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v12, 0x80000000, v12, v41
	v_and_or_b32 v3, 0x80000000, v3, v46
	v_fma_mix_f32 v1, v2, v39, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v2, 28, v10
	v_and_or_b32 v10, 0x80000000, v11, v43
	v_lshlrev_b32_e32 v11, 28, v4
	s_waitcnt vmcnt(5)
	v_fma_mix_f32 v1, v12, v42, v1 op_sel_hi:[0,1,0]
	v_and_or_b32 v2, 0x80000000, v2, v44
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v11, 0x80000000, v11, v47
	v_fma_mix_f32 v1, v10, v42, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 4, v4
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v2, v45, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 8, v4
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v3, v45, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v3, 12, v4
	v_lshlrev_b32_e32 v2, 28, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v10, 0x80000000, v10, v49
	s_waitcnt vmcnt(3)
	v_fma_mix_f32 v1, v11, v48, v1 op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v3, 28, v3
	v_and_or_b32 v2, 0x80000000, v2, v50
	v_lshlrev_b32_e32 v11, 28, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v10, v48, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 20, v4
	v_and_or_b32 v3, 0x80000000, v3, v52
	v_and_or_b32 v11, 0x80000000, v11, v53
	s_waitcnt vmcnt(2)
	v_fma_mix_f32 v1, v2, v51, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 24, v4
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v3, v51, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v2, 28, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_or_b32 v3, 0x80000000, v10, v55
	s_waitcnt vmcnt(1)
	v_fma_mix_f32 v1, v11, v54, v1 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v2, 0x80000000, v2, v56
	v_fma_mix_f32 v1, v3, v54, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_and_or_b32 v3, 0x80000000, v4, v8
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v2, v7, v1 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v10, v3, v7, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_cbranch_scc0 .LBB0_132
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64 v[1:2], 4, v[5:6]
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, s36, v1
	v_add_co_ci_u32_e64 v2, null, s37, v2, vcc_lo
	global_load_b128 v[1:4], v[1:2], off
	s_waitcnt vmcnt(0)
	v_bfe_u32 v8, v1, 1, 2
	v_and_b32_e32 v7, 1, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v8
	s_xor_b32 s0, exec_lo, s0
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 22, v7
	v_lshl_or_b32 v7, v8, 23, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v11, 0.5, v7
                                        ; implicit-def: $vgpr7
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_cndmask_b32_e64 v11, 0.5, 0, vcc_lo
; %bb.7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_lshrrev_b32_e32 v7, 1, v9
	v_mov_b32_e32 v8, v6
	v_bfe_u32 v15, v1, 5, 2
	v_bfe_u32 v14, v1, 4, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, s38, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, s39, v8, vcc_lo
	global_load_b32 v12, v[7:8], off
	v_cmpx_ne_u32_e32 0, v15
	s_xor_b32 s0, exec_lo, s0
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_lshlrev_b32_e32 v13, 22, v14
                                        ; implicit-def: $vgpr14
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v13, v15, 23, v13
	v_add_nc_u32_e32 v13, 0.5, v13
; %bb.9:                                ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	v_cndmask_b32_e64 v13, 0.5, 0, vcc_lo
; %bb.11:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v16, v1, 9, 2
	v_bfe_u32 v15, v1, 8, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr14
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v16
	s_xor_b32 s0, exec_lo, s0
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v14, 22, v15
                                        ; implicit-def: $vgpr15
	v_lshl_or_b32 v14, v16, 23, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v14, 0.5, v14
; %bb.13:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v15
	v_cndmask_b32_e64 v14, 0.5, 0, vcc_lo
; %bb.15:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v15, v[7:8], off offset:4
	v_bfe_u32 v18, v1, 13, 2
	v_bfe_u32 v17, v1, 12, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr16
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v18
	s_xor_b32 s0, exec_lo, s0
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v16, 22, v17
                                        ; implicit-def: $vgpr17
	v_lshl_or_b32 v16, v18, 23, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v16, 0.5, v16
; %bb.17:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	v_cndmask_b32_e64 v16, 0.5, 0, vcc_lo
; %bb.19:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v19, v1, 17, 2
	v_bfe_u32 v18, v1, 16, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr17
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v19
	s_xor_b32 s0, exec_lo, s0
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v17, 22, v18
                                        ; implicit-def: $vgpr18
	v_lshl_or_b32 v17, v19, 23, v17
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v17, 0.5, v17
; %bb.21:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v18
	v_cndmask_b32_e64 v17, 0.5, 0, vcc_lo
; %bb.23:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v18, v[7:8], off offset:8
	v_bfe_u32 v21, v1, 21, 2
	v_bfe_u32 v20, v1, 20, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr19
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v21
	s_xor_b32 s0, exec_lo, s0
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v19, 22, v20
                                        ; implicit-def: $vgpr20
	v_lshl_or_b32 v19, v21, 23, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v19, 0.5, v19
; %bb.25:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v20
	v_cndmask_b32_e64 v19, 0.5, 0, vcc_lo
; %bb.27:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v22, v1, 25, 2
	v_bfe_u32 v21, v1, 24, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr20
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v22
	s_xor_b32 s0, exec_lo, s0
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v20, 22, v21
                                        ; implicit-def: $vgpr21
	v_lshl_or_b32 v20, v22, 23, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v20, 0.5, v20
; %bb.29:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v21
	v_cndmask_b32_e64 v20, 0.5, 0, vcc_lo
; %bb.31:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v21, v[7:8], off offset:12
	v_bfe_u32 v24, v1, 29, 2
	v_bfe_u32 v23, v1, 28, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr22
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v24
	s_xor_b32 s0, exec_lo, s0
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 22, v23
                                        ; implicit-def: $vgpr23
	v_lshl_or_b32 v22, v24, 23, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, 0.5, v22
; %bb.33:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v23
	v_cndmask_b32_e64 v22, 0.5, 0, vcc_lo
; %bb.35:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v25, v2, 1, 2
	v_and_b32_e32 v24, 1, v2
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr23
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v25
	s_xor_b32 s0, exec_lo, s0
; %bb.36:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v23, 22, v24
                                        ; implicit-def: $vgpr24
	v_lshl_or_b32 v23, v25, 23, v23
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v23, 0.5, v23
; %bb.37:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.38:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v24
	v_cndmask_b32_e64 v23, 0.5, 0, vcc_lo
; %bb.39:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v24, v[7:8], off offset:16
	v_bfe_u32 v27, v2, 5, 2
	v_bfe_u32 v26, v2, 4, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr25
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v27
	s_xor_b32 s0, exec_lo, s0
; %bb.40:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 22, v26
                                        ; implicit-def: $vgpr26
	v_lshl_or_b32 v25, v27, 23, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, 0.5, v25
; %bb.41:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.42:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v26
	v_cndmask_b32_e64 v25, 0.5, 0, vcc_lo
; %bb.43:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v28, v2, 9, 2
	v_bfe_u32 v27, v2, 8, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr26
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v28
	s_xor_b32 s0, exec_lo, s0
; %bb.44:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v26, 22, v27
                                        ; implicit-def: $vgpr27
	v_lshl_or_b32 v26, v28, 23, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v26, 0.5, v26
; %bb.45:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.46:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v27
	v_cndmask_b32_e64 v26, 0.5, 0, vcc_lo
; %bb.47:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v27, v[7:8], off offset:20
	v_bfe_u32 v30, v2, 13, 2
	v_bfe_u32 v29, v2, 12, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr28
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v30
	s_xor_b32 s0, exec_lo, s0
; %bb.48:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 22, v29
                                        ; implicit-def: $vgpr29
	v_lshl_or_b32 v28, v30, 23, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 0.5, v28
; %bb.49:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.50:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v29
	v_cndmask_b32_e64 v28, 0.5, 0, vcc_lo
; %bb.51:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v31, v2, 17, 2
	v_bfe_u32 v30, v2, 16, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr29
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v31
	s_xor_b32 s0, exec_lo, s0
; %bb.52:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v29, 22, v30
                                        ; implicit-def: $vgpr30
	v_lshl_or_b32 v29, v31, 23, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v29, 0.5, v29
; %bb.53:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.54:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v30
	v_cndmask_b32_e64 v29, 0.5, 0, vcc_lo
; %bb.55:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v30, v[7:8], off offset:24
	v_bfe_u32 v33, v2, 21, 2
	v_bfe_u32 v32, v2, 20, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr31
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v33
	s_xor_b32 s0, exec_lo, s0
; %bb.56:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 22, v32
                                        ; implicit-def: $vgpr32
	v_lshl_or_b32 v31, v33, 23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 0.5, v31
; %bb.57:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.58:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v32
	v_cndmask_b32_e64 v31, 0.5, 0, vcc_lo
; %bb.59:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v34, v2, 25, 2
	v_bfe_u32 v33, v2, 24, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr32
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v34
	s_xor_b32 s0, exec_lo, s0
; %bb.60:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v32, 22, v33
                                        ; implicit-def: $vgpr33
	v_lshl_or_b32 v32, v34, 23, v32
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v32, 0.5, v32
; %bb.61:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.62:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v33
	v_cndmask_b32_e64 v32, 0.5, 0, vcc_lo
; %bb.63:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v33, v[7:8], off offset:28
	v_bfe_u32 v36, v2, 29, 2
	v_bfe_u32 v35, v2, 28, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr34
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v36
	s_xor_b32 s0, exec_lo, s0
; %bb.64:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v34, 22, v35
                                        ; implicit-def: $vgpr35
	v_lshl_or_b32 v34, v36, 23, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, 0.5, v34
; %bb.65:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.66:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v35
	v_cndmask_b32_e64 v34, 0.5, 0, vcc_lo
; %bb.67:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v37, v3, 1, 2
	v_and_b32_e32 v36, 1, v3
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr35
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v37
	s_xor_b32 s0, exec_lo, s0
; %bb.68:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v35, 22, v36
                                        ; implicit-def: $vgpr36
	v_lshl_or_b32 v35, v37, 23, v35
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v35, 0.5, v35
; %bb.69:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.70:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v36
	v_cndmask_b32_e64 v35, 0.5, 0, vcc_lo
; %bb.71:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v36, v[7:8], off offset:32
	v_bfe_u32 v39, v3, 5, 2
	v_bfe_u32 v38, v3, 4, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr37
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v39
	s_xor_b32 s0, exec_lo, s0
; %bb.72:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v37, 22, v38
                                        ; implicit-def: $vgpr38
	v_lshl_or_b32 v37, v39, 23, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v37, 0.5, v37
; %bb.73:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.74:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v38
	v_cndmask_b32_e64 v37, 0.5, 0, vcc_lo
; %bb.75:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v40, v3, 9, 2
	v_bfe_u32 v39, v3, 8, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr38
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v40
	s_xor_b32 s0, exec_lo, s0
; %bb.76:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v38, 22, v39
                                        ; implicit-def: $vgpr39
	v_lshl_or_b32 v38, v40, 23, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v38, 0.5, v38
; %bb.77:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.78:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v39
	v_cndmask_b32_e64 v38, 0.5, 0, vcc_lo
; %bb.79:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v39, v[7:8], off offset:36
	v_bfe_u32 v42, v3, 13, 2
	v_bfe_u32 v41, v3, 12, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr40
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v42
	s_xor_b32 s0, exec_lo, s0
; %bb.80:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v40, 22, v41
                                        ; implicit-def: $vgpr41
	v_lshl_or_b32 v40, v42, 23, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v40, 0.5, v40
; %bb.81:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.82:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v41
	v_cndmask_b32_e64 v40, 0.5, 0, vcc_lo
; %bb.83:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v43, v3, 17, 2
	v_bfe_u32 v42, v3, 16, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr41
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v43
	s_xor_b32 s0, exec_lo, s0
; %bb.84:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v41, 22, v42
                                        ; implicit-def: $vgpr42
	v_lshl_or_b32 v41, v43, 23, v41
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v41, 0.5, v41
; %bb.85:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.86:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v42
	v_cndmask_b32_e64 v41, 0.5, 0, vcc_lo
; %bb.87:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v42, v[7:8], off offset:40
	v_bfe_u32 v45, v3, 21, 2
	v_bfe_u32 v44, v3, 20, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr43
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v45
	s_xor_b32 s0, exec_lo, s0
; %bb.88:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v43, 22, v44
                                        ; implicit-def: $vgpr44
	v_lshl_or_b32 v43, v45, 23, v43
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v43, 0.5, v43
; %bb.89:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.90:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v44
	v_cndmask_b32_e64 v43, 0.5, 0, vcc_lo
; %bb.91:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v46, v3, 25, 2
	v_bfe_u32 v45, v3, 24, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr44
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v46
	s_xor_b32 s0, exec_lo, s0
; %bb.92:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v44, 22, v45
                                        ; implicit-def: $vgpr45
	v_lshl_or_b32 v44, v46, 23, v44
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v44, 0.5, v44
; %bb.93:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.94:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v45
	v_cndmask_b32_e64 v44, 0.5, 0, vcc_lo
; %bb.95:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v45, v[7:8], off offset:44
	v_bfe_u32 v48, v3, 29, 2
	v_bfe_u32 v47, v3, 28, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr46
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v48
	s_xor_b32 s0, exec_lo, s0
; %bb.96:                               ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v46, 22, v47
                                        ; implicit-def: $vgpr47
	v_lshl_or_b32 v46, v48, 23, v46
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v46, 0.5, v46
; %bb.97:                               ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.98:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v47
	v_cndmask_b32_e64 v46, 0.5, 0, vcc_lo
; %bb.99:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v49, v4, 1, 2
	v_and_b32_e32 v48, 1, v4
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr47
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v49
	s_xor_b32 s0, exec_lo, s0
; %bb.100:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v47, 22, v48
                                        ; implicit-def: $vgpr48
	v_lshl_or_b32 v47, v49, 23, v47
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v47, 0.5, v47
; %bb.101:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.102:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v48
	v_cndmask_b32_e64 v47, 0.5, 0, vcc_lo
; %bb.103:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v48, v[7:8], off offset:48
	v_bfe_u32 v51, v4, 5, 2
	v_bfe_u32 v50, v4, 4, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr49
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v51
	s_xor_b32 s0, exec_lo, s0
; %bb.104:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v49, 22, v50
                                        ; implicit-def: $vgpr50
	v_lshl_or_b32 v49, v51, 23, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v49, 0.5, v49
; %bb.105:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.106:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v50
	v_cndmask_b32_e64 v49, 0.5, 0, vcc_lo
; %bb.107:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v52, v4, 9, 2
	v_bfe_u32 v51, v4, 8, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr50
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v52
	s_xor_b32 s0, exec_lo, s0
; %bb.108:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v50, 22, v51
                                        ; implicit-def: $vgpr51
	v_lshl_or_b32 v50, v52, 23, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v50, 0.5, v50
; %bb.109:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.110:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v51
	v_cndmask_b32_e64 v50, 0.5, 0, vcc_lo
; %bb.111:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v51, v[7:8], off offset:52
	v_bfe_u32 v54, v4, 13, 2
	v_bfe_u32 v53, v4, 12, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr52
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v54
	s_xor_b32 s0, exec_lo, s0
; %bb.112:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v52, 22, v53
                                        ; implicit-def: $vgpr53
	v_lshl_or_b32 v52, v54, 23, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v52, 0.5, v52
; %bb.113:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.114:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v53
	v_cndmask_b32_e64 v52, 0.5, 0, vcc_lo
; %bb.115:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v55, v4, 17, 2
	v_bfe_u32 v54, v4, 16, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr53
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v55
	s_xor_b32 s0, exec_lo, s0
; %bb.116:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v53, 22, v54
                                        ; implicit-def: $vgpr54
	v_lshl_or_b32 v53, v55, 23, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v53, 0.5, v53
; %bb.117:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.118:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v54
	v_cndmask_b32_e64 v53, 0.5, 0, vcc_lo
; %bb.119:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v54, v[7:8], off offset:56
	v_bfe_u32 v57, v4, 21, 2
	v_bfe_u32 v56, v4, 20, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr55
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v57
	s_xor_b32 s0, exec_lo, s0
; %bb.120:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v55, 22, v56
                                        ; implicit-def: $vgpr56
	v_lshl_or_b32 v55, v57, 23, v55
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v55, 0.5, v55
; %bb.121:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.122:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v56
	v_cndmask_b32_e64 v55, 0.5, 0, vcc_lo
; %bb.123:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	v_bfe_u32 v58, v4, 25, 2
	v_bfe_u32 v57, v4, 24, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr56
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v58
	s_xor_b32 s0, exec_lo, s0
; %bb.124:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v56, 22, v57
                                        ; implicit-def: $vgpr57
	v_lshl_or_b32 v56, v58, 23, v56
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v56, 0.5, v56
; %bb.125:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
; %bb.126:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v57
	v_cndmask_b32_e64 v56, 0.5, 0, vcc_lo
; %bb.127:                              ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s0
	global_load_b32 v7, v[7:8], off offset:60
	v_bfe_u32 v58, v4, 29, 2
	v_bfe_u32 v57, v4, 28, 1
	s_mov_b32 s0, exec_lo
                                        ; implicit-def: $vgpr8
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v58
	s_xor_b32 s0, exec_lo, s0
; %bb.128:                              ;   in Loop: Header=BB0_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 22, v57
                                        ; implicit-def: $vgpr57
	v_lshl_or_b32 v8, v58, 23, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, 0.5, v8
; %bb.129:                              ;   in Loop: Header=BB0_3 Depth=1
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB0_2
; %bb.130:                              ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v57
	v_cndmask_b32_e64 v8, 0.5, 0, vcc_lo
	s_branch .LBB0_2
.LBB0_131:
	v_mov_b32_e32 v10, 0
.LBB0_132:
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, s34, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s35, v1, vcc_lo
	global_store_b32 v[0:1], v10, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel fp4_block
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 1
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr max(totalnumvgprs(fp4_block.num_agpr, fp4_block.num_vgpr), 1, 0)
		.amdhsa_next_free_sgpr max(fp4_block.numbered_sgpr+2, 1, 0)-2
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size 30
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	fp4_block, .Lfunc_end0-fp4_block
                                        ; -- End function
	.set fp4_block.num_vgpr, max(60, amdgpu.max_num_vgpr)
	.set fp4_block.num_agpr, max(0, amdgpu.max_num_agpr)
	.set fp4_block.numbered_sgpr, max(40, amdgpu.max_num_sgpr)
	.set fp4_block.num_named_barrier, max(0, amdgpu.max_num_named_barrier)
	.set fp4_block.private_seg_size, 0
	.set fp4_block.uses_vcc, 1
	.set fp4_block.uses_flat_scratch, 1
	.set fp4_block.has_dyn_sized_stack, 1
	.set fp4_block.has_recursion, 1
	.set fp4_block.has_indirect_call, 1
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3816
; TotalNumSgprs: fp4_block.numbered_sgpr+2
; NumVgprs: fp4_block.num_vgpr
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: (alignto(max(max(totalnumvgprs(fp4_block.num_agpr, fp4_block.num_vgpr), 1, 0), 1), 8)/8)-1
; NumSGPRsForWavesPerEU: max(fp4_block.numbered_sgpr+2, 1, 0)
; NumVGPRsForWavesPerEU: max(totalnumvgprs(fp4_block.num_agpr, fp4_block.num_vgpr), 1, 0)
; Occupancy: occupancy(16, 24, 1536, 10, 16, max(fp4_block.numbered_sgpr+extrasgprs(fp4_block.uses_vcc, fp4_block.uses_flat_scratch, 0), 1, 0), max(totalnumvgprs(fp4_block.num_agpr, fp4_block.num_vgpr), 1, 0))
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
	.text
	.protected	__clang_ocl_kern_imp_fp4_block ; -- Begin function __clang_ocl_kern_imp_fp4_block
	.globl	__clang_ocl_kern_imp_fp4_block
	.p2align	2
	.type	__clang_ocl_kern_imp_fp4_block,@function
__clang_ocl_kern_imp_fp4_block:         ; @__clang_ocl_kern_imp_fp4_block
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s0, s33
	s_mov_b32 s33, s32
	s_or_saveexec_b32 s1, -1
	scratch_store_b32 off, v56, s33 offset:32 ; 4-byte Folded Spill
	s_mov_b32 exec_lo, s1
	v_writelane_b32 v56, s0, 2
	s_clause 0x7                            ; 32-byte Folded Spill
	scratch_store_b32 off, v40, s33 offset:28
	scratch_store_b32 off, v41, s33 offset:24
	scratch_store_b32 off, v42, s33 offset:20
	scratch_store_b32 off, v43, s33 offset:16
	scratch_store_b32 off, v44, s33 offset:12
	scratch_store_b32 off, v45, s33 offset:8
	scratch_store_b32 off, v46, s33 offset:4
	scratch_store_b32 off, v47, s33
	v_dual_mov_b32 v46, v0 :: v_dual_mov_b32 v47, 0
	v_mov_b32_e32 v0, 0
	v_writelane_b32 v56, s30, 0
	s_add_i32 s32, s32, 48
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z13get_global_idj@rel32@lo+4
	s_addc_u32 s1, s1, _Z13get_global_idj@rel32@hi+12
	v_dual_mov_b32 v42, v6 :: v_dual_mov_b32 v41, v4
	v_writelane_b32 v56, s31, 1
	v_dual_mov_b32 v40, v5 :: v_dual_mov_b32 v43, v3
	v_dual_mov_b32 v44, v2 :: v_dual_mov_b32 v45, v1
	s_swappc_b64 s[30:31], s[0:1]
	s_mov_b32 s0, exec_lo
	v_cmpx_ne_u32_e32 0, v42
	s_cbranch_execz .LBB1_132
; %bb.1:
	v_mul_lo_u32 v5, v42, v0
	v_mov_b32_e32 v6, 0
	s_mov_b32 s1, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v9, 5, v5
	v_mov_b32_e32 v47, v6
	s_branch .LBB1_3
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_lshlrev_b32_e32 v80, 28, v1
	v_lshrrev_b32_e32 v81, 4, v1
	v_lshrrev_b32_e32 v82, 8, v1
	v_add_nc_u32_e32 v42, -1, v42
	v_add_nc_u32_e32 v9, 32, v9
	v_and_or_b32 v10, 0x80000000, v80, v10
	v_lshlrev_b32_e32 v80, 28, v81
	v_lshrrev_b32_e32 v81, 12, v1
	v_lshlrev_b32_e32 v82, 28, v82
	v_cmp_eq_u32_e32 vcc_lo, 0, v42
	s_waitcnt vmcnt(15)
	v_fma_mix_f32 v10, v10, v11, v47 op_sel_hi:[0,1,0]
	v_and_or_b32 v12, 0x80000000, v80, v12
	v_mov_b16_e32 v80.h, 0
	v_mov_b16_e32 v80.l, v1.h
	v_lshlrev_b32_e32 v81, 28, v81
	v_and_or_b32 v13, 0x80000000, v82, v13
	v_fma_mix_f32 v10, v12, v11, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v1
	v_lshlrev_b32_e32 v12, 28, v80
	v_and_or_b32 v15, 0x80000000, v81, v15
	v_mov_b16_e32 v80.l, v4.h
	s_waitcnt vmcnt(14)
	v_fma_mix_f32 v10, v13, v14, v10 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v13, 24, v1
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v12, 0x80000000, v12, v16
	v_and_or_b32 v1, 0x80000000, v1, v21
	v_fma_mix_f32 v10, v15, v14, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v13, 28, v13
	v_and_or_b32 v11, 0x80000000, v11, v18
	v_add_nc_u32_e32 v5, 1, v5
	s_or_b32 s1, vcc_lo, s1
	s_waitcnt vmcnt(13)
	v_fma_mix_f32 v10, v12, v17, v10 op_sel_hi:[0,1,0]
	v_and_or_b32 v12, 0x80000000, v13, v19
	v_lshlrev_b32_e32 v13, 28, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v10, v11, v17, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 4, v2
	v_and_or_b32 v13, 0x80000000, v13, v22
	s_waitcnt vmcnt(12)
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v10, v12, v20, v10 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v12, 8, v2
	v_lshlrev_b32_e32 v11, 28, v11
	v_fma_mix_f32 v1, v1, v20, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 12, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v12, 28, v12
	v_and_or_b32 v11, 0x80000000, v11, v24
	s_waitcnt vmcnt(11)
	v_fma_mix_f32 v1, v13, v23, v1 op_sel_hi:[0,1,0]
	v_mov_b16_e32 v13.l, v2.h
	v_mov_b16_e32 v13.h, v80.h
	v_lshlrev_b32_e32 v10, 28, v10
	v_and_or_b32 v12, 0x80000000, v12, v25
	v_fma_mix_f32 v1, v11, v23, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v2
	v_lshlrev_b32_e32 v13, 28, v13
	v_and_or_b32 v10, 0x80000000, v10, v27
	s_waitcnt vmcnt(10)
	v_fma_mix_f32 v1, v12, v26, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v12, 24, v2
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v13, 0x80000000, v13, v28
	v_and_or_b32 v2, 0x80000000, v2, v33
	v_fma_mix_f32 v1, v10, v26, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v10, 28, v12
	v_and_or_b32 v11, 0x80000000, v11, v30
	v_lshlrev_b32_e32 v12, 28, v3
	s_waitcnt vmcnt(9)
	v_fma_mix_f32 v1, v13, v29, v1 op_sel_hi:[0,1,0]
	v_and_or_b32 v10, 0x80000000, v10, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v12, 0x80000000, v12, v34
	v_fma_mix_f32 v1, v11, v29, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 4, v3
	s_waitcnt vmcnt(8)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v10, v32, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 8, v3
	v_lshlrev_b32_e32 v11, 28, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v2, v32, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 12, v3
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v11, 0x80000000, v11, v36
	s_waitcnt vmcnt(7)
	v_fma_mix_f32 v1, v12, v35, v1 op_sel_hi:[0,1,0]
	v_mov_b16_e32 v12.l, v3.h
	v_mov_b16_e32 v12.h, v80.h
	v_lshlrev_b32_e32 v2, 28, v2
	v_and_or_b32 v10, 0x80000000, v10, v37
	v_fma_mix_f32 v1, v11, v35, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v11, 20, v3
	v_lshlrev_b32_e32 v12, 28, v12
	v_and_or_b32 v2, 0x80000000, v2, v39
	s_waitcnt vmcnt(6)
	v_fma_mix_f32 v1, v10, v38, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 24, v3
	v_lshlrev_b32_e32 v11, 28, v11
	v_and_or_b32 v12, 0x80000000, v12, v48
	v_and_or_b32 v3, 0x80000000, v3, v53
	v_fma_mix_f32 v1, v2, v38, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v2, 28, v10
	v_and_or_b32 v10, 0x80000000, v11, v50
	v_lshlrev_b32_e32 v11, 28, v4
	s_waitcnt vmcnt(5)
	v_fma_mix_f32 v1, v12, v49, v1 op_sel_hi:[0,1,0]
	v_and_or_b32 v2, 0x80000000, v2, v51
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_or_b32 v11, 0x80000000, v11, v54
	v_fma_mix_f32 v1, v10, v49, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 4, v4
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v2, v52, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 8, v4
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v3, v52, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v3, 12, v4
	v_lshlrev_b32_e32 v2, 28, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_and_or_b32 v10, 0x80000000, v10, v64
	s_waitcnt vmcnt(3)
	v_fma_mix_f32 v1, v11, v55, v1 op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v3, 28, v3
	v_and_or_b32 v2, 0x80000000, v2, v65
	v_lshlrev_b32_e32 v11, 28, v80
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_fma_mix_f32 v1, v10, v55, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v10, 20, v4
	v_and_or_b32 v3, 0x80000000, v3, v67
	v_and_or_b32 v11, 0x80000000, v11, v68
	s_waitcnt vmcnt(2)
	v_fma_mix_f32 v1, v2, v66, v1 op_sel_hi:[0,1,0]
	v_lshrrev_b32_e32 v2, 24, v4
	v_lshlrev_b32_e32 v10, 28, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_fma_mix_f32 v1, v3, v66, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_lshlrev_b32_e32 v2, 28, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_and_or_b32 v3, 0x80000000, v10, v70
	s_waitcnt vmcnt(1)
	v_fma_mix_f32 v1, v11, v69, v1 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_and_or_b32 v2, 0x80000000, v2, v71
	v_fma_mix_f32 v1, v3, v69, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_and_or_b32 v3, 0x80000000, v4, v8
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v2, v7, v1 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v47, v3, v7, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_and_not1_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB1_131
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64 v[1:2], 4, v[5:6]
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr10
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v1, vcc_lo, v46, v1
	v_add_co_ci_u32_e64 v2, null, v45, v2, vcc_lo
	global_load_b128 v[1:4], v[1:2], off
	s_waitcnt vmcnt(0)
	v_bfe_u32 v8, v1, 1, 2
	v_and_b32_e32 v7, 1, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v8
	s_xor_b32 s2, exec_lo, s2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v7, 22, v7
	v_lshl_or_b32 v7, v8, 23, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v10, 0.5, v7
                                        ; implicit-def: $vgpr7
; %bb.5:                                ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.6:                                ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v7
	v_cndmask_b32_e64 v10, 0.5, 0, vcc_lo
; %bb.7:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_lshrrev_b32_e32 v7, 1, v9
	v_mov_b32_e32 v8, v6
	v_bfe_u32 v14, v1, 5, 2
	v_bfe_u32 v13, v1, 4, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[7:8], 2, v[7:8]
	v_add_co_u32 v7, vcc_lo, v44, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, v43, v8, vcc_lo
	global_load_b32 v11, v[7:8], off
	v_cmpx_ne_u32_e32 0, v14
	s_xor_b32 s2, exec_lo, s2
; %bb.8:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshlrev_b32_e32 v12, 22, v13
                                        ; implicit-def: $vgpr13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_or_b32 v12, v14, 23, v12
	v_add_nc_u32_e32 v12, 0.5, v12
; %bb.9:                                ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.10:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v13
	v_cndmask_b32_e64 v12, 0.5, 0, vcc_lo
; %bb.11:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v15, v1, 9, 2
	v_bfe_u32 v14, v1, 8, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr13
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v15
	s_xor_b32 s2, exec_lo, s2
; %bb.12:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v13, 22, v14
                                        ; implicit-def: $vgpr14
	v_lshl_or_b32 v13, v15, 23, v13
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v13, 0.5, v13
; %bb.13:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.14:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	v_cndmask_b32_e64 v13, 0.5, 0, vcc_lo
; %bb.15:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v14, v[7:8], off offset:4
	v_bfe_u32 v17, v1, 13, 2
	v_bfe_u32 v16, v1, 12, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr15
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v17
	s_xor_b32 s2, exec_lo, s2
; %bb.16:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v15, 22, v16
                                        ; implicit-def: $vgpr16
	v_lshl_or_b32 v15, v17, 23, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v15, 0.5, v15
; %bb.17:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.18:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v16
	v_cndmask_b32_e64 v15, 0.5, 0, vcc_lo
; %bb.19:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v18, v1, 17, 2
	v_bfe_u32 v17, v1, 16, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr16
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v18
	s_xor_b32 s2, exec_lo, s2
; %bb.20:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v16, 22, v17
                                        ; implicit-def: $vgpr17
	v_lshl_or_b32 v16, v18, 23, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v16, 0.5, v16
; %bb.21:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.22:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v17
	v_cndmask_b32_e64 v16, 0.5, 0, vcc_lo
; %bb.23:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v17, v[7:8], off offset:8
	v_bfe_u32 v20, v1, 21, 2
	v_bfe_u32 v19, v1, 20, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr18
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v20
	s_xor_b32 s2, exec_lo, s2
; %bb.24:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v18, 22, v19
                                        ; implicit-def: $vgpr19
	v_lshl_or_b32 v18, v20, 23, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v18, 0.5, v18
; %bb.25:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.26:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v19
	v_cndmask_b32_e64 v18, 0.5, 0, vcc_lo
; %bb.27:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v21, v1, 25, 2
	v_bfe_u32 v20, v1, 24, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr19
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v21
	s_xor_b32 s2, exec_lo, s2
; %bb.28:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v19, 22, v20
                                        ; implicit-def: $vgpr20
	v_lshl_or_b32 v19, v21, 23, v19
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v19, 0.5, v19
; %bb.29:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.30:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v20
	v_cndmask_b32_e64 v19, 0.5, 0, vcc_lo
; %bb.31:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v20, v[7:8], off offset:12
	v_bfe_u32 v23, v1, 29, 2
	v_bfe_u32 v22, v1, 28, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr21
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v23
	s_xor_b32 s2, exec_lo, s2
; %bb.32:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v21, 22, v22
                                        ; implicit-def: $vgpr22
	v_lshl_or_b32 v21, v23, 23, v21
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v21, 0.5, v21
; %bb.33:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.34:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v22
	v_cndmask_b32_e64 v21, 0.5, 0, vcc_lo
; %bb.35:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v24, v2, 1, 2
	v_and_b32_e32 v23, 1, v2
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr22
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v24
	s_xor_b32 s2, exec_lo, s2
; %bb.36:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v22, 22, v23
                                        ; implicit-def: $vgpr23
	v_lshl_or_b32 v22, v24, 23, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v22, 0.5, v22
; %bb.37:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.38:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v23
	v_cndmask_b32_e64 v22, 0.5, 0, vcc_lo
; %bb.39:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v23, v[7:8], off offset:16
	v_bfe_u32 v26, v2, 5, 2
	v_bfe_u32 v25, v2, 4, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr24
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v26
	s_xor_b32 s2, exec_lo, s2
; %bb.40:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v24, 22, v25
                                        ; implicit-def: $vgpr25
	v_lshl_or_b32 v24, v26, 23, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v24, 0.5, v24
; %bb.41:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.42:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v25
	v_cndmask_b32_e64 v24, 0.5, 0, vcc_lo
; %bb.43:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v27, v2, 9, 2
	v_bfe_u32 v26, v2, 8, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr25
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v27
	s_xor_b32 s2, exec_lo, s2
; %bb.44:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v25, 22, v26
                                        ; implicit-def: $vgpr26
	v_lshl_or_b32 v25, v27, 23, v25
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v25, 0.5, v25
; %bb.45:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.46:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v26
	v_cndmask_b32_e64 v25, 0.5, 0, vcc_lo
; %bb.47:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v26, v[7:8], off offset:20
	v_bfe_u32 v29, v2, 13, 2
	v_bfe_u32 v28, v2, 12, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr27
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v29
	s_xor_b32 s2, exec_lo, s2
; %bb.48:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v27, 22, v28
                                        ; implicit-def: $vgpr28
	v_lshl_or_b32 v27, v29, 23, v27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v27, 0.5, v27
; %bb.49:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.50:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v28
	v_cndmask_b32_e64 v27, 0.5, 0, vcc_lo
; %bb.51:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v30, v2, 17, 2
	v_bfe_u32 v29, v2, 16, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr28
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v30
	s_xor_b32 s2, exec_lo, s2
; %bb.52:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v28, 22, v29
                                        ; implicit-def: $vgpr29
	v_lshl_or_b32 v28, v30, 23, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v28, 0.5, v28
; %bb.53:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.54:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v29
	v_cndmask_b32_e64 v28, 0.5, 0, vcc_lo
; %bb.55:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v29, v[7:8], off offset:24
	v_bfe_u32 v32, v2, 21, 2
	v_bfe_u32 v31, v2, 20, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr30
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v32
	s_xor_b32 s2, exec_lo, s2
; %bb.56:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v30, 22, v31
                                        ; implicit-def: $vgpr31
	v_lshl_or_b32 v30, v32, 23, v30
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v30, 0.5, v30
; %bb.57:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.58:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v31
	v_cndmask_b32_e64 v30, 0.5, 0, vcc_lo
; %bb.59:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v33, v2, 25, 2
	v_bfe_u32 v32, v2, 24, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr31
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v33
	s_xor_b32 s2, exec_lo, s2
; %bb.60:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v31, 22, v32
                                        ; implicit-def: $vgpr32
	v_lshl_or_b32 v31, v33, 23, v31
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v31, 0.5, v31
; %bb.61:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.62:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v32
	v_cndmask_b32_e64 v31, 0.5, 0, vcc_lo
; %bb.63:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v32, v[7:8], off offset:28
	v_bfe_u32 v35, v2, 29, 2
	v_bfe_u32 v34, v2, 28, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr33
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v35
	s_xor_b32 s2, exec_lo, s2
; %bb.64:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v33, 22, v34
                                        ; implicit-def: $vgpr34
	v_lshl_or_b32 v33, v35, 23, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v33, 0.5, v33
; %bb.65:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.66:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v34
	v_cndmask_b32_e64 v33, 0.5, 0, vcc_lo
; %bb.67:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v36, v3, 1, 2
	v_and_b32_e32 v35, 1, v3
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr34
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v36
	s_xor_b32 s2, exec_lo, s2
; %bb.68:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v34, 22, v35
                                        ; implicit-def: $vgpr35
	v_lshl_or_b32 v34, v36, 23, v34
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v34, 0.5, v34
; %bb.69:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.70:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v35
	v_cndmask_b32_e64 v34, 0.5, 0, vcc_lo
; %bb.71:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v35, v[7:8], off offset:32
	v_bfe_u32 v38, v3, 5, 2
	v_bfe_u32 v37, v3, 4, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr36
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v38
	s_xor_b32 s2, exec_lo, s2
; %bb.72:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 22, v37
                                        ; implicit-def: $vgpr37
	v_lshl_or_b32 v36, v38, 23, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v36, 0.5, v36
; %bb.73:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.74:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v37
	v_cndmask_b32_e64 v36, 0.5, 0, vcc_lo
; %bb.75:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v39, v3, 9, 2
	v_bfe_u32 v38, v3, 8, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr37
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v39
	s_xor_b32 s2, exec_lo, s2
; %bb.76:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v37, 22, v38
                                        ; implicit-def: $vgpr38
	v_lshl_or_b32 v37, v39, 23, v37
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v37, 0.5, v37
; %bb.77:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.78:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v38
	v_cndmask_b32_e64 v37, 0.5, 0, vcc_lo
; %bb.79:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v38, v[7:8], off offset:36
	v_bfe_u32 v49, v3, 13, 2
	v_bfe_u32 v48, v3, 12, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr39
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v49
	s_xor_b32 s2, exec_lo, s2
; %bb.80:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v39, 22, v48
                                        ; implicit-def: $vgpr48
	v_lshl_or_b32 v39, v49, 23, v39
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v39, 0.5, v39
; %bb.81:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.82:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v48
	v_cndmask_b32_e64 v39, 0.5, 0, vcc_lo
; %bb.83:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v50, v3, 17, 2
	v_bfe_u32 v49, v3, 16, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr48
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v50
	s_xor_b32 s2, exec_lo, s2
; %bb.84:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v48, 22, v49
                                        ; implicit-def: $vgpr49
	v_lshl_or_b32 v48, v50, 23, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v48, 0.5, v48
; %bb.85:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.86:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v49
	v_cndmask_b32_e64 v48, 0.5, 0, vcc_lo
; %bb.87:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v49, v[7:8], off offset:40
	v_bfe_u32 v52, v3, 21, 2
	v_bfe_u32 v51, v3, 20, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr50
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v52
	s_xor_b32 s2, exec_lo, s2
; %bb.88:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v50, 22, v51
                                        ; implicit-def: $vgpr51
	v_lshl_or_b32 v50, v52, 23, v50
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v50, 0.5, v50
; %bb.89:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.90:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v51
	v_cndmask_b32_e64 v50, 0.5, 0, vcc_lo
; %bb.91:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v53, v3, 25, 2
	v_bfe_u32 v52, v3, 24, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr51
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v53
	s_xor_b32 s2, exec_lo, s2
; %bb.92:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v51, 22, v52
                                        ; implicit-def: $vgpr52
	v_lshl_or_b32 v51, v53, 23, v51
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v51, 0.5, v51
; %bb.93:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.94:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v52
	v_cndmask_b32_e64 v51, 0.5, 0, vcc_lo
; %bb.95:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v52, v[7:8], off offset:44
	v_bfe_u32 v55, v3, 29, 2
	v_bfe_u32 v54, v3, 28, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr53
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v55
	s_xor_b32 s2, exec_lo, s2
; %bb.96:                               ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v53, 22, v54
                                        ; implicit-def: $vgpr54
	v_lshl_or_b32 v53, v55, 23, v53
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v53, 0.5, v53
; %bb.97:                               ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.98:                               ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v54
	v_cndmask_b32_e64 v53, 0.5, 0, vcc_lo
; %bb.99:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v64, v4, 1, 2
	v_and_b32_e32 v55, 1, v4
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr54
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v64
	s_xor_b32 s2, exec_lo, s2
; %bb.100:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v54, 22, v55
                                        ; implicit-def: $vgpr55
	v_lshl_or_b32 v54, v64, 23, v54
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v54, 0.5, v54
; %bb.101:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.102:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v55
	v_cndmask_b32_e64 v54, 0.5, 0, vcc_lo
; %bb.103:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v55, v[7:8], off offset:48
	v_bfe_u32 v66, v4, 5, 2
	v_bfe_u32 v65, v4, 4, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr64
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v66
	s_xor_b32 s2, exec_lo, s2
; %bb.104:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v64, 22, v65
                                        ; implicit-def: $vgpr65
	v_lshl_or_b32 v64, v66, 23, v64
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v64, 0.5, v64
; %bb.105:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.106:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v65
	v_cndmask_b32_e64 v64, 0.5, 0, vcc_lo
; %bb.107:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v67, v4, 9, 2
	v_bfe_u32 v66, v4, 8, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr65
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v67
	s_xor_b32 s2, exec_lo, s2
; %bb.108:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v65, 22, v66
                                        ; implicit-def: $vgpr66
	v_lshl_or_b32 v65, v67, 23, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v65, 0.5, v65
; %bb.109:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.110:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v66
	v_cndmask_b32_e64 v65, 0.5, 0, vcc_lo
; %bb.111:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v66, v[7:8], off offset:52
	v_bfe_u32 v69, v4, 13, 2
	v_bfe_u32 v68, v4, 12, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr67
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v69
	s_xor_b32 s2, exec_lo, s2
; %bb.112:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v67, 22, v68
                                        ; implicit-def: $vgpr68
	v_lshl_or_b32 v67, v69, 23, v67
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v67, 0.5, v67
; %bb.113:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.114:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v68
	v_cndmask_b32_e64 v67, 0.5, 0, vcc_lo
; %bb.115:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v70, v4, 17, 2
	v_bfe_u32 v69, v4, 16, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr68
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v70
	s_xor_b32 s2, exec_lo, s2
; %bb.116:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v68, 22, v69
                                        ; implicit-def: $vgpr69
	v_lshl_or_b32 v68, v70, 23, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v68, 0.5, v68
; %bb.117:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.118:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v69
	v_cndmask_b32_e64 v68, 0.5, 0, vcc_lo
; %bb.119:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v69, v[7:8], off offset:56
	v_bfe_u32 v80, v4, 21, 2
	v_bfe_u32 v71, v4, 20, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr70
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v80
	s_xor_b32 s2, exec_lo, s2
; %bb.120:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v70, 22, v71
                                        ; implicit-def: $vgpr71
	v_lshl_or_b32 v70, v80, 23, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v70, 0.5, v70
; %bb.121:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.122:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v71
	v_cndmask_b32_e64 v70, 0.5, 0, vcc_lo
; %bb.123:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	v_bfe_u32 v81, v4, 25, 2
	v_bfe_u32 v80, v4, 24, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr71
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v81
	s_xor_b32 s2, exec_lo, s2
; %bb.124:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v71, 22, v80
                                        ; implicit-def: $vgpr80
	v_lshl_or_b32 v71, v81, 23, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v71, 0.5, v71
; %bb.125:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
; %bb.126:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v80
	v_cndmask_b32_e64 v71, 0.5, 0, vcc_lo
; %bb.127:                              ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s2
	global_load_b32 v7, v[7:8], off offset:60
	v_bfe_u32 v81, v4, 29, 2
	v_bfe_u32 v80, v4, 28, 1
	s_mov_b32 s2, exec_lo
                                        ; implicit-def: $vgpr8
	s_delay_alu instid0(VALU_DEP_2)
	v_cmpx_ne_u32_e32 0, v81
	s_xor_b32 s2, exec_lo, s2
; %bb.128:                              ;   in Loop: Header=BB1_3 Depth=1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v8, 22, v80
                                        ; implicit-def: $vgpr80
	v_lshl_or_b32 v8, v81, 23, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v8, 0.5, v8
; %bb.129:                              ;   in Loop: Header=BB1_3 Depth=1
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB1_2
; %bb.130:                              ;   in Loop: Header=BB1_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 0, v80
	v_cndmask_b32_e64 v8, 0.5, 0, vcc_lo
	s_branch .LBB1_2
.LBB1_131:
	s_or_b32 exec_lo, exec_lo, s1
.LBB1_132:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s0
	v_mov_b32_e32 v1, 0
	v_readlane_b32 s31, v56, 1
	v_readlane_b32 s30, v56, 0
	s_mov_b32 s32, s33
	v_readlane_b32 s0, v56, 2
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v0, vcc_lo, v41, v0
	v_add_co_ci_u32_e64 v1, null, v40, v1, vcc_lo
	global_store_b32 v[0:1], v47, off
	s_clause 0x7                            ; 32-byte Folded Reload
	scratch_load_b32 v47, off, s33
	scratch_load_b32 v46, off, s33 offset:4
	scratch_load_b32 v45, off, s33 offset:8
	scratch_load_b32 v44, off, s33 offset:12
	scratch_load_b32 v43, off, s33 offset:16
	scratch_load_b32 v42, off, s33 offset:20
	scratch_load_b32 v41, off, s33 offset:24
	scratch_load_b32 v40, off, s33 offset:28
	s_or_saveexec_b32 s1, -1
	scratch_load_b32 v56, off, s33 offset:32 ; 4-byte Folded Reload
	s_mov_b32 exec_lo, s1
	s_mov_b32 s33, s0
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	__clang_ocl_kern_imp_fp4_block, .Lfunc_end1-__clang_ocl_kern_imp_fp4_block
                                        ; -- End function
	.set __clang_ocl_kern_imp_fp4_block.num_vgpr, max(83, amdgpu.max_num_vgpr)
	.set __clang_ocl_kern_imp_fp4_block.num_agpr, max(0, amdgpu.max_num_agpr)
	.set __clang_ocl_kern_imp_fp4_block.numbered_sgpr, max(34, amdgpu.max_num_sgpr)
	.set __clang_ocl_kern_imp_fp4_block.num_named_barrier, max(0, amdgpu.max_num_named_barrier)
	.set __clang_ocl_kern_imp_fp4_block.private_seg_size, 48
	.set __clang_ocl_kern_imp_fp4_block.uses_vcc, 1
	.set __clang_ocl_kern_imp_fp4_block.uses_flat_scratch, 1
	.set __clang_ocl_kern_imp_fp4_block.has_dyn_sized_stack, 1
	.set __clang_ocl_kern_imp_fp4_block.has_recursion, 1
	.set __clang_ocl_kern_imp_fp4_block.has_indirect_call, 1
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 4044
; TotalNumSgprs: __clang_ocl_kern_imp_fp4_block.numbered_sgpr+2
; NumVgprs: max(83, amdgpu.max_num_vgpr)
; ScratchSize: 48
; MemoryBound: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 83
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 34
	.set amdgpu.max_num_named_barrier, 0
	.text
	.hidden	_Z13get_global_idj
	.ident	"clang version 22.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .is_const:       true
        .offset:         0
        .size:           8
        .type_name:      'uint4*'
        .value_kind:     global_buffer
      - .address_space:  global
        .is_const:       true
        .offset:         8
        .size:           8
        .type_name:      'half2*'
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .type_name:      uint
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         112
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         120
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         128
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         136
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         144
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         232
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           fp4_block
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         fp4_block.kd
    .uses_dynamic_stack: true
    .vgpr_count:     83
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
