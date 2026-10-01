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
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	v_mul_lo_u32 v1, s33, v0
	v_mov_b32_e32 v2, 0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, kE2M1@rel32@lo+4
	s_addc_u32 s1, s1, kE2M1@rel32@hi+12
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 5, v1
	v_mov_b32_e32 v5, v2
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64 v[6:7], 4, v[1:2]
	v_add_nc_u32_e32 v1, 1, v1
	s_add_i32 s33, s33, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	s_cmp_lg_u32 s33, 0
	v_add_co_u32 v6, vcc_lo, s36, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v7, null, s37, v7, vcc_lo
	global_load_b128 v[6:9], v[6:7], off
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v15, 6, v6
	v_mov_b32_e32 v4, v2
	v_lshrrev_b32_e32 v14, 2, v6
	v_lshrrev_b32_e32 v16, 10, v6
	v_lshrrev_b32_e32 v24, 2, v8
	v_and_b32_e32 v15, 60, v15
	v_lshlrev_b64 v[10:11], 1, v[3:4]
	v_and_b32_e32 v4, 15, v6
	v_lshrrev_b32_e32 v46, 2, v9
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v16, 60, v16
	v_and_b32_e32 v24, 60, v24
	v_add_co_u32 v22, vcc_lo, s38, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v23, null, s39, v11, vcc_lo
	v_lshlrev_b32_e32 v4, 2, v4
	v_and_b32_e32 v46, 60, v46
	s_clause 0x5
	global_load_b32 v26, v14, s[0:1]
	global_load_b32 v27, v15, s[0:1]
	global_load_b32 v28, v16, s[0:1]
	global_load_b32 v4, v4, s[0:1]
	global_load_b32 v41, v24, s[0:1]
	global_load_b32 v46, v46, s[0:1]
	global_load_b128 v[10:13], v[22:23], off
	v_lshrrev_b32_e32 v14, 14, v6
	v_lshrrev_b32_e32 v15, 18, v6
	v_lshrrev_b32_e32 v20, 6, v7
	v_lshrrev_b32_e32 v42, 18, v8
	v_lshrrev_b32_e32 v16, 22, v6
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v15, 60, v15
	v_and_b32_e32 v20, 60, v20
	v_and_b32_e32 v42, 60, v42
	v_lshrrev_b32_e32 v18, 2, v7
	s_clause 0x3
	global_load_b32 v29, v14, s[0:1]
	global_load_b32 v30, v15, s[0:1]
	global_load_b32 v34, v20, s[0:1]
	global_load_b32 v42, v42, s[0:1]
	v_and_b32_e32 v15, 15, v7
	v_lshrrev_b32_e32 v24, 14, v8
	v_lshrrev_b32_e32 v48, 6, v9
	v_and_b32_e32 v14, 60, v16
	v_and_b32_e32 v18, 60, v18
	v_lshlrev_b32_e32 v19, 2, v15
	v_and_b32_e32 v24, 60, v24
	v_and_b32_e32 v48, 60, v48
	global_load_b32 v31, v14, s[0:1]
	v_lshrrev_b32_e32 v39, 6, v8
	s_clause 0x3
	global_load_b32 v32, v19, s[0:1]
	global_load_b32 v44, v24, s[0:1]
	global_load_b32 v48, v48, s[0:1]
	global_load_b32 v33, v18, s[0:1]
	v_lshrrev_b32_e32 v19, 10, v7
	v_lshrrev_b32_e32 v18, 14, v7
	v_lshrrev_b32_e32 v51, 18, v9
	v_and_b32_e32 v39, 60, v39
	v_lshrrev_b32_e32 v6, 26, v6
	v_and_b32_e32 v19, 60, v19
	v_and_b32_e32 v18, 60, v18
	v_and_b32_e32 v51, 60, v51
	global_load_b32 v39, v39, s[0:1]
	v_and_b32_e32 v6, 60, v6
	global_load_b32 v35, v19, s[0:1]
	v_lshrrev_b32_e32 v49, 10, v9
	s_clause 0x1
	global_load_b32 v51, v51, s[0:1]
	global_load_b32 v36, v18, s[0:1]
	v_lshrrev_b32_e32 v19, 22, v7
	global_load_b32 v6, v6, s[0:1]
	global_load_b128 v[14:17], v[22:23], off offset:16
	v_and_b32_e32 v49, 60, v49
	v_lshrrev_b32_e32 v50, 14, v9
	v_lshrrev_b32_e32 v20, 18, v7
	v_and_b32_e32 v18, 60, v19
	v_and_b32_e32 v19, 15, v8
	global_load_b32 v49, v49, s[0:1]
	v_and_b32_e32 v50, 60, v50
	v_and_b32_e32 v20, 60, v20
	v_lshrrev_b32_e32 v7, 26, v7
	v_lshlrev_b32_e32 v25, 2, v19
	v_lshrrev_b32_e32 v52, 22, v9
	s_clause 0x1
	global_load_b32 v50, v50, s[0:1]
	global_load_b32 v37, v20, s[0:1]
	v_and_b32_e32 v7, 60, v7
	global_load_b32 v40, v25, s[0:1]
	v_lshrrev_b32_e32 v25, 10, v8
	v_and_b32_e32 v52, 60, v52
	v_add_nc_u32_e32 v3, 32, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_and_b32_e32 v25, 60, v25
	global_load_b32 v43, v25, s[0:1]
	v_lshrrev_b32_e32 v25, 22, v8
	v_lshrrev_b32_e32 v8, 26, v8
	v_and_b32_e32 v24, 60, v25
	v_and_b32_e32 v25, 15, v9
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v8, 60, v8
	v_lshrrev_b32_e32 v9, 26, v9
	global_load_b32 v45, v24, s[0:1]
	v_lshlrev_b32_e32 v47, 2, v25
	v_and_b32_e32 v9, 60, v9
	s_clause 0x4
	global_load_b32 v52, v52, s[0:1]
	global_load_b32 v9, v9, s[0:1]
	global_load_b32 v47, v47, s[0:1]
	global_load_b32 v38, v18, s[0:1]
	global_load_b32 v7, v7, s[0:1]
	global_load_b128 v[18:21], v[22:23], off offset:32
	global_load_b32 v8, v8, s[0:1]
	global_load_b128 v[22:25], v[22:23], off offset:48
	s_waitcnt vmcnt(29)
	v_fma_mix_f32 v4, v4, v10, v5 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v26, v10, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v27, v11, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v28, v11, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(28)
	v_fma_mix_f32 v4, v29, v12, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(27)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v30, v12, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(24)
	v_fma_mix_f32 v4, v31, v13, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(15)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v6, v13, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(14)
	v_fma_mix_f32 v4, v32, v14, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v33, v14, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v34, v15, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v35, v15, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v36, v16, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(11)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v37, v16, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(4)
	v_fma_mix_f32 v4, v38, v17, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(3)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v7, v17, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(2)
	v_fma_mix_f32 v4, v40, v18, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v41, v18, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v39, v19, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v43, v19, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v44, v20, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v42, v20, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v45, v21, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(1)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v8, v21, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, v47, v22, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v46, v22, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v48, v23, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v49, v23, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v50, v24, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v51, v24, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v52, v25, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v5, v9, v25, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_cbranch_scc1 .LBB0_2
	s_branch .LBB0_4
.LBB0_3:
	v_mov_b32_e32 v5, 0
.LBB0_4:
	v_mov_b32_e32 v1, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64 v[0:1], 2, v[0:1]
	v_add_co_u32 v0, vcc_lo, s34, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s35, v1, vcc_lo
	global_store_b32 v[0:1], v5, off
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
		.amdhsa_inst_pref_size 10
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
	.set fp4_block.num_vgpr, max(53, amdgpu.max_num_vgpr)
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
; codeLenInByte = 1244
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
	s_mov_b32 s2, exec_lo
	v_cmpx_ne_u32_e32 0, v42
	s_cbranch_execz .LBB1_4
; %bb.1:
	v_mul_lo_u32 v1, v42, v0
	v_mov_b32_e32 v2, 0
	s_mov_b32 s3, 0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, kE2M1@rel32@lo+4
	s_addc_u32 s1, s1, kE2M1@rel32@hi+12
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v3, 5, v1
	v_mov_b32_e32 v47, v2
.LBB1_2:                                ; =>This Inner Loop Header: Depth=1
	v_lshlrev_b64 v[4:5], 4, v[1:2]
	v_add_nc_u32_e32 v42, -1, v42
	v_add_nc_u32_e32 v1, 1, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v4, vcc_lo, v46, v4
	v_add_co_ci_u32_e64 v5, null, v45, v5, vcc_lo
	global_load_b128 v[5:8], v[4:5], off
	s_waitcnt vmcnt(0)
	v_lshrrev_b32_e32 v13, 2, v5
	v_mov_b32_e32 v4, v2
	v_lshrrev_b32_e32 v14, 6, v5
	v_lshrrev_b32_e32 v65, 14, v8
	v_lshrrev_b32_e32 v66, 18, v8
	v_and_b32_e32 v13, 60, v13
	v_lshlrev_b64 v[9:10], 1, v[3:4]
	v_and_b32_e32 v4, 15, v5
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v65, 60, v65
	v_and_b32_e32 v66, 60, v66
	v_lshrrev_b32_e32 v54, 2, v8
	v_add_co_u32 v21, vcc_lo, v44, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v22, null, v43, v10, vcc_lo
	v_lshlrev_b32_e32 v4, 2, v4
	s_clause 0x1
	global_load_b32 v24, v13, s[0:1]
	global_load_b32 v25, v14, s[0:1]
	global_load_b128 v[9:12], v[21:22], off
	v_lshrrev_b32_e32 v13, 14, v5
	global_load_b32 v23, v4, s[0:1]
	v_lshrrev_b32_e32 v4, 10, v5
	v_lshrrev_b32_e32 v14, 18, v5
	v_lshrrev_b32_e32 v67, 22, v8
	v_and_b32_e32 v13, 60, v13
	v_cmp_eq_u32_e32 vcc_lo, 0, v42
	v_and_b32_e32 v4, 60, v4
	v_and_b32_e32 v14, 60, v14
	v_and_b32_e32 v67, 60, v67
	global_load_b32 v27, v13, s[0:1]
	v_add_nc_u32_e32 v3, 32, v3
	s_clause 0x1
	global_load_b32 v26, v4, s[0:1]
	global_load_b32 v28, v14, s[0:1]
	v_lshrrev_b32_e32 v4, 22, v5
	v_lshrrev_b32_e32 v5, 26, v5
	v_and_b32_e32 v13, 15, v6
	s_or_b32 s3, vcc_lo, s3
	global_load_b32 v65, v65, s[0:1]
	v_and_b32_e32 v4, 60, v4
	v_and_b32_e32 v5, 60, v5
	v_lshlrev_b32_e32 v17, 2, v13
	global_load_b32 v30, v5, s[0:1]
	global_load_b128 v[13:16], v[21:22], off offset:16
	v_lshrrev_b32_e32 v5, 6, v6
	s_clause 0x2
	global_load_b32 v29, v4, s[0:1]
	global_load_b32 v31, v17, s[0:1]
	global_load_b32 v66, v66, s[0:1]
	v_and_b32_e32 v5, 60, v5
	global_load_b32 v33, v5, s[0:1]
	v_lshrrev_b32_e32 v4, 2, v6
	v_lshrrev_b32_e32 v17, 10, v6
	v_lshrrev_b32_e32 v5, 18, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v4, 60, v4
	v_and_b32_e32 v17, 60, v17
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v5, 60, v5
	s_clause 0x2
	global_load_b32 v32, v4, s[0:1]
	global_load_b32 v34, v17, s[0:1]
	global_load_b32 v36, v5, s[0:1]
	v_lshrrev_b32_e32 v4, 14, v6
	v_and_b32_e32 v5, 15, v7
	v_lshrrev_b32_e32 v17, 22, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v4, 60, v4
	v_lshlrev_b32_e32 v5, 2, v5
	s_clause 0x1
	global_load_b32 v35, v4, s[0:1]
	global_load_b32 v39, v5, s[0:1]
	v_lshrrev_b32_e32 v4, 26, v6
	v_and_b32_e32 v6, 60, v17
	v_lshrrev_b32_e32 v5, 10, v7
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v4, 60, v4
	global_load_b32 v38, v4, s[0:1]
	global_load_b128 v[17:20], v[21:22], off offset:32
	v_lshrrev_b32_e32 v4, 6, v7
	global_load_b32 v37, v6, s[0:1]
	v_and_b32_e32 v5, 60, v5
	v_and_b32_e32 v4, 60, v4
	global_load_b32 v50, v5, s[0:1]
	v_lshrrev_b32_e32 v6, 2, v7
	v_lshrrev_b32_e32 v5, 22, v7
	global_load_b32 v49, v4, s[0:1]
	v_lshrrev_b32_e32 v4, 18, v7
	v_and_b32_e32 v6, 60, v6
	v_and_b32_e32 v5, 60, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_and_b32_e32 v4, 60, v4
	s_clause 0x1
	global_load_b32 v48, v6, s[0:1]
	global_load_b32 v53, v5, s[0:1]
	v_lshrrev_b32_e32 v6, 14, v7
	global_load_b32 v52, v4, s[0:1]
	v_and_b32_e32 v4, 15, v8
	v_and_b32_e32 v6, 60, v6
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v55, 2, v4
	global_load_b32 v51, v6, s[0:1]
	v_lshrrev_b32_e32 v6, 26, v7
	v_and_b32_e32 v6, 60, v6
	global_load_b32 v64, v6, s[0:1]
	global_load_b128 v[4:7], v[21:22], off offset:48
	v_lshrrev_b32_e32 v21, 6, v8
	v_and_b32_e32 v22, 60, v54
	global_load_b32 v54, v55, s[0:1]
	v_and_b32_e32 v21, 60, v21
	global_load_b32 v22, v22, s[0:1]
	v_lshrrev_b32_e32 v55, 10, v8
	v_lshrrev_b32_e32 v8, 26, v8
	global_load_b32 v21, v21, s[0:1]
	v_and_b32_e32 v55, 60, v55
	v_and_b32_e32 v8, 60, v8
	s_clause 0x2
	global_load_b32 v67, v67, s[0:1]
	global_load_b32 v8, v8, s[0:1]
	global_load_b32 v55, v55, s[0:1]
	s_waitcnt vmcnt(32)
	v_fma_mix_f32 v23, v23, v9, v47 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v24, v9, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v25, v10, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(30)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v26, v10, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v27, v11, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(29)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v28, v11, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(25)
	v_fma_mix_f32 v9, v29, v12, v9 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v30, v12, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(24)
	v_fma_mix_f32 v9, v31, v13, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(21)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v32, v13, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v33, v14, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(20)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v34, v14, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(18)
	v_fma_mix_f32 v9, v35, v15, v9 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v36, v15, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(14)
	v_fma_mix_f32 v9, v37, v16, v9 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v38, v16, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v39, v17, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(11)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v48, v17, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v49, v18, v9 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v50, v18, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(8)
	v_fma_mix_f32 v9, v51, v19, v9 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v52, v19, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v53, v20, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(7)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v9, v64, v20, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(5)
	v_fma_mix_f32 v9, v54, v4, v9 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(4)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v22, v4, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(3)
	v_fma_mix_f32 v4, v21, v5, v4 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v55, v5, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v65, v6, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v4, v66, v6, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v67, v7, v4 op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v47, v8, v7, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execnz .LBB1_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s3
.LBB1_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s2
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
	.set __clang_ocl_kern_imp_fp4_block.num_vgpr, max(68, amdgpu.max_num_vgpr)
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
; codeLenInByte = 1500
; TotalNumSgprs: __clang_ocl_kern_imp_fp4_block.numbered_sgpr+2
; NumVgprs: max(68, amdgpu.max_num_vgpr)
; ScratchSize: 48
; MemoryBound: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 68
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 34
	.set amdgpu.max_num_named_barrier, 0
	.text
	.hidden	kE2M1                           ; @kE2M1
	.type	kE2M1,@object
	.section	.rodata,"a",@progbits
	.globl	kE2M1
	.p2align	2, 0x0
kE2M1:
	.long	0x00000000                      ; float 0
	.long	0x3f000000                      ; float 0.5
	.long	0x3f800000                      ; float 1
	.long	0x3fc00000                      ; float 1.5
	.long	0x40000000                      ; float 2
	.long	0x40400000                      ; float 3
	.long	0x40800000                      ; float 4
	.long	0x40c00000                      ; float 6
	.long	0x80000000                      ; float -0
	.long	0xbf000000                      ; float -0.5
	.long	0xbf800000                      ; float -1
	.long	0xbfc00000                      ; float -1.5
	.long	0xc0000000                      ; float -2
	.long	0xc0400000                      ; float -3
	.long	0xc0800000                      ; float -4
	.long	0xc0c00000                      ; float -6
	.size	kE2M1, 64

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
        .type_name:      'half*'
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
    .vgpr_count:     68
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
