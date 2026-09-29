BB1:
	s_barrier                                                   ; bfbd0000
	s_waitcnt vmcnt(11)                                         ; bf892ff7
	v_dual_mov_b32 v169, v5 :: v_dual_mov_b32 v168, v4          ; ca100105 a9a80104
	v_dual_mov_b32 v171, v7 :: v_dual_mov_b32 v170, v6          ; ca100107 abaa0106
	ds_store_b128 v1, v[168:171]                                ; db7c0000 0000a801
	s_waitcnt vmcnt(10)                                         ; bf892bf7
	v_dual_mov_b32 v169, v13 :: v_dual_mov_b32 v168, v12        ; ca10010d a9a8010c
	v_dual_mov_b32 v171, v15 :: v_dual_mov_b32 v170, v14        ; ca10010f abaa010e
	ds_store_b128 v1, v[168:171] offset:18432                   ; db7c4800 0000a801
	s_waitcnt vmcnt(9)                                          ; bf8927f7
	v_dual_mov_b32 v169, v61 :: v_dual_mov_b32 v168, v60        ; ca10013d a9a8013c
	v_dual_mov_b32 v171, v63 :: v_dual_mov_b32 v170, v62        ; ca10013f abaa013e
	ds_store_b128 v10, v[168:171]                               ; db7c0000 0000a80a
	s_waitcnt vmcnt(8)                                          ; bf8923f7
	v_dual_mov_b32 v169, v65 :: v_dual_mov_b32 v168, v64        ; ca100141 a9a80140
	v_dual_mov_b32 v171, v67 :: v_dual_mov_b32 v170, v66        ; ca100143 abaa0142
	ds_store_b128 v10, v[168:171] offset:18432                  ; db7c4800 0000a80a
	s_waitcnt vmcnt(7)                                          ; bf891ff7
	v_dual_mov_b32 v169, v69 :: v_dual_mov_b32 v168, v68        ; ca100145 a9a80144
	v_dual_mov_b32 v171, v71 :: v_dual_mov_b32 v170, v70        ; ca100147 abaa0146
	ds_store_b128 v20, v[168:171]                               ; db7c0000 0000a814
	s_waitcnt vmcnt(6)                                          ; bf891bf7
	v_dual_mov_b32 v169, v73 :: v_dual_mov_b32 v168, v72        ; ca100149 a9a80148
	v_dual_mov_b32 v171, v75 :: v_dual_mov_b32 v170, v74        ; ca10014b abaa014a
	ds_store_b128 v20, v[168:171] offset:18432                  ; db7c4800 0000a814
	s_waitcnt vmcnt(5)                                          ; bf8917f7
	v_dual_mov_b32 v169, v77 :: v_dual_mov_b32 v168, v76        ; ca10014d a9a8014c
	v_dual_mov_b32 v171, v79 :: v_dual_mov_b32 v170, v78        ; ca10014f abaa014e
	ds_store_b128 v28, v[168:171]                               ; db7c0000 0000a81c
	s_waitcnt vmcnt(4)                                          ; bf8913f7
	v_dual_mov_b32 v169, v81 :: v_dual_mov_b32 v168, v80        ; ca100151 a9a80150
	v_dual_mov_b32 v171, v83 :: v_dual_mov_b32 v170, v82        ; ca100153 abaa0152
	ds_store_b128 v28, v[168:171] offset:18432                  ; db7c4800 0000a81c
	s_waitcnt vmcnt(3)                                          ; bf890ff7
	v_dual_mov_b32 v169, v85 :: v_dual_mov_b32 v168, v84        ; ca100155 a9a80154
	v_dual_mov_b32 v171, v87 :: v_dual_mov_b32 v170, v86        ; ca100157 abaa0156
	ds_store_b128 v36, v[168:171]                               ; db7c0000 0000a824
	s_waitcnt vmcnt(2)                                          ; bf890bf7
	v_dual_mov_b32 v169, v89 :: v_dual_mov_b32 v168, v88        ; ca100159 a9a80158
	v_dual_mov_b32 v171, v91 :: v_dual_mov_b32 v170, v90        ; ca10015b abaa015a
	ds_store_b128 v41, v[168:171]                               ; db7c0000 0000a829
	s_waitcnt vmcnt(1)                                          ; bf8907f7
	v_dual_mov_b32 v169, v93 :: v_dual_mov_b32 v168, v92        ; ca10015d a9a8015c
	v_dual_mov_b32 v171, v95 :: v_dual_mov_b32 v170, v94        ; ca10015f abaa015e
	ds_store_b128 v46, v[168:171]                               ; db7c0000 0000a82e
	s_waitcnt vmcnt(0)                                          ; bf8903f7
	v_dual_mov_b32 v169, v57 :: v_dual_mov_b32 v168, v56        ; ca100139 a9a80138
	v_dual_mov_b32 v171, v59 :: v_dual_mov_b32 v170, v58        ; ca10013b abaa013a
	ds_store_b128 v51, v[168:171]                               ; db7c0000 0000a833
	s_waitcnt_depctr 0xffe3                                     ; bf88ffe3
	s_barrier                                                   ; bfbd0000
	s_add_u32 s0, s0, 64                                        ; 8000c000
	s_delay_alu instid0(SALU_CYCLE_1)                           ; bf870009
	s_cmpk_lt_u32 s0, 0x1400                                    ; b6801400
	s_cbranch_scc0 BB4                                          ; bfa1008a
BB2:
	v_mov_b32_e32 v5, 0                                         ; 7e0a0280
	s_lshr_b32 s1, s0, 3                                        ; 85018300
	v_mov_b32_e32 v13, 0                                        ; 7e1a0280
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v4, v3, s1, v2                                   ; d6550004 04080303
	v_add3_u32 v12, v3, s1, v8                                  ; d655000c 04200303
	v_add3_u32 v18, v9, s1, v11                                 ; d6550012 042c0309
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_1) ; bf8700c3
	v_lshlrev_b64 v[4:5], 4, v[4:5]                             ; d73c0004 02020884
	v_lshlrev_b64 v[12:13], 4, v[12:13]                         ; d73c000c 02021884
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v6, vcc_lo, s16, v4                            ; d7006a06 02020810
	v_add_co_ci_u32_e32 v7, vcc_lo, s17, v5, vcc_lo             ; 400e0a11
	v_add_co_u32 v14, vcc_lo, s18, v12                          ; d7006a0e 02021812
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1) ; bf8700a1
	v_add_co_ci_u32_e32 v15, vcc_lo, s19, v13, vcc_lo           ; 401e1a13
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	s_clause 0x2                                                ; bf850002
	global_load_b128 v[4:7], v[6:7], off                        ; dc5e0000 047c0006
	global_load_b128 v[12:15], v[14:15], off                    ; dc5e0000 0c7c000e
	global_load_b128 v[60:63], v[22:23], off                    ; dc5e0000 3c7c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v9, s1, v16                                 ; d6550012 04400309
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v22, vcc_lo, s18, v18                          ; d7006a16 02022412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1) ; bf8700c1
	v_add_co_ci_u32_e32 v23, vcc_lo, s19, v19, vcc_lo           ; 402e2613
	global_load_b128 v[64:67], v[22:23], off                    ; dc5e0000 407c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v17, s1, v21                                ; d6550012 04540311
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[68:71], v[22:23], off                    ; dc5e0000 447c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v17, s1, v24                                ; d6550012 04600311
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v22, vcc_lo, s18, v18                          ; d7006a16 02022412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1) ; bf8700c1
	v_add_co_ci_u32_e32 v23, vcc_lo, s19, v19, vcc_lo           ; 402e2613
	global_load_b128 v[72:75], v[22:23], off                    ; dc5e0000 487c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v25, s1, v29                                ; d6550012 04740319
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[76:79], v[22:23], off                    ; dc5e0000 4c7c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v25, s1, v32                                ; d6550012 04800319
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v22, vcc_lo, s18, v18                          ; d7006a16 02022412
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1) ; bf8700c1
	v_add_co_ci_u32_e32 v23, vcc_lo, s19, v19, vcc_lo           ; 402e2613
	global_load_b128 v[80:83], v[22:23], off                    ; dc5e0000 507c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v33, s1, v37                                ; d6550012 04940321
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[84:87], v[22:23], off                    ; dc5e0000 547c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v40, s1, v44                                ; d6550012 04b00328
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1) ; bf8700c1
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[88:91], v[22:23], off                    ; dc5e0000 587c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v45, s1, v47                                ; d6550012 04bc032d
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[92:95], v[22:23], off                    ; dc5e0000 5c7c0016
	v_mov_b32_e32 v19, 0                                        ; 7e260280
	v_add3_u32 v18, v50, s1, v54                                ; d6550012 04d80332
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1) ; bf870091
	v_lshlrev_b64 v[18:19], 4, v[18:19]                         ; d73c0012 02022484
	v_add_co_u32 v22, vcc_lo, s16, v18                          ; d7006a16 02022410
	s_delay_alu instid0(VALU_DEP_1)                             ; bf870001
	v_add_co_ci_u32_e32 v23, vcc_lo, s17, v19, vcc_lo           ; 402e2611
	global_load_b128 v[56:59], v[22:23], off                    ; dc5e0000 387c0016
BB4:
	ds_load_b64 v[168:169], v99 offset:18432                    ; d9d84800 a8000063
	ds_load_b64 v[170:171], v99 offset:18440                    ; d9d84808 aa000063
	ds_load_b64 v[172:173], v99 offset:18448                    ; d9d84810 ac000063
	ds_load_b64 v[174:175], v99 offset:18456                    ; d9d84818 ae000063
	ds_load_b64 v[176:177], v99 offset:20736                    ; d9d85100 b0000063
	ds_load_b64 v[178:179], v99 offset:20744                    ; d9d85108 b2000063
	ds_load_b64 v[180:181], v99 offset:20752                    ; d9d85110 b4000063
	ds_load_b64 v[182:183], v99 offset:20760                    ; d9d85118 b6000063
	ds_load_b64 v[184:185], v100                                ; d9d80000 b8000064
	ds_load_b64 v[186:187], v100 offset:8                       ; d9d80008 ba000064
	ds_load_b64 v[188:189], v100 offset:16                      ; d9d80010 bc000064
	ds_load_b64 v[190:191], v100 offset:24                      ; d9d80018 be000064
	ds_load_b64 v[192:193], v100 offset:2304                    ; d9d80900 c0000064
	ds_load_b64 v[194:195], v100 offset:2312                    ; d9d80908 c2000064
	ds_load_b64 v[196:197], v100 offset:2320                    ; d9d80910 c4000064
	ds_load_b64 v[198:199], v100 offset:2328                    ; d9d80918 c6000064
	ds_load_b64 v[200:201], v100 offset:4608                    ; d9d81200 c8000064
	ds_load_b64 v[202:203], v100 offset:4616                    ; d9d81208 ca000064
	ds_load_b64 v[204:205], v100 offset:4624                    ; d9d81210 cc000064
	ds_load_b64 v[206:207], v100 offset:4632                    ; d9d81218 ce000064
	ds_load_b64 v[208:209], v100 offset:6912                    ; d9d81b00 d0000064
	ds_load_b64 v[210:211], v100 offset:6920                    ; d9d81b08 d2000064
	ds_load_b64 v[212:213], v100 offset:6928                    ; d9d81b10 d4000064
	ds_load_b64 v[214:215], v100 offset:6936                    ; d9d81b18 d6000064
	ds_load_b64 v[216:217], v99 offset:18464                    ; d9d84820 d8000063
	ds_load_b64 v[218:219], v99 offset:18472                    ; d9d84828 da000063
	ds_load_b64 v[220:221], v99 offset:18480                    ; d9d84830 dc000063
	ds_load_b64 v[222:223], v99 offset:18488                    ; d9d84838 de000063
	ds_load_b64 v[224:225], v99 offset:20768                    ; d9d85120 e0000063
	ds_load_b64 v[226:227], v99 offset:20776                    ; d9d85128 e2000063
	ds_load_b64 v[228:229], v99 offset:20784                    ; d9d85130 e4000063
	ds_load_b64 v[230:231], v99 offset:20792                    ; d9d85138 e6000063
	ds_load_b64 v[232:233], v100 offset:32                      ; d9d80020 e8000064
	ds_load_b64 v[234:235], v100 offset:40                      ; d9d80028 ea000064
	ds_load_b64 v[236:237], v100 offset:48                      ; d9d80030 ec000064
	ds_load_b64 v[238:239], v100 offset:56                      ; d9d80038 ee000064
	ds_load_b64 v[240:241], v100 offset:2336                    ; d9d80920 f0000064
	ds_load_b64 v[242:243], v100 offset:2344                    ; d9d80928 f2000064
	ds_load_b64 v[244:245], v100 offset:2352                    ; d9d80930 f4000064
	ds_load_b64 v[246:247], v100 offset:2360                    ; d9d80938 f6000064
	ds_load_b64 v[248:249], v100 offset:4640                    ; d9d81220 f8000064
	ds_load_b64 v[250:251], v100 offset:4648                    ; d9d81228 fa000064
	ds_load_b64 v[252:253], v100 offset:4656                    ; d9d81230 fc000064
	ds_load_b64 v[254:255], v100 offset:4664                    ; d9d81238 fe000064
	ds_load_b64 v[18:19], v100 offset:6944                      ; d9d81b20 12000064
	ds_load_b64 v[22:23], v100 offset:6952                      ; d9d81b28 16000064
	ds_load_b64 v[26:27], v100 offset:6960                      ; d9d81b30 1a000064
	ds_load_b64 v[30:31], v100 offset:6968                      ; d9d81b38 1e000064
	ds_load_b64 v[34:35], v99 offset:18496                      ; d9d84840 22000063
	ds_load_b64 v[38:39], v99 offset:18504                      ; d9d84848 26000063
	ds_load_b64 v[42:43], v99 offset:18512                      ; d9d84850 2a000063
	ds_load_b64 v[48:49], v99 offset:18520                      ; d9d84858 30000063
	ds_load_b64 v[52:53], v99 offset:20800                      ; d9d85140 34000063
	ds_load_b64 v[102:103], v99 offset:20808                    ; d9d85148 66000063
	s_waitcnt lgkmcnt(42)                                       ; bf89fea7
	v_wmma_f32_16x16x16_f16 v[104:111], v[184:191], v[168:175], v[104:111] ; cc404068 1da351b8
	v_wmma_f32_16x16x16_f16 v[112:119], v[184:191], v[176:183], v[112:119] ; cc404070 1dc361b8
	s_waitcnt lgkmcnt(38)                                       ; bf89fe67
	v_wmma_f32_16x16x16_f16 v[120:127], v[192:199], v[168:175], v[120:127] ; cc404078 1de351c0
	v_wmma_f32_16x16x16_f16 v[128:135], v[192:199], v[176:183], v[128:135] ; cc404080 1e0361c0
	s_waitcnt lgkmcnt(34)                                       ; bf89fe27
	v_wmma_f32_16x16x16_f16 v[136:143], v[200:207], v[168:175], v[136:143] ; cc404088 1e2351c8
	v_wmma_f32_16x16x16_f16 v[144:151], v[200:207], v[176:183], v[144:151] ; cc404090 1e4361c8
	s_waitcnt lgkmcnt(30)                                       ; bf89fde7
	v_wmma_f32_16x16x16_f16 v[152:159], v[208:215], v[168:175], v[152:159] ; cc404098 1e6351d0
	ds_load_b64 v[188:189], v99 offset:20816                    ; d9d85150 bc000063
	ds_load_b64 v[190:191], v99 offset:20824                    ; d9d85158 be000063
	ds_load_b64 v[184:185], v100 offset:64                      ; d9d80040 b8000064
	ds_load_b64 v[186:187], v100 offset:72                      ; d9d80048 ba000064
	ds_load_b64 v[196:197], v100 offset:80                      ; d9d80050 c4000064
	ds_load_b64 v[198:199], v100 offset:88                      ; d9d80058 c6000064
	ds_load_b64 v[192:193], v100 offset:2368                    ; d9d80940 c0000064
	ds_load_b64 v[194:195], v100 offset:2376                    ; d9d80948 c2000064
	ds_load_b64 v[204:205], v100 offset:2384                    ; d9d80950 cc000064
	ds_load_b64 v[206:207], v100 offset:2392                    ; d9d80958 ce000064
	ds_load_b64 v[200:201], v100 offset:4672                    ; d9d81240 c8000064
	ds_load_b64 v[202:203], v100 offset:4680                    ; d9d81248 ca000064
	ds_load_b64 v[172:173], v100 offset:4688                    ; d9d81250 ac000064
	ds_load_b64 v[174:175], v100 offset:4696                    ; d9d81258 ae000064
	ds_load_b64 v[168:169], v100 offset:6976                    ; d9d81b40 a8000064
	ds_load_b64 v[170:171], v100 offset:6984                    ; d9d81b48 aa000064
	v_wmma_f32_16x16x16_f16 v[160:167], v[208:215], v[176:183], v[160:167] ; cc4040a0 1e8361d0
	s_waitcnt lgkmcnt(34)                                       ; bf89fe27
	v_wmma_f32_16x16x16_f16 v[104:111], v[232:239], v[216:223], v[104:111] ; cc404068 1da3b1e8
	v_wmma_f32_16x16x16_f16 v[112:119], v[232:239], v[224:231], v[112:119] ; cc404070 1dc3c1e8
	s_waitcnt lgkmcnt(30)                                       ; bf89fde7
	v_wmma_f32_16x16x16_f16 v[120:127], v[240:247], v[216:223], v[120:127] ; cc404078 1de3b1f0
	v_wmma_f32_16x16x16_f16 v[128:135], v[240:247], v[224:231], v[128:135] ; cc404080 1e03c1f0
	ds_load_b64 v[180:181], v100 offset:6992                    ; d9d81b50 b4000064
	ds_load_b64 v[182:183], v100 offset:7000                    ; d9d81b58 b6000064
	ds_load_b64 v[208:209], v99 offset:18528                    ; d9d84860 d0000063
	ds_load_b64 v[210:211], v99 offset:18536                    ; d9d84868 d2000063
	ds_load_b64 v[212:213], v99 offset:18544                    ; d9d84870 d4000063
	ds_load_b64 v[214:215], v99 offset:18552                    ; d9d84878 d6000063
	ds_load_b64 v[176:177], v99 offset:20832                    ; d9d85160 b0000063
	ds_load_b64 v[178:179], v99 offset:20840                    ; d9d85168 b2000063
	ds_load_b64 v[236:237], v99 offset:20848                    ; d9d85170 ec000063
	ds_load_b64 v[238:239], v99 offset:20856                    ; d9d85178 ee000063
	ds_load_b64 v[232:233], v100 offset:96                      ; d9d80060 e8000064
	ds_load_b64 v[234:235], v100 offset:104                     ; d9d80068 ea000064
	ds_load_b64 v[244:245], v100 offset:112                     ; d9d80070 f4000064
	ds_load_b64 v[246:247], v100 offset:120                     ; d9d80078 f6000064
	ds_load_b64 v[240:241], v100 offset:2400                    ; d9d80960 f0000064
	ds_load_b64 v[242:243], v100 offset:2408                    ; d9d80968 f2000064
	s_waitcnt lgkmcnt(42)                                       ; bf89fea7
	v_wmma_f32_16x16x16_f16 v[136:143], v[248:255], v[216:223], v[136:143] ; cc404088 1e23b1f8
	v_wmma_f32_16x16x16_f16 v[144:151], v[248:255], v[224:231], v[144:151] ; cc404090 1e43c1f8
	v_mov_b32_e32 v97, v20                                      ; 7ec20314
	v_mov_b32_e32 v101, v21                                     ; 7eca0315
	s_waitcnt lgkmcnt(40)                                       ; bf89fe87
	v_dual_mov_b32 v21, v23 :: v_dual_mov_b32 v20, v22          ; ca100117 15140116
	s_waitcnt lgkmcnt(39)                                       ; bf89fe77
	v_dual_mov_b32 v23, v27 :: v_dual_mov_b32 v22, v26          ; ca10011b 1716011a
	v_dual_mov_b32 v27, v25 :: v_dual_mov_b32 v26, v24          ; ca100119 1b1a0118
	s_waitcnt lgkmcnt(38)                                       ; bf89fe67
	v_dual_mov_b32 v25, v31 :: v_dual_mov_b32 v24, v30          ; ca10011f 1918011e
	s_delay_alu instid0(VALU_DEP_1)                             ; bf870001
	v_wmma_f32_16x16x16_f16 v[152:159], v[18:25], v[216:223], v[152:159] ; cc404098 1e63b112
	v_wmma_f32_16x16x16_f16 v[160:167], v[18:25], v[224:231], v[160:167] ; cc4040a0 1e83c112
	ds_load_b64 v[252:253], v100 offset:2416                    ; d9d80970 fc000064
	ds_load_b64 v[254:255], v100 offset:2424                    ; d9d80978 fe000064
	ds_load_b64 v[248:249], v100 offset:4704                    ; d9d81260 f8000064
	ds_load_b64 v[250:251], v100 offset:4712                    ; d9d81268 fa000064
	ds_load_b64 v[220:221], v100 offset:4720                    ; d9d81270 dc000064
	ds_load_b64 v[222:223], v100 offset:4728                    ; d9d81278 de000064
	ds_load_b64 v[30:31], v100 offset:7008                      ; d9d81b60 1e000064
	ds_load_b64 v[216:217], v100 offset:7016                    ; d9d81b68 d8000064
	ds_load_b64 v[218:219], v100 offset:7024                    ; d9d81b70 da000064
	ds_load_b64 v[230:231], v100 offset:7032                    ; d9d81b78 e6000064
	s_waitcnt lgkmcnt(47)                                       ; bf89fef7
	v_dual_mov_b32 v19, v35 :: v_dual_mov_b32 v18, v34          ; ca100123 13120122
	s_waitcnt lgkmcnt(46)                                       ; bf89fee7
	v_dual_mov_b32 v21, v39 :: v_dual_mov_b32 v20, v38          ; ca100127 15140126
	s_waitcnt lgkmcnt(45)                                       ; bf89fed7
	v_dual_mov_b32 v23, v43 :: v_dual_mov_b32 v22, v42          ; ca10012b 1716012a
	s_waitcnt lgkmcnt(44)                                       ; bf89fec7
	v_dual_mov_b32 v25, v49 :: v_dual_mov_b32 v24, v48          ; ca100131 19180130
	s_waitcnt lgkmcnt(39)                                       ; bf89fe77
	v_swap_b32 v53, v185                                        ; 7e6acbb9
	s_waitcnt lgkmcnt(38)                                       ; bf89fe67
	v_swap_b32 v102, v186                                       ; 7ecccbba
	v_swap_b32 v103, v187                                       ; 7ececbbb
	s_waitcnt lgkmcnt(35)                                       ; bf89fe37
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3) ; bf8701d3
	v_swap_b32 v53, v193                                        ; 7e6acbc1
	s_waitcnt lgkmcnt(34)                                       ; bf89fe27
	v_swap_b32 v102, v194                                       ; 7ecccbc2
	v_swap_b32 v103, v195                                       ; 7ececbc3
	s_waitcnt lgkmcnt(31)                                       ; bf89fdf7
	v_swap_b32 v53, v201                                        ; 7e6acbc9
	v_swap_b32 v52, v184                                        ; 7e68cbb8
	s_waitcnt lgkmcnt(3)                                        ; bf89fc37
	v_dual_mov_b32 v225, v31 :: v_dual_mov_b32 v224, v30        ; ca10011f e1e0011e
	v_swap_b32 v102, v202                                       ; 7ecccbca
	v_swap_b32 v103, v203                                       ; 7ececbcb
	v_swap_b32 v53, v169                                        ; 7e6acba9
	v_swap_b32 v52, v192                                        ; 7e68cbc0
	s_delay_alu instid0(VALU_DEP_2)                             ; bf870002
	v_swap_b32 v53, v177                                        ; 7e6acbb1
	v_wmma_f32_16x16x16_f16 v[112:119], v[192:199], v[184:191], v[112:119] ; cc404070 1dc371c0
	v_swap_b32 v52, v200                                        ; 7e68cbc8
	v_wmma_f32_16x16x16_f16 v[104:111], v[192:199], v[18:25], v[104:111] ; cc404068 1da225c0
	v_swap_b32 v53, v233                                        ; 7e6acbe9
	v_swap_b32 v102, v170                                       ; 7ecccbaa
	v_swap_b32 v103, v171                                       ; 7ececbab
	v_wmma_f32_16x16x16_f16 v[120:127], v[200:207], v[18:25], v[120:127] ; cc404078 1de225c8
	v_wmma_f32_16x16x16_f16 v[128:135], v[200:207], v[184:191], v[128:135] ; cc404080 1e0371c8
	v_swap_b32 v52, v168                                        ; 7e68cba8
	s_cmpk_ge_u32 s0, 0x1400                                    ; b6001400
	v_swap_b32 v53, v241                                        ; 7e6acbf1
	v_swap_b32 v102, v178                                       ; 7ecccbb2
	v_swap_b32 v103, v179                                       ; 7ececbb3
	v_swap_b32 v52, v176                                        ; 7e68cbb0
	v_wmma_f32_16x16x16_f16 v[144:151], v[168:175], v[184:191], v[144:151] ; cc404090 1e4371a8
	v_wmma_f32_16x16x16_f16 v[136:143], v[168:175], v[18:25], v[136:143] ; cc404088 1e2225a8
	v_swap_b32 v53, v249                                        ; 7e6acbf9
	v_swap_b32 v102, v234                                       ; 7ecccbea
	v_swap_b32 v103, v235                                       ; 7ececbeb
	v_swap_b32 v52, v232                                        ; 7e68cbe8
	v_wmma_f32_16x16x16_f16 v[152:159], v[176:183], v[18:25], v[152:159] ; cc404098 1e6225b0
	v_wmma_f32_16x16x16_f16 v[160:167], v[176:183], v[184:191], v[160:167] ; cc4040a0 1e8371b0
	s_waitcnt lgkmcnt(2)                                        ; bf89fc27
	v_swap_b32 v53, v217                                        ; 7e6acbd9
	v_swap_b32 v102, v242                                       ; 7ecccbf2
	v_swap_b32 v103, v243                                       ; 7ececbf3
	v_swap_b32 v52, v240                                        ; 7e68cbf0
	s_delay_alu instid0(VALU_DEP_3)                             ; bf870003
	v_swap_b32 v102, v250                                       ; 7ecccbfa
	v_swap_b32 v103, v251                                       ; 7ececbfb
	v_wmma_f32_16x16x16_f16 v[104:111], v[240:247], v[208:215], v[104:111] ; cc404068 1da3a1f0
	v_wmma_f32_16x16x16_f16 v[112:119], v[240:247], v[232:239], v[112:119] ; cc404070 1dc3d1f0
	v_swap_b32 v52, v248                                        ; 7e68cbf8
	s_waitcnt lgkmcnt(1)                                        ; bf89fc17
	v_swap_b32 v102, v218                                       ; 7ecccbda
	v_swap_b32 v103, v219                                       ; 7ececbdb
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_2) ; bf870143
	v_wmma_f32_16x16x16_f16 v[120:127], v[248:255], v[208:215], v[120:127] ; cc404078 1de3a1f8
	v_wmma_f32_16x16x16_f16 v[128:135], v[248:255], v[232:239], v[128:135] ; cc404080 1e03d1f8
	v_swap_b32 v52, v216                                        ; 7e68cbd8
	v_dual_mov_b32 v229, v103 :: v_dual_mov_b32 v228, v102      ; ca100167 e5e40166
	v_wmma_f32_16x16x16_f16 v[136:143], v[216:223], v[208:215], v[136:143] ; cc404088 1e23a1d8
	v_wmma_f32_16x16x16_f16 v[144:151], v[216:223], v[232:239], v[144:151] ; cc404090 1e43d1d8
	v_dual_mov_b32 v227, v53 :: v_dual_mov_b32 v226, v52        ; ca100135 e3e20134
	s_waitcnt lgkmcnt(0)                                        ; bf89fc07
	s_delay_alu instid0(VALU_DEP_1)                             ; bf870001
	v_wmma_f32_16x16x16_f16 v[152:159], v[224:231], v[208:215], v[152:159] ; cc404098 1e63a1e0
	v_wmma_f32_16x16x16_f16 v[160:167], v[224:231], v[232:239], v[160:167] ; cc4040a0 1e83d1e0
	s_cbranch_scc0 BB6                                          ; bfa1fdbc
