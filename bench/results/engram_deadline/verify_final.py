from pathlib import Path
import datetime, hashlib, json, subprocess

root = Path(__file__).resolve().parents[3]
base = Path(__file__).resolve().parent

def read(path):
    return json.loads((base / path).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = read('source_manifest.json')
assert sha(root / 'build/deepmoe') == manifest['binary_sha256']
assert sha(root / 'build/tests/deepmoe_tests') == manifest['tests_sha256']
assert all(sha(root / name) == value for name, value in manifest['source_files'].items())
assert all(sha(root / 'build/shaders' / name) == value for name, value in manifest['shader_files'].items())
assert len(manifest['source_files']) == 259
assert len(manifest['shader_files']) == 50

receipts = {}
for folder, names in {
    'guarded_gate': ['wrap_selected', 'adaptive_zero', 'l3_off', 'mmlu6_baseline', 'mmlu6_fix'],
    'final_decode_gate': ['decode', 'decode_longctx'],
    'final_power': ['fix_mask', 'fix_spec', 'spec_k3'],
    'mmlu57': ['mmlu57'],
}.items():
    jobs = read(folder + '/check_results.json')
    assert [job['name'] for job in jobs] == names
    assert all(job['rc'] == 0 and job['failure'] is None and job['thermal_pauses'] == 0 and not job['ac_changed'] for job in jobs)
    assert read(folder + '/profile_receipt.json') == {'original': 'performance', 'final': 'performance'}
    receipts[folder] = jobs
balanced = read('balanced/check_results.json')[0]
assert balanced['name'] == 'fix_mask' and balanced['rc'] == 0 and balanced['thermal_pauses'] == 0 and not balanced['ac_changed']
receipts['balanced_mask_only'] = balanced

assert '30/30 gates passed' in (base / 'final_cpu_gates.log').read_text()
assert '100% tests passed' in (base / 'cpu_guarded.log').read_text()
l3 = read('guarded_gate/l3_off.json')['rows'][0]
assert l3['nll'] == 0.622784 and l3['top1'] == 56 and l3['top1_of'] == 64
mmlu = read('mmlu57/evaluation/summary.json')[0]
assert mmlu['n'] == 57 and mmlu['correct'] == 48 and mmlu['invalid'] == 2
comparison = read('mmlu57_comparison.json')
assert comparison['same_predictions'] == 57 and comparison['same_correctness'] == 57
assert comparison['all_speculated'] and comparison['all_exact_prefix']

provenance_paths = ['final_power/fix_mask/provenance.json', 'final_power/fix_spec/provenance.json',
                    'final_power/spec_k3/provenance.json', 'balanced/fix_mask/provenance.json',
                    'mmlu57/evaluation/mask-spec/provenance.json']
for name in provenance_paths:
    p = read(name)
    assert manifest['binary_sha256'].startswith(p['exe_sha'])
    assert p['shaders']['count'] == 50 and p['shaders']['sha'] == '214a8d1c095876be'

assert subprocess.run(['pgrep', '-x', 'deepmoe'], stdout=subprocess.DEVNULL).returncode == 1
assert subprocess.run(['pgrep', '-x', 'deepmoe_tests'], stdout=subprocess.DEVNULL).returncode == 1
profile = subprocess.check_output(['powerprofilesctl', 'get'], text=True).strip()
assert profile == 'performance'
assert subprocess.run(['python', 'tools/perf_ledger.py', '--check'], cwd=root).returncode == 0
assert subprocess.run(['git', 'diff', '--check'], cwd=root).returncode == 0

data = {
    'time': datetime.datetime.now().astimezone().isoformat(),
    'final_binary_sha256': manifest['binary_sha256'],
    'tests_sha256': manifest['tests_sha256'],
    'source_sha256': manifest['source_sha256'],
    'source_files_verified': 259,
    'shader_files_verified': 50,
    'provenance_paths': provenance_paths,
    'cpu_ctest': '25/25',
    'cpu_and_tools': '30/30',
    'off64': {'nll': l3['nll'], 'top1': 56, 'steps': 64},
    'gpu_receipts': receipts,
    'mmlu': mmlu,
    'mmlu_comparison': comparison,
    'performance_comparison': 'comparison.json',
    'target_trace_analysis': 'verify_trace.json',
    'excluded_speed_cells': {
        'final_performance/fix_mask': '5 thermal pauses',
        'final_performance/fix_spec': '7 thermal pauses',
        'balanced/fix_spec': 'GPU reached 80 C; deliberately stopped on first thermal pause',
        'measure/fix_mask': 'policy inactive: 0 deadline chunks',
        'measure/fix_spec': 'policy inactive: 0 deadline chunks',
    },
    'profile_restored': profile,
    'no_engine_processes': True,
    'notes': [
        'One primary read source; absent external mirror. Native FP4/FP8, total cache 5500.',
        'New GPU receipts use this final binary. Earlier validation/ failed wrap receipt is preserved.',
        'Speed controls and final runs differ in output/routes and masked mass; not bit-identical kernel comparisons.',
        'Speculation remains a single main path and one target forward; no 1.6x or losslessness claim.',
        'MMLU is 57-subject zero-shot generated sample, not full canonical five-shot evaluation.',
        'Short decode oracle passes existing gate with 6/8 loaded and 7/8 own-prefill; no threshold change.',
    ],
}
(base / 'final_acceptance.json').write_text(json.dumps(data, indent=2) + '\n')
print('Final source, binary, shader, quality, performance and restore receipts verified.')
