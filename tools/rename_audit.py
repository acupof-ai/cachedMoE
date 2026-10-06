#!/usr/bin/env python3
"""Read-only per-hit rename audit of the current tree or a committed Git tree."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import re
import subprocess

OLD = re.compile('deepmoe', re.I)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES = Path(__file__).with_name("rename_retention_rules.json")


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def blobs(repo, commit):
    entries = git(repo, 'ls-tree', '-r', '-z', commit).split(b'\0')
    child = subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for entry in filter(None, entries):
            metadata, raw_path = entry.split(b'\t', 1)
            mode, kind, oid = metadata.split()
            if kind != b'blob':
                continue
            child.stdin.write(oid + b'\n')
            child.stdin.flush()
            header = child.stdout.readline().split()
            size = int(header[2])
            data = child.stdout.read(size)
            if len(data) != size or child.stdout.read(1) != b'\n':
                raise RuntimeError('incomplete git cat-file output')
            yield raw_path.decode(), oid.decode(), mode.decode(), data
    finally:
        child.stdin.close()
        child.stdout.close()
        child.wait()


def token_span(line, match):
    start, end = match.span()
    while start > 0 and (line[start - 1].isalnum() or line[start - 1] in '_.-'):
        start -= 1
    while end < len(line) and (line[end].isalnum() or line[end] in '_.-'):
        end += 1
    return start, end


def inside(line, match, literal):
    return any(found.start() <= match.start() and match.end() <= found.end()
               for found in re.finditer(re.escape(literal), line))


def disposition(path, line, number, match, token, rules):
    # Rules are token-local, and may add a line constraint. A whole mixed line
    # never acquires permission merely because a different token was retained.
    for rule in rules:
        if (fnmatchcase(path, rule['path_glob'])
                and re.fullmatch(rule['token_regex'], token)
                and ('line_regex' not in rule or re.search(rule['line_regex'], line))
                and number >= rule.get('line_min', 1)
                and (rule.get('line_max') is None or number <= rule['line_max'])):
            return rule.get('status', 'RETAIN'), rule['role'], rule['reason'], rule['id']
    if inside(line, match, 'deepmoe_manifest.json'):
        return ('RETAIN', 'checkpoint_format',
                'Native checkpoint manifest filename is immutable; no checkpoint write or format migration.', 'builtin:manifest')
    if inside(line, match, '/mnt/deepmoe2'):
        return ('RETAIN', 'hardware_mount',
                'Actual second-drive read-only mount path; branding does not rename mounted hardware.', 'builtin:mirror-mount')
    if inside(line, match, 'deepmoe-mesa/'):
        return ('RETAIN', 'custom_driver_location',
                'Actual custom Mesa installation path used by the recorded driver experiment.', 'builtin:mesa-path')
    if path.startswith('tests/data/'):
        return ('RETAIN', 'immutable_golden',
                'Tracked fixed golden/tokenizer input; its bytes are protected by the migration acceptance contract.', 'builtin:golden')
    if path.startswith(('bench/results/', 'traces/')):
        return ('RETAIN', 'historical_raw_evidence',
                'Tracked historical run/trace evidence or frozen command uses its original names and hashes.', 'builtin:raw-history')
    if path == 'docs/codex_todo.md':
        return ('RETAIN', 'owner_text_or_recorded_receipt',
                'Owner original task text and appended past receipts remain verbatim; no mechanical rewrite.', 'builtin:owner-todo')
    if path == 'tools/oracle_dspark.py' and ('TREE_SOURCE_REV' in line or inside(line, match, 'deepMoE 是')):
        return ('RETAIN', 'fixed_git_oracle_locator',
                'Exact old source revision/path or oracle prompt text anchors a fixed-git input, not a live default.', 'builtin:oracle-source')
    if path.startswith('docs/') or path in ('README.md', 'AGENTS.md', 'CLAUDE.md'):
        return ('NEEDS_REVIEW', 'documentation_scope',
                'Mixed current instructions and historical evidence require a reviewed per-hit decision; no blanket STATUS/doc retention.', None)
    return ('NEEDS_REVIEW', 'live_source_or_explicit_compatibility',
            'Current source/default/helper/query or compatibility intent needs a concrete reviewed role; not assumed historical.', None)


def working_files(repo):
    names = sorted(set(filter(None, git(repo, 'ls-files', '--cached', '--others',
                                      '--exclude-standard', '-z').decode().split('\0'))))
    for name in names:
        path = repo / name
        if path.is_symlink():
            data = path.readlink().as_posix().encode()
            mode = "120000"
        elif path.is_file():
            data = path.read_bytes()
            mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        else:
            # A tracked deletion has no current content to audit.
            continue
        yield name, None, mode, data


def load_rules(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rules = data['rules'] if isinstance(data, dict) else data
    seen = set()
    for rule in rules:
        if not all(rule.get(key) for key in ('id', 'path_glob', 'token_regex', 'role', 'reason')):
            raise ValueError('each retention rule needs id/path/token/role/reason')
        if rule['id'] in seen:
            raise ValueError('duplicate retention rule id: ' + rule['id'])
        seen.add(rule['id'])
        re.compile(rule['token_regex'])
        if 'line_regex' in rule:
            re.compile(rule['line_regex'])
        if 'after_marker' in rule and not rule['after_marker']:
            raise ValueError('retention section marker must be nonempty')
    return rules


def file_rules(text, rules):
    """Keep archival section rules from covering current instructions."""
    lines = text.split('\n')
    selected = []
    for rule in rules:
        marker = rule.get('after_marker')
        if marker is None:
            selected.append(rule)
        elif marker in lines:
            selected.append(dict(rule, line_min=lines.index(marker) + 2))
    return selected


def audit(repo, ref, rules):
    commit = git(repo, 'rev-parse', f'{ref or "HEAD"}^{{commit}}').decode().strip()
    status = git(repo, 'status', '--porcelain').decode().splitlines()
    source = blobs(repo, commit) if ref else working_files(repo)
    fingerprints = []
    hits, path_hits, skipped = [], [], []
    text_files = 0
    for path, oid, mode, data in source:
        fingerprints.append((path, hashlib.sha256(data).hexdigest()))
        if OLD.search(path):
            wrapper = path == "cmake/deepmoe_options.cmake" and (
                b'include("${CMAKE_CURRENT_LIST_DIR}/cachedmoe_options.cmake")' in data)
            path_hits.append(dict(path=path, blob=oid, mode=mode,
                                  status='RETAIN' if wrapper else 'NEEDS_REVIEW',
                                  role='legacy_cmake_module_path' if wrapper else 'legacy_filename',
                                  reason='Thin compatibility module includes canonical options.' if wrapper
                                  else 'Legacy filename needs an explicit compatibility/history role.'))
        if b'\0' in data:
            skipped.append(dict(path=path, reason='binary NUL bytes; equivalent to rg --text not requested'))
            continue
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            skipped.append(dict(path=path, reason='not UTF-8 text'))
            continue
        text_files += 1
        file_sha = hashlib.sha256(data).hexdigest()
        scoped_rules = file_rules(text, rules)
        for number, line in enumerate(text.split('\n'), 1):
            for match in OLD.finditer(line):
                start, end = token_span(line, match)
                token = line[start:end]
                status, role, reason, rule_id = disposition(path, line, number, match, token, scoped_rules)
                hits.append(dict(path=path, line=number,
                                 column=len(line[:match.start()].encode('utf-8')) + 1,
                                 character_column=match.start() + 1,
                                 exact_match=match.group(), exact_token=token,
                                 token_character_columns=[start + 1, end + 1],
                                 text=line, status=status, role=role, reason=reason,
                                 rule_id=rule_id, source_blob=oid, source_sha256=file_sha,
                                 line_sha256=hashlib.sha256(line.encode()).hexdigest()))
    fingerprint = hashlib.sha256(json.dumps(fingerprints, separators=(",", ":")).encode()).hexdigest()
    if not ref:
        # Verify the whole current-tree snapshot again: a concurrent edit must
        # fail this run instead of producing a mixed-generation final receipt.
        after = [(path, hashlib.sha256(data).hexdigest())
                 for path, _, _, data in working_files(repo)]
        if after != fingerprints:
            raise RuntimeError("working tree changed during audit; retry after edits finish")
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), repo=str(repo),
                source_commit=commit, snapshot='committed Git tree' if ref else 'tracked and nonignored current tree',
                working_tree_status=status if not ref else [], source_fingerprint_sha256=fingerprint,
                column_definition='one-based UTF-8 byte column, as rg JSON submatch.start+1; character_column is also retained',
                counts=dict(text_files=text_files, content_hits=len(hits),
                            by_status=dict(Counter(row['status'] for row in hits)),
                            by_role=dict(Counter(row['role'] for row in hits)),
                            tracked_legacy_filenames=len(path_hits),
                            unreviewed_filenames=sum(row['status'] == 'NEEDS_REVIEW' for row in path_hits),
                            skipped_binary_files=len(skipped)),
                rules=rules, hits=hits, tracked_path_hits=path_hits, skipped_binary=skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument('--ref', help='committed tree; default audits the current tree')
    scope.add_argument('--worktree', action='store_true', help='audit tracked and nonignored current files')
    parser.add_argument('--rules', type=Path, default=DEFAULT_RULES)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        raise RuntimeError('refusing to overwrite an existing audit receipt')
    rules = load_rules(args.rules)
    report = audit(args.repo.resolve(), args.ref, rules)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(source_commit=report['source_commit'], **report['counts']), ensure_ascii=False))
    return int(report['counts']['by_status'].get('NEEDS_REVIEW', 0) > 0
               or report['counts']['unreviewed_filenames'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
