import __main__ as owner
import json
import time
assert owner.job['name'] == 'mmlu57_spec5'
previous = owner.job['timeout_s']
owner.job['timeout_s'] = 1300
(owner.base / 'jobs.json').write_text(json.dumps(owner.plan, indent=2) + '\n')
(owner.base / 'deadline_extension.json').write_text(json.dumps(dict(job=owner.job['name'], old_seconds=previous, new_seconds=1300, reason='57 exact prefix prefills take longer than the initial budget; complete the same single run without repeating GPU configurations', elapsed_seconds=time.monotonic()-owner.t0), indent=2) + '\n')
