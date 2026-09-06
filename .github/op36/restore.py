"""Restore exact reviewed source commits from readable, checksum-verified patches."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

STAGING = 'ops/rc6-optimizer-20260906'
TARGET = 'refs/heads/release/5.0.0rc6'


def git(*args, data=None):
    return subprocess.check_output(['git', *args], input=data).decode().strip()


def load(path):
    plan = json.loads(path.read_text())
    if plan['repository'] not in {'s7cret/optimizer', 's7cret/openpine'} or plan['repository'] != os.environ['GITHUB_REPOSITORY']:
        raise ValueError('repository mismatch')
    if plan['target'] != TARGET.removeprefix('refs/heads/') or plan['expected'] not in (None, plan['base']):
        raise ValueError('target mismatch')
    parent = plan['base']
    for item in plan['commits']:
        if item['parent'] != parent or any(not re.fullmatch('[0-9a-f]{40}', item[key]) for key in ('sha','parent','tree')):
            raise ValueError('noncontiguous commit identities')
        raw = item['raw'].encode()
        if not raw.startswith(f"tree {item['tree']}\nparent {parent}\nauthor ".encode()):
            raise ValueError('commit header mismatch')
        if hashlib.sha1(b'commit '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != item['sha']:
            raise ValueError('commit SHA mismatch')
        parent = item['sha']
    if parent != plan['head']:
        raise ValueError('head mismatch')
    return plan


def restore(path, evidence):
    plan = load(path)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    patches = []
    for item in plan['commits']:
        pieces = []
        for name in item['parts']:
            if not re.fullmatch(r'[0-9]{2}-[0-9]+\.patch', name):
                raise ValueError('invalid patch path')
            data = (path.parent/name).read_bytes()
            if len(data) > 1000000:
                raise ValueError('patch too large')
            pieces.append(data)
            (evidence/name).write_bytes(data)
        patch = b''.join(pieces)
        if hashlib.sha256(patch).hexdigest() != item['sha256']:
            raise ValueError('patch checksum mismatch')
        patches.append(patch)
    git('checkout', '--detach', plan['base'])
    for item, patch in zip(plan['commits'], patches, strict=True):
        if git('rev-parse','HEAD') != item['parent']:
            raise ValueError('parent drift')
        git('apply','--check','--index','-',data=patch)
        git('apply','--index','--whitespace=nowarn','-',data=patch)
        if git('write-tree') != item['tree']:
            raise ValueError('tree mismatch')
        if git('hash-object','-t','commit','-w','--stdin',data=item['raw'].encode()) != item['sha']:
            raise ValueError('restored commit mismatch')
        git('checkout','--detach',item['sha'])
    git('update-ref','refs/heads/review-candidate',plan['head'])
    git('bundle','create',str(evidence/'verified.bundle'),'refs/heads/review-candidate')
    git('bundle','verify',str(evidence/'verified.bundle'))
    (evidence/'code-head.txt').write_text(plan['head']+'\n')


def publish(evidence):
    plan = load(evidence/'plan.json')
    if os.environ.get('GITHUB_REF') != 'refs/heads/'+STAGING:
        raise ValueError('wrong publication branch')
    git('fetch',str(evidence/'verified.bundle'),'refs/heads/review-candidate')
    if git('rev-parse','FETCH_HEAD') != plan['head']:
        raise ValueError('bundle mismatch')
    git('merge-base','--is-ancestor',plan['base'],plan['head'])
    before = git('ls-remote','--heads','origin')
    heads = {ref: sha for sha,ref in (line.split() for line in before.splitlines())}
    if heads.get(TARGET) not in (plan['expected'],plan['head']):
        raise ValueError('release changed concurrently')
    if heads.get(TARGET) != plan['head']:
        git('push','origin',plan['head']+':'+TARGET)
    if git('ls-remote','--refs','origin',TARGET).split()[0] != plan['head']:
        raise ValueError('published ref mismatch')
    (evidence/'before-heads.txt').write_text(before+'\n')
    branch,tag='refs/heads/'+STAGING,'refs/tags/'+STAGING
    sha=os.environ['GITHUB_SHA']
    git('push','--atomic',f'--force-with-lease={branch}:{sha}',f'--force-with-lease={tag}:','origin',sha+':'+tag,':'+branch)
    after = git('ls-remote','--heads','origin')
    expected = {**heads,TARGET:plan['head']}
    expected.pop(branch)
    if {ref: value for value,ref in (line.split() for line in after.splitlines())} != expected:
        raise ValueError('final branch inventory mismatch')
    if git('ls-remote','--refs','origin',tag).split()[0] != sha:
        raise ValueError('archive mismatch')
    (evidence/'final-heads.txt').write_text(after+'\n')


if __name__ == '__main__':
    if sys.argv[1]=='restore': restore(Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
    elif sys.argv[1]=='publish': publish(Path(sys.argv[2]).resolve())
    else: raise ValueError('unknown operation')
