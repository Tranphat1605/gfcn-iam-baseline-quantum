"""Verify the four trained checkpoints against their recorded SHA-256 hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for record in json.loads((root/'results/checkpoints.json').read_text()):
    file = root/record['path']
    assert file.stat().st_size == record['bytes'], file
    assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256'], file
    print('OK', record['path'])
