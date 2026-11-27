"""manifest.json agrees with the code: identity, version, settings, model.

Copyright 2026 Subtitld contributors
SPDX-License-Identifier: Apache-2.0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from deepfilternet_addon import ADDON_ID, __version__, engine, spec
from deepfilternet_addon import __main__ as addon

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / 'manifest.json').read_text(encoding='utf-8'))
FIELDS = {field['key']: field for field in MANIFEST['config_schema']['fields']}


def test_identity_and_version():
    assert MANIFEST['id'] == ADDON_ID
    assert MANIFEST['executable'] == f'bin/{ADDON_ID}-addon'
    assert MANIFEST['tasks'] == ['audio.separate']
    assert 'languages' not in MANIFEST  # the sound of any language
    pyproject = (REPO / 'pyproject.toml').read_text(encoding='utf-8')
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == __version__
    assert MANIFEST['version'] == __version__ == addon.ADDON_VERSION
    assert re.fullmatch(r'\d\d\.\d{1,2}\.\d+', __version__)


def test_the_settings_are_the_ones_the_code_reads():
    assert set(FIELDS) == {'threads'}
    field = FIELDS['threads']
    assert field['type'] == 'int' and field['min'] == 0 and field['default'] == 0 and field['max'] == 256


def test_the_model_entry_matches_the_spec():
    model = MANIFEST['models'][0]
    assert model['license'] and MANIFEST['compute'] == 'cpu'
    if engine.downloads():
        assert model['repo'] == spec.MODEL_REPO
        assert model['revision'] == spec.MODEL_REVISION
        assert model['size_mb'] == round(engine.MODEL_BYTES / 2**20)
        assert model['bundled'] is False
        assert MANIFEST['disk_required_mb'] > model['size_mb']
    else:
        assert model['bundled'] is True
        assert 'repo' not in model and 'revision' not in model
    assert MANIFEST['ram_required_mb'] >= spec.RAM_GB * 1024


def test_the_licence_files_are_committed():
    assert (REPO / 'THIRD_PARTY_LICENSES' / 'models' / spec.MODEL_LICENCE).is_file()
    assert (REPO / 'THIRD_PARTY_LICENSES' / 'models' / 'README.md').is_file()


def test_the_spec_is_consistent():
    assert spec.TASK in ('sep', 's2s')
    assert spec.CHANNELS in (1, 2) and spec.RATE in (16000, 44100, 48000)
    assert 0 < spec.OVERLAP_SEC < spec.CHUNK_SEC / 2
    assert bool(spec.MODEL_FILES) != bool(spec.BUNDLED_WEIGHTS)
    if spec.BUNDLED_WEIGHTS:
        assert re.fullmatch(r'[0-9a-f]{64}', spec.BUNDLED_SHA256)
    for _name, size, sha in spec.MODEL_FILES:
        assert size > 0 and re.fullmatch(r'[0-9a-f]{64}', sha)


def test_cpu_lists_parse():
    assert engine._cpu_list('0-3,8,10-11\n') == [0, 1, 2, 3, 8, 10, 11]
