import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'actions'))
from freeze_plan import freeze,verify,claim_execution


def test_frozen_evaluation_rejects_plan_and_controller_changes(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'plan.npz').write_bytes(b'fixed targets')
    (inputs/'manifest.json').write_text('{}')
    executor=tmp_path/'executor.py';executor.write_text('# fixed controller')
    freeze(inputs,executor)
    verify(inputs,executor,1.)
    (inputs/'plan.npz').write_bytes(b'adapted after failure')
    with pytest.raises(ValueError,match='Frozen input'):verify(inputs,executor,1.)
    (inputs/'plan.npz').write_bytes(b'fixed targets')
    executor.write_text('# adjusted controller')
    with pytest.raises(ValueError,match='executor/configuration'):verify(inputs,executor,1.)
    with pytest.raises(FileExistsError):freeze(inputs,executor)


def test_frozen_plan_cannot_be_retried_in_another_output(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'plan.npz').write_bytes(b'fixed targets')
    (inputs/'manifest.json').write_text('{}')
    executor=tmp_path/'executor.py';executor.write_text('# controller')
    freeze(inputs,executor)
    claim_execution(inputs,executor,1.,tmp_path/'first')
    verify(inputs,executor,1.)
    with pytest.raises(FileExistsError):
        claim_execution(inputs,executor,1.,tmp_path/'second')


def test_changed_plan_does_not_consume_execution(tmp_path):
    inputs=tmp_path/'inputs';inputs.mkdir()
    (inputs/'plan.npz').write_bytes(b'fixed targets')
    (inputs/'manifest.json').write_text('{}')
    executor=tmp_path/'executor.py';executor.write_text('# controller')
    freeze(inputs,executor)
    (inputs/'plan.npz').write_bytes(b'changed targets')
    with pytest.raises(ValueError,match='Frozen input'):
        claim_execution(inputs,executor,1.,tmp_path/'first')
    assert not (inputs/'execution-started.json').exists()
