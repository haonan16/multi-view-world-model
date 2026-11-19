"""Mechanical port of the saved pushed RobotFitter; retain an exact source diff.

Free wrist translation is part of the generalized coordinate vector. The outer
base transform is identity, NOT a claim that the human wrist is stationary.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path


def build(run):
    src = run / 'upstream/artirecon/fit_robot.py'
    original = src.read_text()
    s = original
    def change(a, b):
        nonlocal s
        if a not in s:
            raise ValueError('Unrecognized upstream code: ' + a)
        s = s.replace(a, b)
    change('from .data import backproject', 'from .mano_adapter import backproject')
    change('from .robot import PANDA_READY, URDFRobot, axis_angle',
           'from .mano_adapter import MANORobot as URDFRobot, axis_angle')
    change('return torch.cat([q8, q8[:, -1:]], 1)', 'return q8')
    change('self.lo[:8] - q8', 'self.lo - q8')
    change('q8 - self.hi[:8]', 'q8 - self.hi')
    change('return ch + w_depth * dl + lim, ch, dl',
           'return ch + w_depth * dl + lim + 0.001 * self.robot.betas.square().mean() + 0.0001 * q8[:, 3:-3].square().mean(1), ch, dl')
    change('q8 = q8.clone().requires_grad_(True)',
           'qraw = (q8.clone() / self.robot.optim_scale).requires_grad_(True)\n        q8 = qraw * self.robot.optim_scale')
    change('params = [{"params": [q8], "lr": lr}]',
           'params = [{"params": [qraw], "lr": lr}]\n        fit_shape = len(set(frames)) > 1\n        self.robot.betas.requires_grad_(fit_shape)\n        if fit_shape:\n            params.append({"params": [self.robot.betas], "lr": 0.01})')
    change('for it in range(iters):\n            s =',
           'for it in range(iters):\n            q8 = qraw * self.robot.optim_scale\n            s =')
    change('            opt.step()\n            sched.step()',
           '            opt.step()\n            with torch.no_grad():\n                self.robot.betas.clamp_(-3, 3)\n            sched.step()')
    change('        with torch.no_grad():\n            L, ch, dl = self.loss(q8,',
           '        q8 = qraw * self.robot.optim_scale\n        with torch.no_grad():\n            L, ch, dl = self.loss(q8,')
    change('ready = torch.tensor(PANDA_READY[:8], device=self.dev)', 'ready = self.robot.initial[0].clone()')
    change('torch.randn(H, 8, generator=g).to(self.dev)', 'torch.randn(H, self.robot.dof, generator=g).to(self.dev) * self.robot.noise_scale')
    change('self.hi[:8]', 'self.hi')
    change('self.lo[:8]', 'self.lo')
    change('inits = [qs[-1], ready]', 'inits = [qs[-1], self.robot.initial[t]]')
    change('torch.randn(8, generator=g).to(self.dev)', 'torch.randn(self.robot.dof, generator=g).to(self.dev) * self.robot.noise_scale')
    change('                z[7] = torch.rand(1, generator=g).item() * 0.04 - qs[-1][7]\n', '')
    # Unreachable fixed-Panda base search is removed, not silently executed for hands.
    start = s.index('    def search_base(')
    end = s.index('    def optimise(', start)
    s = s[:start] + '    def search_base(self, *args, **kwargs):\n        raise RuntimeError("MANO uses free wrist coordinates, not a table-mounted Panda base search")\n\n' + s[end:]
    dest = run / 'runtime/amani_comparison/fit_robot_mano.py'
    compile(s, str(dest), 'exec')
    dest.write_text(s)
    geom=dest.parent/'geom.py'
    geom.write_text(geom.read_text().replace('from .data import backproject','from .mano_adapter import backproject'))
    (dest.parent / 'robotfitter-mano.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), s.splitlines(True), fromfile='pushed/fit_robot.py', tofile='port/fit_robot_mano.py')))
    audit = dict(upstream_sha256=hashlib.sha256(original.encode()).hexdigest(),
                 port_sha256=hashlib.sha256(s.encode()).hexdigest(),
                 preserved=['24 first-frame hypotheses', '10 forward starts, 300 iterations',
                            '6 backward starts, 150 iterations, accept only improvement',
                            '300-iteration all-frame refinement', 'original depth and robust Chamfer',
                            'Adam and cosine learning-rate schedule'],
                 adaptations=['MANO surface and free wrist instead of URDF FK and fixed base',
                              'RGB triangulation supplies each-frame initialization; never previous RGB-D fit',
                              'PCA coefficient bounds replace robot limits; no tied fingers',
                              'translation-specific step/noise scale in metres',
                              'one shared MANO shape optimized in joint refinement',
                              'no Panda base/yaw gauge search; no robot-proprioception camera ICP',
                              'publisher camera calibration fixed; depth self-calibration disabled for comparison'])
    (dest.parent / 'robotfitter-mano-provenance.json').write_text(json.dumps(audit, indent=2))
    print(dest)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    build(p.parse_args().run)
