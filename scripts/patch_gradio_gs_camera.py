#!/usr/bin/env python3
"""Patch gradio-model3dgscamera 0.0.3 with less restrictive controls."""

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path


REPLACEMENTS = (
    (
        "this.minAngle = -90, this.maxAngle = 90, this.minZoom = 0.1, "
        "this.maxZoom = 30, this.orbitSpeed = 1, this.panSpeed = 1, "
        "this.zoomSpeed = 1, this.dampening = 0.12,",
        "this.minAngle = -90, this.maxAngle = 90, this.minZoom = 0.02, "
        "this.maxZoom = 300, this.orbitSpeed = 1, this.panSpeed = 1.25, "
        "this.zoomSpeed = 1, this.dampening = 0.18,",
    ),
    (
        "const N = {};\n    let M = !1;",
        'const N = {};\n    l.tabIndex = 0, l.style.outline = "none";\n    let M = !1;',
    ),
    (
        "const y = () => 0.1 + 0.9 * (o - this.minZoom) / "
        "(this.maxZoom - this.minZoom), x = (i) => {\n"
        "      N[i.code] = !0,",
        "const y = () => Math.max(0.1, Math.min(3, o * 0.05)), x = (i) => {\n"
        '      ["KeyW", "KeyA", "KeyS", "KeyD", "KeyQ", "KeyE", '
        '"KeyR", "KeyF", "KeyC", "Space", "ArrowUp", "ArrowDown", '
        '"ArrowLeft", "ArrowRight"].includes(i.code) && p(i), N[i.code] = !0,',
    ),
    (
        "p(i), I = !0, c = i.button === 2, C = i.clientX, X = i.clientY,",
        "p(i), l.focus(), I = !0, c = i.button === 2 || i.button === 1 "
        "|| i.shiftKey, C = i.clientX, X = i.clientY,",
    ),
    (
        "const O = 0.025, P = 0.01, _ = eU.RotationFromQuaternion(U.rotation).buffer, "
        "j = new m(-_[2], -_[5], -_[8]), cU = new m(_[0], _[3], _[6]);\n"
        "      N.KeyS && (e = e.add(j.multiply(O))), N.KeyW && (e = e.subtract(j.multiply(O))), "
        "N.KeyA && (e = e.subtract(cU.multiply(O))), N.KeyD && (e = e.add(cU.multiply(O))), "
        "N.KeyE && (A += P), N.KeyQ && (A -= P), N.KeyR && (R += P), N.KeyF && (R -= P), M = !1;",
        "const O = Math.max(0.005, Math.min(0.25, B * 0.015)) * "
        "(N.ShiftLeft || N.ShiftRight ? 4 : 1), P = 0.015, "
        "_ = eU.RotationFromQuaternion(U.rotation).buffer, "
        "j = new m(-_[2], -_[5], -_[8]), cU = new m(_[0], _[3], _[6]), "
        "oU = new m(0, 1, 0);\n"
        "      N.KeyS && (e = e.add(j.multiply(O))), N.KeyW && (e = e.subtract(j.multiply(O))), "
        "N.KeyA && (e = e.subtract(cU.multiply(O))), N.KeyD && (e = e.add(cU.multiply(O))), "
        "N.Space && (e = e.add(oU.multiply(O))), N.KeyC && (e = e.subtract(oU.multiply(O))), "
        "N.KeyE && (A += P), N.KeyQ && (A -= P), N.KeyR && (R += P), N.KeyF && (R -= P), M = !1;",
    ),
    (
        't && (window.removeEventListener("keydown", x), '
        'window.removeEventListener("keyup", K));',
        't && (l.removeEventListener("keydown", x), '
        'window.removeEventListener("keyup", K));',
    ),
    (
        't && (window.addEventListener("keydown", x), '
        'window.addEventListener("keyup", K)),',
        't && (l.addEventListener("keydown", x), '
        'window.addEventListener("keyup", K)),',
    ),
    (
        "R += f * this.orbitSpeed * 3e-3, R = Math.min(Math.max(R, "
        "this.minAngle * Math.PI / 180), this.maxAngle * Math.PI / 180);\n"
        "      C = i.clientX",
        "R += f * this.orbitSpeed * 3e-3;\n      C = i.clientX",
    ),
    (
        "R += f * this.orbitSpeed * 3e-3, R = Math.min(Math.max(R, "
        "this.minAngle * Math.PI / 180), this.maxAngle * Math.PI / 180), "
        "C = i.touches[0].clientX",
        "R += f * this.orbitSpeed * 3e-3, C = i.touches[0].clientX",
    ),
    (
        "const u = n.subtract(U.position).normalize(), QU = Math.asin(-u.y), "
        "AU = Math.atan2(u.x, u.z);\n"
        "      U.rotation = L.FromEuler(new m(QU, AU, 0));",
        "U.rotation = L.FromEuler(new m(-Q, -F, 0));",
    ),
)


def main():
    installed_version = version("gradio-model3dgscamera")
    if installed_version != "0.0.3":
        raise SystemExit(
            "Camera-control patch requires gradio-model3dgscamera==0.0.3, "
            f"found {installed_version}."
        )

    spec = find_spec("gradio_model3dgscamera")
    if spec is None or spec.submodule_search_locations is None:
        raise SystemExit("gradio_model3dgscamera is not installed.")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    bundles = list((package_dir / "templates" / "component").glob("Canvas3DGSCamera-*.js"))
    if len(bundles) != 1:
        raise SystemExit(f"Expected one Canvas3DGSCamera bundle, found {len(bundles)}.")

    bundle = bundles[0]
    source = bundle.read_text()
    changed = False
    for entry in REPLACEMENTS:
        original, replacement = entry[:2]
        expected_count = entry[2] if len(entry) == 3 else 1
        original_count = source.count(original)
        replacement_count = source.count(replacement)
        if original_count == expected_count and replacement_count == 0:
            source = source.replace(original, replacement)
            changed = True
        elif original_count == 0 and replacement_count == expected_count:
            continue
        else:
            raise SystemExit(
                f"Unexpected bundle contents for {bundle.name}: "
                f"original={original_count}, patched={replacement_count}."
            )

    if changed:
        bundle.write_text(source)
        print(f"Patched camera controls: {bundle}")
    else:
        print(f"Camera controls already patched: {bundle}")


if __name__ == "__main__":
    main()
