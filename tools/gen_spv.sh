#!/usr/bin/env bash
# gen_spv.sh — compile *.comp shaders to SPIR-V and emit C headers.
# Requires glslangValidator + python3 (both in MSYS2).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build/spv
for comp in src/*.comp; do
    name=$(basename "$comp" .comp)
    out="build/spv/${name}.spv"
    hdr="src/${name}_spv.h"
    glslangValidator -V "$comp" -o "$out"
    python3 - "$out" "$hdr" "$comp" <<'PY'
import sys
spv, hdr, comp = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(spv, 'rb').read()
base = hdr.split('/')[-1].removeprefix('wubu_vk_').removesuffix('_spv.h').upper()
var = f"WUBU_VK_{base}_SPV"
lines = [f"/* auto-generated from {comp} - do not edit */",
         f"static const unsigned char {var}[{len(data)}] = {{"]
for i in range(0, len(data), 12):
    chunk = data[i:i+12]
    lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
lines.append("};")
open(hdr, 'w').write("\n".join(lines) + "\n")
print(f"  {comp} -> {hdr} ({len(data)//4} words)")
PY
done
