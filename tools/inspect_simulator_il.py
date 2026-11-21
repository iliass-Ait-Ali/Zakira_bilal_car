"""Optional audit utility: pip install dnfile==0.17.0 dncil==1.0.2.

Reads metadata/instructions only. Does not load or execute .NET assemblies.
"""
import argparse
import json
import hashlib
from pathlib import Path


def inspect(path):
    import dnfile
    from dncil.cil.body.reader import read_method_body_from_bytes
    pe=dnfile.dnPE(str(path))
    def resolve(token):
        if not hasattr(token,"value"): return str(token)
        value=token.value
        if value>>24 == 0x70: return repr(pe.net.user_strings.get(value&0xffffff).value)
        try:
            row=pe.net.mdtables.tables[value>>24].rows[(value&0xffffff)-1]
            return str(getattr(row,"Name",getattr(row,"TypeName",token)))
        except (KeyError,IndexError): return str(token)
    lines=[f"Assembly: {path.name}", f"SHA256: {hashlib.sha256(path.read_bytes()).hexdigest()}"]
    names={"TcpCarHandler","Car","UnityStandardCarAdapter","CarController","VersionCheck"}
    methods={"OnControlsRecv","SendTelemetry","FixedUpdate","RequestSteering","RequestFootBrake",".ctor","Move","Start"}
    for t in pe.net.mdtables.TypeDef:
        if str(t.TypeName) not in names: continue
        for method in t.MethodList:
            m=method.row
            if str(m.Name) not in methods or not m.Rva: continue
            body=read_method_body_from_bytes(pe.get_data(m.Rva,20000))
            lines.append(f"\n{t.TypeName}.{m.Name}")
            lines.extend(f"{i.offset:04x}: {i.opcode.name} {resolve(i.operand)}" for i in body.instructions)
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--managed",default="DonkeySimWin/DonkeySimWin/donkey_sim_Data/Managed")
    p.add_argument("--out",default="docs/binary_protocol_il.txt")
    a=p.parse_args()
    text="\n\n".join(inspect(Path(a.managed)/name) for name in ("Assembly-CSharp.dll","Assembly-CSharp-firstpass.dll"))
    Path(a.out).write_text(text,encoding="utf-8")
    print(json.dumps({"output":a.out,"lines":len(text.splitlines()),"execution":"static metadata only"}))


if __name__ == "__main__": main()
