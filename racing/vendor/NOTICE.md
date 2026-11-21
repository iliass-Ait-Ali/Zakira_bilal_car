`client.py` and `util.py` originate from
https://github.com/tawnkramer/gym-donkeycar at commit
`a1f4ca6961e17a6929c1f2e883358a1edafb479d`.
The MIT licence is included in `LICENSE`.

Local changes on 2026-11-15: bounded partial nonblocking writes, EOF handling,
byte buffering and safe UTF-8 decoding, scalar-only locale-decimal repair,
payload-free errors, and shutdown/join before descriptor close.
The locale-repair expression consumes whole runs inside strings rather than
one character per alternation, avoiding long GIL stalls on base64 images.
Receive/decode timing is exposed for latency diagnostics.
The unchanged upstream reference files remain in `docs/upstream/`.

The installed gym-donkeycar 1.0.13 distribution lacks both client modules in
the inspected environment. This small pinned copy preserves SDClient without
installing Gym. The public SDClient interfaces are retained while its receive
loop and lifecycle are hardened. Application startup/framing/camera changes
are in `racing/transport.py` and `racing/session.py`. The user's original script
is unchanged.
