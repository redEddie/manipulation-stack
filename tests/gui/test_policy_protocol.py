"""Policy server <-> client wire contract (mstack/comm/policy_protocol.py).

No GPU, no model, no robot: a fake backend stands in for the policy, and the
real HTTP layer of apps/policy_server.py serves it on a free local port.

What it pins down:
  1. images survive encode -> JSON -> decode bit-exactly;
  2. PolicyInfo refuses an action type the client cannot execute;
  3. GET /info round-trips into a PolicyInfo the client accepts;
  4. a wrong-size image is answered 400 with a message, and the server keeps
     serving (a bad request must not kill the policy process);
  5. /predict before /reset is a 400, not a crash.
"""
import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "apps"))

from mstack.comm.policy_protocol import (  # noqa: E402
    JOINT_ABSOLUTE,
    STATE_KEY,
    PolicyInfo,
    decode_observation,
    encode_image,
    image_key,
)
import policy_server  # noqa: E402

# 1 ------------------------------------------------------------------------
img = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
back = decode_observation(json.loads(json.dumps({"x": encode_image(img)})))["x"]
assert back.dtype == np.uint8 and np.array_equal(back, img)
print("1. image encode/decode bit-exact OK")

# 2 ------------------------------------------------------------------------
base = dict(policy="fake", checkpoint="-", fps=20, image_size=224, cameras=["agent", "wrist"],
            state_dim=8, action_type=JOINT_ABSOLUTE, action_dim=8, chunk_size=4)
for bad in ({"action_type": "joint_velocity"}, {"action_dim": 7}, {"fps": 0}):
    try:
        PolicyInfo.from_dict({**base, **bad})
    except ValueError:
        continue
    raise AssertionError(f"PolicyInfo accepted {bad}")
print("2. PolicyInfo rejects unexecutable conventions OK")


# 3-5 ----------------------------------------------------------------------
class FakeBackend:
    def __init__(self):
        self.instruction = None

    def info(self):
        return PolicyInfo(**base)

    def reset(self, instruction):
        self.instruction = instruction or self.instruction
        return self.instruction

    def predict(self, obs):
        if not self.instruction:
            raise policy_server.RequestError("no instruction yet")
        if obs[image_key("agent")].shape != (224, 224, 3):
            raise policy_server.RequestError("wrong size")
        return np.tile(np.asarray(obs[STATE_KEY], float), (4, 1))


srv = policy_server.PolicyServer(FakeBackend())
httpd = None


def serve():
    global httpd
    # PolicyServer.run() blocks on serve_forever; reuse its handler on port 0.
    orig = policy_server.HTTPServer

    def capture(addr, handler):
        global httpd
        httpd = orig(("127.0.0.1", 0), handler)
        return httpd

    policy_server.HTTPServer = capture
    try:
        srv.run("127.0.0.1", 0)
    finally:
        policy_server.HTTPServer = orig


threading.Thread(target=serve, daemon=True).start()
for _ in range(100):
    if httpd is not None:
        break
    threading.Event().wait(0.02)
url = f"http://127.0.0.1:{httpd.server_address[1]}"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


code, body = call("/info")
assert code == 200 and PolicyInfo.from_dict(body).chunk_size == 4, (code, body)
print("3. GET /info -> PolicyInfo OK")

obs = {STATE_KEY: list(range(8)), image_key("agent"): encode_image(img), image_key("wrist"): encode_image(img)}
code, body = call("/predict", obs)
assert code == 400 and "instruction" in body["error"], (code, body)
print("5. /predict before /reset -> 400 OK")

assert call("/reset", {"instruction": "pick"})[0] == 200
small = dict(obs, **{image_key("agent"): encode_image(img[:100, :100])})
code, body = call("/predict", small)
assert code == 400 and body["error"], (code, body)
code, body = call("/predict", obs)
assert code == 200 and np.asarray(body["actions"]).shape == (4, 8), (code, body)
print("4. wrong-size image -> 400, server still serving OK")

httpd.shutdown()
print("\n정책 프로토콜 계약 통과")
