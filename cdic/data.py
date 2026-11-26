"""MSC (Multi-Session Chat, Xu et al. 2022) pulled from the official ParlAI tarball.

Each episode chains previous_dialogs (sessions 1..k-1) with dialog (session k), all run through one memory in order.
  train: msc_dialogue/session_4/train.txt -> 1,001 episodes (= the paper's "official training split with 1 001 episodes"),
         sessions 1-4 each.
  eval : msc_dialogue/session_5/{valid,test}.txt -> 500/501 episodes, sessions 1-5.
Turn split (CHOICE, paper silent): within each session, utterances pair up by position (u0,u1), (u2,u3), ...;
q_t = even utterance (session opener, "Speaker 1"), r_t = next odd utterance ("Speaker 2" = the model).
A leftover unpaired utterance gets dropped. Speaker roles alternate in all files (checked).
"""
import json
import os
import tarfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/parlai/msc/msc_v0.1.tar.gz"


def msc_episodes(root, session=4, split="train"):
    """Returns [[{'session': s, 'q': str, 'r': str}, ...], ...]."""
    path = os.path.join(root, "msc", "msc_dialogue", f"session_{session}", f"{split}.txt")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "msc_v0.1.tar.gz")
        urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as f:
            f.extractall(root, filter="data")
    eps = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            sessions = [p["dialog"] for p in d["previous_dialogs"]] + [d["dialog"]]
            eps.append([{"session": s + 1, "q": u[i]["text"], "r": u[i + 1]["text"]}
                        for s, u in enumerate(sessions) for i in range(0, len(u) - 1, 2)])
    return eps


def tokenize(tok, ep, max_len=128):
    """CHOICE: each utterance is cut to max_len tokens (paper gives no max length)."""
    ids = lambda s: tok(s, add_special_tokens=False).input_ids[:max_len]
    return [(ids(x["q"]), ids(x["r"])) for x in ep]


if __name__ == "__main__":
    import sys
    eps = msc_episodes(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(len(eps), "episodes;", sum(map(len, eps)) / len(eps), "turns/episode;", eps[0][0])
