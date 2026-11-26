"""LLM explanation + claim check on the displayed Top-10 paths of a saved UPGPR beam. Read-only on UPGPR data.

  prepare   extract MOOCCube relations/entities from the zip and run preprocess_mooc.py (vocab files)
  sample    rebuild the frozen Top-10 (as m5_paths.py), check gates, sample displayed paths  -> sample.jsonl
  generate  writer + extractor with Qwen, rule-based check (GPU)                              -> results.jsonl
  report    per-group summary and the hand-label sheet                                       -> summary.json, hand_labels.csv
  score     checker precision/recall against a filled hand-label sheet                       -> checker_scores.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "src" / "UPGPR")]
import m5_metrics as mm  # noqa: E402
import m5_paths as mp  # noqa: E402
import spike_b_posthoc as sb  # noqa: E402
from llm_explain import core, prompts, report  # noqa: E402

QUOTAS = {"co_learner": 100, "concept": 100, "teacher": 25, "school": 25}
# docs/m5/S0_PATH_TYPES.md, control arm, seed 23, selector S0
EXPECTED_S0_SEED23 = {"co_learner": 56877, "concept": 7603, "teacher": 27, "school": 4}
EMB_SHA = ("5769c715", "e787")  # docs/m5/REPORT.md
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def dump(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


# ------------------------------------------------------------------ sample
def displayed(results, scores, labels, train, self_loop="self_loop"):
    pred, frozen = mm.build(results, scores, labels, train)
    out = []
    for uid, top in frozen.items():
        for rank, s0 in enumerate(top, 1):
            item = s0[2][-1][2]
            ch = sb.select_path("S0", pred[uid][item], s0, self_loop)
            out.append({"uid": int(uid), "rank": rank, "item": int(item),
                        "group": mp.group_of(mp.pattern(ch[2], self_loop)), "path": ch[2]})
    return out


def check_counts(counts, expected=EXPECTED_S0_SEED23):
    got = {g: n for g, n in counts.items() if n}
    if got != expected:
        raise ValueError(f"Top-10 group counts differ from the published M5 S0 counts: got {got}, expected {expected}")


def sample(shown, quotas, seed):
    rng = random.Random(seed)
    out = []
    for g, q in quotas.items():
        pool = sorted((d for d in shown if d["group"] == g), key=lambda d: (d["uid"], d["rank"]))
        out += rng.sample(pool, min(q, len(pool)))
    return [{"example_id": i, **d} for i, d in enumerate(out)]


def check_edges(paths, tables, self_loop="self_loop"):
    """Every edge must exist in the regenerated processed_files, else ids and names are misaligned."""
    for p in paths:
        for (_, t0, i0), (rel, t1, i1) in zip(p, p[1:]):
            if rel == self_loop:
                continue
            if rel == "enroll":
                ok = ((i0, i1) if t0 == "user" else (i1, i0)) in tables["enroll"]
            else:
                c, other = (i0, i1) if t0 == "item" else (i1, i0)
                ok = other in tables[rel][c]
            if not ok:
                raise ValueError(f"edge not in processed_files: {t0} {i0} -{rel}-> {t1} {i1}")


def load_tables(vocab_dir):
    lines = lambda n: Path(vocab_dir, n).read_text(encoding="utf-8").split("\n")
    sets = lambda n: [set(map(int, l.split())) for l in lines(n)]
    return {"item_concept": sets("course_concepts.txt"), "item_teacher": sets("course_teachers.txt"),
            "item_school": sets("course_school.txt"),
            "enroll": {tuple(map(int, l.split())) for l in lines("enrolments.txt") if l}}


def namer(vocab_dir, entity_dir):
    files = {"item": ("courses", "course"), "concept": ("concepts", "concept"),
             "teacher": ("teachers", "teacher"), "school": ("schools", "school")}
    vocab, ents = {}, {}
    for t, (v, e) in files.items():
        vocab[t] = Path(vocab_dir, f"{v}.txt").read_text(encoding="utf-8").split("\n")
        rows = read_jsonl(Path(entity_dir, f"{e}.json"))
        ents[t] = {o["id"].replace(" ", "_") if t == "teacher" else o["id"]: o for o in rows}

    def name_of(t, i):
        if t == "user":
            return f"learner {i}"
        o = ents[t][vocab[t][i]]  # KeyError = missing entity, stop rather than show a raw id
        raw = (o.get("en") if t == "concept" else None) or o["name"]
        return " ".join(raw.split()) or o["id"]

    return name_of


def cmd_prepare(a):
    want = {"relations": ["course-concept", "school-course", "teacher-course", "user-course"],
            "entities": ["course", "concept", "teacher", "school"]}
    need = {(d, f"{n}.json") for d, ns in want.items() for n in ns}
    found = set()
    with zipfile.ZipFile(a.zip) as z:
        for info in z.infolist():
            p = Path(info.filename)
            key = (p.parent.name, p.name)
            if key in need and "__MACOSX" not in p.parts:
                dst = REPO / "data/mooc/MOOCCube" / key[0]
                dst.mkdir(parents=True, exist_ok=True)
                with z.open(info) as s, open(dst / key[1], "wb") as d:
                    shutil.copyfileobj(s, d)
                found.add(key)
    if need - found:
        raise ValueError(f"missing in zip: {need - found}")
    subprocess.run([sys.executable, "src/UPGPR/preprocess_mooc.py"], cwd=REPO, check=True)


def cmd_sample(a):
    import pickle
    import m5_funnel_run as fr
    import m5_run
    emb_sha = sha256(os.path.join(a.tmp_dir, "transe_embed.pkl"))
    if not (emb_sha.startswith(EMB_SHA[0]) and emb_sha.endswith(EMB_SHA[1])):
        raise ValueError(f"transe_embed.pkl sha256 {emb_sha} is not the M5 embedding")
    _, kg, scores, train, test = fr.load_inputs(a.root, a.tmp_dir)
    beam = os.path.join(m5_run.arm_dir(a.root, 23, 0.0), "beam_test", "beam.pkl")
    with open(beam, "rb") as f:
        shown = displayed(pickle.load(f), scores, test, train, kg["self_loop"])
    counts = Counter(d["group"] for d in shown)
    print("displayed slots by group:", dict(counts))
    check_counts(counts)
    picked = sample(shown, QUOTAS, a.seed)
    check_edges([d["path"] for d in picked], load_tables(a.vocab_dir), kg["self_loop"])
    name_of = namer(a.vocab_dir, a.entity_dir)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "sample.jsonl"), "w", encoding="utf-8") as f:
        for d in picked:
            f.write(json.dumps({**d, "facts": core.facts(d["path"], name_of)}, ensure_ascii=False) + "\n")
    dump({"beam": beam, "beam_sha256": sha256(beam), "embed_sha256": emb_sha, "arm": "control (lambda0)",
          "model_seed": 23, "split": "test", "sample_seed": a.seed, "quotas": QUOTAS,
          "displayed_slot_counts": counts, "sampled": Counter(d["group"] for d in picked)},
         os.path.join(a.out, "meta_sample.json"))
    print("sampled:", dict(Counter(d["group"] for d in picked)))


# ------------------------------------------------------------------ generate
def process(ex, chat):
    f = ex["facts"]
    text = chat(prompts.writer(f), 300)
    raw = chat(prompts.extractor(text), 600)
    claims = core.parse_claims(raw)
    if claims is not None:
        claims = [{**c, "label": core.label_claim(c, f, text)} for c in claims]
    return {**ex, "path_facts": prompts.fact_lines(f), "explanation": text, "extractor_raw": raw, "claims": claims,
            "cjk_leak": core.cjk_leak(text, f), "flag_wording": core.has_flag_wording(text, f),
            "code_name_mismatches": core.code_name_mismatches(text, f)}


def load_model(name, revision):
    import torch
    import transformers
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    transformers.set_seed(0)
    revision = HfApi().model_info(name, revision=revision).sha  # resolve first, so the logged sha is what loads
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16))

    def chat(prompt, max_new):
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    info = {"model": name, "revision": revision, "quantization": "bitsandbytes 4-bit, compute float16",
            "decoding": "greedy (do_sample=False)", "seed": 0, "torch": torch.__version__,
            "transformers": transformers.__version__}
    return chat, info


def cmd_generate(a):
    examples = read_jsonl(os.path.join(a.out, "sample.jsonl"))
    if a.smoke:
        first = {}
        for e in examples:
            first.setdefault(e["group"], e)
        examples = list(first.values())
    dest = os.path.join(a.out, "results_smoke.jsonl" if a.smoke else "results.jsonl")
    meta_path = os.path.join(a.out, "meta_generate.json")
    fixed = {"writer_prompt_template": prompts.writer(
        {"entities": [{"code": "L1", "name": "<learner>"}, {"code": "C1", "name": "<course>"}],
         "triples": [("L1", "enrolled_in", "C1")], "target": "L1", "final": "C1"}),
        "extractor_prompt": prompts.EXTRACT, "sample_sha256": sha256(os.path.join(a.out, "sample.jsonl"))}
    done = set()
    if a.smoke and os.path.exists(dest):
        os.remove(dest)
    elif os.path.exists(dest):  # resume only under the same prompts and sample
        with open(meta_path, encoding="utf-8") as f:
            old = json.load(f)
        if {k: old[k] for k in fixed} != fixed:
            raise ValueError("results.jsonl was made with other prompts or another sample; use a new --out")
        done = {r["example_id"] for r in read_jsonl(dest)}
        a.revision = old["revision"]
    chat, info = load_model(MODEL, a.revision)
    if not a.smoke:
        dump({**info, **fixed}, meta_path)
    with open(dest, "a", encoding="utf-8") as f:
        for n, ex in enumerate((e for e in examples if e["example_id"] not in done), 1):
            r = process(ex, chat)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if a.smoke:
                print(f"== {r['group']}\n{r['path_facts']}\n-> {r['explanation']}")
                for c in r["claims"] or ["EXTRACTION FAILED: " + r["extractor_raw"]]:
                    print("  ", c)
            elif n % 10 == 0:
                print(n, "done", flush=True)
    print("wrote", dest)


# ------------------------------------------------------------------ report / score
def cmd_report(a):
    rows = read_jsonl(os.path.join(a.out, "results.jsonl"))
    n = len(read_jsonl(os.path.join(a.out, "sample.jsonl")))
    if len(rows) != n:
        raise ValueError(f"results incomplete: {len(rows)}/{n}; rerun generate")
    sheet = os.path.join(a.out, "hand_labels.csv")
    s = report.summarize(rows)
    dump({"status": "UNVALIDATED: checker not yet compared with hand labels; not findings", "groups": s},
         os.path.join(a.out, "summary.json"))
    print("UNVALIDATED checker output, not findings")
    print(json.dumps(s, indent=1))
    if os.path.exists(sheet):
        print("kept existing", sheet, "(delete it yourself to regenerate)")
        return
    with open(sheet, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=report.CSV_FIELDS)
        w.writeheader()
        w.writerows(report.label_rows(report.pick_for_labeling(rows)))
    print("wrote", sheet)


def cmd_score(a):
    rows = read_jsonl(os.path.join(a.out, "results.jsonl"))
    with open(a.labels, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    s = report.score_labels(sheet, rows)
    odd = Counter(r["group"] for r in {r["example_id"]: r for r in sheet}.values() if r["path_odd"].strip())
    missed = sum(bool(r["missed_claim"].strip()) for r in sheet)
    out = {"checker_flags_vs_hand_labels": s, "examples_marked_path_odd": odd, "rows_with_missed_claim": missed,
           "note": "precision/recall cover extracted claims only; claims the extractor missed are not in them"}
    dump(out, os.path.join(a.out, "checker_scores.json"))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--zip", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--root", required=True, help="M5 root (has effective_config.json)")
    p.add_argument("--tmp_dir", required=True, help="M5 tmp/mooc_train4_eval3")
    p.add_argument("--vocab_dir", default=str(REPO / "data/mooc/processed_files"))
    p.add_argument("--entity_dir", default=str(REPO / "data/mooc/MOOCCube/entities"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p = sub.add_parser("generate"); p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--revision", default="main")
    p = sub.add_parser("report"); p.add_argument("--out", required=True)
    p = sub.add_parser("score"); p.add_argument("--out", required=True); p.add_argument("--labels", required=True)
    a = ap.parse_args()
    {"prepare": cmd_prepare, "sample": cmd_sample, "generate": cmd_generate, "report": cmd_report,
     "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
