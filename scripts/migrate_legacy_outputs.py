#!/usr/bin/env python3
"""Archive historical generated outputs and remove unsupported active claims.

Original bytes are preserved once under output/legacy_archive with SHA-256.
Measured 3B vectors/metrics are retained; random 7B vectors are removed from
the active artifact. This migration never creates replacement measurements.
"""
import hashlib
import html
import json
from pathlib import Path
import shutil


def main():
    output = Path(__file__).resolve().parents[1] / "output"
    archive = output / "legacy_archive"
    archive.mkdir(exist_ok=True)
    manifest = {}
    for stem, synthetic in (("Qwen2_5-7B",True),("Qwen_Qwen2.5-3B",False)):
        for suffix in ("_map.json","_soulprint.json","_atlas.html"):
            source = output/(stem+suffix)
            original = archive/source.name
            if source.exists() and not original.exists():
                shutil.copy2(source,original)
            if original.exists():
                manifest[original.name] = hashlib.sha256(original.read_bytes()).hexdigest()
        map_path = output/(stem+"_map.json")
        if not map_path.exists():
            continue
        data = json.loads(map_path.read_text())
        data["evidence_level"] = "historical_metrics_without_geometry" if synthetic else "legacy_activation_probe"
        data["invariant_markers"] = []
        data["layer_depth_profile"] = {}
        data["provenance"] = {"migration":"2026-10-11", "original_archive":str(archive/map_path.name),
                              "layer_labels_valid":False,"held_out_confirmation":False,
                              "direction_source":"removed random placeholders" if synthetic else "historical final-hidden-state forward passes"}
        for record in data["concept_directions"].values():
            record["layer"] = None
            if synthetic:
                record["direction"] = []
                record["contexts"] = []
                record["metrics"]["statistic_note"] = "Historical whole-layer maximum of measured LOO, not a mean or independent test"
        map_path.write_text(json.dumps(data,indent=2,ensure_ascii=False))
        candidates = []
        if not synthetic:
            for record in data["concept_directions"].values():
                candidates.append({"marker_type":"candidate_consistency",
                    "description":"Historical template-limited statistic, not a confirmed invariant",
                    "concepts_involved":[record["concept_a"],record["concept_b"]],
                    "strength":record["metrics"].get("loo_cosine_mean"),"confidence":None,
                    "evidence":record["metrics"]})
        identity = {"model_name":data["model_name"],"timestamp":data["timestamp"],"version":"0.2.0",
                    "status":"historical_unconfirmed_probe_summary","invariant_markers":[],
                    "differential_markers":[],"candidate_markers":candidates,
                    "depth_profiles":{},"behavioral_tendencies":{},
                    "identity_summary":"No confirmed global invariants, depth plateau or behavioral identity. Original outputs archived; unsupported defaults removed."}
        (output/(stem+"_soulprint.json")).write_text(json.dumps(identity,indent=2,ensure_ascii=False))
        rows = "".join(f'<tr><td>{html.escape(key)}</td><td>{r["metrics"].get("loo_cosine_mean")}</td></tr>'
                       for key,r in data["concept_directions"].items())
        note = ("历史 demo 的方向是随机占位，已从活动 JSON 移除；这里只保留历史实测指标。"
                if synthetic else "保留历史实测的最终层向量和指标；旧层标签不可信，已留空。")
        page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>IOTA historical probes</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:20px;line-height:1.8}}td{{padding:8px;border-bottom:1px solid #aaa}}</style>
<h1>IOTA · 历史探测结果</h1><p>{html.escape(note)}</p><p>未测量的深度平台、人格分数和 invariant 声明已撤回。原始产物保存在 legacy_archive 中，仅供溯源。</p>
<p>这些统计量不构成 Model Semantic Identity。条件地图与探索轨迹使用 scripts/discover.py 的新报告。</p><table><tr><th>词对</th><th>历史 LOO 指标</th></tr>{rows}</table></html>'''
        (output/(stem+"_atlas.html")).write_text(page)
    (archive/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (archive/"README.md").write_text("# Historical outputs — invalid identity claims\n\nThese original bytes are preserved for audit only. 7B directions were random placeholders; both original Soulprints contained unmeasured defaults. Do not use them as measured geometry or model identity. See ../ active corrected artifacts and ../../REVIEW_2026-10-11.md.\n")
    print(f"Corrected legacy outputs; original hashes retained in {archive/'manifest.json'}")


if __name__ == "__main__":
    main()
