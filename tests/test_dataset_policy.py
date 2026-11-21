import copy
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from PIL import Image
from racing.config import load_config, dump_json, sha256
from racing.control import Command
from racing.data import checked_record, load_sessions, prepare, read_prepared, require_baseline_policy
from racing.recording import Recorder
from test_pipeline import workspace_temp


def record(i=0, sequence="sequence"):
    t=i*.05
    return {"sequence_id":sequence,"frame_id":i+1,"sim_time":t,"source":"baseline","label_valid":True,
            "received_monotonic":100+t,"sent_monotonic":100+t+.01,"decision_age_s":.01,"speed":1,
            "lane_confidence":17,"command":asdict(Command(.1,.1,0)),"telemetry":{"hit":"none","cte":.1,"time":t,"speed":1}}


class DatasetPolicyTests(unittest.TestCase):
    def setUp(self): self.cfg=load_config()

    def test_cannot_enable_manual_by_old_configuration(self):
        for sources in (["manual"],["manual","baseline"],["cnn"],[]):
            cfg=copy.deepcopy(self.cfg); cfg["data"]["allowed_sources"]=sources
            with self.assertRaises(ValueError): require_baseline_policy(cfg)

    def test_bad_record_sources_clocks_safety_and_pixels_rejected(self):
        with workspace_temp() as td:
            path=Path(td); Image.new("RGB",(255,255)).save(path/"image.png")
            good=dict(record(),session_id="s",image="image.png")
            meta={"source":"baseline","session_id":"s"}
            self.assertEqual(checked_record(good,meta,path,self.cfg)["source"],"baseline")
            for key,value in [("source","manual"),("source","cnn"),("label_valid",False),("decision_age_s",.02),
                              ("sent_monotonic",99),("speed",float("nan")),("lane_confidence",0)]:
                bad=dict(good); bad[key]=value
                with self.subTest(key=key,value=value), self.assertRaises(ValueError): checked_record(bad,meta,path,self.cfg)
            for key,value in [("hit","wall"),("cte",1.01),("cte",float("nan")),("time",3),("speed",2)]:
                bad=copy.deepcopy(good); bad["telemetry"][key]=value
                with self.subTest(key=key), self.assertRaises(ValueError): checked_record(bad,meta,path,self.cfg)
            with self.assertRaises(ValueError): checked_record(good,{**meta,"source":"manual"},path,self.cfg)
            Image.new("RGB",(160,120)).save(path/"image.png")
            with self.assertRaises(ValueError): checked_record(good,meta,path,self.cfg)

    def test_manual_frame_cuts_future_targets_without_relabeling(self):
        with workspace_temp() as td:
            writer=Recorder(td,"group",self.cfg,"baseline")
            for i in range(30):
                r=record(i)
                if i==12: r["source"]="manual"
                writer.submit(Image.new("RGB",(255,255)),r)
            writer.close()
            rows,rejected,_=load_sessions(td,self.cfg)
            self.assertEqual(len(rejected),1); self.assertEqual(rejected[0]["line"],13)
            self.assertTrue(all(r["source"]=="baseline" for r in rows))
            self.assertTrue(all(r["mask"][2]==0 for r in rows if .15<=r["sim_time"]<.6))

    def test_optional_trim_drops_frames_just_before_an_invalid_frame(self):
        with workspace_temp() as td:
            writer=Recorder(td,"group",self.cfg,"baseline")
            for i in range(40):
                r=record(i)
                if i==30: r["telemetry"]["cte"]=5.0  # expert left the CTE bound
                writer.submit(Image.new("RGB",(255,255),(i,i,i)),r)
            writer.close()
            _,plain,_=load_sessions(td,self.cfg)
            cfg=copy.deepcopy(self.cfg); cfg["data"]["trim_before_invalid_s"]=.5
            rows,rejected,_=load_sessions(td,cfg)
            self.assertEqual(len(plain),1)
            trimmed=sorted(r["line"] for r in rejected if "before an invalid" in r["reason"])
            self.assertEqual(trimmed,list(range(21,31)))  # frames 20..29 (t 1.0..1.45 s)
            self.assertTrue(all(not 1.0<=r["sim_time"]<1.5 for r in rows))
            self.assertTrue(any(r["sim_time"]>=1.55 for r in rows))

    def test_revalidation_only_for_listed_scenes_and_still_checks_safety(self):
        with workspace_temp() as td:
            path=Path(td); Image.new("RGB",(255,255)).save(path/"image.png")
            rec=dict(record(),session_id="s",image="image.png",label_valid=False); rec["telemetry"]["cte"]=3.0
            meta={"source":"baseline","session_id":"s","config":{"runtime":{"scene":"waveshare"}}}
            with self.assertRaises(ValueError): checked_record(rec,meta,path,self.cfg)
            cfg=copy.deepcopy(self.cfg); cfg["data"]["revalidate_scenes"]=["waveshare"]
            cfg["data"]["cte_reference"]={"waveshare":{"center":0.0,"max_abs":6.0}}
            self.assertEqual(checked_record(rec,meta,path,cfg)["source"],"baseline")
            for key,value in [("hit","wall"),("cte",7.0)]:
                bad=copy.deepcopy(rec); bad["telemetry"][key]=value
                with self.subTest(key=key), self.assertRaises(ValueError): checked_record(bad,meta,path,cfg)
            with self.assertRaises(ValueError): checked_record(dict(rec,lane_lost=True),meta,path,cfg)
            with self.assertRaises(ValueError): checked_record(dict(rec,source="manual"),meta,path,cfg)

    def test_same_sequence_cannot_be_split_across_independent_groups(self):
        with workspace_temp() as td:
            for group in ("a","b"):
                writer=Recorder(td,group,self.cfg,"baseline"); writer.submit(Image.new("RGB",(255,255)),record()); writer.close()
            with self.assertRaisesRegex(RuntimeError,"physical sequence"): load_sessions(td,self.cfg)

    def test_prepared_source_is_rechecked_even_if_hash_is_updated(self):
        with workspace_temp() as td:
            root=Path(td)/"sessions"
            for g in range(5):
                writer=Recorder(root,str(g),self.cfg,"baseline")
                for i in range(15): writer.submit(Image.new("RGB",(255,255),(g*30,i,10)),record(i,f"q{g}"))
                writer.close()
            out=Path(td)/"prepared"; prepare(root,out,self.cfg)
            manifest,splits=read_prepared(out,self.cfg)
            self.assertEqual(set(splits), {"train","val","test"})
            for split in ("train","val","test"):
                original=(out/f"{split}.jsonl").read_text()
                rows=[json.loads(s) for s in original.splitlines()]; rows[0]["source"]="manual"
                (out/f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
                altered=copy.deepcopy(manifest); altered["split_hashes"][split]=sha256(out/f"{split}.jsonl")
                dump_json(out/"dataset.json",altered)
                with self.assertRaisesRegex(ValueError,"Non-baseline"): read_prepared(out,self.cfg)
                (out/f"{split}.jsonl").write_text(original); dump_json(out/"dataset.json",manifest)

    def test_duplicate_pixels_removed_across_groups_and_cut_future_labels(self):
        with workspace_temp() as td:
            for g in range(2):
                writer=Recorder(td,str(g),self.cfg,"baseline")
                for i in range(15):
                    color=(0,0,0) if i==7 else (20+g*30,i,10)
                    writer.submit(Image.new("RGB",(255,255),color),record(i,f"q{g}"))
                writer.close()
            rows,rejected,summaries=load_sessions(td,self.cfg)
            self.assertEqual(len(rows),28); self.assertEqual(len(rejected),2)
            self.assertTrue(all(s["accepted"]==14 for s in summaries))
            self.assertTrue(all(r["mask"][2]==0 for r in rows if r["sim_time"]<.35))


if __name__ == "__main__": unittest.main()
