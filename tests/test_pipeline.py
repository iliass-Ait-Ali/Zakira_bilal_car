"""Synthetic, offline checks only. No optimizer step, socket or simulator."""
import base64
import copy
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path
from contextlib import contextmanager
import shutil
import uuid
import threading
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from racing.config import load_config, validate_config, contract_hash, dump_json
from racing.control import Controller, Command
from racing.data import nearest, sequence_samples, split_groups, prepare, read_jsonl, sampling_weights
from racing.dataset import DrivingDataset
from racing.drive import decode_telemetry
from racing.model import RacingCNN
from racing.preprocess import stack_images
from racing.recording import Recorder
from racing.train import masked_loss, atomic_checkpoint
from racing.inference import load_checkpoint
from racing.transport import RaceClient
from racing.vendor.client import SDClient
from racing.diagnostics import Diagnostics
from racing.evaluate import summarize

torch.set_num_threads(2)


@contextmanager
def workspace_temp():
    # Python 3.13's Windows mode-0700 temp dirs exclude the restricted runner.
    root = (Path(__file__).parent/".tmp").resolve()
    root.mkdir(exist_ok=True)
    path = root/uuid.uuid4().hex
    path.mkdir()
    try: yield str(path)
    finally:
        if path.resolve().parent != root: raise RuntimeError("Unexpected temporary path")
        shutil.rmtree(path)


class PipelineTests(unittest.TestCase):
    def setUp(self): self.cfg = load_config()

    def test_invalid_config(self):
        for key,value in (("speed_straight",-1),("brake_max",2),("steering_tau_s",float("nan"))):
            c=copy.deepcopy(self.cfg); c["control"][key]=value
            with self.assertRaises(ValueError): validate_config(c)

    def test_contract_catches_camera_change_but_not_speed_tuning(self):
        c=copy.deepcopy(self.cfg); c["control"]["speed_straight"]=4
        self.assertEqual(contract_hash(c),contract_hash(self.cfg))
        c["camera"]["fov"]=100
        self.assertNotEqual(contract_hash(c),contract_hash(self.cfg))

    def test_preprocess_rgb_shape_range(self):
        image=np.zeros((255,255,3),dtype=np.uint8); image[:,:,0]=255
        x,_=stack_images([image],self.cfg)
        self.assertEqual(x.shape,(3,96,160))
        self.assertTrue((x[0]==1).all() and (x[1]==-1).all())
        with self.assertRaises(ValueError): stack_images([image.astype(float)],self.cfg)

    def test_flip_changes_all_horizon_signs_and_keeps_speed(self):
        with workspace_temp() as td:
            p=Path(td)/"image.png"; Image.new("RGB",(255,255)).save(p)
            c=copy.deepcopy(self.cfg); c["data"]["flip_probability"]=1
            row={"images":[str(p)],"steering":[.2,-.3,.4],"mask":[1,1,1],"speed":2}
            _,speed,y,_=DrivingDataset([row],c,True)[0]
            np.testing.assert_allclose(y.numpy(),[-.2,.3,-.4])
            self.assertAlmostEqual(speed.item(),.2)

    def test_pedal_output_is_learned_but_never_mirrored(self):
        c=copy.deepcopy(self.cfg); c["model"]["throttle_output"]=True; c["data"]["flip_probability"]=1
        with self.assertRaises(ValueError): validate_config(c)  # needs a 4th loss weight
        c["training"]["loss_weights"]=c["training"]["loss_weights"]+[.3]; validate_config(c)
        with workspace_temp() as td:
            p=Path(td)/"image.png"; Image.new("RGB",(255,255)).save(p)
            row={"images":[str(p)],"steering":[.2,-.3,.4],"mask":[1,1,1],"speed":2,"pedal":.25}
            _,_,y,mask=DrivingDataset([row],c,True)[0]
            np.testing.assert_allclose(y.numpy(),[-.2,.3,-.4,.25]); self.assertEqual(mask.tolist(),[1,1,1,1])
        model=RacingCNN(c).eval()
        with torch.inference_mode(): self.assertEqual(model(torch.zeros(1,3,96,160),torch.zeros(1,1)).shape,(1,4))
        samples=sequence_samples([{"sim_time":i*.05,"command":{"steering":0,"throttle":.3,"brake":0},"session_id":"s","group_id":"g","sequence_id":"q","image":str(i),"speed":1,"source":"baseline"} for i in range(12)],c)
        self.assertAlmostEqual(samples[0]["pedal"],.3)
        c["control"]["throttle_source"]="network"
        ctl=Controller(c)
        go,_=ctl.step([0,0,0,.4],1,.05); self.assertAlmostEqual(go.throttle,.4); self.assertEqual(go.brake,0)
        stop,_=ctl.step([0,0,0,-.5],1,.05); self.assertEqual(stop.throttle,0); self.assertGreater(stop.brake,0)
        with self.assertRaises(ValueError): ctl.step([0,0,0],1,.05)  # the pedal output is required

    def test_model_one_and_three_frames_forward_only(self):
        for frames in (1,3):
            c=copy.deepcopy(self.cfg); c["model"]["frames"]=frames
            model=RacingCNN(c).eval()
            with torch.inference_mode(): y=model(torch.zeros(2,3*frames,96,160),torch.zeros(2,1))
            self.assertEqual(y.shape,(2,3)); self.assertTrue((y.abs()<=1).all())
            with self.assertRaises(ValueError): model(torch.zeros(2,3*frames,96,160),torch.zeros(2))

    def test_masked_loss_excludes_missing_future(self):
        loss=masked_loss(torch.tensor([[.1,1.]]),torch.zeros(1,2),torch.tensor([[1.,0.]]),torch.tensor([.6,.4]),.1)
        self.assertAlmostEqual(loss.item(),.05,places=6)

    def test_commands_reject_invalid_or_conflicting_values(self):
        for values in ((float("nan"),0,0),(2,0,0),(0,-1,0),(0,.2,.1)):
            with self.assertRaises(ValueError): Command(*values)
        self.assertEqual(Command(-1,0,1).packet()["steering"],"-1")

    def test_controller_brakes_on_future_turn_and_slew_is_bounded(self):
        c=Controller(self.cfg)
        straight,_=c.step([.8,0,0],1,.05)
        self.assertLessEqual(abs(straight.steering),.2)
        braking,info=c.step([0,1,1],5,.05)
        self.assertEqual(braking.throttle,0); self.assertGreater(braking.brake,0)
        self.assertLess(info["target_speed"],5)
        c.stop(); self.assertEqual(c.integral,0)

    def test_invalid_prediction_speed_or_dt(self):
        c=Controller(self.cfg)
        for pred,speed,dt in (([float("nan")]*3,1,.05),([0]*2,1,.05),([0]*3,-1,.05),([0]*3,1,0)):
            with self.assertRaises(ValueError): c.step(pred,speed,dt)

    def test_future_masks_and_distinct_history(self):
        rows=[{"sim_time":i*.05,"command":{"steering":i/20},"session_id":"s","group_id":"g","sequence_id":"q","image":str(i),"speed":1,"source":"baseline"} for i in range(12)]
        samples=sequence_samples(rows,self.cfg)
        self.assertEqual(samples[0]["steering"],[0,.25,.5])
        self.assertEqual(samples[-1]["mask"],[1,0,0])
        c=copy.deepcopy(self.cfg); c["model"]["frames"]=3
        samples=sequence_samples(rows,c)
        self.assertTrue(all(len(set(r["images"]))==3 for r in samples))
        self.assertEqual(nearest([0,.1],.5,.02),None)

    def test_group_split_no_leakage(self):
        rows=[{"group_id":str(i//3)} for i in range(15)]
        splits,groups=split_groups(rows,self.cfg)
        self.assertEqual(len(groups),5)
        sets=[{r["group_id"] for r in splits[k]} for k in ("train","val","test")]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
        with self.assertRaises(ValueError): split_groups(rows[:3],self.cfg)

    def test_record_prepare_dataset_roundtrip_and_corrupt_gap(self):
        with workspace_temp() as td:
            root=Path(td)/"sessions"
            for g in range(5):
                writer=Recorder(root,str(g),self.cfg,"baseline")
                for i in range(15):
                    record={"sequence_id":f"q{g}","frame_id":i+1,"sim_time":i*.05,"source":"baseline","label_valid":True,
                            "received_monotonic":100+i*.05,"sent_monotonic":100+i*.05+.01,"decision_age_s":.01,
                            "speed":1,"lane_confidence":17,"command":asdict(Command(.1, .2,0)),
                            "telemetry":{"hit":"none","cte":.1,"time":i*.05,"speed":1}}
                    writer.submit(Image.new("RGB",(255,255),(g*30,i,10)),record)
                writer.close()
            corrupt=next(root.glob("*/images/00000007.png")); corrupt.write_bytes(b"corrupt")
            out=Path(td)/"prepared"
            report=prepare(root,out,self.cfg)
            self.assertEqual(report["rejected"],1)
            rows=list(read_jsonl(out/"train.jsonl"))
            x,speed,y,mask=DrivingDataset(rows,self.cfg)[0]
            self.assertEqual(x.shape,(3,96,160)); self.assertEqual(y.shape,mask.shape)
            self.assertTrue((sampling_weights(rows,self.cfg)<=4).all())
            all_rows=sum([list(read_jsonl(out/f"{k}.jsonl")) for k in ("train","val","test")],[])
            sid=corrupt.parent.parent.name
            before=[r for r in all_rows if r["session_id"]==sid and r["sim_time"]<.35]
            self.assertTrue(all(r["mask"][2]==0 for r in before))

    def test_unlabeled_images_are_not_a_dataset(self):
        with workspace_temp() as td:
            Image.new("RGB",(160,120)).save(Path(td)/"x.png")
            from racing.data import load_sessions
            with self.assertRaises(ValueError): load_sessions(td,self.cfg)
            with self.assertRaises(ValueError): prepare(td,Path(td)/"out",self.cfg)
            self.assertFalse((Path(td)/"out").exists())

    def test_decode_telemetry_requires_real_speed_and_resolution(self):
        b=BytesIO(); Image.new("RGB",(255,255)).save(b,format="PNG")
        packet={"image":base64.b64encode(b.getvalue()).decode(),"speed":1,"time":2}
        image,speed,stamp,tel=decode_telemetry(packet,self.cfg)
        self.assertEqual(image.size,(255,255)); self.assertNotIn("image",tel)
        packet["speed"]="nan"
        with self.assertRaises(ValueError): decode_telemetry(packet,self.cfg)

    def test_transport_latches_emergency_no_real_socket(self):
        client=RaceClient.__new__(RaceClient)
        client.cfg=self.cfg; client.send_lock=threading.Lock(); client.fault=None; client.aborted=False
        client.closed=False; client.diagnostics=Diagnostics()
        with patch.object(SDClient,"send_now") as sender:
            client.send_packet(Command(0,.1,0).packet(),control=True)
            self.assertEqual(json.loads(sender.call_args.args[0])["throttle"],"0.1")
            client.emergency("stale")
            with self.assertRaises(ConnectionError): client.send_packet(Command(0,.1,0).packet(),control=True)
            self.assertEqual(json.loads(sender.call_args.args[0])["brake"],"1.0")

    def test_untrained_checkpoint_refused(self):
        with workspace_temp() as td:
            p=Path(td)/"synthetic.pt"; torch.save({"format_version":1,"trained":False},p)
            with self.assertRaises(ValueError): load_checkpoint(p)

    def test_checkpoint_serialization_without_training(self):
        with workspace_temp() as td:
            model=RacingCNN(self.cfg).eval()
            p=Path(td)/"synthetic_untrained.pt"
            atomic_checkpoint(p,{"trained":False,"model_state":model.state_dict(),"config":self.cfg})
            payload=torch.load(p,weights_only=True)
            clone=RacingCNN(payload["config"]).eval(); clone.load_state_dict(payload["model_state"])
            self.assertFalse(p.with_suffix(".tmp").exists())
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(model.parameters(),clone.parameters())))

    def test_watchdog_stops_on_stale_control_without_socket(self):
        client=RaceClient.__new__(RaceClient)
        client.cfg=self.cfg; client.armed=True; client.fault=None
        client.last_rx=10; client.last_control=9
        from unittest.mock import Mock
        client.watch_stop=Mock(); client.watch_stop.wait.side_effect=[False,True]
        with patch("racing.transport.time.monotonic",return_value=10), patch.object(client,"emergency") as stop:
            client._watchdog()
            stop.assert_called_once_with("watchdog_timeout")

    def test_validation_augmentation_is_off(self):
        with workspace_temp() as td:
            p=Path(td)/"im.png"; Image.new("RGB",(255,255),(40,80,120)).save(p)
            row={"images":[str(p)],"steering":[.1,.2,.3],"mask":[1,1,1],"speed":2}
            ds=DrivingDataset([row],self.cfg,False)
            self.assertTrue(torch.equal(ds[0][0],ds[0][0]))

    def test_image_cache_matches_direct_loading(self):
        from racing.cache import build_cache
        with workspace_temp() as td:
            p=Path(td)/"im.png"
            rng=np.random.default_rng(0)
            Image.fromarray(rng.integers(0,256,(255,255,3),dtype=np.uint8)).save(p)
            row={"images":[str(p)],"image_sha256":["x"],"steering":[.1,.2,.3],"mask":[1,1,1],"speed":2}
            cache=build_cache([row],self.cfg,Path(td)/"cache")
            direct=DrivingDataset([row],self.cfg,False)[0]
            cached=DrivingDataset([row],self.cfg,False,cache)[0]
            for a,b in zip(direct,cached): self.assertTrue(torch.equal(a,b))
            self.assertEqual(cache[0],build_cache([row],self.cfg,Path(td)/"cache")[0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_batches_match_dataset_and_augment_safely(self):
        from racing.cache import build_cache
        from racing.gpu_data import GPUBatches, load_cache_to_device
        with workspace_temp() as td:
            rng=np.random.default_rng(1); rows=[]
            for i in range(5):
                p=Path(td)/f"im{i}.png"; Image.fromarray(rng.integers(0,256,(255,255,3),dtype=np.uint8)).save(p)
                rows.append({"images":[str(p)],"image_sha256":[str(i)],"steering":[.1*i,.2,-.3],"mask":[1,1,0],"speed":1+i})
            cache=build_cache(rows,self.cfg,Path(td)/"cache")
            images,index=load_cache_to_device(cache,torch.device("cuda"))
            x,speed,y,mask=next(iter(GPUBatches(rows,self.cfg,images,index,False,8)))
            for k,row in enumerate(rows):
                ref=DrivingDataset([row],self.cfg,False)[0]
                for a,b in zip(ref,(x[k],speed[k],y[k],mask[k])): self.assertTrue(torch.allclose(a,b.cpu(),atol=1e-6))
            ax,_,ay,_=next(iter(GPUBatches(rows,self.cfg,images,index,True,8,[1]*5,3)))
            self.assertEqual(ax.shape,x.shape); self.assertTrue(bool(ax.isfinite().all()))
            self.assertLessEqual(float(ax.abs().max()),1.0)
            self.assertTrue(torch.allclose(ay[:,1:].abs(),torch.tensor([.2,.3],device=ay.device).expand(len(ay),2)))
            del images

    def test_single_line_expert_keeps_center_when_one_line_disappears(self):
        from racing.baseline import BaselinePilot
        def road(lines):
            im=np.full((255,255,3),90,np.uint8)
            for x in lines: im[:,x:x+3]=255
            return im
        cfg=copy.deepcopy(self.cfg); cfg["baseline"]=dict(cfg.get("baseline",{}),single_line=True)
        pilot=BaselinePilot(cfg)
        _,both=pilot.step(road([60,180]),1.0,.05)
        _,left_only=pilot.step(road([60]),1.0,.05)
        self.assertFalse(left_only.get("lane_lost",False))
        self.assertAlmostEqual(both["lane_error"],left_only["lane_error"],places=3)
        cfg["baseline"]["single_line"]=False
        _,legacy=BaselinePilot(cfg).step(road([60]),1.0,.05)
        self.assertTrue(legacy["lane_lost"])

    def test_yellow_guide_expert_follows_center_line(self):
        from racing.baseline import BaselinePilot
        im=np.full((255,255,3),90,np.uint8); im[:,150:154]=(230,200,40)
        cfg=copy.deepcopy(self.cfg); cfg["baseline"]=dict(cfg.get("baseline",{}),guide="yellow")
        cmd,info=BaselinePilot(cfg).step(im,1.0,.05)
        self.assertFalse(info.get("lane_lost",False))
        self.assertAlmostEqual(info["lane_error"],(151.5-127.5)/127.5,places=3)
        self.assertGreater(cmd.steering,0)
        blank=np.full((255,255,3),90,np.uint8)
        self.assertTrue(BaselinePilot(cfg).step(blank,1.0,.05)[1]["lane_lost"])

    def test_cte_expert_slides_from_spawn_lane_to_target(self):
        from racing.baseline import BaselinePilot
        cfg=copy.deepcopy(self.cfg); cfg["baseline"]=dict(cfg.get("baseline",{}),guide="cte",cte_center="spawn",cte_target=-1.0,cte_ramp_per_s=.5,cte_abort=10)
        pilot=BaselinePilot(cfg); im=np.zeros((255,255,3),np.uint8)
        tel={"cte":2.0,"pos_x":0.0,"pos_z":0.0}
        _,first=pilot.step(im,1.0,.05,tel)
        self.assertAlmostEqual(first["lane_error"],2.0-(2.0-.025),places=6)  # moved 0.5*0.05 toward -1
        for _ in range(200): pilot.step(im,1.0,.05,tel)
        _,late=pilot.step(im,1.0,.05,tel)
        self.assertAlmostEqual(late["lane_error"],3.0,places=6)  # reached the fixed lane and stays there
        plain=copy.deepcopy(cfg); del plain["baseline"]["cte_target"]
        _,spawn=BaselinePilot(plain).step(im,1.0,.05,tel)
        self.assertAlmostEqual(spawn["lane_error"],0.0,places=6)  # default unchanged

    def test_path_expert_steers_toward_reference_line(self):
        from racing.baseline import BaselinePilot
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path=os.path.join(d,"line.json")
            json.dump({"points":[[0.0,z*.25] for z in range(40)]},open(path,"w"))  # straight line along +z
            cfg=copy.deepcopy(self.cfg); cfg["baseline"]=dict(cfg.get("baseline",{}),guide="path",path_file=path,lookahead=.8)
            im=np.zeros((255,255,3),np.uint8)
            # Heading +z (yaw 0); line is 0.3 to the left (x smaller): steer left (negative).
            cmd,info=BaselinePilot(cfg).step(im,1.0,.05,{"pos_x":.3,"pos_z":1.0,"yaw":0.0})
            self.assertLess(cmd.steering,0); self.assertFalse(info.get("lane_lost",False))
            cmd,_=BaselinePilot(cfg).step(im,1.0,.05,{"pos_x":-.3,"pos_z":1.0,"yaw":0.0})
            self.assertGreater(cmd.steering,0)
            _,far=BaselinePilot(cfg).step(im,1.0,.05,{"pos_x":3.0,"pos_z":1.0,"yaw":0.0})
            self.assertTrue(far["lane_lost"])  # never a label far from the line

    def test_cnn_lstm_variant_shapes_and_time_order(self):
        cfg=copy.deepcopy(self.cfg); cfg["model"].update(frames=5,temporal="lstm",lstm_hidden=32)
        validate_config(cfg)
        torch.manual_seed(0); model=RacingCNN(cfg).eval()
        x=torch.randn(2,15,cfg["preprocess"]["height"],cfg["preprocess"]["width"]); v=torch.ones(2,1)
        y=model(x,v); self.assertEqual(tuple(y.shape),(2,3)); self.assertTrue(bool(y.abs().le(1).all()))
        reversed_x=x.reshape(2,5,3,*x.shape[2:]).flip(1).reshape_as(x)
        self.assertFalse(torch.allclose(model(reversed_x,v),y))  # the order of frames matters
        bad=copy.deepcopy(cfg); bad["model"]["temporal"]="gru"
        with self.assertRaises(ValueError): validate_config(bad)

    def test_evaluation_missing_metrics_are_not_invented(self):
        with workspace_temp() as td:
            p=Path(td)
            dump_json(p/"run.json",{"mode":"synthetic","config":self.cfg,"checkpoint_sha256":None})
            (p/"decisions.jsonl").write_text('',encoding="utf-8")
            r=summarize(p)
            for key in ("success_rate","lap_mean_s","official_score","speed_mean_time_weighted","offtrack_events_cte_proxy"):
                self.assertIsNone(r[key])


if __name__ == "__main__": unittest.main()
