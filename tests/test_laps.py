import copy
import unittest
from PIL import Image, ImageDraw
from racing.baseline import BaselinePilot
from racing.config import load_config
from racing.laps import validate_laps
from racing.session import SimulatorSession, TelemetryError
from test_connection import FakeClient, image_packet
from unittest.mock import patch


def lap_rows():
    rows = [{"kind":"event", "event":{"msg_type":"collision_with_starting_line", "starting_line_index":0,
                                     "received_monotonic":0, "timeStamp":100}}]
    for index, node in enumerate([1,2,3,0,1]):
        rows.append({"kind":"decision", "received_monotonic":index+1, "sim_time":index+1,
                     "source":"baseline", "speed":1, "decision_age_s":.03,
                     "telemetry":{"activeNode":node, "totalNodes":4, "pos_x":node, "pos_z":0, "cte":.2, "hit":"none"}})
    rows.append({"kind":"event", "event":{"msg_type":"collision_with_starting_line", "starting_line_index":0,
                                          "received_monotonic":6, "timeStamp":120}})
    return rows


class LapTests(unittest.TestCase):
    def test_full_ordered_closed_lap(self):
        result=validate_laps(lap_rows())
        self.assertEqual(result["validated_laps"],1)
        self.assertEqual(result["laps"][0]["observed_nodes"],4)

    def test_starting_line_alone_never_proves_lap(self):
        self.assertEqual(validate_laps(lap_rows()[:-1])["validated_laps"],0)
        rows=lap_rows()
        for row in rows[1:-1]: row["telemetry"]["activeNode"]=1
        self.assertEqual(validate_laps(rows)["validated_laps"],0)

    def test_skipped_or_reversed_nodes_rejected(self):
        for nodes in ([1,2,2,0,1],[1,0,3,2,1]):
            rows=lap_rows()
            for row,node in zip(rows[1:-1],nodes): row["telemetry"]["activeNode"]=node
            self.assertEqual(validate_laps(rows)["validated_laps"],0)

    def test_collision_manual_stale_stop_and_departure_rejected(self):
        for field,value,telemetry in [("hit","wall",True),("cte",1.2,True),("source","manual",False),
                                      ("decision_age_s",.2,False),("speed",0,False),("totalNodes",5,True)]:
            rows=lap_rows()
            (rows[2]["telemetry"] if telemetry else rows[2])[field]=value
            self.assertEqual(validate_laps(rows)["validated_laps"],0,field)

    def test_cnn_laps_need_explicit_opt_in(self):
        rows=lap_rows()
        for row in rows:
            if row.get("kind")=="decision": row["source"]="cnn"
        self.assertEqual(validate_laps(rows)["validated_laps"],0)
        self.assertEqual(validate_laps(rows,allowed_sources=("cnn",))["validated_laps"],1)
        rows[2]["source"]="baseline"
        lap=validate_laps(rows,allowed_sources=("baseline","cnn"))["laps"][0]
        self.assertIn("mixed_sources",lap["reasons"])
        rows[2]["source"]="manual"
        self.assertEqual(validate_laps(rows,allowed_sources=("cnn",))["validated_laps"],0)
        with self.assertRaises(ValueError): validate_laps(rows,allowed_sources=("manual",))


class ExpertTests(unittest.TestCase):
    def test_large_bright_wall_does_not_pull_lane_center(self):
        image=Image.new("RGB",(255,255),(50,50,50)); draw=ImageDraw.Draw(image)
        for box in [(0,130,40,165),(70,130,74,165),(180,130,184,165)]: draw.rectangle(box,fill="white")
        command,info=BaselinePilot(load_config()).step(image,.8,.05)
        self.assertLess(abs(command.steering),.05)
        self.assertGreaterEqual(info["lane_confidence"],4)

    def test_lane_loss_brakes_and_turn_is_slew_limited(self):
        pilot=BaselinePilot(load_config())
        command,_=pilot.step(Image.new("RGB",(255,255)),1,.05)
        self.assertEqual(command.throttle,0); self.assertEqual(command.brake,1)
        image=Image.new("RGB",(255,255)); draw=ImageDraw.Draw(image)
        for box in [(90,130,94,165),(210,130,214,165)]: draw.rectangle(box,fill="white")
        command,info=pilot.step(image,1.5,.05)
        self.assertGreater(command.steering,0); self.assertLessEqual(command.steering,.25)
        self.assertEqual(command.throttle,0); self.assertGreater(command.brake,0)
        self.assertLessEqual(info["target_speed"],1)

    def test_stale_observation_still_stops_after_optimization(self):
        client=FakeClient(); session=SimulatorSession(client,load_config())
        session.camera_sent_at=0; session.ready=True
        client.latest=(1,1,image_packet())
        with patch("racing.session.time.monotonic",return_value=1.2):
            with self.assertRaises(TelemetryError) as error: session.poll()
        self.assertEqual(error.exception.reason,"stale_frame")


if __name__ == "__main__": unittest.main()
