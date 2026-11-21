"""Regression tests with synthetic packets/local TCP peer; never SDSandbox."""
import base64
from contextlib import contextmanager
from io import BytesIO
import json
import socket
import threading
import time
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from racing.config import load_config
from racing.diagnostics import Diagnostics
from racing.session import SimulatorSession, ControlSchedule, TelemetryError, decode_telemetry
from racing.transport import RaceClient, encode_packet
from racing.vendor.client import SDClient
from racing.vendor.util import replace_float_notation


def image_packet(size=(255,255), stamp=1, encoding="PNG"):
    stream=BytesIO()
    Image.new("RGB",size,(40,50,60)).save(stream,format=encoding)
    return {"msg_type":"telemetry","image":base64.b64encode(stream.getvalue()).decode(),"speed":0,"time":stamp}


class CaptureClient(SDClient):
    def __init__(self, port):
        self.received=[]
        self.errors=[]
        super().__init__("127.0.0.1",port,poll_socket_sleep_time=.002)

    def on_msg_recv(self, packet): self.received.append(packet)
    def on_protocol_error(self, reason): self.errors.append(reason)


@contextmanager
def peer_client():
    server=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server.bind(("127.0.0.1",0)); server.listen(1); server.settimeout(2)
    client=CaptureClient(server.getsockname()[1])
    peer,_=server.accept(); peer.settimeout(2)
    try: yield peer,client
    finally:
        client.stop(); peer.close(); server.close()


def wait_for(predicate, timeout=1):
    deadline=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>deadline: raise AssertionError("Local test peer deadline exceeded")
        time.sleep(.002)


class FakeClient:
    def __init__(self):
        self.diagnostics=Diagnostics()
        self.events=[]; self.latest=None; self.sent=[]
        self.fault=None; self.aborted=False
        self.car_configs=0; self.camera_configs=0

    def snapshot(self):
        events,self.events=self.events,[]
        return self.latest,events

    def send_packet(self, packet): self.sent.append(packet); return time.monotonic()
    def configure_car(self): self.car_configs+=1
    def configure_camera(self): self.camera_configs+=1; return time.monotonic()


class ConnectionTests(unittest.TestCase):
    def setUp(self): self.cfg=load_config()

    def test_outgoing_commands_end_at_brace_without_newline(self):
        packets=[{"msg_type":"load_scene","scene_name":"roboracingleague_1"},CommandPacket()]
        wire="".join(encode_packet(p) for p in packets)
        self.assertNotIn("\n",wire)
        # Mirrors the observed C# split/last-character check, not Python's parser.
        chunks=["{"+x for x in wire.split("{") if x]
        self.assertTrue(all(x[-1]=="}" for x in chunks))
        self.assertFalse((encode_packet(packets[0])+"\n").endswith("}"))

    def test_locale_repair_preserves_strings_arrays_and_exponents(self):
        # Build escaped string safely rather than relying on an ambiguous fixture.
        text=json.dumps('escaped "x":1,25')
        raw='{"speed":1,25,"time":2,3e+02,"image":"1,25/+123,4==","a":[1,2],"text":'+text+',"last":-1,2E-05}'
        result=json.loads(replace_float_notation(raw))
        self.assertEqual(result["speed"],1.25); self.assertEqual(result["time"],230)
        self.assertEqual(result["image"],"1,25/+123,4==")
        self.assertEqual(result["a"],[1,2]); self.assertEqual(result["text"],'escaped "x":1,25')
        self.assertAlmostEqual(result["last"],-.000012)

    def test_control_keeps_early_frame_until_deadline_without_reusing_it(self):
        schedule=ControlSchedule(20)
        schedule.offer("frame1"); self.assertEqual(schedule.take(0),"frame1")
        schedule.offer("frame2"); self.assertIsNone(schedule.take(.049))
        self.assertEqual(schedule.take(.051),"frame2")
        self.assertIsNone(schedule.take(.11))
        schedule.offer("frame3"); schedule.offer("frame4")
        self.assertEqual(schedule.take(.12),"frame4")

    def test_tcp_fragmented_coalesced_and_utf8_messages(self):
        with peer_client() as (peer,client):
            first=json.dumps({"msg_type":"note","name":"café"},ensure_ascii=False).encode()+b"\r\n"
            cut=first.index("é".encode())+1
            peer.sendall(first[:cut]); peer.sendall(first[cut:]+b'{"msg_type":"telemetry","speed":1,25}\n')
            wait_for(lambda:len(client.received)==2)
            self.assertEqual(client.received[0]["name"],"café")
            self.assertEqual(client.received[1]["speed"],1.25)
            self.assertEqual(client.errors,[])

    def test_malformed_packet_is_counted_and_image_not_logged(self):
        with peer_client() as (peer,client):
            peer.sendall(b'{"msg_type":"telemetry","image":"SECRET_BASE64",BAD}\n{"msg_type":"car_loaded"}\n')
            wait_for(lambda:len(client.received)==1)
            self.assertEqual(len(client.errors),1)
            self.assertNotIn("SECRET_BASE64",client.errors[0])
            self.assertFalse(client.aborted)

    def test_peer_eof_stops_once_and_blocks_further_sends(self):
        with peer_client() as (peer,client):
            peer.shutdown(socket.SHUT_RDWR)
            wait_for(lambda:client.aborted)
            client.th.join(timeout=1)
            self.assertFalse(client.th.is_alive())
            self.assertEqual(len(client.received),1)
            self.assertIn("EOF",client.received[0]["reason"])
            with self.assertRaises(ConnectionError): client.send_now("{}")

    def test_idle_shutdown_joins_before_close_no_negative_descriptor(self):
        with peer_client() as (peer,client):
            client.stop(); client.stop()
            self.assertFalse(client.th.is_alive()); self.assertEqual(client.s.fileno(),-1)
            self.assertFalse(client.aborted); self.assertEqual(client.received,[])
            with self.assertRaises(ConnectionError): client.send_now("{}")

    def test_partial_nonblocking_send_is_completed(self):
        client=SDClient.__new__(SDClient)
        client._write_lock=threading.Lock(); client._stopping=threading.Event()
        client.aborted=False; client.send_timeout=.2
        class Partial:
            def __init__(self): self.result=bytearray(); self.block_once=True
            def send(self, data):
                if self.block_once: self.block_once=False; raise BlockingIOError()
                n=min(2,len(data)); self.result.extend(data[:n]); return n
        client.s=Partial()
        with patch("racing.vendor.client.select.select",return_value=([],[client.s],[])):
            client.send_now("abcdef")
        self.assertEqual(client.s.result,b"abcdef")

    def test_nonblocking_send_has_deadline(self):
        client=SDClient.__new__(SDClient)
        client._write_lock=threading.Lock(); client._stopping=threading.Event()
        client.aborted=False; client.send_timeout=.01; client.s=Mock()
        with patch("racing.vendor.client.select.select",return_value=([],[],[])):
            with self.assertRaises(TimeoutError): client.send_now("{}")

    def test_camera_config_includes_required_rot_x_and_no_invalid_disable_packets(self):
        client=RaceClient.__new__(RaceClient)
        client.cfg=self.cfg; client.diagnostics=Diagnostics(); client.send_packet=Mock(return_value=1)
        client.configure_camera()
        packet=client.send_packet.call_args.args[0]
        self.assertEqual(packet["rot_x"],"0")
        self.assertEqual(packet["img_w"],"255")
        self.assertEqual(packet["img_h"],"255")
        self.assertEqual(packet["img_enc"],"PNG")
        client.send_packet.assert_called_once()

    def test_menu_handshake_and_camera_readiness(self):
        c=FakeClient(); s=SimulatorSession(c,self.cfg)
        c.events=[{"msg_type":"scene_selection_ready"}]; s.poll()
        self.assertEqual(c.sent[-1]["msg_type"],"get_scene_names")
        c.events=[{"msg_type":"scene_names","scene_names":["roboracingleague_1"]}]; s.poll()
        self.assertEqual(c.sent[-1]["msg_type"],"load_scene")
        c.events=[{"msg_type":"car_loaded"},{"msg_type":"need_car_config"}]; s.poll()
        self.assertEqual(c.car_configs,1)
        s.camera_due=0; s.poll(); self.assertEqual(c.camera_configs,1)
        c.latest=(1,time.monotonic(),image_packet((160,120))); observation,_=s.poll()
        self.assertIsNone(observation); self.assertFalse(s.ready)
        self.assertEqual(c.diagnostics.rejections["wrong_resolution"],1)
        c.latest=(2,time.monotonic(),image_packet()); observation,_=s.poll()
        self.assertIsNotNone(observation); self.assertTrue(s.ready)
        c.events=[{"msg_type":"car_loaded"}]
        with self.assertRaises(RuntimeError): s.poll()

    def test_preloaded_track_returns_to_menu_then_waits_for_names(self):
        c=FakeClient(); s=SimulatorSession(c,self.cfg)
        c.events=[{"msg_type":"car_loaded"}]; s.poll()
        self.assertEqual(c.sent,[{"msg_type":"exit_scene"}])
        s.menu_query_at=0; s.poll()
        self.assertEqual(c.sent[-1]["msg_type"],"get_scene_names")
        # No scene_selection_ready is required on return to menu in this build.
        c.events=[{"msg_type":"scene_names","scene_names":["roboracingleague_1"]}]; s.poll()
        self.assertEqual(c.sent[-1]["msg_type"],"load_scene")

    def test_missing_track_is_explicit_error(self):
        c=FakeClient(); s=SimulatorSession(c,self.cfg)
        s.state="waiting_for_scene_names"
        c.events=[{"msg_type":"scene_names","scene_names":["other"]}]
        with self.assertRaisesRegex(ValueError,"not in simulator"): s.poll()

    def test_decode_reports_distinct_reasons(self):
        cases=[({},"missing_image"),({"image":"!"},"invalid_base64"),
               ({"image":base64.b64encode(b"not image").decode()},"undecodable_image"),
               (image_packet((160,120)),"wrong_resolution"),
               (image_packet(encoding="JPEG"),"wrong_encoding")]
        for packet,reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaises(TelemetryError) as error: decode_telemetry(packet,self.cfg)
                self.assertEqual(error.exception.reason,reason)

    def test_startup_timeout_includes_actual_stage_and_counts(self):
        c=FakeClient(); s=SimulatorSession(c,self.cfg)
        s.started-=21
        with self.assertRaisesRegex(TimeoutError,"waiting_for_server.*telemetry=0"): s.poll()


def CommandPacket():
    return {"msg_type":"control","steering":"0","throttle":"0","brake":"1"}


if __name__=="__main__": unittest.main()
