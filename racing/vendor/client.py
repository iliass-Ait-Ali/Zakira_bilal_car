"""SDClient from gym-donkeycar, with local framing/I/O/lifecycle fixes.

Original author: Tawn Kramer. MIT licence and provenance: LICENSE, NOTICE.md.
The public callback and send interfaces are preserved.
"""
import json
import logging
import select
import socket
import threading
import time
from .util import replace_float_notation

logger = logging.getLogger(__name__)


class SDClient:
    def __init__(self, host, port, poll_socket_sleep_time=0.001,
                 connect_timeout=3.0, send_timeout=1.0):
        self.msg = None
        self.host, self.port = host, port
        self.poll_socket_sleep_sec = poll_socket_sleep_time
        self.connect_timeout, self.send_timeout = connect_timeout, send_timeout
        self.th = None
        self.aborted = False
        self.s = None
        self.do_process_msgs = False
        self._stopping = threading.Event()
        self._write_lock = threading.Lock()
        self._queue_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self.connect()

    def connect(self):
        logger.debug("[CONNECT] Opening TCP socket to %s:%s", self.host, self.port)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.settimeout(self.connect_timeout)
            self.s.connect((self.host, self.port))
            self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.s.setblocking(False)
        except OSError:
            self.s.close()
            raise
        self.do_process_msgs = True
        self.on_connected()
        self.th = threading.Thread(target=self.proc_msg, args=(self.s,), daemon=True,
                                   name="sdsandbox-receiver")
        self.th.start()

    def on_connected(self):
        logger.info("[CONNECTED] TCP connection established")

    def send(self, message):
        """Legacy latest-command slot; RaceClient uses synchronous send_now."""
        with self._queue_lock:
            if self._stopping.is_set(): raise ConnectionError("Client is stopping")
            self.msg = message

    def send_now(self, message):
        """Send all bytes on a nonblocking socket, with bounded partial-write retries."""
        payload = memoryview(message.encode("utf-8"))
        deadline = time.monotonic() + self.send_timeout
        with self._write_lock:
            while payload:
                if self._stopping.is_set() or self.aborted:
                    raise ConnectionError("Cannot send after disconnect/shutdown")
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError("TCP send deadline exceeded")
                _, writable, exceptional = select.select([], [self.s], [self.s], min(.05, remaining))
                if exceptional: raise ConnectionError("Socket failed during send")
                if not writable: continue
                try: sent = self.s.send(payload)
                except (BlockingIOError, InterruptedError): continue
                if sent == 0: raise ConnectionError("Socket closed during send")
                payload = payload[sent:]

    def on_msg_recv(self, packet):
        logger.debug("[SIM] Message received: %s", packet["msg_type"])

    def on_protocol_error(self, reason):
        logger.warning("[PROTOCOL] %s", reason)

    def _disconnect(self, reason):
        if self._stopping.is_set(): return
        self.aborted = True
        self.do_process_msgs = False
        self.on_msg_recv({"msg_type": "aborted", "reason": reason})

    def _decode_line(self, line):
        started = time.monotonic()
        self.last_message_received_at = started
        try:
            value = line.decode("utf-8")
            try: packet = json.loads(value)
            except json.JSONDecodeError:
                packet = json.loads(replace_float_notation(value))
            if not isinstance(packet, dict) or not isinstance(packet.get("msg_type"), str):
                raise ValueError("Expected a JSON object with a string msg_type")
            self.last_json_decode_s = time.monotonic()-started
            return packet
        except (UnicodeError, ValueError) as error:
            self.on_protocol_error(f"Invalid incoming JSON ({len(line)} bytes): {error}")
            return None

    def proc_msg(self, sock):
        buffer = bytearray()
        try:
            while self.do_process_msgs and not self._stopping.is_set():
                readable, _, exceptional = select.select([sock], [], [sock], max(.001, self.poll_socket_sleep_sec))
                if self._stopping.is_set(): break
                if exceptional: raise ConnectionError("Socket exception during receive")
                if readable:
                    try: data = sock.recv(1024 * 256)
                    except (BlockingIOError, InterruptedError): continue
                    if not data:
                        self._disconnect("Simulator closed the TCP connection (EOF)")
                        break
                    buffer.extend(data)
                    if len(buffer) > 16_000_000:
                        raise ValueError("Incoming message buffer exceeds 16 MB")
                    while b"\n" in buffer:
                        line, _, rest = buffer.partition(b"\n")
                        buffer = bytearray(rest)
                        if not line.strip(): continue
                        packet = self._decode_line(line)
                        if packet is not None: self.on_msg_recv(packet)
                with self._queue_lock:
                    pending, self.msg = self.msg, None
                if pending is not None: self.send_now(pending)
        except (OSError, ValueError, RuntimeError) as error:
            if not self._stopping.is_set():
                self._disconnect(f"{type(error).__name__}: {error}")
        finally:
            self.do_process_msgs = False

    def stop(self):
        """Wake/join the receiver before closing its descriptor. Idempotent."""
        with self._stop_lock:
            if self._stopping.is_set(): return
            self._stopping.set()
            self.do_process_msgs = False
            if self.s is not None:
                try: self.s.shutdown(socket.SHUT_RDWR)
                except OSError as error:
                    logger.debug("[DISCONNECT] Shutdown on an already disconnected socket: %s", error)
            if self.th is not None and self.th is not threading.current_thread():
                self.th.join(timeout=2)
                if self.th.is_alive():
                    raise RuntimeError("Receiver did not stop; descriptor retained to avoid a close/select race")
            with self._write_lock:
                if self.s is not None: self.s.close()
            logger.info("[DISCONNECT] Receiver stopped; socket closed")
