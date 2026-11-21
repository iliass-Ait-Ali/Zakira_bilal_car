"""Deterministic camera expert: thin lane markings, lookahead and speed feedback."""
import math
import numpy as np
from .control import Command, clip


def lane_centers(image):
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    first, last = int(.51*h), int(.65*h)
    roi = rgb[first:last]
    low, high = roi.min(axis=2), roi.max(axis=2)
    white = (low >= 180) & (high-low < 65)
    centers = []
    # Separate thin paint from large bright walls, signs and ceiling panels.
    for y in range(first, last, 2):
        row = white[y-first]
        boundaries = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)))
        runs = [(a+b-1)/2 for a,b in zip(boundaries[::2], boundaries[1::2]) if 1 <= b-a <= .055*w]
        left = [x for x in runs if x < w/2-5]
        right = [x for x in runs if x > w/2+5]
        if left and right:
            l, r = max(left), min(right)
            if r-l >= .08*w: centers.append((y/h, ((l+r)/2-w/2)/(w/2), l/w, r/w))
    return centers


def paint_mask(roi, paint="white", white_min=180, white_spread=65, yellow_ratio=.4):
    """Thin painted lines. "white_yellow" also accepts yellow/orange paint, which
    bounds the lane on several tracks (beige floors stay below the 0.4 ratio).
    white_min/white_spread/yellow_ratio adapt to cream paint or dim indoor light."""
    roi = roi.astype(int)
    low, high = roi.min(axis=2), roi.max(axis=2)
    mask = (low >= white_min) & (high-low < white_spread)
    if paint == "white_yellow":
        r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
        mask |= (r > 120) & (r >= g) & (g > .45*r) & (np.minimum(r, g)-b > yellow_ratio*np.maximum(r, g))
    return mask


def lane_rows(image, split_x=None, paint="white", white_min=180, white_spread=65, yellow_ratio=.4):
    """Per-row (y, left, right) paint positions in pixels; left/right may be None.
    Runs are classified around split_x (the last estimated lane centre) instead of
    the image centre, so a line that drifts past the middle keeps its side."""
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    split_x = w/2 if split_x is None else split_x
    first, last = int(.51*h), int(.65*h)
    white = paint_mask(rgb[first:last], paint, white_min, white_spread, yellow_ratio)
    rows = []
    for y in range(first, last, 2):
        row = white[y-first]
        boundaries = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)))
        runs = [(a+b-1)/2 for a,b in zip(boundaries[::2], boundaries[1::2]) if 1 <= b-a <= .055*w]
        left = [x for x in runs if x < split_x-5]
        right = [x for x in runs if x > split_x+5]
        rows.append((y, max(left) if left else None, min(right) if right else None))
    return rows


def yellow_runs(image, pale=False, strengths=False):
    """Per-row centres of yellow paint runs (road centre line) in the lower view.
    pale=True accepts the faded ochre line of generated_road (too noisy on beige roads)."""
    rgb = np.asarray(image).astype(int)
    h, w = rgb.shape[:2]
    # Above the car hood. The pale mode looks further ahead to bridge dashes; the
    # saturated mode stays below the horizon, where trees can look yellow.
    first, last = int((.42 if pale else .50)*h), int(.61*h)
    r, g, b = rgb[first:last, :, 0], rgb[first:last, :, 1], rgb[first:last, :, 2]
    if pale: mask = (r > 150) & (g > 120) & (b < 170) & (r-b > 75) & (g-b > 40) & (r >= g)
    else:  # paint that is yellower than the rest of its row: works in sun and in shade
        yellowness = (r+g)/2-b
        relative = yellowness-np.median(yellowness, axis=1, keepdims=True)
        brightest = np.maximum(np.maximum(r, g), 1)
        mask = ((r > .85*g) & (relative > 25) & (relative > .2*brightest)
                & (np.minimum(r, g)-b > .2*brightest))  # really yellow, not just less blue than the wall
    rows = []
    for y in range(first, last, 2):
        row = mask[y-first]
        boundaries = np.flatnonzero(np.diff(np.r_[False, row, False].astype(int)))
        runs = [(a+b-1)/2 for a, b in zip(boundaries[::2], boundaries[1::2]) if 1 <= b-a <= .08*w]
        if strengths:  # mean yellow intensity: bright paint beats dull olive road edges
            yel = np.minimum(r[y-first], g[y-first])-b[y-first]
            runs = [((a+b2-1)/2, float(yel[a:b2].mean())) for a, b2 in zip(boundaries[::2], boundaries[1::2]) if 1 <= b2-a <= .08*w]
        if runs: rows.append((y, runs))
    return rows


class BaselinePilot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.p = {"gain": 10.0, "speed_straight": 1.0, "speed_turn": .7,
                  "steering_tau_s": .06, "steering_rate": 5.0,
                  "throttle_max": .18, "throttle_ff": .08, "speed_kp": .35, "speed_ki": .08,
                  **cfg.get("baseline", {})}
        self.steering, self.integral = 0.0, 0.0
        self.split_x, self.widths, self.elapsed, self.last_seen = None, {}, 0.0, 0.0

    def _single_line_centers(self, image):
        """Optional mode (baseline.single_line): when one line leaves the view, keep
        following the other one using the lane width last measured on that row."""
        w = np.asarray(image).shape[1]
        centers = []
        for y, l, r in lane_rows(image, self.split_x, self.p.get("paint", "white"), self.p.get("white_min", 180),
                                 self.p.get("white_spread", 65), self.p.get("yellow_ratio", .4)):
            if l is not None and r is not None and r-l >= .08*w:
                self.widths[y] = r-l if y not in self.widths else .8*self.widths[y]+.2*(r-l)
                c = (l+r)/2
            elif y in self.widths and (l is not None) != (r is not None):
                c = l+self.widths[y]/2 if l is not None else r-self.widths[y]/2
            else:
                continue
            centers.append((y/np.asarray(image).shape[0], (c-w/2)/(w/2)))
        if len(centers) >= self.p.get("min_rows", 4):
            self.split_x = float(np.clip(w/2*(1+np.median([c[1] for c in centers])), .15*w, .85*w))
        return centers

    def _yellow_centers(self, image):
        """Optional mode (baseline.guide == "yellow"): follow the yellow centre line.
        Candidate points are chained row by row from the bottom of the view (a painted
        line is continuous, noise is not); the longest chain wins, ties go to the one
        closest to the previous estimate."""
        h, w = np.asarray(image).shape[:2]
        ref = w/2 if self.split_x is None else self.split_x
        rows = sorted(yellow_runs(image, self.p.get("yellow_pale", False), strengths=True), reverse=True)  # bottom first
        if not rows: return []
        window = (.45 if self.split_x is None else .12)*w
        best, best_key = [], None
        for start_row in range(min(3, len(rows))):  # the bottom row may be partly hidden
            for x0, s0 in rows[start_row][1]:
                if abs(x0-ref) > window: continue
                chain, x, last_y, score = [(rows[start_row][0], x0)], x0, rows[start_row][0], s0
                for y, runs in rows[start_row+1:]:
                    nxt, st = min(runs, key=lambda c: abs(c[0]-x))
                    if abs(nxt-x) <= 2+1.5*(last_y-y):  # a few pixels per row, allows slant
                        chain.append((y, nxt)); x, last_y, score = nxt, y, score+st
                key = (score, -abs(x0-ref))
                if best_key is None or key > best_key: best, best_key = chain, key
        centers = [(y/h, (x-w/2)/(w/2)) for y, x in best]
        if len(centers) >= 4:
            new_x = w/2*(1+np.median([c[1] for c in centers]))
            if self.split_x is not None and abs(new_x-self.split_x) > .06*w: return []  # jump: treat as lost
            self.split_x = float(np.clip(new_x, .1*w, .9*w))
        return centers

    def _cte_step(self, speed, dt, telemetry):
        """Optional mode (baseline.guide == "cte"): a privileged teacher that steers
        from the simulator's cross-track error to the track path (PD control), for
        tracks whose markings are hard to see. The CNN still learns from images only."""
        p = self.p
        if telemetry is None or "cte" not in telemetry: raise ValueError("guide=cte needs telemetry")
        center = p.get("cte_center", 0.0)
        if center == "spawn":  # follow a path parallel to the reference, through the spawn lane
            if not hasattr(self, "_cte_spawn"): self._cte_spawn = float(telemetry["cte"])
            # Optional (cte_target + cte_ramp_per_s): start in the spawn lane, then slide
            # gently to one fixed lane so every session gives the same labels.
            if "cte_target" in p:
                step = p.get("cte_ramp_per_s", .3)*dt
                self._cte_spawn += clip(float(p["cte_target"])-self._cte_spawn, -step, step)
            center = self._cte_spawn
        raw_cte = float(telemetry["cte"])
        # A jump the car cannot physically make in one frame means the simulator switched
        # to another segment of its reference path: shift the reference, do not steer.
        # The cross-track error cannot change by more than the distance actually driven
        # (measured from the simulator position); any excess is a path artifact.
        pos = (float(telemetry.get("pos_x", 0.0)), float(telemetry.get("pos_z", 0.0)))
        if not hasattr(self, "_cte_shift"): self._cte_shift, self._cte_raw, self._pos = 0.0, raw_cte, pos
        moved = math.hypot(pos[0]-self._pos[0], pos[1]-self._pos[1])+p.get("cte_slack", .03)
        change = raw_cte-self._cte_raw
        self._cte_shift += change-clip(change, -moved, moved)
        self._cte_raw, self._pos = raw_cte, pos
        error = raw_cte-self._cte_shift-center
        if not hasattr(self, "_cte_prev"): self._cte_prev, self._cte_rate = error, 0.0
        rate = (error-self._cte_prev)/max(dt, 1e-3)
        self._cte_rate += min(1.0, dt/.1)*(rate-self._cte_rate)  # 0.1 s low-pass on the derivative
        self._cte_prev = error
        if abs(error) > p.get("cte_abort", 3.0):  # far off the path: stop, never a label
            return Command(self.steering, 0, self.cfg["control"]["emergency_brake"]), {"lane_confidence": 0, "lane_lost": True}
        limit = p.get("max_steering", 1.0)
        if self.elapsed < p.get("startup_s", 0.0): limit = min(limit, p.get("startup_max_steering", limit))
        raw = clip(p.get("cte_sign", -1.0)*(p.get("cte_kp", .4)*error+p.get("cte_kd", .25)*self._cte_rate), -limit, limit)
        alpha = 1-math.exp(-dt/p["steering_tau_s"])
        self.steering += clip(alpha*(raw-self.steering), -p["steering_rate"]*dt, p["steering_rate"]*dt)
        # Slow down when steering hard (curves) like the camera expert.
        target = p["speed_straight"]-(p["speed_straight"]-p["speed_turn"])*min(1.0, abs(self.steering)/max(limit, 1e-6))
        throttle, brake = self._speed(target, speed, dt)
        return Command(self.steering, throttle, brake), {"lane_confidence": 20, "lane_error": error, "target_speed": target, "steering_raw": raw}

    def _path_step(self, speed, dt, telemetry):
        """Optional mode (baseline.guide == "path"): privileged pure-pursuit teacher.
        It aims at a point `lookahead` metres ahead on a reference line (x, z points
        built from earlier runs, tools/build_path.py), so it starts turning before a
        hairpin instead of after it. The CNN still learns and drives from images only."""
        p = self.p
        if telemetry is None or "pos_x" not in telemetry: raise ValueError("guide=path needs telemetry")
        if not hasattr(self, "_path"):
            import json
            from pathlib import Path
            line = json.loads(Path(p["path_file"]).read_text(encoding="utf-8"))
            self._path = np.asarray(line["points"], dtype=float)
            self._path_visited = line.get("visited")
            self._path_i = None
        node = telemetry.get("activeNode")
        # Optional exploration (path_fallback_cte): on nodes nobody has driven yet, the
        # line is only interpolated, so steer from the cross-track error instead.
        if p.get("path_fallback_cte") and self._path_visited is not None and node is not None and not self._path_visited[int(node) % len(self._path_visited)]:
            self._path_i = None
            return self._cte_step(speed, dt, telemetry)
        pts, n = self._path, len(self._path)
        pos = np.array([float(telemetry["pos_x"]), float(telemetry["pos_z"])])
        if self._path_i is None: self._path_i = int(np.argmin(((pts-pos)**2).sum(1)))
        window = (self._path_i+np.arange(-5, 40)) % n  # search forward only: no jump across the track
        self._path_i = int(window[np.argmin(((pts[window]-pos)**2).sum(1))])
        distance = float(np.sqrt(((pts[self._path_i]-pos)**2).sum()))
        k, travelled = self._path_i, 0.0
        while travelled < p.get("lookahead", .8):
            nxt = (k+1) % n
            if p.get("path_fallback_cte") and self._path_visited is not None and not self._path_visited[nxt]: break  # never aim into unexplored nodes
            travelled += float(np.sqrt(((pts[nxt]-pts[k])**2).sum())); k = nxt
            if k == self._path_i: break
        yaw = math.radians(float(telemetry["yaw"]))
        d = pts[k]-pos
        lateral = d[0]*math.cos(yaw)-d[1]*math.sin(yaw)  # > 0: target on the right
        ld2 = max(float((d**2).sum()), 1e-4)
        if distance > p.get("path_abort", 1.5):  # far off the line: stop, never a label
            return Command(self.steering, 0, self.cfg["control"]["emergency_brake"]), {"lane_confidence": 0, "lane_lost": True}
        limit = p.get("max_steering", 1.0)
        if self.elapsed < p.get("startup_s", 0.0): limit = min(limit, p.get("startup_max_steering", limit))
        raw = clip(2*lateral/ld2/p.get("curvature_per_steering", 1.372), -limit, limit)
        alpha = 1-math.exp(-dt/p["steering_tau_s"])
        self.steering += clip(alpha*(raw-self.steering), -p["steering_rate"]*dt, p["steering_rate"]*dt)
        target = p["speed_straight"]-(p["speed_straight"]-p["speed_turn"])*min(1.0, abs(self.steering)/max(limit, 1e-6))
        throttle, brake = self._speed(target, speed, dt)
        return Command(self.steering, throttle, brake), {"lane_confidence": 20, "lane_error": distance, "target_speed": target, "steering_raw": raw}

    def _speed(self, target, speed, dt):
        speed_error = target-speed
        if speed_error < -.15:
            self.integral = 0
            return 0, min(.5, -speed_error*.4)
        self.integral = clip(self.integral+speed_error*dt, -.4, .7)
        return clip(self.p["throttle_ff"]+self.p["speed_kp"]*speed_error+self.p["speed_ki"]*self.integral, 0, self.p["throttle_max"]), 0

    def step(self, image, speed, dt, telemetry=None):
        self.elapsed += dt
        if self.p.get("guide") == "cte": return self._cte_step(speed, dt, telemetry)
        if self.p.get("guide") == "path": return self._path_step(speed, dt, telemetry)
        if self.p.get("guide") == "yellow": centers = self._yellow_centers(image)
        else: centers = self._single_line_centers(image) if self.p.get("single_line") else lane_centers(image)
        if len(centers) < self.p.get("min_rows", 4):
            # Optional grace (baseline.lost_grace_s) for dashed centre lines: coast on the
            # last steering without throttle. These frames stay lane_lost (never labels).
            if self.elapsed-self.last_seen < self.p.get("lost_grace_s", 0.0):
                return Command(self.steering, 0, 0), {"lane_confidence": len(centers), "lane_lost": True, "coasting": True}
            if self.p.get("guide") == "yellow": self.split_x = None  # search the whole view again
            return Command(self.steering, 0, self.cfg["control"]["emergency_brake"]), {"lane_confidence": len(centers), "lane_lost": True}
        self.last_seen = self.elapsed
        # Median rejects isolated transverse/checkered paint and thin sign edges.
        error = float(np.median([row[1] for row in centers]))
        # Optional, for wide tracks (defaults unchanged): a gentle limit while the car
        # recentres after spawning, then the normal limit.
        limit = self.p.get("max_steering", 1.0)
        if self.elapsed < self.p.get("startup_s", 0.0): limit = min(limit, self.p.get("startup_max_steering", limit))
        raw = clip(math.tanh(error)*self.p["gain"], -limit, limit)
        alpha = 1-math.exp(-dt/self.p["steering_tau_s"])
        self.steering += clip(alpha*(raw-self.steering), -self.p["steering_rate"]*dt, self.p["steering_rate"]*dt)
        target = self.p["speed_straight"]-(self.p["speed_straight"]-self.p["speed_turn"])*abs(raw)
        speed_error = target-speed
        if speed_error < -.15:
            throttle, brake = 0, min(.5, -speed_error*.4)
            self.integral = 0
        else:
            self.integral = clip(self.integral+speed_error*dt, -.4, .7)
            throttle = clip(self.p["throttle_ff"]+self.p["speed_kp"]*speed_error+self.p["speed_ki"]*self.integral, 0, self.p["throttle_max"])
            brake = 0
        return Command(self.steering, throttle, brake), {"lane_confidence": len(centers), "lane_error": error,
                                                       "target_speed": target, "steering_raw": raw}


def legacy_baseline(image, cfg):
    rgb = np.asarray(image)
    h, w = rgb.shape[:2]
    mask = np.all(rgb[int(.5*h):int(.7*h)] >= 180, axis=2)
    mid = w//2
    left = np.nonzero(mask[:, :mid-5])[1]
    right = np.nonzero(mask[:, mid+5:])[1]
    if not len(left) or not len(right): return Command(0, 0, cfg["control"]["emergency_brake"])
    center = (left.mean()+right.mean()+mid+5)/2
    steer = float(np.clip(np.tanh((center-mid)/mid)*cfg["runtime"]["baseline_gain"], -1, 1))
    return Command(steer, cfg["runtime"]["baseline_throttle"], 0)
