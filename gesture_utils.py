"""
gesture_utils.py  ──  UPGRADED CINEMATIC EDITION
-------------------------------------------------
Anime-style cinematic effects: particles, shockwaves, lightning bolts,
screen flash, dramatic text animations, glitch overlays, aura trails.
"""

import numpy as np
import cv2
import math
import random
import time


# ──────────────────────────────────────────────
# Landmark Index Constants
# ──────────────────────────────────────────────
WRIST           = 0
THUMB_TIP       = 4;  THUMB_IP  = 3;  THUMB_MCP = 2;  THUMB_CMC = 1
INDEX_TIP       = 8;  INDEX_PIP = 6;  INDEX_MCP = 5
MIDDLE_TIP      = 12; MIDDLE_PIP= 10; MIDDLE_MCP= 9
RING_TIP        = 16; RING_PIP  = 14; RING_MCP  = 13
PINKY_TIP       = 20; PINKY_PIP = 18; PINKY_MCP = 17


# ──────────────────────────────────────────────
# Core Geometry Helpers
# ──────────────────────────────────────────────

def distance(point1, point2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))


def is_finger_up(landmarks, finger_tip_idx, finger_pip_idx):
    tip_y = landmarks[finger_tip_idx][1]
    pip_y = landmarks[finger_pip_idx][1]
    return tip_y < pip_y


def is_thumb_up(landmarks):
    tip_x   = landmarks[THUMB_TIP][0]
    mcp_x   = landmarks[THUMB_MCP][0]
    wrist_x = landmarks[WRIST][0]
    if wrist_x > mcp_x:
        return tip_x < mcp_x
    else:
        return tip_x > mcp_x


def count_fingers(landmarks):
    fingers = [
        is_thumb_up(landmarks),
        is_finger_up(landmarks, INDEX_TIP,  INDEX_PIP),
        is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP),
        is_finger_up(landmarks, RING_TIP,   RING_PIP),
        is_finger_up(landmarks, PINKY_TIP,  PINKY_PIP),
    ]
    return sum(fingers), fingers


# ──────────────────────────────────────────────
# Gesture Classifiers
# ──────────────────────────────────────────────

def detect_gesture(landmarks):
    if is_domain_expansion(landmarks):
        return "DOMAIN EXPANSION"
    if is_cursed_technique(landmarks):
        return "CURSED TECHNIQUE"
    if is_energy_release(landmarks):
        return "ENERGY RELEASE"
    return None


def is_domain_expansion(landmarks):
    index_up   = is_finger_up(landmarks, INDEX_TIP,  INDEX_PIP)
    middle_up  = is_finger_up(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_down  = not is_finger_up(landmarks, RING_TIP,  RING_PIP)
    pinky_down = not is_finger_up(landmarks, PINKY_TIP, PINKY_PIP)
    index_tip_pos  = landmarks[INDEX_TIP][:2]
    middle_tip_pos = landmarks[MIDDLE_TIP][:2]
    fingers_spread = distance(index_tip_pos, middle_tip_pos) > 0.04
    return index_up and middle_up and ring_down and pinky_down and fingers_spread


def is_cursed_technique(landmarks):
    count, fingers = count_fingers(landmarks)
    if count <= 1:
        return True
    thumb_tip = landmarks[THUMB_TIP][:2]
    index_tip = landmarks[INDEX_TIP][:2]
    if distance(thumb_tip, index_tip) < 0.06:
        return True
    return False


def is_energy_release(landmarks):
    count, fingers = count_fingers(landmarks)
    return fingers[1] and fingers[2] and fingers[3] and fingers[4]


# ──────────────────────────────────────────────
# Gesture Config
# ──────────────────────────────────────────────

GESTURE_COLORS = {
    "DOMAIN EXPANSION": (255, 140, 0),
    "CURSED TECHNIQUE": (60,  0,   220),
    "ENERGY RELEASE":   (0,   220, 80),
}

GESTURE_SUBTITLES = {
    "DOMAIN EXPANSION": "Infinite Void",
    "CURSED TECHNIQUE": "Malevolent Shrine",
    "ENERGY RELEASE":   "Divergent Fist",
}

GESTURE_KANJI = {
    "DOMAIN EXPANSION": "無量空処",
    "CURSED TECHNIQUE": "呪術廻戦",
    "ENERGY RELEASE":   "発散拳",
}


# ──────────────────────────────────────────────
# Particle System
# ──────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2, 9)
        self.x = float(x)
        self.y = float(y)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.06)
        self.radius = random.randint(2, 6)
        self.color = color
        # Optional: slight gravity pull
        self.gravity = random.uniform(0.05, 0.15)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.vx *= 0.97
        self.life -= self.decay

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        if not self.is_alive():
            return
        alpha = max(0.0, self.life)
        r = max(1, int(self.radius * alpha))
        cx, cy = int(self.x), int(self.y)
        h, w = frame.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r, self.color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha * 0.9, frame, 1 - alpha * 0.9, 0, frame)


class LightningBolt:
    """Jagged lightning segment between two points."""
    def __init__(self, start, end, color, segments=8, jaggedness=15):
        self.color = color
        self.life = 1.0
        self.decay = 0.08
        self.points = self._generate(start, end, segments, jaggedness)

    def _generate(self, start, end, segments, jaggedness):
        pts = [start]
        for i in range(1, segments):
            t = i / segments
            mx = int(start[0] + (end[0] - start[0]) * t + random.randint(-jaggedness, jaggedness))
            my = int(start[1] + (end[1] - start[1]) * t + random.randint(-jaggedness, jaggedness))
            pts.append((mx, my))
        pts.append(end)
        return pts

    def update(self):
        self.life -= self.decay
        # Regenerate for flicker effect
        if self.life > 0:
            start, end = self.points[0], self.points[-1]
            self.points = self._generate(start, end, 8, 15)

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        if not self.is_alive():
            return
        alpha = max(0.0, self.life)
        overlay = frame.copy()
        for i in range(len(self.points) - 1):
            cv2.line(overlay, self.points[i], self.points[i+1],
                     self.color, max(1, int(3 * alpha)), cv2.LINE_AA)
            # Glow
            cv2.line(overlay, self.points[i], self.points[i+1],
                     (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.85, frame, 1 - alpha * 0.85, 0, frame)


# ──────────────────────────────────────────────
# Effect State — Full Cinematic Engine
# ──────────────────────────────────────────────

class EffectState:
    def __init__(self):
        # Ring animation
        self.radius = 0
        self.max_radius = 140
        self.speed = 4
        self.alpha = 1.0

        # Particles
        self.particles: list[Particle] = []
        self.lightning_bolts: list[LightningBolt] = []

        # Screen flash
        self.flash_alpha = 0.0
        self.flash_color = (255, 255, 255)

        # Text animation
        self.text_scale = 0.0
        self.text_alpha = 0.0
        self.text_target_scale = 1.0

        # Aura pulse
        self.aura_phase = 0.0

        # Shockwave rings
        self.shockwaves: list[dict] = []

        # Gesture trigger tracking
        self.last_gesture = None
        self.gesture_frame = 0
        self.trigger_cooldown = 0  # frames until next burst

    def trigger(self, gesture, hand_center, color):
        """Called once when a new gesture is detected."""
        self.flash_alpha = 0.45
        self.flash_color = color
        self.text_scale = 0.3
        self.text_alpha = 1.0

        # Burst of particles
        cx, cy = hand_center
        for _ in range(50):
            self.particles.append(Particle(cx, cy, color))
        # Shockwave
        self.shockwaves.append({"r": 0, "max_r": 200, "speed": 8,
                                 "alpha": 1.0, "color": color, "cx": cx, "cy": cy})
        self.trigger_cooldown = 25

    def update(self, gesture, hand_center, color):
        # Ring animation
        self.radius += self.speed
        self.alpha = 1.0 - (self.radius / self.max_radius)
        if self.radius >= self.max_radius:
            self.radius = 0
            self.alpha = 1.0

        # Aura pulse
        self.aura_phase = (self.aura_phase + 0.08) % (2 * math.pi)

        # Screen flash fade
        self.flash_alpha = max(0.0, self.flash_alpha - 0.04)

        # Text animation
        if self.text_scale < self.text_target_scale:
            self.text_scale = min(self.text_target_scale, self.text_scale + 0.06)
        self.text_alpha = max(0.0, self.text_alpha - 0.008)

        # Update particles
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.is_alive()]

        # Spawn continuous particles near hand
        if hand_center and gesture and self.trigger_cooldown == 0:
            cx, cy = hand_center
            if random.random() < 0.4:
                self.particles.append(Particle(
                    cx + random.randint(-30, 30),
                    cy + random.randint(-30, 30),
                    color
                ))

        # Spawn random lightning near hand
        if hand_center and gesture and random.random() < 0.08:
            cx, cy = hand_center
            angle = random.uniform(0, 2 * math.pi)
            dist = random.randint(40, 100)
            ex = int(cx + math.cos(angle) * dist)
            ey = int(cy + math.sin(angle) * dist)
            self.lightning_bolts.append(LightningBolt((cx, cy), (ex, ey), color))

        # Update lightning
        for bolt in self.lightning_bolts:
            bolt.update()
        self.lightning_bolts = [b for b in self.lightning_bolts if b.is_alive()]

        # Update shockwaves
        for sw in self.shockwaves:
            sw["r"] += sw["speed"]
            sw["alpha"] = max(0.0, 1.0 - sw["r"] / sw["max_r"])
        self.shockwaves = [sw for sw in self.shockwaves if sw["alpha"] > 0]

        # Countdown
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            # Extra particles during trigger burst
            if hand_center:
                cx, cy = hand_center
                for _ in range(3):
                    self.particles.append(Particle(cx, cy, color))

        self.gesture_frame += 1


# ──────────────────────────────────────────────
# Drawing: Cinematic Effects
# ──────────────────────────────────────────────

def draw_aura(frame, gesture, hand_center, effect_state):
    """Multi-layered pulsing aura around the hand."""
    if not hand_center or gesture not in GESTURE_COLORS:
        return
    color = GESTURE_COLORS[gesture]
    cx, cy = hand_center
    pulse = math.sin(effect_state.aura_phase)

    for i in range(3):
        r = int(55 + i * 22 + pulse * (8 + i * 4))
        a = max(0.0, 0.18 - i * 0.05)
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, a, frame, 1 - a, 0, frame)

    # Hard outline ring
    ring_r = int(60 + pulse * 8)
    cv2.circle(frame, (cx, cy), ring_r, color, 2, cv2.LINE_AA)


def draw_shockwaves(frame, effect_state):
    """Expanding shockwave rings."""
    for sw in effect_state.shockwaves:
        overlay = frame.copy()
        thickness = max(1, int(4 * sw["alpha"]))
        cv2.circle(overlay, (sw["cx"], sw["cy"]), sw["r"], sw["color"], thickness, cv2.LINE_AA)
        # Second thinner ring offset
        if sw["r"] > 20:
            cv2.circle(overlay, (sw["cx"], sw["cy"]), sw["r"] - 15,
                       sw["color"], 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, sw["alpha"] * 0.85, frame, 1 - sw["alpha"] * 0.85, 0, frame)


def draw_screen_flash(frame, effect_state):
    """Whole-screen color flash on gesture trigger."""
    if effect_state.flash_alpha <= 0:
        return
    overlay = frame.copy()
    overlay[:] = effect_state.flash_color
    cv2.addWeighted(overlay, effect_state.flash_alpha, frame, 1 - effect_state.flash_alpha, 0, frame)


def draw_particles(frame, effect_state):
    for p in effect_state.particles:
        p.draw(frame)


def draw_lightning(frame, effect_state):
    for bolt in effect_state.lightning_bolts:
        bolt.draw(frame)


def draw_gesture_effects(frame, gesture, hand_center, effect_state):
    """
    Master effect renderer — called every frame when a gesture is active.
    """
    if gesture not in GESTURE_COLORS:
        return

    color = GESTURE_COLORS[gesture]
    cx, cy = hand_center

    # Detect gesture change → trigger burst
    if gesture != effect_state.last_gesture:
        effect_state.trigger(gesture, hand_center, color)
        effect_state.last_gesture = gesture
        effect_state.radius = 0

    # Update all animation state
    effect_state.update(gesture, hand_center, color)

    # 1. Aura (bottom layer)
    draw_aura(frame, gesture, hand_center, effect_state)

    # 2. Shockwaves
    draw_shockwaves(frame, effect_state)

    # 3. Main expanding ring
    ring_alpha = max(0.0, effect_state.alpha)
    if effect_state.radius > 0:
        overlay = frame.copy()
        thickness = max(2, int(4 * ring_alpha))
        cv2.circle(overlay, (cx, cy), effect_state.radius, color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, ring_alpha * 0.9, frame, 1 - ring_alpha * 0.9, 0, frame)
        # Inner ring
        inner_r = max(0, effect_state.radius - 25)
        if inner_r > 0:
            cv2.circle(frame, (cx, cy), inner_r, color, 1, cv2.LINE_AA)

    # 4. Particles (above rings)
    draw_particles(frame, effect_state)

    # 5. Lightning bolts
    draw_lightning(frame, effect_state)

    # 6. Screen flash (top layer)
    draw_screen_flash(frame, effect_state)


# ──────────────────────────────────────────────
# Drawing: Cinematic HUD / Labels
# ──────────────────────────────────────────────

def draw_gesture_label(frame, gesture, effect_state=None):
    """
    Cinematic gesture label with:
    - Animated scale-in text
    - Colored glow/shadow
    - Kanji subtitle
    - Decorative side lines
    - Gesture-specific corner badge
    """
    if gesture not in GESTURE_COLORS:
        return

    color    = GESTURE_COLORS[gesture]
    subtitle = GESTURE_SUBTITLES.get(gesture, "")
    kanji    = GESTURE_KANJI.get(gesture, "")
    h, w     = frame.shape[:2]

    # ── Gradient banner ─────────────────────────────────────────────
    banner_h = 90
    banner_overlay = frame.copy()
    for i in range(banner_h):
        y_pos = h - banner_h + i
        progress = i / banner_h
        darkness = int(200 * (1 - progress * 0.5))
        cv2.line(banner_overlay, (0, y_pos), (w, y_pos),
                 (0, 0, 0), 1)
    cv2.addWeighted(banner_overlay, 0.7, frame, 0.3, 0, frame)

    # ── Colored top border on banner ────────────────────────────────
    cv2.line(frame, (0, h - banner_h), (w, h - banner_h), color, 2)
    # Subtle second line
    cv2.line(frame, (0, h - banner_h + 4), (w, h - banner_h + 4), color, 1)

    # ── Text scale from effect state ────────────────────────────────
    scale = effect_state.text_scale if effect_state else 1.0
    scale = max(0.3, min(1.0, scale))

    # ── Main gesture name ────────────────────────────────────────────
    font       = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.3 * scale
    thickness  = 2
    text_size  = cv2.getTextSize(gesture, font, font_scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = h - banner_h + 38

    # Multi-layer shadow for depth
    for offset, alpha_mul in [(4, 0.3), (2, 0.5)]:
        shadow_overlay = frame.copy()
        cv2.putText(shadow_overlay, gesture, (tx + offset, ty + offset),
                    font, font_scale, (0, 0, 0), thickness + 2)
        cv2.addWeighted(shadow_overlay, alpha_mul, frame, 1 - alpha_mul, 0, frame)

    # Colored glow
    glow_overlay = frame.copy()
    cv2.putText(glow_overlay, gesture, (tx, ty), font, font_scale, color, thickness + 4)
    cv2.addWeighted(glow_overlay, 0.4, frame, 0.6, 0, frame)

    # Main white text
    cv2.putText(frame, gesture, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    # ── Decorative lines flanking text ──────────────────────────────
    line_y = ty - text_size[1] // 2
    left_end  = tx - 20
    right_start = tx + text_size[0] + 20
    if left_end > 10:
        cv2.line(frame, (10, line_y), (left_end, line_y), color, 1)
        cv2.circle(frame, (10, line_y), 3, color, -1)
    if right_start < w - 10:
        cv2.line(frame, (right_start, line_y), (w - 10, line_y), color, 1)
        cv2.circle(frame, (w - 10, line_y), 3, color, -1)

    # ── Subtitle ─────────────────────────────────────────────────────
    sub_scale = 0.6 * scale
    sub_size  = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, sub_scale, 1)[0]
    sx = (w - sub_size[0]) // 2
    sy = h - banner_h + 65
    cv2.putText(frame, subtitle, (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 200, 200), 1)

    # ── Corner badge (top-right) ─────────────────────────────────────
    _draw_corner_badge(frame, gesture, color)


def _draw_corner_badge(frame, gesture, color):
    """Small energy badge in the top-right with gesture initial."""
    h, w = frame.shape[:2]
    bx, by = w - 70, 20
    bs = 50

    # Background
    badge_overlay = frame.copy()
    pts = np.array([[bx, by], [bx + bs, by],
                    [bx + bs, by + bs], [bx, by + bs]], dtype=np.int32)
    cv2.fillPoly(badge_overlay, [pts], (0, 0, 0))
    cv2.addWeighted(badge_overlay, 0.65, frame, 0.35, 0, frame)

    # Border
    cv2.polylines(frame, [pts], True, color, 2)

    # Gesture letter
    initial = gesture[0]
    ts = cv2.getTextSize(initial, cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)[0]
    ix = bx + (bs - ts[0]) // 2
    iy = by + (bs + ts[1]) // 2
    cv2.putText(frame, initial, (ix, iy), cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2)


def draw_fps(frame, fps):
    """FPS counter — styled."""
    h, w = frame.shape[:2]
    label = f"FPS  {fps:.0f}"
    # Small background
    ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (18 + ts[0], 26 + ts[1]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1)


def draw_idle_hint(frame):
    """Hint overlay when idle — with subtle scan line effect."""
    h, w = frame.shape[:2]

    # Subtle vignette
    overlay = np.zeros_like(frame)
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    # Center of frame is clear, edges darkened
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (101, 101), 0)
    for c in range(3):
        frame[:, :, c] = np.clip(
            frame[:, :, c].astype(np.float32) * (0.75 + 0.25 * mask),
            0, 255
        ).astype(np.uint8)

    # Bottom hint
    hint = "✦  Show a gesture to activate cursed energy  ✦"
    font_scale = 0.5
    size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
    x = (w - size[0]) // 2
    y = h - 18

    # Pulsing hint (uses time for flicker)
    pulse = abs(math.sin(time.time() * 1.5))
    gray_val = int(100 + 80 * pulse)
    cv2.putText(frame, hint, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (gray_val, gray_val, gray_val), 1)

    # Top title
    title = "⚡  JJK GESTURE SYSTEM  ⚡"
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
    tx = (w - ts[0]) // 2
    cv2.putText(frame, title, (tx, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.6, (60, 60, 60), 1)


def draw_scanlines(frame, strength=0.07):
    """Subtle anime-style scanline effect."""
    h, w = frame.shape[:2]
    for y in range(0, h, 4):
        cv2.line(frame, (0, y), (w, y), (0, 0, 0), 1)
    overlay = frame.copy()
    cv2.addWeighted(overlay, 1 - strength, frame, strength, 0, frame)


def draw_hand_trails(frame, pixel_landmarks, trail_points, color):
    """
    Draw fading trails behind key fingertip positions.
    trail_points: deque of previous fingertip positions
    """
    if len(trail_points) < 2:
        return
    pts_list = list(trail_points)
    for i in range(1, len(pts_list)):
        alpha = i / len(pts_list)
        thickness = max(1, int(3 * alpha))
        overlay = frame.copy()
        cv2.line(overlay, pts_list[i-1], pts_list[i], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)