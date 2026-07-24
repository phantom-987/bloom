import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
import sys

# =============================================================================
# GLOBAL CONSTANTS & SETUP
# =============================================================================
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ~137.5 degrees
WINDOW_NAME = "Interactive Realist Sunflower Garden"

# Realistic Sunflower Palette (BGR) -- bright daytime scene
COLOR_PALETTE = {
    'bg_top': (235, 175, 90),          # Warm Bright Sky Blue (Top)
    'bg_bottom': (170, 235, 245),      # Pale Sunlit Horizon (Bottom)
    'ground': (70, 165, 95),           # Sunlit Grass
    'petal_base': (0, 140, 235),       # Deep Golden Amber
    'petal_mid': (0, 195, 255),        # Rich Vibrant Yellow
    'petal_tip': (60, 230, 255),       # Bright Golden Sun
    'petal_shine': (170, 250, 255),    # Hot highlight streak
    'center_dark': (18, 30, 50),       # Deep Chocolate / Dark Core
    'center_rim': (25, 75, 130),       # Warm Brown Seed Rim
    'floret_gold': (0, 205, 255),      # Golden Pollen Florets
    'stem': (30, 120, 55),             # Realistic Stem Green
    'stem_rim': (90, 220, 140),        # Rim-light edge on stem
    'leaf_dark': (25, 95, 45),         # Foliage Shadow
    'leaf_light': (55, 160, 70),       # Foliage Highlight
    'particle': (120, 230, 255),       # Floating Pollen Glow
    'sun': (225, 250, 255),            # Bright sun core
    'sun_glow': (170, 235, 255),       # Warm sun halo
    'cloud': (250, 245, 240)           # Soft white clouds
}

def lerp(start: float, end: float, factor: float) -> float:
    return start + (end - start) * factor

def distance_2d(p1: tuple, p2: tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =============================================================================
# PARTICLE SYSTEM
# =============================================================================
class ParticleSystem:
    def __init__(self, max_particles: int = 160):
        self.max_particles = max_particles
        self.particles = []

    def spawn_pollen(self, pos: tuple, color: tuple, count: int = 2):
        px, py = pos
        for _ in range(count):
            if len(self.particles) < self.max_particles:
                vx = random.uniform(-1.2, 1.2)
                vy = random.uniform(-2.5, -0.5)
                life = random.uniform(1.0, 2.5)
                size = random.randint(1, 3)
                self.particles.append([px, py, vx, vy, life, life, size, color])

    def spawn_firefly(self, bounds: tuple):
        w, h = bounds
        if len(self.particles) < self.max_particles and random.random() < 0.18:
            px = random.randint(0, w)
            py = random.randint(int(h * 0.3), h)
            vx = random.uniform(-0.4, 0.4)
            vy = random.uniform(-0.6, -0.2)
            life = random.uniform(3.0, 5.0)
            size = random.randint(2, 3)
            self.particles.append([px, py, vx, vy, life, life, size, (120, 245, 200)])

    def update_and_draw(self, canvas: np.ndarray, dt: float, wind_x: float = 0.0):
        alive_particles = []
        for p in self.particles:
            px, py, vx, vy, life, max_life, size, color = p
            px += (vx + wind_x)
            py += vy
            life -= dt
            if life > 0:
                p[0], p[1], p[4] = px, py, life
                alive_particles.append(p)
                alpha = max(0.0, life / max_life)
                glow_color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
                # soft outer halo + bright core for a real glow feel
                cv2.circle(canvas, (int(px), int(py)), size + 3, glow_color, -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, (int(px), int(py)), size, glow_color, -1, lineType=cv2.LINE_AA)
        self.particles = alive_particles

# =============================================================================
# BACKGROUND: BRIGHT SKY + SUN + CLOUDS + GROUND
# =============================================================================
def draw_dark_background(width: int, height: int) -> np.ndarray:
    """Bright day-sky gradient with a sunlit ground strip along the bottom.
    (Kept the old function name so the rest of the file / call sites don't
    need to change.)"""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    top_color = np.array(COLOR_PALETTE['bg_top'], dtype=np.float32)
    bottom_color = np.array(COLOR_PALETTE['bg_bottom'], dtype=np.float32)

    horizon = int(height * 0.80)
    for y in range(horizon):
        t = y / max(1, horizon)
        row_color = (1.0 - t) * top_color + t * bottom_color
        bg[y] = row_color.astype(np.uint8)

    ground = np.array(COLOR_PALETTE['ground'], dtype=np.float32)
    for y in range(horizon, height):
        t = (y - horizon) / max(1, height - horizon)
        row_color = ground * (1.0 - 0.25 * t)
        bg[y] = row_color.astype(np.uint8)
    return bg


def make_starfield(width: int, height: int, count: int = 6):
    """Precompute cloud positions/sizes/drift-speed so they move smoothly
    frame to frame instead of re-randomizing. (Name kept for compatibility;
    these are now clouds, not stars.)"""
    rng = random.Random(42)
    clouds = []
    for _ in range(count):
        x = rng.uniform(0, width)
        y = rng.uniform(height * 0.05, height * 0.35)
        scale = rng.uniform(0.7, 1.6)
        speed = rng.uniform(4.0, 10.0)
        clouds.append([x, y, scale, speed])
    return clouds


def _draw_cloud_puff(canvas: np.ndarray, cx: int, cy: int, scale: float):
    color = COLOR_PALETTE['cloud']
    shadow = tuple(max(0, c - 35) for c in color)
    lobes = [(-1.0, 0.05, 1.0), (-0.4, -0.25, 1.2), (0.3, -0.2, 1.3), (0.9, 0.05, 1.0), (0.2, 0.15, 1.4)]
    for dx, dy, r in lobes:
        cv2.circle(canvas, (int(cx + dx * 40 * scale), int(cy + dy * 40 * scale + 6)),
                   int(r * 24 * scale), shadow, -1, lineType=cv2.LINE_AA)
    for dx, dy, r in lobes:
        cv2.circle(canvas, (int(cx + dx * 40 * scale), int(cy + dy * 40 * scale)),
                   int(r * 24 * scale), color, -1, lineType=cv2.LINE_AA)


def draw_stars(canvas: np.ndarray, clouds: list, time_sec: float):
    """Drifts and draws the clouds. (Function name kept for compatibility.)"""
    w = canvas.shape[1]
    for c in clouds:
        c[0] = (c[0] + c[3] * 0.02) % (w + 200) 
        _draw_cloud_puff(canvas, int(c[0] - 100), int(c[1]), c[2])


def draw_moon(canvas: np.ndarray, center: tuple, radius: int):
    """Draws a bright sun with a warm glow halo. (Function name kept for
    compatibility with the rest of the file.)"""
    cx, cy = center
    for r, alpha in [(int(radius * 3.2), 0.12), (int(radius * 2.2), 0.20), (int(radius * 1.5), 0.35)]:
        overlay = canvas.copy()
        cv2.circle(overlay, (cx, cy), r, COLOR_PALETTE['sun_glow'], -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    # simple sun rays
    for i in range(12):
        ang = (2 * math.pi / 12) * i
        x1 = int(cx + math.cos(ang) * radius * 1.3)
        y1 = int(cy + math.sin(ang) * radius * 1.3)
        x2 = int(cx + math.cos(ang) * radius * 1.9)
        y2 = int(cy + math.sin(ang) * radius * 1.9)
        cv2.line(canvas, (x1, y1), (x2, y2), COLOR_PALETTE['sun_glow'], 3, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), radius, COLOR_PALETTE['sun'], -1, lineType=cv2.LINE_AA)

# =============================================================================
# REALISTIC DRAWING
# =============================================================================
def draw_filled_petal(img: np.ndarray, center: tuple, length: float, width: float,
                      angle: float, color_fill: tuple, color_edge: tuple, shine: bool = False):
    """Renders a solid, organic teardrop petal with smooth shading and an
    optional glossy highlight streak down the middle."""
    cx, cy = center
    pts_local = [
        (0, 0),
        (-width * 0.45, length * 0.3),
        (-width * 0.5, length * 0.65),
        (0, length),
        (width * 0.5, length * 0.65),
        (width * 0.45, length * 0.3)
    ]

    pts_world = []
    for lx, ly in pts_local:
        rx = lx * math.cos(angle) - ly * math.sin(angle)
        ry = lx * math.sin(angle) + ly * math.cos(angle)
        pts_world.append([int(cx + rx), int(cy - ry)])

    pts_arr = np.array(pts_world, dtype=np.int32)

    cv2.fillPoly(img, [pts_arr], color=color_fill, lineType=cv2.LINE_AA)
    cv2.polylines(img, [pts_arr], isClosed=True, color=color_edge, thickness=1, lineType=cv2.LINE_AA)

    if shine:
        # thin bright streak slightly off-center = glossy petal highlight
        s0_local = (-width * 0.12, length * 0.15)
        s1_local = (-width * 0.05, length * 0.75)
        s0 = (int(cx + s0_local[0] * math.cos(angle) - s0_local[1] * math.sin(angle)),
              int(cy - (s0_local[0] * math.sin(angle) + s0_local[1] * math.cos(angle))))
        s1 = (int(cx + s1_local[0] * math.cos(angle) - s1_local[1] * math.sin(angle)),
              int(cy - (s1_local[0] * math.sin(angle) + s1_local[1] * math.cos(angle))))
        cv2.line(img, s0, s1, COLOR_PALETTE['petal_shine'], 1, lineType=cv2.LINE_AA)


def draw_sunflower_disk(img: np.ndarray, center: tuple, radius: float, time_sec: float):
    """Renders a dense, dark chocolate seed center with golden florets and a
    subtle domed radial-shading so it reads as 3D, not a flat disc."""
    cx, cy = center

    # Domed shading: several concentric rings, darkening toward the rim
    steps = 6
    for i in range(steps, 0, -1):
        t = i / steps
        r = radius * t
        shade = lerp(1.15, 0.55, t)
        base = COLOR_PALETTE['center_rim']
        ring_color = tuple(min(255, int(c * shade)) for c in base)
        cv2.circle(img, (int(cx), int(cy)), int(r), ring_color, -1, lineType=cv2.LINE_AA)

    num_seeds = 160
    for i in range(1, num_seeds):
        r = math.sqrt(i / float(num_seeds)) * radius
        theta = i * GOLDEN_ANGLE

        sx = int(cx + r * math.cos(theta))
        sy = int(cy + r * math.sin(theta))

        if r > radius * 0.65:
            seed_color = COLOR_PALETTE['floret_gold']
            seed_r = 2
        else:
            seed_color = COLOR_PALETTE['center_dark']
            seed_r = max(1, int(2.5 * (r / radius)))

        cv2.circle(img, (sx, sy), seed_r, seed_color, -1, lineType=cv2.LINE_AA)

    cv2.circle(img, (int(cx), int(cy)), int(radius * 0.35), COLOR_PALETTE['center_dark'], -1, lineType=cv2.LINE_AA)
    # crisp rim ring for definition against the glow pass
    cv2.circle(img, (int(cx), int(cy)), int(radius), (10, 10, 15), 1, lineType=cv2.LINE_AA)


def draw_stem_and_leaves(img: np.ndarray, base_x: int, base_y: int, head_x: int, head_y: int,
                        thickness: int, sway_offset: float, scale: float, time_sec: float):
    mid_x = int((base_x + head_x) / 2 + sway_offset * 0.5)
    mid_y = int((base_y + head_y) / 2)

    points = []
    num_segments = 25
    for i in range(num_segments + 1):
        t = i / float(num_segments)
        px = (1 - t)**2 * base_x + 2 * (1 - t) * t * mid_x + t**2 * head_x
        py = (1 - t)**2 * base_y + 2 * (1 - t) * t * mid_y + t**2 * head_y
        points.append([int(px), int(py)])

    pts_arr = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=False, color=COLOR_PALETTE['stem'],
                  thickness=thickness, lineType=cv2.LINE_AA)
    # thin rim-light on one side of the stem for a rounded, lit look
    rim_pts = pts_arr.copy()
    rim_pts[:, 0] -= max(1, thickness // 3)
    cv2.polylines(img, [rim_pts], isClosed=False, color=COLOR_PALETTE['stem_rim'],
                  thickness=max(1, thickness // 4), lineType=cv2.LINE_AA)

    leaf_configs = [
        {'seg': 0.3, 'deg': -60, 'mult': 1.0, 'color': COLOR_PALETTE['leaf_dark']},
        {'seg': 0.45, 'deg': 55, 'mult': 0.9, 'color': COLOR_PALETTE['leaf_light']},
        {'seg': 0.65, 'deg': -50, 'mult': 0.75, 'color': COLOR_PALETTE['leaf_dark']},
        {'seg': 0.8, 'deg': 45, 'mult': 0.6, 'color': COLOR_PALETTE['leaf_light']}
    ]

    for idx, l_cfg in enumerate(leaf_configs):
        seg_idx = int(num_segments * l_cfg['seg'])
        attach_point = tuple(points[seg_idx])
        angle_rad = math.radians(l_cfg['deg'] + sway_offset * 0.1)
        length = 40.0 * scale * l_cfg['mult']
        width = length * 0.5

        bx, by = attach_point
        leaf_pts = [
            (bx, by),
            (int(bx + length * 0.5 * math.cos(angle_rad - 0.4)), int(by - length * 0.5 * math.sin(angle_rad - 0.4))),
            (int(bx + length * math.cos(angle_rad)), int(by - length * math.sin(angle_rad))),
            (int(bx + length * 0.5 * math.cos(angle_rad + 0.4)), int(by - length * 0.5 * math.sin(angle_rad + 0.4)))
        ]
        cv2.fillPoly(img, [np.array(leaf_pts, dtype=np.int32)], color=l_cfg['color'], lineType=cv2.LINE_AA)
        # leaf spine highlight
        cv2.line(img, (bx, by), (int(bx + length * math.cos(angle_rad)), int(by - length * math.sin(angle_rad))),
                 COLOR_PALETTE['leaf_light'], 1, lineType=cv2.LINE_AA)


def draw_realistic_sunflower(img: np.ndarray, head_pos: tuple, stem_base: tuple,
                             bloom: float, sway: float, time_sec: float, config: dict):
    hx, hy = head_pos
    bx, by = stem_base

    draw_stem_and_leaves(img, bx, by, hx, hy, config['stem_thickness'], sway, config['scale'], time_sec)

    disk_radius = lerp(22.0, 42.0, bloom) * config['scale']
    petal_len = lerp(30.0, 95.0, bloom) * config['scale']
    petal_width = lerp(12.0, 26.0, bloom) * config['scale']

    num_petals = config['num_petals']

    layers = [
        {'scale': 1.0,  'color': COLOR_PALETTE['petal_base'], 'offset': 0.0, 'shine': False},
        {'scale': 0.9,  'color': COLOR_PALETTE['petal_mid'],  'offset': math.pi / num_petals, 'shine': False},
        {'scale': 0.78, 'color': COLOR_PALETTE['petal_tip'],  'offset': (math.pi / num_petals) * 0.5, 'shine': True}
    ]

    for layer in layers:
        l_len = petal_len * layer['scale']
        l_width = petal_width * layer['scale']

        for i in range(num_petals):
            angle = (2.0 * math.pi / num_petals) * i + layer['offset'] + config['base_rotation']

            px = hx + (disk_radius * 0.75) * math.cos(angle)
            py = hy - (disk_radius * 0.75) * math.sin(angle)

            draw_filled_petal(img, (px, py), l_len, l_width, angle - (math.pi / 2.0),
                              layer['color'], COLOR_PALETTE['petal_base'], shine=layer['shine'])

    draw_sunflower_disk(img, (hx, hy), disk_radius, time_sec)

# =============================================================================
# POST-PROCESSING: BLOOM/GLOW + VIGNETTE
# =============================================================================
def apply_bloom_glow(canvas: np.ndarray, threshold: int = 230, blur_size: int = 21, intensity: float = 0.35) -> np.ndarray:
    """Classic 'bloom' effect: pull out the bright pixels, blur them into a
    soft halo, then screen-blend back on top. This is what makes the petals
    and pollen actually look like they're glowing instead of flat-shaded."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    bright_pass = cv2.bitwise_and(canvas, mask_3ch)

    k = blur_size if blur_size % 2 == 1 else blur_size + 1
    blurred = cv2.GaussianBlur(bright_pass, (k, k), 0)
    blurred2 = cv2.GaussianBlur(bright_pass, (k * 2 + 1, k * 2 + 1), 0)
    glow = cv2.addWeighted(blurred, 0.6, blurred2, 0.4, 0)

    # screen blend: result = 1 - (1-a)(1-b), done in float space
    canvas_f = canvas.astype(np.float32) / 255.0
    glow_f = (glow.astype(np.float32) / 255.0) * intensity
    screened = 1.0 - (1.0 - canvas_f) * (1.0 - glow_f)
    return np.clip(screened * 255.0, 0, 255).astype(np.uint8)


def apply_vignette(canvas: np.ndarray, strength: float = 0.22) -> np.ndarray:
    """Darkens the corners so the eye is pulled toward the flowers in the
    center of frame -- a cheap but very effective 'cinematic' cue."""
    h, w = canvas.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    max_dist = math.hypot(cx, cy)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
    mask = 1.0 - strength * np.clip(dist, 0, 1) ** 2
    mask_3ch = np.dstack([mask] * 3)
    return np.clip(canvas.astype(np.float32) * mask_3ch, 0, 255).astype(np.uint8)


def apply_color_grade(canvas: np.ndarray, saturation_boost: float = 1.25, contrast: float = 1.08) -> np.ndarray:
    """Slight saturation + contrast push for a punchier, more 'graded' look."""
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_boost, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    graded = np.clip((boosted - 127.5) * contrast + 127.5, 0, 255)
    return graded.astype(np.uint8)

# =============================================================================
# GARDEN CLASS
# =============================================================================
class SunflowerGarden:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

        self.configs = [
            {'x_ratio': 0.50, 'min_h': 360, 'max_h': 180, 'scale': 1.25, 'num_petals': 24, 'stem_thickness': 9, 'base_rotation': 0.0},
            {'x_ratio': 0.24, 'min_h': 270, 'max_h': 120, 'scale': 0.95, 'num_petals': 20, 'stem_thickness': 7, 'base_rotation': 0.2},
            {'x_ratio': 0.76, 'min_h': 280, 'max_h': 130, 'scale': 1.05, 'num_petals': 22, 'stem_thickness': 7, 'base_rotation': -0.3},
            {'x_ratio': 0.10, 'min_h': 190, 'max_h': 60,  'scale': 0.80, 'num_petals': 18, 'stem_thickness': 5, 'base_rotation': 0.5},
            {'x_ratio': 0.90, 'min_h': 200, 'max_h': 70,  'scale': 0.85, 'num_petals': 18, 'stem_thickness': 5, 'base_rotation': -0.1}
        ]
        self.smooth_control = 0.5

    def update_and_draw(self, canvas: np.ndarray, raw_control: float, time_sec: float, wind_bias: float) -> list:
        self.smooth_control = lerp(self.smooth_control, raw_control, 0.08)
        bloom_val = self.smooth_control
        growth_val = 1.0 - self.smooth_control

        flower_heads = []
        for i, cfg in enumerate(self.configs):
            base_x = int(self.width * cfg['x_ratio'])
            base_y = self.height

            stem_h = lerp(cfg['min_h'], cfg['max_h'], growth_val)
            sway = math.sin(time_sec * 1.8 + i * 1.2) * 12.0 + wind_bias

            head_x = int(base_x + sway)
            head_y = int(base_y - stem_h)
            flower_heads.append((head_x, head_y))

            draw_realistic_sunflower(canvas, (head_x, head_y), (base_x, base_y),
                                     bloom_val, sway, time_sec, cfg)
        return flower_heads

# =============================================================================
# MAIN LOOP
# =============================================================================
def render_frame(garden, particles, stars, bg_gradient, width, height, raw_control, current_time, wind_force, dt):
    canvas = bg_gradient.copy()
    draw_stars(canvas, stars, current_time)
    draw_moon(canvas, (int(width * 0.86), int(height * 0.16)), 34)

    flower_heads = garden.update_and_draw(canvas, raw_control, current_time, wind_force)

    if garden.smooth_control > 0.7:
        for head in flower_heads:
            if random.random() < 0.25:
                particles.spawn_pollen(head, COLOR_PALETTE['particle'], count=1)

    particles.update_and_draw(canvas, dt, wind_x=wind_force * 0.05)

    canvas = apply_color_grade(canvas)
    canvas = apply_bloom_glow(canvas)
    canvas = apply_vignette(canvas)

    cv2.putText(canvas, f"Bloom: {int(garden.smooth_control * 100)}%",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 240, 255), 2, cv2.LINE_AA)
    return canvas


def run_test_mode():
    """No webcam needed: renders a handful of frames at different bloom
    levels to sunflower_test.png, so the look can be checked without a
    camera or a display."""
    width, height = 1280, 720
    garden = SunflowerGarden(width, height)
    particles = ParticleSystem(max_particles=160)
    stars = make_starfield(width, height)
    bg_gradient = draw_dark_background(width, height)

    samples = [(0.05, "closed / grown"), (0.5, "mid bloom"), (0.95, "full bloom")]
    tiles = []
    for control, label in samples:
        garden.smooth_control = control  # force it so the test is deterministic
        frame = render_frame(garden, particles, stars, bg_gradient, width, height,
                              raw_control=control, current_time=1.0, wind_force=5.0, dt=0.016)
        cv2.putText(frame, label, (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (width // 2, height // 2)))

    grid = np.hstack(tiles)
    cv2.imwrite("sunflower_test.png", grid)
    print(f"Saved sunflower_test.png ({grid.shape[1]}x{grid.shape[0]}) - open it to check the look.")


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

    garden = SunflowerGarden(width, height)
    particles = ParticleSystem(max_particles=160)
    stars = make_starfield(width, height)
    bg_gradient = draw_dark_background(width, height)

    start_time = time.time()
    last_frame_time = start_time
    raw_control = 0.5
    wind_force = 0.0

    # Pinch is measured as a RATIO (thumb-index distance / hand size), not
    # raw pixels -- this makes it scale-invariant, so it doesn't matter how
    # close you are to the camera or how big your hand looks on screen.
    # These starting bounds work for most webcams/hands, but you can also
    # live-calibrate: hold your fingers fully spread and press 'm' to set
    # that as 100%, or fully pinched and press 'n' to set that as 0%.
    pinch_ratio_min = 0.12
    pinch_ratio_max = 0.95
    last_pinch_ratio = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        current_time = now - start_time

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            thumb = (int(lm[4].x * width), int(lm[4].y * height))
            index = (int(lm[8].x * width), int(lm[8].y * height))
            wrist = (int(lm[0].x * width), int(lm[0].y * height))
            mcp = (int(lm[9].x * width), int(lm[9].y * height))

            pinch_dist = distance_2d(thumb, index)
            hand_scale = max(1.0, distance_2d(wrist, mcp))  # avoid div-by-zero
            pinch_ratio = pinch_dist / hand_scale
            last_pinch_ratio = pinch_ratio

            span = max(1e-4, pinch_ratio_max - pinch_ratio_min)
            raw_control = float(np.clip((pinch_ratio - pinch_ratio_min) / span, 0.0, 1.0))

            tilt = math.degrees(math.atan2(mcp[0] - wrist[0], -(mcp[1] - wrist[1])))
            wind_force = np.clip(tilt / 45.0, -1.0, 1.0) * 30.0

        canvas = render_frame(garden, particles, stars, bg_gradient, width, height,
                               raw_control, current_time, wind_force, dt)

        cv2.putText(canvas, f"pinch ratio: {last_pinch_ratio:.2f}  (min {pinch_ratio_min:.2f} / max {pinch_ratio_max:.2f})",
                    (30, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "spread fingers + press 'm' = set 100%   pinch + press 'n' = set 0%",
                    (30, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord('m'):
            pinch_ratio_max = max(pinch_ratio_min + 0.05, last_pinch_ratio)
        elif key == ord('n'):
            pinch_ratio_min = min(pinch_ratio_max - 0.05, last_pinch_ratio)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_test_mode()
    else:
        main()