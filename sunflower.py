import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random

# =============================================================================
# GLOBAL CONSTANTS & SETUP
# =============================================================================
GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ~137.5 degrees
WINDOW_NAME = "Interactive Realist Sunflower Garden"

# Realistic Sunflower Palette (BGR)
COLOR_PALETTE = {
    'bg_top': (25, 15, 10),            # Deep Midnight Blue (Top)
    'bg_bottom': (15, 35, 20),         # Dark Forest Dusk (Bottom)
    'petal_base': (0, 120, 230),       # Deep Golden Amber
    'petal_mid': (0, 185, 255),        # Rich Vibrant Yellow
    'petal_tip': (30, 220, 255),       # Bright Golden Sun
    'center_dark': (12, 20, 35),       # Deep Chocolate / Dark Core
    'center_rim': (20, 60, 110),       # Warm Brown Seed Rim
    'floret_gold': (0, 200, 255),      # Golden Pollen Florets
    'stem': (25, 100, 45),             # Realistic Stem Green
    'leaf_dark': (20, 75, 35),         # Foliage Shadow
    'leaf_light': (40, 130, 55),       # Foliage Highlight
    'particle': (80, 220, 255)         # Floating Pollen Glow
}

def lerp(start: float, end: float, factor: float) -> float:
    return start + (end - start) * factor

def distance_2d(p1: tuple, p2: tuple) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# =============================================================================
# PARTICLE SYSTEM
# =============================================================================
class ParticleSystem:
    def __init__(self, max_particles: int = 120):
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
        if len(self.particles) < self.max_particles and random.random() < 0.15:
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
                cv2.circle(canvas, (int(px), int(py)), size, glow_color, -1, lineType=cv2.LINE_AA)
        self.particles = alive_particles

# =============================================================================
# BACKGROUND & REALISTIC DRAWING
# =============================================================================
def draw_dark_background(width: int, height: int) -> np.ndarray:
    """Creates a subtle dark twilight gradient instead of pitch black."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    top_color = np.array(COLOR_PALETTE['bg_top'], dtype=np.float32)
    bottom_color = np.array(COLOR_PALETTE['bg_bottom'], dtype=np.float32)
    
    for y in np.linspace(0, 1, height):
        row_color = (1.0 - y) * top_color + y * bottom_color
        bg[int(y * (height - 1))] = row_color.astype(np.uint8)
    return bg

def draw_filled_petal(img: np.ndarray, center: tuple, length: float, width: float, 
                      angle: float, color_fill: tuple, color_edge: tuple):
    """Renders a solid, organic teardrop petal with smooth shading."""
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
    
    # Solid Petal Body
    cv2.fillPoly(img, [pts_arr], color=color_fill, lineType=cv2.LINE_AA)
    # Subtle Outer Edge for Definition
    cv2.polylines(img, [pts_arr], isClosed=True, color=color_edge, thickness=1, lineType=cv2.LINE_AA)

def draw_sunflower_disk(img: np.ndarray, center: tuple, radius: float, time_sec: float):
    """Renders a dense, dark chocolate seed center with golden florets."""
    cx, cy = center
    num_seeds = 160
    
    # Outer Brown Disk Shadow
    cv2.circle(img, (int(cx), int(cy)), int(radius), COLOR_PALETTE['center_rim'], -1, lineType=cv2.LINE_AA)
    
    # Dense Seed Spiral
    for i in range(1, num_seeds):
        r = math.sqrt(i / float(num_seeds)) * radius
        theta = i * GOLDEN_ANGLE
        
        sx = int(cx + r * math.cos(theta))
        sy = int(cy + r * math.sin(theta))
        
        # Outer ring florets glow gold, inner seeds stay dark
        if r > radius * 0.65:
            seed_color = COLOR_PALETTE['floret_gold']
            seed_r = 2
        else:
            seed_color = COLOR_PALETTE['center_dark']
            seed_r = max(1, int(2.5 * (r / radius)))
            
        cv2.circle(img, (sx, sy), seed_r, seed_color, -1, lineType=cv2.LINE_AA)
        
    # Dark Indented Core
    cv2.circle(img, (int(cx), int(cy)), int(radius * 0.35), COLOR_PALETTE['center_dark'], -1, lineType=cv2.LINE_AA)

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
    
    # Leaves Along Stem
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
        
        # Leaf Polygon
        bx, by = attach_point
        leaf_pts = [
            (bx, by),
            (int(bx + length * 0.5 * math.cos(angle_rad - 0.4)), int(by - length * 0.5 * math.sin(angle_rad - 0.4))),
            (int(bx + length * math.cos(angle_rad)), int(by - length * math.sin(angle_rad))),
            (int(bx + length * 0.5 * math.cos(angle_rad + 0.4)), int(by - length * 0.5 * math.sin(angle_rad + 0.4)))
        ]
        cv2.fillPoly(img, [np.array(leaf_pts, dtype=np.int32)], color=l_cfg['color'], lineType=cv2.LINE_AA)

def draw_realistic_sunflower(img: np.ndarray, head_pos: tuple, stem_base: tuple, 
                             bloom: float, sway: float, time_sec: float, config: dict):
    hx, hy = head_pos
    bx, by = stem_base
    
    draw_stem_and_leaves(img, bx, by, hx, hy, config['stem_thickness'], sway, config['scale'], time_sec)
    
    # Dimensions
    disk_radius = lerp(22.0, 42.0, bloom) * config['scale']
    petal_len = lerp(30.0, 95.0, bloom) * config['scale']
    petal_width = lerp(12.0, 26.0, bloom) * config['scale']
    
    num_petals = config['num_petals']
    
    # 3 Staggered Layers of Filled Petals for High Density
    layers = [
        {'scale': 1.0,  'color': COLOR_PALETTE['petal_base'], 'offset': 0.0},
        {'scale': 0.9,  'color': COLOR_PALETTE['petal_mid'],  'offset': math.pi / num_petals},
        {'scale': 0.78, 'color': COLOR_PALETTE['petal_tip'],  'offset': (math.pi / num_petals) * 0.5}
    ]
    
    for layer in layers:
        l_len = petal_len * layer['scale']
        l_width = petal_width * layer['scale']
        
        for i in range(num_petals):
            angle = (2.0 * math.pi / num_petals) * i + layer['offset'] + config['base_rotation']
            
            # Anchor at edge of the disk
            px = hx + (disk_radius * 0.75) * math.cos(angle)
            py = hy - (disk_radius * 0.75) * math.sin(angle)
            
            draw_filled_petal(img, (px, py), l_len, l_width, angle - (math.pi / 2.0), 
                              layer['color'], COLOR_PALETTE['petal_base'])

    # Center Seed Disc
    draw_sunflower_disk(img, (hx, hy), disk_radius, time_sec)

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
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
    
    garden = SunflowerGarden(width, height)
    particles = ParticleSystem(max_particles=120)
    bg_gradient = draw_dark_background(width, height)
    
    start_time = time.time()
    last_frame_time = start_time
    raw_control = 0.5
    wind_force = 0.0
    active_mode = 'PINCH'

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
            
            pinch_dist = distance_2d(thumb, index)
            raw_control = np.clip((pinch_dist - 20.0) / 200.0, 0.0, 1.0)
            
            wrist = (int(lm[0].x * width), int(lm[0].y * height))
            mcp = (int(lm[9].x * width), int(lm[9].y * height))
            tilt = math.degrees(math.atan2(mcp[0] - wrist[0], -(mcp[1] - wrist[1])))
            wind_force = np.clip(tilt / 45.0, -1.0, 1.0) * 30.0

        particles.spawn_firefly((width, height))
        
        # Start with the dark gradient canvas
        canvas = bg_gradient.copy()
        
        flower_heads = garden.update_and_draw(canvas, raw_control, current_time, wind_force)
        
        if garden.smooth_control > 0.7:
            for head in flower_heads:
                if random.random() < 0.25:
                    particles.spawn_pollen(head, COLOR_PALETTE['particle'], count=1)

        particles.update_and_draw(canvas, dt, wind_x=wind_force * 0.05)
        
        cv2.putText(canvas, f"Bloom: {int(garden.smooth_control * 100)}%", 
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 240, 255), 2, cv2.LINE_AA)
        
        cv2.imshow(WINDOW_NAME, canvas)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()