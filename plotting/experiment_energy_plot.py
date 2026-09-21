"""Lifetime experiment sensor-energy snapshots.

This module intentionally depends only on Pillow, so normal algorithm runs do not
need to import a plotting backend.  A snapshot uses one fixed red-to-green scale:
zero energy is red and the configured full energy is green.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = (1120, 920)
PLOT_BOX = (70, 90, 830, 850)


def _load_font(size, bold=False):
    """Load a readable bundled/system font, falling back to Pillow's default."""
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _energy_color(ratio):
    """Map an energy ratio to red -> yellow -> green using a fixed scale."""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    red = np.asarray((215, 48, 39), dtype=float)
    yellow = np.asarray((255, 224, 80), dtype=float)
    green = np.asarray((26, 152, 80), dtype=float)
    if ratio <= 0.5:
        color = red + (yellow - red) * (ratio / 0.5)
    else:
        color = yellow + (green - yellow) * ((ratio - 0.5) / 0.5)
    return tuple(np.rint(color).astype(int))


def _format_energy(value):
    """Format energy compactly while retaining four significant digits."""
    return f"{float(value):.4g} J"


def _map_position(point, boundary):
    """Convert a WSN coordinate to a pixel inside the map panel."""
    left, top, right, bottom = PLOT_BOX
    x = left + float(point[0]) / boundary * (right - left)
    # Match the project's experiment visualization: (0, 0) is the upper-left
    # corner and y increases downward on the image.
    y = top + float(point[1]) / boundary * (bottom - top)
    return int(round(x)), int(round(y))


def _draw_base_station(draw, position):
    """Draw the orange radio-beacon symbol used by experiment visualizations."""
    x, y = position
    orange = (255, 144, 30)
    for radius in (5, 8, 12, 14):
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=orange if radius == 5 else None,
            outline=orange,
            width=1,
        )
    draw.line((x, y + 5, x - 5, y + 21), fill=(20, 20, 20), width=2)
    draw.line((x, y + 5, x + 5, y + 21), fill=(20, 20, 20), width=2)


def _draw_dashed_ellipse(draw, bbox, fill, width=2, dash=8, gap=6):
    """Draw a dashed ellipse using Pillow's arc primitive."""
    period = max(int(dash) + int(gap), 1)
    for start in range(0, 360, period):
        draw.arc(
            bbox,
            start=start,
            end=min(start + int(dash), 360),
            fill=fill,
            width=width,
        )


def save_energy_snapshot(
    problem,
    output_path=None,
    *,
    algorithm,
    run_id,
    seed,
    stage,
    lifetime,
    segment_id=None,
    segment_length=None,
    state=None,
    display=False,
    window_name="WSN_live_energy",
):
    """Save one spatial sensor-energy map and return its displayed statistics.

    ``weakest_target_energy`` is the minimum, over all targets, of the sum of
    current energy held by sensors that can cover that target.  This is the same
    aggregation exposed by ``Problem.calculate_target_remaining_energy``.
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    energy = np.asarray(problem.energy, dtype=float).reshape(-1)
    full_energy = np.asarray(problem.initial_energy, dtype=float)
    if full_energy.ndim == 0:
        denominator = np.full_like(energy, float(full_energy))
    else:
        denominator = np.broadcast_to(full_energy, energy.shape)
    energy_ratio = np.divide(
        energy,
        denominator,
        out=np.zeros_like(energy),
        where=denominator > 0,
    )
    energy_ratio = np.clip(energy_ratio, 0.0, 1.0)

    target_energy = np.asarray(
        problem.calculate_target_remaining_energy(energy), dtype=float
    ).reshape(-1)
    if target_energy.size:
        weakest_target_id = int(np.argmin(target_energy))
        weakest_target_energy = float(target_energy[weakest_target_id])
    else:
        weakest_target_id = None
        weakest_target_energy = float("nan")

    total_energy = float(np.sum(energy))
    image = Image.new("RGB", CANVAS_SIZE, "white")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(25, bold=True)
    heading_font = _load_font(18, bold=True)
    text_font = _load_font(16)
    summary_font = _load_font(15)
    small_font = _load_font(11)

    title = f"{algorithm} sensor energy | run {run_id} | seed {seed}"
    draw.text((70, 25), title, fill=(20, 20, 20), font=title_font)
    draw.text(
        (70, 58),
        f"Stage: {stage}    Lifetime: {lifetime}",
        fill=(45, 45, 45),
        font=text_font,
    )

    left, top, right, bottom = PLOT_BOX
    draw.rectangle(PLOT_BOX, outline=(60, 60, 60), width=2)
    boundary = max(float(problem.BOUNDARY), 1.0)

    # Live mode extends the energy map with the selected solution's sensing
    # ranges and routing tree.  Ordinary saved snapshots do not pass a
    # state, so their previous appearance stays unchanged.
    if state is not None:
        levels = np.asarray(getattr(state, "levels", []), dtype=int).reshape(-1)
        next_hops = np.asarray(
            getattr(state, "next_hops", []), dtype=int
        ).reshape(-1)

        def _pixel_radius(radius):
            return max(1, int(round(float(radius) / boundary * (right - left))))

        for sensor_id, level in enumerate(levels):
            if level <= 0 or sensor_id >= len(problem.sensor):
                continue
            try:
                radius = problem.sensing_radius(sensor_id, int(level))
            except (IndexError, TypeError, ValueError):
                continue
            x, y = _map_position(problem.sensor[sensor_id], boundary)
            pixel_radius = _pixel_radius(radius)
            _draw_dashed_ellipse(
                draw,
                (
                    x - pixel_radius,
                    y - pixel_radius,
                    x + pixel_radius,
                    y + pixel_radius,
                ),
                fill=(0, 174, 190),
                width=2,
            )

        device = np.asarray(getattr(problem, "device", []))
        if next_hops.size and device.ndim == 2:
            for sensor_id, next_hop in enumerate(next_hops):
                next_hop = int(next_hop)
                if (
                    sensor_id >= len(device)
                    or next_hop < 0
                    or next_hop >= len(device)
                ):
                    continue
                start = _map_position(device[sensor_id], boundary)
                end = _map_position(device[next_hop], boundary)
                draw.line((start, end), fill=(100, 100, 170), width=2)
                # Small arrowhead pointing toward the next hop.
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length = max((dx * dx + dy * dy) ** 0.5, 1.0)
                ux, uy = dx / length, dy / length
                px, py = -uy, ux
                tip = end
                base_x = end[0] - ux * 10
                base_y = end[1] - uy * 10
                draw.polygon(
                    [
                        tip,
                        (int(base_x + px * 4), int(base_y + py * 4)),
                        (int(base_x - px * 4), int(base_y - py * 4)),
                    ],
                    fill=(100, 100, 170),
                )

    # Reproduce the existing experiment map's coordinate ticks: x is printed
    # below the map and y on the right, with the origin at the upper-left.
    for tick_id in range(11):
        ratio = tick_id / 10
        x = left + round(ratio * (right - left))
        y = top + round(ratio * (bottom - top))
        label = f"{boundary * ratio:g}"
        draw.line((x, bottom, x, bottom + 7), fill=(30, 30, 30), width=1)
        draw.text(
            (x - 8, bottom + 10),
            label,
            fill=(30, 30, 30),
            font=small_font,
        )
        draw.line((right, y, right + 7, y), fill=(30, 30, 30), width=1)
        draw.text(
            (right + 11, y - 6),
            label,
            fill=(30, 30, 30),
            font=small_font,
        )

    # Targets follow the experiment view's filled-dot style. The weakest one
    # keeps a purple outline so the energy-specific information remains clear.
    for target_id, point in enumerate(np.asarray(problem.target)):
        x, y = _map_position(point, boundary)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 20, 20))
        if target_id == weakest_target_id:
            draw.ellipse(
                (x - 7, y - 7, x + 7, y + 7),
                outline=(135, 45, 170),
                width=2,
            )

    bs_x, bs_y = _map_position(problem.BS, boundary)
    _draw_base_station(draw, (bs_x, bs_y))

    for sensor_id, (point, ratio) in enumerate(
        zip(np.asarray(problem.sensor), energy_ratio)
    ):
        x, y = _map_position(point, boundary)
        radius = 7
        diamond = [
            (x, y - radius),
            (x - radius, y),
            (x, y + radius),
            (x + radius, y),
        ]
        draw.polygon(
            diamond,
            fill=_energy_color(ratio),
        )
        draw.line(diamond + [diamond[0]], fill=(20, 20, 20), width=1)
        draw.text(
            (x + radius + 2, y - radius - 11),
            str(sensor_id),
            fill=(20, 20, 20),
            font=small_font,
        )

    panel_x = 890
    draw.text((panel_x, 100), "Energy summary", fill=(20, 20, 20), font=heading_font)
    lines = [
        f"Total energy: {_format_energy(total_energy)}",
        f"Mean energy: {_format_energy(np.mean(energy))}",
        f"Minimum energy: {_format_energy(np.min(energy))}",
    ]
    if weakest_target_id is None:
        lines.append("Weakest target: N/A")
    else:
        lines.append(
            f"Weakest target T{weakest_target_id}: "
            f"{_format_energy(weakest_target_energy)}"
        )
    if segment_id is not None:
        lines.append(f"Segment: {segment_id}")
    if segment_length is not None:
        lines.append(f"Segment lifetime: {segment_length}")
    for index, line in enumerate(lines):
        draw.text(
            (panel_x, 140 + index * 29),
            line,
            fill=(35, 35, 35),
            font=summary_font,
        )

    legend_y = 405
    # Compact legend for the four requested map symbols.
    icon_x = panel_x + 35
    icon_y = legend_y + 12
    draw.ellipse((icon_x - 4, icon_y - 4, icon_x + 4, icon_y + 4), fill=(20, 20, 20))
    draw.text((panel_x + 65, icon_y - 8), "Target", fill=(20, 20, 20), font=text_font)
    weak_y = icon_y + 30
    draw.ellipse((icon_x - 4, weak_y - 4, icon_x + 4, weak_y + 4), fill=(20, 20, 20))
    draw.ellipse((icon_x - 9, weak_y - 9, icon_x + 9, weak_y + 9), outline=(135, 45, 170), width=2)
    draw.text((panel_x + 65, weak_y - 8), "Weak target", fill=(20, 20, 20), font=text_font)
    bs_y = icon_y + 60
    _draw_base_station(draw, (icon_x, bs_y))
    draw.text((panel_x + 65, bs_y - 8), "Base station", fill=(20, 20, 20), font=text_font)
    sensor_y = icon_y + 93
    diamond = [(icon_x, sensor_y - 8), (icon_x - 8, sensor_y),
               (icon_x, sensor_y + 8), (icon_x + 8, sensor_y)]
    draw.polygon(diamond, fill=(34, 139, 34), outline=(20, 20, 20))
    draw.text((panel_x + 65, sensor_y - 8), "Sensor", fill=(20, 20, 20), font=text_font)

    draw.text((panel_x, 545), "Remaining energy", fill=(20, 20, 20), font=heading_font)
    # Keep the energy scale vertically aligned with the map panel.
    bar_left, bar_top, bar_right, bar_bottom = panel_x, 580, panel_x + 52, bottom
    steps = 100
    for step in range(steps):
        ratio = 1.0 - step / (steps - 1)
        y0 = bar_top + round(step * (bar_bottom - bar_top) / steps)
        y1 = bar_top + round((step + 1) * (bar_bottom - bar_top) / steps)
        draw.rectangle((bar_left, y0, bar_right, y1), fill=_energy_color(ratio))
    draw.rectangle((bar_left, bar_top, bar_right, bar_bottom), outline=(30, 30, 30), width=1)
    for ratio in (1.0, 0.75, 0.5, 0.25, 0.0):
        y = bar_top + round((1.0 - ratio) * (bar_bottom - bar_top))
        draw.line((bar_right, y, bar_right + 8, y), fill=(30, 30, 30), width=1)
        draw.text((bar_right + 13, y - 8), f"{ratio:.0%}", fill=(30, 30, 30), font=text_font)

    if output_path is not None:
        image.save(output_path, format="PNG", optimize=True)
    if display:
        import cv2

        frame = np.asarray(image)[:, :, ::-1]
        cv2.imshow(window_name, frame)
        cv2.waitKey(10)
    return {
        "total_energy": total_energy,
        "weakest_target_id": weakest_target_id,
        "weakest_target_energy": weakest_target_energy,
    }
