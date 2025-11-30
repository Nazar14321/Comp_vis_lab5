import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def apply_gaussian_smoothing(frame, kernel_size=5, sigma=3):
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)


def kmeans_color_quantization(rgb_image, clusters=4, attempts=10):
    flat_pixels = rgb_image.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )

    _compactness, labels, centers = cv2.kmeans(
        flat_pixels,
        clusters,
        None,
        criteria,
        attempts,
        cv2.KMEANS_RANDOM_CENTERS,
    )

    centers_u8 = np.uint8(centers)
    quantized = centers_u8[labels.flatten()]
    return quantized.reshape(rgb_image.shape)


def canny_edge_mask(
        src_image,
        low_thr=10,
        high_thr=20,
        aperture=3,
        use_l2=True,
        merge_kernel=3,
        ignore_rgb=None,
        ignore_tolerance=30,
        ignore_dilate_radius=10,
):
    if len(src_image.shape) == 3:
        gray = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = src_image.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 1.0)

    edges = cv2.Canny(
        blurred,
        threshold1=low_thr,
        threshold2=high_thr,
        apertureSize=aperture,
        L2gradient=use_l2,
    )

    if merge_kernel and merge_kernel > 1:
        if merge_kernel % 2 == 0:
            merge_kernel += 1
        fused = cv2.GaussianBlur(edges, (merge_kernel, merge_kernel), 0)
        _, edges = cv2.threshold(
            fused, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if ignore_rgb is not None:
        if len(src_image.shape) == 3:
            img_int16 = src_image.astype(np.int16)
            target = np.array(ignore_rgb, dtype=np.int16).reshape(1, 1, 3)
            diff = img_int16 - target
            dist = np.linalg.norm(diff, axis=2)
            color_mask = dist < ignore_tolerance
        else:
            col = np.uint8([[ignore_rgb[:3]]])
            gray_target = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)[0, 0]
            dist = np.abs(gray.astype(np.int16) - int(gray_target))
            color_mask = dist < ignore_tolerance

        if ignore_dilate_radius and ignore_dilate_radius > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (ignore_dilate_radius, ignore_dilate_radius),
            )
            color_mask = cv2.dilate(
                color_mask.astype(np.uint8), kernel
            ).astype(bool)

        edges[color_mask] = 0

    return edges


def pick_forest_like_cluster(
        rgb_source,
        kmeans_rgb,
        forest_target=(32, 92, 24),
        primary_tolerance=80.0,
        texture_min=30.0,
        texture_max=300.0,
        neighbor_tolerance=80.0,
        border_erosion_iters=2,
        save_overlay=False,
):
    if rgb_source.shape[:2] != kmeans_rgb.shape[:2]:
        kmeans_rgb = cv2.resize(
            kmeans_rgb,
            (rgb_source.shape[1], rgb_source.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    flat = kmeans_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    brightness = (
            0.299 * uniq[:, 0] + 0.587 * uniq[:, 1] + 0.114 * uniq[:, 2]
    )
    sort_idx = np.argsort(brightness)

    target_vec = np.array(forest_target, dtype=np.float32)

    primary_color = None
    secondary_color = None

    for idx in sort_idx:
        color_vec = uniq[idx].astype(np.float32)
        dist = np.linalg.norm(color_vec - target_vec)
        if dist <= primary_tolerance:
            primary_color = uniq[idx]
            break

    if primary_color is None:
        primary_color = uniq[sort_idx[0]]

    primary_vec = primary_color.astype(np.float32)
    for idx in sort_idx:
        color = uniq[idx]
        if np.array_equal(color, primary_color):
            continue
        cv = color.astype(np.float32)
        d2 = np.linalg.norm(cv - primary_vec)
        if d2 <= neighbor_tolerance:
            secondary_color = color
            break

    mask_primary = np.all(
        kmeans_rgb == primary_color.reshape(1, 1, 3), axis=2
    ).astype(np.uint8) * 255

    combined_mask = mask_primary.copy()

    if secondary_color is not None:
        secondary_bool = np.all(
            kmeans_rgb == secondary_color.reshape(1, 1, 3), axis=2
        ).astype(np.uint8)

        if np.any(secondary_bool):
            gray_src = cv2.cvtColor(rgb_source, cv2.COLOR_RGB2GRAY)
            sec_u8 = (secondary_bool * 255).astype(np.uint8)

            if border_erosion_iters > 0:
                kernel = np.ones((3, 3), np.uint8)
                inner = cv2.erode(sec_u8, kernel, iterations=border_erosion_iters)
            else:
                inner = sec_u8

            if not np.any(inner):
                inner = sec_u8

            inner_bool = inner.astype(bool)
            lap = cv2.Laplacian(gray_src, cv2.CV_64F)
            lap_values = lap[inner_bool]
            noise_var = float(lap_values.var()) if lap_values.size > 0 else 0.0

            if texture_min <= noise_var <= texture_max:
                combined_mask = cv2.bitwise_or(
                    combined_mask, (secondary_bool * 255).astype(np.uint8)
                )

    return combined_mask


def knock_out_mask_as_black(rgb_image, binary_mask):
    if binary_mask.ndim == 3:
        mask_gray = cv2.cvtColor(binary_mask, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = binary_mask.copy()

    if rgb_image.shape[:2] != mask_gray.shape[:2]:
        mask_gray = cv2.resize(
            mask_gray,
            (rgb_image.shape[1], rgb_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    result = rgb_image.copy()
    result[mask_gray == 255] = (0, 0, 0)
    return result


def fuse_similar_palette_colors(kmeans_rgb, distance_thr=20.0):
    flat = kmeans_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    if len(uniq) <= 1:
        return kmeans_rgb.copy()

    centers = uniq.copy()

    while True:
        centers_f = centers.astype(np.float32)
        diff = centers_f[:, None, :] - centers_f[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e9)

        min_idx = np.argmin(dist)
        i, j = divmod(min_idx, dist.shape[1])
        min_dist = dist[i, j]

        if min_dist > distance_thr:
            break

        new_center = ((centers_f[i] + centers_f[j]) / 2.0).astype(np.uint8)

        keep = np.ones(len(centers), dtype=bool)
        keep[i] = False
        keep[j] = False
        centers = np.concatenate(
            [centers[keep], new_center.reshape(1, 3)], axis=0
        )

    h, w, _ = kmeans_rgb.shape
    centers_f = centers.astype(np.float32)
    pixels_f = flat.astype(np.float32)

    diff_all = pixels_f[:, None, :] - centers_f[None, :, :]
    dist_all = np.sum(diff_all ** 2, axis=2)
    nearest_idx = np.argmin(dist_all, axis=1)

    merged_flat = centers_f[nearest_idx].astype(np.uint8)
    return merged_flat.reshape(h, w, 3)


def select_regions_excluding_palette(
        src_rgb,
        palette_rgb,
        excluded_colors,
        exclusion_tolerance=40.0,
        min_region_area=100,
        max_hole_ratio=0.2,
        min_rect_fill=0.4,
        max_rect_fill=1.0,
        alpha=0.25,
        highlight_color=(255, 0, 0),
):
    if src_rgb.shape[:2] != palette_rgb.shape[:2]:
        palette_rgb = cv2.resize(
            palette_rgb,
            (src_rgb.shape[1], src_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if isinstance(excluded_colors, (tuple, list)) and excluded_colors:
        if not isinstance(excluded_colors[0], (tuple, list, np.ndarray)):
            excluded_colors = [excluded_colors]

    if not excluded_colors:
        excluded_arr = None
    else:
        excluded_arr = np.array(excluded_colors, dtype=np.float32).reshape(-1, 3)

    h, w, _ = palette_rgb.shape
    flat = palette_rgb.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)

    result_mask = np.zeros((h, w), dtype=np.uint8)

    for color in uniq:
        cluster_mask = np.all(
            palette_rgb == color.reshape(1, 1, 3), axis=2
        ).astype(np.uint8) * 255

        if not np.any(cluster_mask):
            continue

        num_labels, labels = cv2.connectedComponents(cluster_mask)

        for lbl in range(1, num_labels):
            region_mask = (labels == lbl).astype(np.uint8) * 255
            area = int(np.count_nonzero(region_mask))

            if area < min_region_area:
                continue

            if excluded_arr is not None:
                cf = color.astype(np.float32)
                dists = np.linalg.norm(excluded_arr - cf, axis=1)
                if float(np.min(dists)) <= exclusion_tolerance:
                    continue

            contours, _ = cv2.findContours(
                region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            filled = np.zeros_like(region_mask)
            cv2.drawContours(filled, contours, -1, 255, thickness=-1)

            filled_area = int(np.count_nonzero(filled))
            if filled_area == 0:
                continue

            holes_area = filled_area - area
            hole_ratio = holes_area / float(filled_area)

            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            w_rect, h_rect = rect[1]
            rect_area = float(w_rect * h_rect) if w_rect > 0 and h_rect > 0 else 0.0
            if rect_area <= 0:
                rect_fill_ratio = 0.0
            else:
                rect_fill_ratio = area / rect_area

            if (
                    hole_ratio <= max_hole_ratio
                    and min_rect_fill <= rect_fill_ratio <= max_rect_fill
            ):
                result_mask = cv2.bitwise_or(result_mask, region_mask)

    return result_mask


def overlay_masks_with_legend(
        base_rgb,
        masks,
        colors,
        labels=None,
        alpha=0.25,
        output_dir=".",
        filename="",
        font_path=None,
        font_size=18,
):
    h, w = base_rgb.shape[:2]
    blended = base_rgb.copy().astype(np.float32)

    prepared_masks = []
    for m in masks:
        if isinstance(m, tuple):
            m = m[-1]
        if m.ndim == 3:
            g = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        else:
            g = m.copy()

        if g.shape[:2] != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)

        prepared_masks.append(g)

    for idx, g in enumerate(prepared_masks):
        color_vec = np.array(colors[idx], dtype=np.float32)
        mask_bool = g > 0
        blended[mask_bool] = (
                (1.0 - alpha) * blended[mask_bool] + alpha * color_vec
        )

    blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
    final_img = blended_uint8

    if labels is not None:
        rows = max(1, len(labels))
        row_h = 30
        legend_h = rows * row_h + 10
        legend = np.ones((legend_h, w, 3), dtype=np.uint8) * 255

        for idx, _ in enumerate(labels):
            y = (idx + 1) * row_h
            x0, x1 = 10, 40
            y0, y1 = y - 15, y + 5

            color = tuple(int(c) for c in colors[idx])
            cv2.rectangle(
                legend,
                (x0, max(5, y0)),
                (x1, min(legend_h - 5, y1)),
                color,
                thickness=-1,
            )

        legend_pil = Image.fromarray(legend)
        draw = ImageDraw.Draw(legend_pil)

        if font_path is None:
            possible_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    font_path = p
                    break

        if font_path is not None and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()

        for idx, label in enumerate(labels):
            y = (idx + 1) * row_h
            text_pos = (50, y - 10)
            draw.text(text_pos, str(label), font=font, fill=(0, 0, 0))

        legend = np.array(legend_pil)

        final_img = np.vstack([blended_uint8, legend])

    if filename:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        img_pil = Image.fromarray(final_img)
        img_pil.save(out_path)

    return final_img


def gray_mask(image, tolerance=35):
    if image.ndim == 3:
        img_tmp = image.astype(np.int16)
        c1, c2, c3 = img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2]
        max_c = np.maximum(np.maximum(c1, c2), c3)
        min_c = np.minimum(np.minimum(c1, c2), c3)
        spread = max_c - min_c
        not_black = (c1 != 0) | (c2 != 0) | (c3 != 0)
        mask_bool = (spread <= tolerance) & not_black
        return (mask_bool.astype(np.uint8) * 255)
    else:
        gray_b = image if image.ndim == 2 else image[:, :, 0]
        return (gray_b > 0).astype(np.uint8) * 255


NUM_IMAGES = 10

for idx in range(NUM_IMAGES):
    bgr = cv2.imread(f"{idx}.png")
    if bgr is None:
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 1. Ліс
    smoothed_forest = apply_gaussian_smoothing(rgb, kernel_size=15, sigma=15)
    forest_kmeans = kmeans_color_quantization(smoothed_forest, clusters=4, attempts=5)
    forest_mask = pick_forest_like_cluster(
        rgb,
        forest_kmeans,
        texture_min=2.5,
        texture_max=5.0,
        neighbor_tolerance=30,
        save_overlay=False,
    )
    rgb_without_forest = knock_out_mask_as_black(rgb, forest_mask)

    # 2. Поля
    blurred_fields = apply_gaussian_smoothing(rgb_without_forest, kernel_size=15, sigma=15)
    fields_kmeans = kmeans_color_quantization(blurred_fields, clusters=5, attempts=5)
    merged_fields = fuse_similar_palette_colors(fields_kmeans, distance_thr=40)
    edges_for_fields = canny_edge_mask(rgb, merge_kernel=20)
    fields_without_edges = knock_out_mask_as_black(merged_fields, edges_for_fields)
    field_mask = select_regions_excluding_palette(
        rgb,
        fields_without_edges,
        excluded_colors=[(190, 190, 190), (0, 0, 0)],
        exclusion_tolerance=60,
        min_region_area=3000,
        max_hole_ratio=0.1,
        min_rect_fill=0.45,
    )

    # 3. Будівлі
    land_without_forest_and_fields = knock_out_mask_as_black(
        rgb_without_forest, field_mask
    )
    building_edges = canny_edge_mask(
        land_without_forest_and_fields,
        low_thr=40,
        high_thr=55,
        merge_kernel=15,
        ignore_rgb=(0, 0, 0),
    )
    inverted_edges = cv2.bitwise_not(building_edges)
    buildings_only = knock_out_mask_as_black(
        land_without_forest_and_fields, inverted_edges
    )

    buildings_mask = gray_mask(buildings_only)

    # 4. Вивід
    _final_image = overlay_masks_with_legend(
        rgb,
        [forest_mask, field_mask, buildings_mask],
        [(0, 255, 0), (255, 255, 0), (0, 0, 0)],
        labels=["Ліс", "Полія", "Будівлі"],
        alpha=0.5,
        filename=f"{idx}_класифіковано.png",
    )
