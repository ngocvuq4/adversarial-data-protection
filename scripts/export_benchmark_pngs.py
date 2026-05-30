from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIRS = [
    Path("reports/assets/benchmark_tables"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def save_table_png(headers: list[str], rows: list[list[object]], title: str, filename: str) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
    title_font = font(28, bold=True)
    head_font = font(17, bold=True)
    body_font = font(17)

    padding_x = 18
    padding_y = 13
    temp = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(temp)

    string_rows = [[str(v) for v in row] for row in rows]
    widths = []
    for col_idx, header in enumerate(headers):
        max_w = text_size(draw, header, head_font)[0]
        for row in string_rows:
            max_w = max(max_w, text_size(draw, row[col_idx], body_font)[0])
        widths.append(max_w + padding_x * 2)

    title_h = 64
    row_h = 48
    table_w = sum(widths)
    image_w = max(table_w + 80, text_size(draw, title, title_font)[0] + 80)
    image_h = title_h + row_h * (len(rows) + 1) + 70

    img = Image.new("RGB", (image_w, image_h), "white")
    draw = ImageDraw.Draw(img)

    title_w, title_text_h = text_size(draw, title, title_font)
    draw.text(((image_w - title_w) / 2, 24), title, fill="#24292F", font=title_font)

    x0 = (image_w - table_w) // 2
    y0 = title_h + 24
    x = x0
    for col_idx, header in enumerate(headers):
        draw.rectangle([x, y0, x + widths[col_idx], y0 + row_h], fill="#24292F", outline="#D0D7DE")
        tw, th = text_size(draw, header, head_font)
        draw.text((x + (widths[col_idx] - tw) / 2, y0 + (row_h - th) / 2 - 1), header, fill="white", font=head_font)
        x += widths[col_idx]

    for row_idx, row in enumerate(string_rows):
        y = y0 + row_h * (row_idx + 1)
        fill = "#F6F8FA" if row_idx % 2 else "white"
        x = x0
        for col_idx, value in enumerate(row):
            draw.rectangle([x, y, x + widths[col_idx], y + row_h], fill=fill, outline="#D0D7DE")
            tw, th = text_size(draw, value, body_font)
            draw.text((x + (widths[col_idx] - tw) / 2, y + (row_h - th) / 2 - 1), value, fill="#24292F", font=body_font)
            x += widths[col_idx]

    for out_dir in OUT_DIRS:
        img.save(out_dir / filename)


def save_line_chart() -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(30, bold=True)
    label_font = font(18)
    tick_font = font(15)

    title = "Unlearnable Epsilon Ablation With Validation"
    tw, th = text_size(draw, title, title_font)
    draw.text(((1400 - tw) / 2, 30), title, fill="#24292F", font=title_font)

    left, top, right, bottom = 120, 115, 1020, 620
    draw.rectangle([left, top, right, bottom], outline="#D0D7DE", width=2)

    eps = [0.01, 0.03, 0.05]
    baseline = [0.6457, 0.6457, 0.6457]
    protected = [0.1375, 0.1055, 0.1030]
    psnr = [40.5664, 31.1111, 26.6897]
    ssim = [0.9922, 0.9348, 0.8619]

    def x_map(v: float) -> float:
        return left + (v - 0.01) / 0.04 * (right - left)

    def y_acc(v: float) -> float:
        return bottom - v * (bottom - top)

    def y_quality(v: float, min_v: float, max_v: float) -> float:
        return bottom - (v - min_v) / (max_v - min_v) * (bottom - top)

    for i in range(6):
        y = bottom - i / 5 * (bottom - top)
        draw.line([left, y, right, y], fill="#EAEEF2", width=1)
        draw.text((70, y - 9), f"{i/5:.1f}", fill="#57606A", font=tick_font)

    for value in eps:
        x = x_map(value)
        draw.line([x, bottom, x, bottom + 8], fill="#57606A", width=2)
        draw.text((x - 18, bottom + 16), f"{value:.2f}", fill="#57606A", font=tick_font)

    draw.text((440, 680), "Epsilon", fill="#24292F", font=label_font)
    draw.text((20, 335), "Accuracy", fill="#24292F", font=label_font)
    draw.text((1045, 335), "Quality scale", fill="#24292F", font=label_font)

    def draw_series(values: list[float], color: str, mapper) -> None:
        pts = [(x_map(e), mapper(v)) for e, v in zip(eps, values)]
        draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color)

    draw_series(baseline, "#0366D6", y_acc)
    draw_series(protected, "#D73A49", y_acc)
    draw_series(psnr, "#22863A", lambda v: y_quality(v, 25, 42))
    draw_series(ssim, "#6F42C1", lambda v: y_quality(v, 0.80, 1.00))

    legend_x, legend_y = 1070, 170
    legend = [
        ("Baseline clean acc", "#0366D6"),
        ("Protected clean acc", "#D73A49"),
        ("PSNR", "#22863A"),
        ("SSIM", "#6F42C1"),
    ]
    for i, (name, color) in enumerate(legend):
        y = legend_y + i * 42
        draw.line([legend_x, y + 10, legend_x + 42, y + 10], fill=color, width=5)
        draw.ellipse([legend_x + 16, y + 3, legend_x + 30, y + 17], fill=color)
        draw.text((legend_x + 55, y), name, fill="#24292F", font=label_font)

    for out_dir in OUT_DIRS:
        img.save(out_dir / "unlearnable_validation_epsilon_line_chart.png")


def main() -> None:
    save_table_png(
        ["epsilon", "train_size", "victim", "baseline_acc", "protected_acc", "drop", "asr", "psnr", "ssim", "linf"],
        [
            [0.01, 5000, "ResNet-18", 0.3690, 0.1160, 0.2530, 0.8840, 40.6858, 0.9928, 0.01],
            [0.03, 5000, "ResNet-18", 0.4900, 0.1230, 0.3670, 0.8770, 31.4192, 0.9413, 0.03],
            [0.05, 5000, "ResNet-18", 0.3850, 0.1170, 0.2680, 0.8830, 27.1187, 0.8834, 0.05],
        ],
        "Preliminary Benchmark Before Validation - No Fixed Seed",
        "benchmark_pre_validation_no_seed.png",
    )
    save_table_png(
        ["epsilon", "train_size", "seed", "victim", "baseline_acc", "protected_acc", "drop", "asr", "psnr", "ssim", "linf"],
        [
            [0.01, 5000, 42, "ResNet-18", 0.4180, 0.1000, 0.3180, 0.9000, 40.7485, 0.9918, 0.01],
            [0.03, 5000, 42, "ResNet-18", 0.4180, 0.0850, 0.3330, 0.9150, 31.2963, 0.9424, 0.03],
            [0.05, 5000, 42, "ResNet-18", 0.4180, 0.1050, 0.3130, 0.8950, 27.0640, 0.8818, 0.05],
        ],
        "Preliminary Benchmark Before Validation - Fixed Seed 42",
        "benchmark_pre_validation_seed42.png",
    )
    save_table_png(
        [
            "epsilon",
            "train",
            "val",
            "test",
            "baseline_val",
            "baseline_test_ref",
            "protected_val",
            "val_drop",
            "val_asr",
            "psnr",
            "ssim",
            "quality_ok",
        ],
        [
            [0.01, 10000, 2000, 10000, 0.6510, 0.6457, 0.1375, 0.5135, 0.8625, 40.5664, 0.9922, "True"],
            [0.03, 10000, 2000, 10000, 0.6510, 0.6457, 0.1055, 0.5455, 0.8945, 31.1111, 0.9348, "True"],
            [0.05, 10000, 2000, 10000, 0.6510, 0.6457, 0.1030, 0.5480, 0.8970, 26.6897, 0.8619, "False"],
        ],
        "Validation Epsilon Ablation - Train 10000 / Val 2000 / Test 10000",
        "benchmark_validation_epsilon_ablation.png",
    )
    save_table_png(
        ["selected_epsilon", "baseline_test_acc", "final_protected_test_acc", "final_drop", "final_asr", "psnr", "ssim"],
        [[0.03, 0.6457, 0.1012, 0.5445, 0.8988, 31.1111, 0.9348]],
        "Final Test After Validation Selection",
        "benchmark_validation_final_test.png",
    )
    save_table_png(
        [
            "epsilon",
            "train",
            "val",
            "test",
            "victim",
            "baseline_val",
            "baseline_test",
            "protected_val",
            "val_drop",
            "val_asr",
            "psnr",
            "ssim",
            "linf",
        ],
        [[0.03, 45000, 5000, 10000, "ResNet-18", 0.8632, 0.8528, 0.0966, 0.7666, 0.9034, 31.5549, 0.9419, 0.03]],
        "Full Final Benchmark - Train 45000 / Val 5000 / Test 10000",
        "benchmark_full_final.png",
    )
    save_table_png(
        ["selected_epsilon", "baseline_test_acc", "protected_test_acc", "test_drop", "test_asr", "runtime"],
        [[0.03, 0.8528, 0.1037, 0.7491, 0.8963, "3121.89s"]],
        "Full Final Test Result",
        "benchmark_full_final_test.png",
    )
    save_table_png(
        ["surrogate", "victim", "epsilon", "train", "test", "baseline_test_acc", "protected_test_acc", "drop", "asr_proxy"],
        [
            ["ResNet-18", "MobileNetV2", 0.03, 45000, 10000, 0.6570, 0.1008, 0.5562, 0.8992],
            ["ResNet-18", "VGG-16", 0.03, 45000, 10000, 0.8291, 0.1000, 0.7291, 0.9000],
        ],
        "Transfer Victim Benchmark",
        "benchmark_transfer_victims.png",
    )
    save_line_chart()
    for out_dir in OUT_DIRS:
        print(f"Saved PNG files to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
