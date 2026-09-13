"""README results animation (Manim Community). See media/manim/README.md to regenerate.

Run from the repository root:
    uvx --python 3.12 manim -r 960,540 --fps 15 --format=gif \
        --media_dir media/manim/.media media/manim/results_scene.py Results
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    FadeIn,
    FadeOut,
    GrowFromEdge,
    Line,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
    config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from results_data import MODEL_NAMES, MODEL_ORDER, Panel, load_results  # noqa: E402

FONT = "Microsoft JhengHei"
BACKGROUND = "#0b1220"
BAR_COLORS = ("#8b93a7", "#3b82f6", "#22c55e")
TEXT = "#f3f4f6"
MUTED = "#9ca3af"
ACCENT = "#facc15"

config.background_color = BACKGROUND


def bar_panel(
    panel: Panel, *, width: float, height: float, center
) -> tuple[VGroup, list[Rectangle]]:
    """Baseline, title, model names, and three bars positioned around ``center``."""

    baseline = Line(LEFT * width / 2, RIGHT * width / 2, color=MUTED, stroke_width=2)
    baseline.move_to(center)
    title = Text(f"{panel.title}（{panel.questions:,} 題）", font=FONT, font_size=22, color=TEXT)
    title.next_to(baseline, UP, buff=height + 0.55)
    bars: list[Rectangle] = []
    names = VGroup()
    slot = width / 3
    for index, model in enumerate(MODEL_ORDER):
        accuracy = panel.accuracy_pct[index]
        bar = Rectangle(
            width=slot * 0.62,
            height=max(accuracy / 100 * height, 0.02),
            fill_color=BAR_COLORS[index],
            fill_opacity=0.92,
            stroke_width=0,
        )
        x = baseline.get_left()[0] + slot * (index + 0.5)
        bar.move_to([x, baseline.get_y() + bar.height / 2, 0])
        bars.append(bar)
        name = Text(MODEL_NAMES[model], font=FONT, font_size=17, color=TEXT)
        name.next_to([x, baseline.get_y(), 0], DOWN, buff=0.18)
        names.add(name)
    return VGroup(baseline, title, names), bars


def value_labels(panel: Panel, bars: list[Rectangle]) -> VGroup:
    labels = VGroup()
    for index, bar in enumerate(bars):
        label = Text(
            f"{panel.accuracy_pct[index]:.2f}%",
            font=FONT,
            font_size=20,
            color=ACCENT if index == 2 else TEXT,
        )
        label.next_to(bar, UP, buff=0.1)
        labels.add(label)
    return labels


def gain_arrow(panel: Panel, labels: VGroup) -> VGroup:
    """Arrow from the base value label to the adapter value label, text above it."""

    base_label, adapter_label = labels[1], labels[2]
    start = base_label.get_top() + UP * 0.12 + RIGHT * 0.15
    end = adapter_label.get_left() + LEFT * 0.08 + UP * 0.05
    arrow = Arrow(
        start, end, color=ACCENT, buff=0.0, stroke_width=4, max_tip_length_to_length_ratio=0.25
    )
    label = Text(f"+{panel.adapter_minus_base_pp:.2f} pp", font=FONT, font_size=18, color=ACCENT)
    label.next_to(arrow.get_center(), UP, buff=0.12)
    return VGroup(arrow, label)


class Results(Scene):
    def construct(self) -> None:
        data = load_results()

        headline = Text("臺灣醫療選擇題 QLoRA 領域微調", font=FONT, font_size=40, color=TEXT)
        sub = Text(
            "同一份 prompt、parser 與生成設定，三個模型的正式評估",
            font=FONT,
            font_size=22,
            color=MUTED,
        )
        VGroup(headline, sub).arrange(DOWN, buff=0.35)
        self.play(Write(headline), run_time=1.2)
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=0.6)
        self.wait(0.6)
        self.play(FadeOut(sub), headline.animate.scale(0.6).to_edge(UP, buff=0.3), run_time=0.7)

        panel_w, panel_h = 5.6, 2.7
        left, left_bars = bar_panel(
            data.medqa, width=panel_w, height=panel_h, center=LEFT * 3.35 + DOWN * 1.05
        )
        right, right_bars = bar_panel(
            data.tmmlu, width=panel_w, height=panel_h, center=RIGHT * 3.35 + DOWN * 1.05
        )
        self.play(FadeIn(left), FadeIn(right), run_time=0.6)
        self.play(*[GrowFromEdge(bar, DOWN) for bar in left_bars + right_bars], run_time=1.4)
        left_labels = value_labels(data.medqa, left_bars)
        right_labels = value_labels(data.tmmlu, right_bars)
        self.play(FadeIn(left_labels), FadeIn(right_labels), run_time=0.5)
        self.play(
            FadeIn(gain_arrow(data.medqa, left_labels)),
            FadeIn(gain_arrow(data.tmmlu, right_labels)),
            run_time=0.7,
        )
        self.wait(1.2)

        footer_lines = [
            f"Adapter 答案解析率 {data.medqa.parse_rate_pct[2]:.0f}% / "
            f"{data.tmmlu.parse_rate_pct[2]:.0f}%；TAIDE Base 僅 "
            f"{data.medqa.parse_rate_pct[1]:.1f}% / {data.tmmlu.parse_rate_pct[1]:.1f}%",
            "非醫學控制 5 科通過預先定義的 −2 pp non-inferiority 判準"
            f"（95% CI 下界 +{data.control_ci_lower_pp:.2f} pp）",
            f"TMMLU+ v1.1 重新計分（{data.tmmlu_v1_1_questions:,} 題）："
            f"Adapter {data.tmmlu_v1_1_adapter_accuracy_pct:.2f}%，結論不變",
        ]
        footer = VGroup(
            *[Text(line, font=FONT, font_size=17, color=MUTED) for line in footer_lines]
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        footer.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(footer, shift=UP * 0.15), run_time=0.8)
        self.wait(2.2)
