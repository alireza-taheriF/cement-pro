"""Renders the Persian evaluation report (docs/phase1/06) from ``metrics.json`` so numbers never drift from code."""

from __future__ import annotations

from pathlib import Path

_FA = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")


def fa(x, nd: int | None = None) -> str:
    """Format a number with Persian digits and decimal separator."""
    if isinstance(x, float):
        if x != x:
            return "—"
        x = f"{x:.{nd}f}" if nd is not None else f"{x:g}"
    return str(x).translate(_FA)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def render(m: dict) -> str:
    ing, det = m["ingest"], {d["fault"]: d for d in m["detection"]}
    imp = {d["gap_class"]: d for d in m["imputation"]}
    cmp_ = {(d["method"], d["gap_class"]): d for d in m["imputer_comparison"]}
    med, short, long_ = "medium (6-60 min)", "short (<=5 min)", "long (>60 min)"
    gain = 100 * (1 - cmp_[("hybrid", med)]["nrmse"] / cmp_[("linear", med)]["nrmse"])
    gain_ff = 100 * (1 - cmp_[("hybrid", med)]["nrmse"] / cmp_[("ffill", med)]["nrmse"])

    det_rows = [[{"spike": "Spike (پرش)", "flatline": "Flat-line (سنسور گیرکرده)", "missing": "Gap (مفقود)"}[k],
                 fa(det[k]["true_positive"]), fa(det[k]["false_positive"]), fa(det[k]["false_negative"]),
                 fa(det[k]["precision"], 3), fa(det[k]["recall"], 3)] for k in ("spike", "flatline", "missing")]

    imp_rows = [["کوتاه (≤ ۵ دقیقه)", fa(imp[short]["invalid_samples"]), fa(imp[short]["filled_pct"], 1) + "٪", fa(imp[short]["nrmse"], 3)],
                ["متوسط (۶–۶۰ دقیقه)", fa(imp[med]["invalid_samples"]), fa(imp[med]["filled_pct"], 1) + "٪", fa(imp[med]["nrmse"], 3)],
                ["بلند (> ۶۰ دقیقه)", fa(imp[long_]["invalid_samples"]), fa(imp[long_]["filled_pct"], 1) + "٪", "پر نشد (عمدی)"]]

    cmp_rows = [[mth, fa(cmp_[(mth, short)]["nrmse"], 3), fa(cmp_[(mth, med)]["nrmse"], 3)]
                for mth in ("ffill", "linear", "hybrid")]

    q_rows = [[q["tag"], q["unit"], fa(q["completeness_pct"], 2), fa(q["gaps"]), fa(q["max_gap_min"], 0),
               fa(q["spikes"]), fa(q["flatline_samples"]), fa(q["duplicates"]), fa(q["out_of_range"])]
              for q in m["quality"]]

    la, lb, lr = m["lab_alignment"], m["lab_alignment_timezone_ignored"], m["lab_relation_r"]
    return f"""# ۰۶ · گزارش ارزیابی نمونهٔ آزمایشگاهی

> [!CAUTION]
> **این نتایج روی داده‌ی مصنوعی هستند.** خطاها را خودمان تزریق کرده‌ایم، پس عدد بالا نشان می‌دهد **خط‌لوله درست کار می‌کند**، نه اینکه روی کارخانهٔ واقعی همین دقت را دارد. عملکرد روی داده‌ی واقعی: **UNVERIFIED**.
> این فایل خودکار از `reports/metrics.json` ساخته می‌شود؛ برای بازتولید: `python -m cementpro evaluate`.

## ۱. آزمایش چگونه انجام شد

```mermaid
flowchart LR
    A[شبیه‌ساز فرایند<br/>۱۱ تگ + ۲ پارامتر lab] --> B[تزریق خطا<br/>با ثبت دقیق ground truth]
    B --> C[خط‌لولهٔ cementpro]
    C --> D[مقایسه با حقیقت]
    A -.->|داده‌ی تمیز| D
```

| مورد | مقدار |
|---|---|
| مدت داده | {fa(m["config"]["days"])} روز، seed = {fa(m["config"]["seed"])} |
| ردیف خام سیگنال | {fa(ing["signal_rows_raw"])} |
| خطاهای تزریق‌شده | gap (کوتاه/متوسط/بلند)، spike، flat-line، مقدار محال، ردیف تکراری، ردیف نامعتبر، تگ ناشناس، واحد متفاوت (°F و kg/s)، زمان آزمایشگاه به وقت تهران |
| ردیف رد‌شده در ورود | {fa(ing["signal_rows_rejected"])} (همه با دلیل) |
| ردیف تگ ناشناس (جداشده) | {fa(ing["unknown_tag_rows"])} |
| ردیف تکراری حذف‌شده | {fa(ing["duplicates_removed"])} |

> [!NOTE]
> پارامتر آستانهٔ spike روی seedهای ۱ تا ۳ تنظیم شد و این گزارش روی seed = {fa(m["config"]["seed"])} (دیده‌نشده) است.

## ۲. تشخیص خطا

{_table(["نوع خطا", "TP", "FP", "FN", "Precision", "Recall"], det_rows)}

- همهٔ gapها و نمونه‌های flat-line یافته شدند و هیچ هشدار کاذبی برای آن‌ها نبود.
- برای spike، تعداد هشدار کاذب {fa(det["spike"]["false_positive"])} است (چرخش‌های طبیعی سیگنال). این یک مصالحه‌ی آستانه است: آستانهٔ پایین‌تر هشدار کاذب بیشتری می‌دهد.

## ۳. جبران داده‌ی مفقود

خطای نرمال‌شده (NRMSE = RMSE ÷ انحراف معیار حقیقت؛ کمتر بهتر) روی نمونه‌هایی که نامعتبر بودند:

{_table(["طول gap", "نمونهٔ نامعتبر", "درصد پرشده", "NRMSE"], imp_rows)}

![مقایسهٔ روش‌ها](assets/fig_imputers.png)

{_table(["روش", "NRMSE کوتاه", "NRMSE متوسط"], cmp_rows)}

- روش ترکیبی (`hybrid`) روی gap متوسط حدود **{fa(gain, 0)}٪** بهتر از درون‌یابی خطی و حدود **{fa(gain_ff, 0)}٪** بهتر از نگه‌داشتن آخرین مقدار است.
- برتری روش ترکیبی **متوسط** است و بخشی از آن به این برمی‌گردد که تگ‌های شبیه‌ساز عمداً همبسته ساخته شده‌اند. روی داده‌ی واقعی ممکن است کمتر باشد.
- روی gap بلند عمداً هیچ داده‌ای ساخته نشد.

![نمونهٔ ترمیم](assets/fig_cleaning_example.png)

## ۴. هم‌ترازی آزمایشگاه

| سناریو | نمونه در خروجی | درست هم‌تراز | همبستگی (free lime با دمای ناحیهٔ پخت) |
|---|---|---|---|
| منطقهٔ زمانی اعمال شد | {fa(la["lab_samples_in_output"])} | {fa(la["alignment_rate_pct"], 1)}٪ | {fa(lr["timezone_handled"], 2)} |
| منطقهٔ زمانی نادیده گرفته شد | {fa(lb["lab_samples_in_output"])} | {fa(lb["alignment_rate_pct"], 1)}٪ | {fa(lr["timezone_ignored"], 2)} |

![هم‌ترازی](assets/fig_lab_alignment.png)

> [!NOTE]
> رابطهٔ منفی free lime و دما را خودمان در شبیه‌ساز گذاشته‌ایم. هدف این آزمایش فقط نشان دادن این است که **اشتباه در منطقهٔ زمانی** این رابطه را کاملاً از بین می‌برد و خط‌لوله آن را درست می‌کند.

## ۵. استانداردسازی واحد

سه تگ با نام و واحد متفاوت وارد شدند (`BZT_F` به °F، `KLN_FUEL_KGS` به kg/s، `KLN_FEED_TPH`). بعد از استانداردسازی، میانهٔ خطای مطلق نسبت به حقیقت برابر {fa(m["units"][0]["median_abs_error"], 4)} است (خطای عددی ممیز شناور).

## ۶. کامل‌بودن داده

![کامل‌بودن](assets/fig_completeness.png)

- سهم نمونه‌های پرنشده (gap بلند): **{fa(m["unrecoverable_pct"], 2)}٪** از کل سلول‌های فرایندی.
- انحراف ساختار همبستگی بین تگ‌ها نسبت به حقیقت (میانگین قدر مطلق تفاوت): **{fa(m["correlation_drift"], 3)}**؛ یعنی ترمیم، ارتباط بین متغیرها را به‌طور محسوس خراب نکرده است.

## ۷. گزارش کیفیت به‌ازای هر تگ

{_table(["تگ", "واحد", "کامل‌بودن ٪", "تعداد gap", "بزرگ‌ترین gap (دقیقه)", "spike", "نمونهٔ flat-line", "تکراری", "خارج از بازه"], q_rows)}

## ۸. محدودیت‌ها (صادقانه)

1. **داده‌ی مصنوعی:** ساختار نویز، خطا و همبستگی را ما ساخته‌ایم؛ داده‌ی واقعی نویزی‌تر و ناهمگن‌تر است.
2. **فقط ۱۱ تگ و ۲ پارامتر آزمایشگاهی:** کارخانه‌ی واقعی صدها تگ دارد.
3. **تأخیر آزمایشگاه صفر فرض شد.** اگر تأخیر واقعی نامعلوم باشد، `lab_lag_minutes` باید کالیبره شود.
4. **آستانه‌ها (spike، flat-line) برای فرایند واقعی باید کالیبره شوند.**
5. **عملکرد زمانی روی سخت‌افزار کارخانه** اندازه‌گیری نشده است.
6. روش جبران متوسط به **وجود تگ‌های همبسته‌ی سالم** در همان لحظه وابسته است؛ در قطعی همزمان چند تگ به درون‌یابی خطی برمی‌گردد.
"""


def write(metrics: dict, path: str | Path) -> None:
    Path(path).write_text(render(metrics), encoding="utf-8")
