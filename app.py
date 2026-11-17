import io
import os
import random
import re
import urllib.parse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st
import weasyprint
from google import genai
from google.genai import types
from pypdf import PdfReader

GEMINI_MODEL = "gemini-3.6-flash"

MOTIVATIONAL_QUOTES = [
    "مع تمنياتي لكم بالنجاح والتوفيق الباهر.",
    "دعواتي لكم بالنجاح والتألق الدائم.",
    "مع أطيب التمنيات بالتوفيق والنجاح الدائم.",
]

FORMATTING_INSTRUCTIONS = """
تعليمات الصياغة:
1. اكتب عنوان كل سؤال في سطر منفصل بصيغة: س^1 ، س^2 الخ.
2. الفروع الكبيرة اكتبها بحروف إنجليزية كبيرة متبوعة بنقطة مثل: A. أو B.
3. الفروع الصغيرة التابعة لها اكتبها بحروف إنجليزية صغيرة متبوعة بنقطة مثل: a. أو b.
4. اكتب المعادلات الرياضية بصيغة LaTeX محصورة بين علامتي دولار $...$.
"""


# --- Math & LaTeX Rendering ---

def render_latex_to_svg_url(latex_code: str) -> str:
    """Convert a LaTeX string into a data-URI SVG via CodeCogs, falling back to Matplotlib."""
    cleaned = re.sub(r"<[^>]+>", "", latex_code).strip()
    cleaned = cleaned.replace(r"^{\wedge}", "^").replace(r"^{\hat{}}", "^")

    # Try CodeCogs first for crisp typesetting
    try:
        encoded = urllib.parse.quote(cleaned)
        url = f"https://latex.codecogs.com/svg.latex?\\dpi{{150}}\\color{{black}}{encoded}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200 and "<svg" in response.text:
            return f"data:image/svg+xml;utf8,{urllib.parse.quote(response.text)}"
    except requests.RequestException:
        pass

    # Fallback to local Matplotlib rendering if offline or CodeCogs times out
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)
    try:
        text = fig.text(0, 0, f"${cleaned}$", fontsize=15, color="black", math_fontfamily="dejavuserif")
        fig.canvas.draw()
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        fig.set_size_inches(max(bbox.width / fig.dpi + 0.05, 0.1), max(bbox.height / fig.dpi + 0.05, 0.1))

        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
        svg_data = buf.getvalue().decode("utf-8")
        return f"data:image/svg+xml;utf8,{urllib.parse.quote(svg_data)}"
    except Exception:
        return f"<span>${cleaned}$</span>"
    finally:
        plt.close(fig)


def process_math_in_html(html_content: str) -> str:
    """Replace inline ($...$) and block ($$...$$) LaTeX expressions with inline SVG images."""
    def replace_match(match: re.Match) -> str:
        svg_uri = render_latex_to_svg_url(match.group(1))
        if svg_uri.startswith("data:image"):
            return f'<img src="{svg_uri}" style="vertical-align: middle; margin: 0 4px;" />'
        return svg_uri

    html_content = re.sub(r"\$\$([^\$]+?)\$\$", replace_match, html_content)
    html_content = re.sub(r"\$([^\$]+?)\$", replace_match, html_content)
    return html_content


# --- Plain Text to HTML Parsing ---

def parse_text_to_html(text: str) -> str:
    """Convert structured exam plain text into semantic HTML blocks."""
    blocks = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "س^" in line and len(line) < 20:
            num_match = re.search(r"\d+", line)
            num = num_match.group() if num_match else ""
            blocks.append(f"<h4>س<sup>{num}</sup>:</h4>")
        elif major_match := re.match(r"^([A-Z])[\.\-]\s*(.*)", line):
            blocks.append(f"<div class='major-branch'><strong>{major_match.group(1)}.</strong> {major_match.group(2)}</div>")
        elif sub_match := re.match(r"^([a-z])[\.\-]\s*(.*)", line):
            blocks.append(f"<div class='sub-branch'><strong>{sub_match.group(1)}.</strong> {sub_match.group(2)}</div>")
        elif list_match := re.match(r"^(\d+)[\.\-]\s*(.*)", line):
            blocks.append(f"<div class='numbered-item'>{list_match.group(1)}. {list_match.group(2)}</div>")
        else:
            blocks.append(f"<p>{line}</p>")

    return "\n".join(blocks)


# --- Gemini AI Extraction & Book Analysis ---

def extract_handwritten_questions(images: list[tuple[bytes, str]], api_key: str) -> str:
    """Extract and sequentially number exam questions from one or more handwritten page images."""
    client = genai.Client(api_key=api_key)
    prompt = f"""
    أنت خبير في قراءة وتحويل صور الأسئلة الامتحانية.
    استخرج الأسئلة كنص عادي (Plain Text) منظم.
    إذا كانت هناك أكثر من صورة، اعتبرها صفحات متتالية لنفس ورقة الأسئلة، ورقّم الأسئلة بشكل متسلسل عبر جميع الصور دون تكرار الترقيم.
    {FORMATTING_INSTRUCTIONS}
    """
    parts = [types.Part.from_bytes(data=img_bytes, mime_type=mime) for img_bytes, mime in images]
    response = client.models.generate_content(model=GEMINI_MODEL, contents=[*parts, prompt])
    return response.text


def extract_text_from_pdf_book(pdf_bytes: bytes, max_chars: int = 45000) -> str:
    """Extract selectable text from a PDF file. Returns an empty string for scanned image-only PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if len(full_text) > max_chars:
            return full_text[:max_chars] + "\n...[تم اختصار نص الكتاب لطوله]"
        return full_text
    except Exception:
        return ""


def select_questions_from_book(
    images: list[tuple[bytes, str]],
    pdf_texts: list[str],
    num_questions: int,
    criteria: str,
    api_key: str,
) -> str:
    """Analyze textbook pages (images and/or PDF text) and generate a curated set of exam questions."""
    client = genai.Client(api_key=api_key)

    criteria_line = f"\nمعيار اختيار إضافي طلبه المدرس: {criteria.strip()}" if criteria and criteria.strip() else ""
    combined_book_text = "\n\n---\n\n".join(pdf_texts)
    text_section = f"\n\nنص مستخرج من الكتاب:\n{combined_book_text}" if combined_book_text else ""

    prompt = f"""
    أنت خبير في إعداد الأسئلة الامتحانية العراقية.
    أمامك محتوى كتاب دراسي (قد يكون على شكل صور صفحات و/أو نص مستخرج من ملف PDF).
    اختر بالضبط {num_questions} سؤال/أسئلة مناسبة لامتحان من هذا المحتوى، بحيث تكون متنوعة وتغطي المحتوى بشكل جيد.{criteria_line}
    اكتب الأسئلة المختارة فقط بصيغة نص عادي (Plain Text)، رقّمها من 1 وحتى {num_questions}، ولا تكتب أي مقدمة أو تعليق خارج الصيغة المطلوبة.
    {FORMATTING_INSTRUCTIONS}
    {text_section}
    """
    parts = [types.Part.from_bytes(data=img_bytes, mime_type=mime) for img_bytes, mime in images]
    response = client.models.generate_content(model=GEMINI_MODEL, contents=[*parts, prompt])
    return response.text


# --- HTML & PDF Generation ---

def build_exam_html(data: dict, text_content: str, font_scale: float = 1.0, page_css_size: str = "A4 portrait") -> str:
    """Render the complete Iraqi Ministry-style exam sheet HTML with dynamic font/spacing scaling."""
    quote = random.choice(MOTIVATIONAL_QUOTES)
    questions_html = process_math_in_html(parse_text_to_html(text_content))

    base_font = round(13 * font_scale, 2)
    h4_font = round(13.5 * font_scale, 2)
    header_font = round(11.5 * font_scale, 2)
    footer_font = round(11 * font_scale, 2)
    sup_font = round(10 * font_scale, 2)

    margin_v = max(4, round(6 * font_scale))
    margin_h = max(7, round(10 * font_scale))
    h4_spacing = max(2, round(6 * font_scale))
    major_top = max(2, round(6 * font_scale))
    major_bottom = max(2, round(4 * font_scale))
    sub_bottom = max(2, round(4 * font_scale))
    p_spacing = max(2, round(6 * font_scale))
    footer_top = max(4, round(10 * font_scale))
    footer_pad_top = max(2, round(4 * font_scale))

    return f"""
    <!DOCTYPE html>
    <html lang="ar" dir="rtl">
    <head>
        <meta charset="UTF-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap');

            @page {{ size: {page_css_size}; margin: {margin_v}mm {margin_h}mm; }}
            * {{ box-sizing: border-box; }}
            body {{
                font-family: 'Amiri', serif !important;
                direction: rtl;
                font-size: {base_font:g}pt;
                line-height: 1.45;
                margin: 0;
                padding: 0;
            }}
            .header-table {{ width: 100%; border-collapse: collapse; }}
            .header-table td {{
                width: 33.33%;
                vertical-align: top;
                font-weight: bold;
                font-size: {header_font:g}pt;
                line-height: 1.25;
            }}
            .header-right {{ text-align: right; }}
            .header-center {{ text-align: center; }}
            .header-left {{ text-align: left; }}
            .divider {{
                border-top: 1.5px solid #000;
                border-bottom: 0.5px solid #000;
                height: 2px;
                margin: 4px 0 6px 0;
            }}
            .note-box {{ font-weight: bold; font-size: {header_font:g}pt; margin-bottom: 4px; }}

            .questions-body h4 {{
                font-size: {h4_font:g}pt;
                margin: {h4_spacing}px 0 0 0;
                padding-top: {h4_spacing}px;
                font-weight: bold;
                page-break-after: avoid;
                border-top: 1.5px solid #444;
            }}
            .questions-body h4:first-of-type {{ border-top: none; padding-top: 0; margin-top: 0; }}
            .questions-body h4 sup {{ font-size: {sup_font:g}pt; }}

            .major-branch {{ margin-top: {major_top}px; margin-bottom: {major_bottom}px; font-weight: bold; }}
            .sub-branch {{ margin-right: 25px; margin-bottom: {sub_bottom}px; }}
            .numbered-item {{ margin-right: 15px; margin-bottom: 6px; }}
            .questions-body p {{ margin: {p_spacing}px 0; }}

            .footer-section {{
                margin-top: {footer_top}px;
                width: 100%;
                border-top: 1px dashed #000;
                padding-top: {footer_pad_top}px;
                page-break-inside: avoid;
            }}
            .footer-table {{ width: 100%; }}
            .footer-table td {{ font-size: {footer_font:g}pt; font-weight: bold; }}
        </style>
    </head>
    <body>
        <table class="header-table">
            <tr>
                <td class="header-right">جمهورية العراق<br>وزارة التربية<br>{data['school_name']}</td>
                <td class="header-center">الامتحانات {data['exam_type']}<br>العام الدراسي: {data['academic_year']}<br>الدور: {data['session']}</td>
                <td class="header-left">المادة: {data['subject']}<br>الصف: {data['grade']}<br>الوقت: {data['time_limit']}</td>
            </tr>
        </table>
        <div class="divider"></div>
        <div class="note-box">ملاحظة: {data.get('general_note', 'أجب عن جميع الأسئلة.')}</div>
        <div class="questions-body">{questions_html}</div>
        <div class="footer-section">
            <table class="footer-table">
                <tr>
                    <td style="text-align: right;">مدرس المادة: {data['teacher_name']}</td>
                    <td style="text-align: left; font-style: italic;">{quote}</td>
                </tr>
            </table>
        </div>
    </body>
    </html>
    """


def generate_exam_pdf(
    data: dict,
    text_content: str,
    page_size_mode: str = "full",
    target_pages: int | None = None,
) -> tuple[bytes, int | None, bool]:
    """
    Generate the exam PDF in memory and optionally shrink typography/margins to fit `target_pages`.
    Returns (pdf_bytes, final_page_count, fit_succeeded).
    """
    if page_size_mode == "auto":
        question_count = len(re.findall(r"س\^", text_content))
        resolved_mode = "half" if (question_count <= 1 and len(text_content.strip()) < 500) else "full"
    else:
        resolved_mode = page_size_mode

    page_css_size = "210mm 148.5mm" if resolved_mode == "half" else "A4 portrait"

    scale = 1.0
    min_scale = 0.72
    pdf_bytes = b""
    final_page_count = None

    for _ in range(8 if target_pages else 1):
        html = build_exam_html(data, text_content, font_scale=scale, page_css_size=page_css_size)
        pdf_bytes = weasyprint.HTML(string=html).write_pdf()

        if not target_pages:
            break

        try:
            final_page_count = len(PdfReader(io.BytesIO(pdf_bytes)).pages)
        except Exception:
            final_page_count = None
            break

        if final_page_count <= target_pages or scale <= min_scale:
            break
        scale = max(min_scale, round(scale - 0.06, 2))

    fit_ok = (final_page_count is None) or (target_pages is None) or (final_page_count <= target_pages)
    return pdf_bytes, final_page_count, fit_ok


# --- Streamlit UI ---

def append_extracted_text(new_text: str) -> None:
    existing = st.session_state["extracted_text"]
    st.session_state["extracted_text"] = f"{existing}\n\n{new_text}".strip() if existing else new_text


st.set_page_config(page_title="صانع الأسئلة الامتحانية العراقية", page_icon="📄", layout="wide")
st.title("📄 برنامج تحويل الأسئلة اليدوية إلى PDF رسمي")

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = ""

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 1. بيانات الهيدر والفوتر")
    school_name = st.text_input("اسم المدرسة", "اعدادية بردى للبنات")

    c_sub1, c_sub2 = st.columns(2)
    with c_sub1:
        subject = st.text_input("المادة", "الرياضيات")
        grade = st.text_input("الصف", "الخامس علمي")
        academic_year = st.text_input("العام الدراسي", "2025 - 2026")
    with c_sub2:
        exam_type = st.selectbox("نوع الامتحان", ["نصف السنة", "الشهرية", "النهائية"])
        session = st.selectbox("الدور", ["الدور الأول", "الدور الثاني", "الدور الثالث"])
        time_limit = st.text_input("الوقت", "ساعتان")

    teacher_name = st.text_input("اسم المدرس", "الاء صالح")
    general_note = st.text_area("الملاحظة العامة", "الإجابة عن خمسة أسئلة فقط، ولكل سؤال 20 درجة.")

with col2:
    st.subheader("📸 2. مصدر الأسئلة ومفتاح API")
    api_key_input = st.text_input("مفتاح Gemini API Key", type="password")
    api_key = api_key_input.strip() or os.getenv("GEMINI_API_KEY", "")

    tab_handwritten, tab_book = st.tabs(["📷 صورة مكتوبة بخط اليد", "📚 كتاب دراسي (PDF أو صور)"])

    with tab_handwritten:
        uploaded_files = st.file_uploader(
            "اختر صورة/صور الأسئلة (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="handwritten_uploader",
        )

        if st.button("🔍 قراءة واستخراج الأسئلة", use_container_width=True, key="extract_handwritten_btn"):
            if not uploaded_files or not api_key:
                st.error("⚠️ يرجى رفع صورة واحدة على الأقل وإدخال مفتاح الـ API أولاً!")
            else:
                with st.spinner("جاري قراءة خط اليد واستخراج النصوص..."):
                    try:
                        images = [(f.getvalue(), f.type) for f in uploaded_files]
                        append_extracted_text(extract_handwritten_questions(images, api_key))
                        st.success("✅ تم الاستخراج بنجاح!")
                    except Exception as exc:
                        st.error(f"خطأ في الاستخراج: {exc}")

    with tab_book:
        book_files = st.file_uploader(
            "ارفع صفحات الكتاب (PDF أو صور JPG/PNG)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="book_uploader",
        )
        st.caption("يمكن رفع أكثر من صفحة/صورة، أو أكثر من ملف PDF في نفس الوقت.")
        num_questions = st.number_input("كم عدد الأسئلة المطلوب اختيارها؟", min_value=1, max_value=30, value=5, step=1)
        criteria = st.text_input(
            "تحديد اختياري لنوع الأسئلة (مثال: فصل الاشتقاق، أسئلة صعبة فقط) - اتركه فارغاً للاختيار العام",
            key="book_criteria",
        )

        if st.button("🔍 تحليل الكتاب واختيار الأسئلة", use_container_width=True, key="extract_book_btn"):
            if not book_files or not api_key:
                st.error("⚠️ يرجى رفع ملف الكتاب (PDF أو صور) وإدخال مفتاح الـ API أولاً!")
            else:
                with st.spinner("جاري تحليل الكتاب واختيار الأسئلة المناسبة..."):
                    try:
                        images, pdf_texts = [], []
                        for f in book_files:
                            if f.type == "application/pdf" or f.name.lower().endswith(".pdf"):
                                if extracted := extract_text_from_pdf_book(f.getvalue()):
                                    pdf_texts.append(extracted)
                            else:
                                images.append((f.getvalue(), f.type))

                        if not images and not pdf_texts:
                            st.error("⚠️ لم يتم العثور على نص قابل للقراءة داخل ملفات PDF. تأكد أن الملف نصي أو ارفع صوراً بدلاً منه.")
                        else:
                            selected = select_questions_from_book(images, pdf_texts, int(num_questions), criteria, api_key)
                            append_extracted_text(selected)
                            st.success("✅ تم اختيار الأسئلة بنجاح!")
                    except Exception as exc:
                        st.error(f"خطأ في تحليل الكتاب: {exc}")

st.divider()

if st.session_state["extracted_text"]:
    st.subheader("✏️ 3. محرر النصوص")
    st.info(
        "💡 **طريقة الكتابة:**\n"
        "- اكتب `س^1` في سطر منفصل للعنوان.\n"
        "- اكتب الفرع الرئيسي بحرف كبير مثل `A. باستخدام قواعد المشتقة...`\n"
        "- اكتب الفروع التابعة بحرف صغير مثل `a. $f(x) = 3x^2 + 5x + 8$` لتأخذ إزاحة تلقائية للداخل!"
    )

    edited_text = st.text_area(
        "نص الأسئلة",
        value=st.session_state["extracted_text"],
        height=400,
        label_visibility="collapsed",
    )

    if st.button("🗑️ مسح النص والبدء من جديد"):
        st.session_state["extracted_text"] = ""
        st.rerun()

    with st.expander("⚙️ إعدادات حجم وعدد الصفحات (اختياري)"):
        col_size, col_pages = st.columns(2)
        with col_size:
            page_size_choice = st.selectbox(
                "حجم الصفحة",
                ["A4 كاملة", "تلقائي (نصف صفحة عند قلة الأسئلة)", "نصف صفحة دائماً"],
                index=0,
            )
        with col_pages:
            target_pages_choice = st.selectbox(
                "عدد صفحات الامتحان",
                ["بدون تحديد", "صفحة واحدة", "صفحتان"],
                index=0,
            )

    page_size_map = {
        "A4 كاملة": "full",
        "تلقائي (نصف صفحة عند قلة الأسئلة)": "auto",
        "نصف صفحة دائماً": "half",
    }
    target_pages_map = {"بدون تحديد": None, "صفحة واحدة": 1, "صفحتان": 2}

    if st.button("🚀 إنشاء ملف الـ PDF النهائي", type="primary", use_container_width=True):
        with st.spinner("جاري إنشاء ملف الـ PDF..."):
            try:
                metadata = {
                    "school_name": school_name,
                    "subject": subject,
                    "grade": grade,
                    "academic_year": academic_year,
                    "exam_type": exam_type,
                    "session": session,
                    "time_limit": time_limit,
                    "teacher_name": teacher_name,
                    "general_note": general_note,
                }

                chosen_target = target_pages_map[target_pages_choice]
                pdf_bytes, final_pages, fit_ok = generate_exam_pdf(
                    metadata,
                    edited_text,
                    page_size_mode=page_size_map[page_size_choice],
                    target_pages=chosen_target,
                )

                st.success("🎉 تم تجهيز الملف بنجاح!")
                if chosen_target and not fit_ok:
                    st.warning(
                        f"⚠️ تعذّر ضغط الأسئلة بالكامل ضمن العدد المطلوب من الصفحات حتى مع تصغير الخط لأقل حجم مقروء "
                        f"(عدد الصفحات الحالي: {final_pages}). يُفضّل تقليل عدد الأسئلة."
                    )

                st.download_button(
                    label="📥 تحميل ملف PDF للطباعة",
                    data=pdf_bytes,
                    file_name=f"اسئلة_{subject}_{grade}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"حدث خطأ أثناء الإنشاء: {exc}")