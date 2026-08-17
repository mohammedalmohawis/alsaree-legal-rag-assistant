"""Bilingual interface copy.

Every user-visible string in the product is defined here, once per language, and
reached through :func:`t`. Nothing in ``sanad.ui`` hard-codes English or Arabic
text, which is what keeps the two catalogues verifiably in step — a test asserts
they have identical key sets and identical placeholders.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sanad.config import SUPPORTED_LANGUAGES

EN: dict[str, str] = {
    # --- shell ---------------------------------------------------------------
    "app_title": "Sanad — Legal Document Intelligence",
    "tagline": "Retrieval-grounded analysis of your own legal documents.",
    "language_label": "Language",
    "language_en": "English",
    "language_ar": "العربية",
    # --- workspace / uploads --------------------------------------------------
    "upload_heading": "Documents",
    "upload_label": "Add PDF or DOCX files",
    "upload_help": "PDF and DOCX are supported, up to {size} MB per file.",
    "process": "Process documents",
    "reprocess": "Re-index documents",
    "processing_parse": "Reading and structuring documents…",
    "processing_embed": "Building the search index…",
    "processed": "Indexed {passages} across {documents}.",
    "count_passages_one": "1 passage",
    "count_passages_many": "{count} passages",
    "count_documents_one": "1 document",
    "count_documents_many": "{count} documents",
    "count_pages_one": "1 page",
    "count_pages_many": "{count} pages",
    "no_documents": "No documents loaded yet.",
    "document_list": "Loaded documents",
    "doc_meta_pages": "{pages} · {passages}",
    "doc_meta_nopages": "{passages} · no page numbers in DOCX",
    "remove_document": "Remove",
    "clear_all": "Clear workspace",
    "scope_label": "Search within",
    "scope_all": "All documents",
    "index_stale": "Documents have changed. Re-index to search the latest version.",
    # --- tabs ----------------------------------------------------------------
    "tab_chat": "Ask",
    "tab_summary": "Summary",
    "tab_compare": "Compare",
    "tab_facts": "Key facts",
    # --- chat ----------------------------------------------------------------
    "chat_heading": "Ask your documents",
    "chat_placeholder": "Ask about a clause, obligation, date or amount…",
    "chat_empty_title": "Nothing asked yet",
    "chat_empty_body": "Every answer is drawn only from the documents you upload, with a citation for each statement.",
    "chat_locked": "Upload and process a document to start asking questions.",
    "answering": "Searching the documents and drafting a cited answer…",
    "sources_heading": "Sources",
    "show_passage": "View the cited passage",
    "ungrounded_notice": "No passage in the uploaded documents supports an answer to this question.",
    "clear_chat": "Clear conversation",
    # --- summary --------------------------------------------------------------
    "summary_heading": "Structured summary",
    "summary_intro": "Produce a structured brief of a single document. The summary states only what the document says.",
    "summary_document": "Document",
    "summary_focus": "Focus (optional)",
    "summary_focus_help": "For example: concentrate on termination and penalties.",
    "summary_run": "Generate summary",
    "summarizing": "Reading the document and drafting the brief…",
    "summary_truncated": "The document exceeded the analysis budget; the brief covers the opening portion only.",
    # --- comparison -----------------------------------------------------------
    "compare_heading": "Compare two documents",
    "compare_intro": "Identify differing clauses, obligations, dates, amounts and parties between two versions.",
    "compare_first": "First document",
    "compare_second": "Second document",
    "compare_run": "Compare",
    "comparing": "Comparing the two documents clause by clause…",
    "compare_empty_title": "Not enough documents",
    "compare_need_two": "Load at least two documents to run a comparison.",
    "compare_same": "Select two different documents.",
    "compare_truncated": "One or both documents exceeded the analysis budget; the comparison covers the opening portions.",
    # --- key facts ------------------------------------------------------------
    "facts_heading": "Key information",
    "facts_intro": "Extract the contractual facts a reviewer checks first. Fields the document does not state are left out.",
    "facts_run": "Extract key facts",
    "extracting": "Extracting the key contractual facts…",
    "facts_empty": "No key facts could be extracted from this document.",
    "field_contract_type": "Contract type",
    "field_parties": "Parties",
    "field_effective_date": "Effective date",
    "field_term": "Term",
    "field_key_dates": "Key dates",
    "field_obligations": "Obligations",
    "field_payment_terms": "Payment terms",
    "field_termination": "Termination",
    "field_renewal": "Renewal",
    "field_penalties": "Penalties",
    "field_governing_law": "Governing law",
    "field_deadlines": "Deadlines",
    "field_notes": "Other notes",
    # --- errors ---------------------------------------------------------------
    "error_generic": "Something went wrong: {detail}",
    "error_api_key_missing": "No Gemini API key is configured. Set GOOGLE_API_KEY in your environment or in Streamlit Secrets.",
    "error_document_read": "Could not read {filename}: {detail}",
    "error_unsupported_format": "{filename} is not a supported format. Upload a PDF or DOCX file.",
    "error_empty_document": "{filename} contains no extractable text. Scanned PDFs need OCR before upload.",
    "error_model": "The model request failed: {detail}",
    "error_model_unreachable": "Could not reach the Gemini service. Check your internet connection, then try again.",
    "error_model_auth": "The Gemini API key was rejected. Check GOOGLE_API_KEY in your .env file or in Streamlit Secrets.",
    "error_model_rate_limit": "Gemini is rate limiting requests right now. Wait a moment, then try again.",
    "technical_detail": "Technical detail",
    "error_retrieval": "No documents are indexed yet. Process a document first.",
    "error_too_large": "{filename} exceeds the {size} MB limit and was skipped.",
    # --- footer ---------------------------------------------------------------
    "disclaimer_title": "Important notice",
    "disclaimer": "Sanad produces document analysis, not legal advice. Verify every citation against the source document before relying on it.",
}

AR: dict[str, str] = {
    # --- shell ---------------------------------------------------------------
    "app_title": "سند — منصة تحليل المستندات القانونية",
    "tagline": "تحليل مستنداتك القانونية استناداً إلى محتواها وحده.",
    "language_label": "اللغة",
    "language_en": "English",
    "language_ar": "العربية",
    # --- workspace / uploads --------------------------------------------------
    "upload_heading": "المستندات",
    "upload_label": "أضف ملفات PDF أو DOCX",
    "upload_help": "الصيغ المدعومة PDF وDOCX، بحد أقصى {size} ميجابايت للملف.",
    "process": "معالجة المستندات",
    "reprocess": "إعادة فهرسة المستندات",
    "processing_parse": "جارٍ قراءة المستندات وتحليل بنيتها…",
    "processing_embed": "جارٍ بناء فهرس البحث…",
    "processed": "تمت فهرسة {passages} من {documents}.",
    "count_passages_one": "مقطع واحد",
    "count_passages_many": "{count} مقاطع",
    "count_documents_one": "مستند واحد",
    "count_documents_many": "{count} مستندات",
    "count_pages_one": "صفحة واحدة",
    "count_pages_many": "{count} صفحات",
    "no_documents": "لم يتم تحميل أي مستند بعد.",
    "document_list": "المستندات المحمّلة",
    "doc_meta_pages": "{pages} · {passages}",
    "doc_meta_nopages": "{passages} · لا تتضمن ملفات DOCX أرقام صفحات",
    "remove_document": "إزالة",
    "clear_all": "إفراغ مساحة العمل",
    "scope_label": "نطاق البحث",
    "scope_all": "جميع المستندات",
    "index_stale": "تغيّرت المستندات. أعد الفهرسة للبحث في أحدث نسخة.",
    # --- tabs ----------------------------------------------------------------
    "tab_chat": "اسأل",
    "tab_summary": "الملخص",
    "tab_compare": "المقارنة",
    "tab_facts": "المعلومات الأساسية",
    # --- chat ----------------------------------------------------------------
    "chat_heading": "اسأل مستنداتك",
    "chat_placeholder": "اسأل عن بند أو التزام أو تاريخ أو مبلغ…",
    "chat_empty_title": "لم تطرح أي سؤال بعد",
    "chat_empty_body": "كل إجابة مستمدة من المستندات التي ترفعها فقط، مع مصدر لكل معلومة.",
    "chat_locked": "ارفع مستنداً وعالجه لبدء طرح الأسئلة.",
    "answering": "جارٍ البحث في المستندات وصياغة إجابة موثّقة…",
    "sources_heading": "المصادر",
    "show_passage": "عرض المقطع المُستشهد به",
    "ungrounded_notice": "لا يوجد في المستندات المرفوعة ما يدعم إجابة على هذا السؤال.",
    "clear_chat": "مسح المحادثة",
    # --- summary --------------------------------------------------------------
    "summary_heading": "ملخص منظّم",
    "summary_intro": "أنشئ ملخصاً منظّماً لمستند واحد. لا يذكر الملخص إلا ما ورد في المستند.",
    "summary_document": "المستند",
    "summary_focus": "التركيز (اختياري)",
    "summary_focus_help": "مثال: ركّز على أحكام الإنهاء والغرامات.",
    "summary_run": "إنشاء الملخص",
    "summarizing": "جارٍ قراءة المستند وصياغة الملخص…",
    "summary_truncated": "تجاوز المستند حد التحليل؛ يغطي الملخص الجزء الأول منه فقط.",
    # --- comparison -----------------------------------------------------------
    "compare_heading": "مقارنة مستندين",
    "compare_intro": "حدّد الفروق في البنود والالتزامات والتواريخ والمبالغ والأطراف بين نسختين.",
    "compare_first": "المستند الأول",
    "compare_second": "المستند الثاني",
    "compare_run": "مقارنة",
    "comparing": "جارٍ مقارنة المستندين بنداً بنداً…",
    "compare_empty_title": "عدد المستندات غير كافٍ",
    "compare_need_two": "حمّل مستندين على الأقل لإجراء المقارنة.",
    "compare_same": "اختر مستندين مختلفين.",
    "compare_truncated": "تجاوز أحد المستندين أو كلاهما حد التحليل؛ تغطي المقارنة الأجزاء الأولى منهما.",
    # --- key facts ------------------------------------------------------------
    "facts_heading": "المعلومات الأساسية",
    "facts_intro": "استخرج الوقائع التعاقدية التي يراجعها المحامي أولاً. تُستبعد الحقول التي لم يذكرها المستند.",
    "facts_run": "استخراج المعلومات",
    "extracting": "جارٍ استخراج الوقائع التعاقدية الأساسية…",
    "facts_empty": "تعذّر استخراج معلومات أساسية من هذا المستند.",
    "field_contract_type": "نوع العقد",
    "field_parties": "الأطراف",
    "field_effective_date": "تاريخ النفاذ",
    "field_term": "المدة",
    "field_key_dates": "التواريخ المهمة",
    "field_obligations": "الالتزامات",
    "field_payment_terms": "شروط الدفع",
    "field_termination": "الإنهاء",
    "field_renewal": "التجديد",
    "field_penalties": "الغرامات",
    "field_governing_law": "القانون الواجب التطبيق",
    "field_deadlines": "المواعيد النهائية",
    "field_notes": "ملاحظات أخرى",
    # --- errors ---------------------------------------------------------------
    "error_generic": "حدث خطأ: {detail}",
    "error_api_key_missing": "لم يتم ضبط مفتاح Gemini. أضف GOOGLE_API_KEY إلى متغيرات البيئة أو إلى أسرار Streamlit.",
    "error_document_read": "تعذّرت قراءة {filename}: {detail}",
    "error_unsupported_format": "صيغة الملف {filename} غير مدعومة. ارفع ملف PDF أو DOCX.",
    "error_empty_document": "لا يحتوي {filename} على نص قابل للاستخراج. تحتاج ملفات PDF الممسوحة ضوئياً إلى معالجة OCR قبل رفعها.",
    "error_model": "فشل طلب النموذج: {detail}",
    "error_model_unreachable": "تعذّر الوصول إلى خدمة Gemini. تحقّق من اتصالك بالإنترنت ثم أعد المحاولة.",
    "error_model_auth": "تم رفض مفتاح Gemini. تحقّق من GOOGLE_API_KEY في ملف ‎.env‎ أو في أسرار Streamlit.",
    "error_model_rate_limit": "تجاوزت الطلبات الحد المسموح في خدمة Gemini حالياً. انتظر قليلاً ثم أعد المحاولة.",
    "technical_detail": "التفاصيل التقنية",
    "error_retrieval": "لا توجد مستندات مفهرسة بعد. عالج مستنداً أولاً.",
    "error_too_large": "تجاوز {filename} الحد الأقصى {size} ميجابايت وتم تخطيه.",
    # --- footer ---------------------------------------------------------------
    "disclaimer_title": "تنبيه مهم",
    "disclaimer": "يقدّم «سند» تحليلاً للمستندات ولا يقدّم استشارة قانونية. تحقّق من كل مصدر في المستند الأصلي قبل الاعتماد عليه.",
}

CATALOGUES: Mapping[str, Mapping[str, str]] = {"en": EN, "ar": AR}


def translate(key: str, language: str = "en", **values: Any) -> str:
    """Look up ``key`` in ``language`` and interpolate ``values``.

    Falls back to English, then to the key itself, so a missing string degrades
    to something diagnosable instead of raising inside a Streamlit rerun. A
    formatting failure returns the uninterpolated template for the same reason.
    """
    catalogue = CATALOGUES.get(language if language in SUPPORTED_LANGUAGES else "en", EN)
    template = catalogue.get(key) or EN.get(key) or key
    if not values:
        return template
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template


#: Short alias used throughout the UI layer.
t = translate


def count_phrase(noun: str, value: int, language: str = "en") -> str:
    """Render a counted noun with the right agreement, e.g. "1 page" / "3 pages".

    English distinguishes one from everything else. Arabic's dual form is not
    modelled: "مستندان" would be strictly correct for two, but the plural
    reads acceptably and avoids a third form that English has no use for.
    """
    variant = "one" if value == 1 else "many"
    return translate(f"count_{noun}_{variant}", language, count=value)


def field_label(field: str, language: str = "en") -> str:
    """Translated heading for an extraction field name."""
    return translate(f"field_{field}", language)
