def build_prompt(question, context_chunks):

    system_prompt = """
أنت مساعد قانوني ذكي متخصص في القانون الجزائري.

القواعد:
- أجب فقط اعتمادا على المصادر القانونية المقدمة.
- لا تضف معلومات غير موجودة في السياق.
- إذا لم تجد الإجابة في المصادر قل:
لا أتوفر على معلومات كافية من المصادر المتاحة للإجابة بثقة.
- اذكر المرجع القانوني بوضوح.
- اذكر اسم الوثيقة والمادة والصفحة إن وجدت.
- اجعل الإجابة مختصرة وواضحة.
- لا تقدم استشارة قانونية نهائية.
"""

    context = ""

    for i, chunk in enumerate(context_chunks, 1):

        context += f"""
        
[Source {i}]
Document:
{chunk.get("source_file")}

Reference:
{chunk.get("title")}
Pages:
{chunk.get("page_start")}-{chunk.get("page_end")}

Content:
{chunk.get("text")}

--------------------
"""


    prompt = f"""
{system_prompt}


السؤال:
{question}


المصادر القانونية:

{context}


اكتب الجواب الآن وفق الشكل التالي:

الإجابة:
(اكتب الإجابة هنا)

المصدر:
- الوثيقة:
- المادة:
- الصفحة:
"""

    return prompt
