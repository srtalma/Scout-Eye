import streamlit as st
import google.generativeai as genai
import os
import tempfile
import time
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
import logging
import re
import pandas as pd # Added for better display formatting

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="AI League Scout Eye (Gemini Flex - عربي)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Constants (Arabic) ---

# Page Names (for session state)
PAGE_HOME = "home"
PAGE_LEGEND = "اسطورة_الغد"
PAGE_STAR = "نجم_لا_يغيب"
PAGE_PERSON = "الشخص_المناسب"

# Age Groups (for Legend page)
AGE_GROUP_5_8 = "5 إلى 8 سنوات"
AGE_GROUP_8_PLUS = "8 سنوات وأكثر"

# --- Skills for Age Group: 5 to 8 Years ---
SKILLS_AGE_5_8_EN = [
    "Running_Basic", "Ball_Feeling", "Focus_On_Task", "First_Touch_Simple"
]
SKILLS_LABELS_AGE_5_8_AR = {
    "Running_Basic": "الجري",
    "Ball_Feeling": "الإحساس بالكرة",
    "Focus_On_Task": "التركيز وتنفيذ المطلوب",
    "First_Touch_Simple": "اللمسة الأولى (استلام بسيط)"
}

# --- Skills for Age Group: 8 Years and Older ---
SKILLS_AGE_8_PLUS_EN = [
    "Jumping", "Running_Control", "Passing", "Receiving", "Zigzag"
]
SKILLS_LABELS_AGE_8_PLUS_AR = {
    "Jumping": "القفز بالكرة (تنطيط الركبة)",
    "Running_Control": "الجري بالكرة (التحكم)",
    "Passing": "التمرير",
    "Receiving": "استقبال الكرة",
    "Zigzag": "المراوغة (زجزاج)"
}

# --- Biomechanics Metrics (for Star page) ---
# Use consistent keys, preferably English for internal use
BIOMECHANICS_METRICS_EN = [
    "Right_Knee_Angle_Avg", "Left_Knee_Angle_Avg", "Asymmetry_Avg_Percent",
    "Contact_Angle_Avg", "Max_Acceleration", "Steps_Count",
    "Step_Frequency", "Hip_Flexion_Avg", "Trunk_Lean_Avg",
    "Pelvic_Tilt_Avg", "Thorax_Rotation_Avg", "Risk_Level", "Risk_Score"
]
# Arabic labels for display
BIOMECHANICS_LABELS_AR = {
    "Right_Knee_Angle_Avg": "متوسط زاوية الركبة اليمنى (°)",
    "Left_Knee_Angle_Avg": "متوسط زاوية الركبة اليسرى (°)",
    "Asymmetry_Avg_Percent": "متوسط عدم التماثل (%)",
    "Contact_Angle_Avg": "متوسط زاوية التلامس (°)",
    "Max_Acceleration": "أقصى تسارع (قيمة نسبية)",
    "Steps_Count": "عدد الخطوات",
    "Step_Frequency": "تردد الخطوات (خطوة/ثانية)",
    "Hip_Flexion_Avg": "متوسط ثني الورك (°)",
    "Trunk_Lean_Avg": "متوسط ميل الجذع (°)",
    "Pelvic_Tilt_Avg": "متوسط إمالة الحوض (°)",
    "Thorax_Rotation_Avg": "متوسط دوران الصدر (°)",
    "Risk_Level": "مستوى الخطورة",
    "Risk_Score": "درجة الخطورة"
}
# --- Biomechanics Metrics (English Labels for Star page Display) ---
BIOMECHANICS_LABELS_EN = {
    "Right_Knee_Angle_Avg": "Right Knee Angle Avg (°)",
    "Left_Knee_Angle_Avg": "Left Knee Angle Avg (°)",
    "Asymmetry_Avg_Percent": "Asymmetry Avg (%)",
    "Contact_Angle_Avg": "Contact Angle Avg (°)",
    "Max_Acceleration": "Max Acceleration (Relative)",
    "Steps_Count": "Steps Count",
    "Step_Frequency": "Step Frequency (steps/sec)",
    "Hip_Flexion_Avg": "Hip Flexion Avg (°)",
    "Trunk_Lean_Avg": "Trunk Lean Avg (°)",
    "Pelvic_Tilt_Avg": "Pelvic Tilt Avg (°)",
    "Thorax_Rotation_Avg": "Thorax Rotation Avg (°)",
    "Risk_Level": "Risk Level",
    "Risk_Score": "Risk Score"
}
NOT_CLEAR_EN = "Not Clear"
# Mapping from potential Arabic values received from Gemini to English display values
BIO_VALUE_MAP_AR_TO_EN = {
    'غير واضح': NOT_CLEAR_EN,
    'منخفض': 'Low',
    'متوسط': 'Medium',
    'مرتفع': 'High'
    # Add any other potential Arabic text values Gemini might return here
}
# Placeholder for non-detected values
NOT_CLEAR_AR = "غير واضح"

# --- General Constants ---
MAX_SCORE_PER_SKILL = 5
MODEL_NAME = "models/gemini-1.5-pro" # Make sure this model supports video analysis

# --- Analysis Modes (Simplified - Arabic) ---
MODE_SINGLE_VIDEO_ALL_SKILLS_AR = "تقييم جميع مهارات الفئة العمرية (فيديو واحد)"
MODE_SINGLE_VIDEO_ONE_SKILL_AR = "تقييم مهارة محددة (فيديو واحد)"

# --- Gemini API Configuration ---
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    logging.info("Gemini API Key loaded successfully.")
except KeyError:
    st.error("❗️ لم يتم العثور على مفتاح Gemini API في أسرار Streamlit. الرجاء إضافة `GEMINI_API_KEY`.")
    st.stop()
except Exception as e:
    st.error(f"❗️ فشل في إعداد Gemini API: {e}")
    logging.error(f"Gemini API configuration failed: {e}")
    st.stop()

# --- Gemini Model Setup ---
if "model_name" not in st.session_state:
    # Default to your usual model name, e.g. "models/gemini-1.5-pro"
    st.session_state.model_name = "models/gemini-1.5-pro"
    
@st.cache_resource
def load_gemini_model(model_name):
    """Loads the Gemini model with specific configurations."""
    try:
        # Increased output tokens for biomechanics list
        generation_config = {
             "temperature": 0.2, # Slightly higher for more descriptive potential but still controlled
             "top_p": 1,
             "top_k": 1,
             "max_output_tokens": 800, # Increased significantly for the list output
             # "response_mime_type": "application/json", # Could try this for structured output later
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        logging.info(f"Gemini Model '{MODEL_NAME}' loaded with MINIMUM safety settings (BLOCK_NONE).")
        return model
    except Exception as e:
        st.error(f"❗️ فشل تحميل نموذج Gemini '{MODEL_NAME}': {e}")
        logging.error(f"Gemini model loading failed: {e}")
        return None

model = load_gemini_model(st.session_state.model_name)
if not model:
    st.stop()
    
def test_gemini_connection(chosen_model=None):
    """
    Test basic Gemini API connectivity with a simple text prompt.
    chosen_model can override the default if you want.
    """
    try:
        # If you wanted to re-initialize the model with chosen_model:
        # if chosen_model:
        #     test_model = genai.GenerativeModel(model_name=chosen_model, ...)
        # else:
        #     test_model = model  # use your existing global 'model'

        test_prompt = "Please respond with the number 5 to test API connectivity."
        test_response = model.generate_content(test_prompt)

        st.success(f"✅ Gemini API test successful. Response: {test_response.text}")
        logging.info(f"API test successful. Raw response: {test_response}")
        return True

    except Exception as e:
        st.error(f"❌ Gemini API test failed: {e}")
        logging.error(f"API test failed: {e}", exc_info=True)
        return False


# --- CSS Styling (Remains the same) ---
st.markdown("""
<style>
    /* ... (Existing CSS styles) ... */
    body { direction: rtl; }
    .stApp { background-color: #1a1a4a; color: white; }
    /* ... other styles ... */
    /* Style for biomechanics table */
    .dataframe table { width: 100%; border-collapse: collapse; }
    .dataframe th { background-color: rgba(216, 184, 216, 0.2); /* Light purple tint */ color: white; text-align: right !important; padding: 8px; border: 1px solid rgba(255, 255, 255, 0.2); font-weight: bold; }
    .dataframe td { color: white; text-align: right !important; padding: 8px; border: 1px solid rgba(255, 255, 255, 0.2); }
    .dataframe tr:nth-child(even) { background-color: rgba(255, 255, 255, 0.05); }
    .dataframe tr:hover { background-color: rgba(255, 255, 255, 0.1); }
</style>
""", unsafe_allow_html=True)

# =========== Gemini Interaction Functions ============================

# --- Prompt function for Skill Evaluation (Legend Page) ---
def create_prompt_for_skill(skill_key_en, age_group):
    # --- (Code from previous step - no changes needed here) ---
    specific_rubric = "لا توجد معايير محددة لهذه المهارة في هذه الفئة العمرية." # Default
    skill_name_ar = skill_key_en # Default

    # --- Rubrics for Age Group: 5 to 8 Years ---
    if age_group == AGE_GROUP_5_8:
        skill_name_ar = SKILLS_LABELS_AGE_5_8_AR.get(skill_key_en, skill_key_en)
        rubrics_5_8 = {
            "Running_Basic": """
            **معايير تقييم الجري (5-8 سنوات):**
            - 0: لا يستطيع الجري أو يمشي فقط.
            - 1: يجري بشكل غير متزن أو بطيء جدًا.
            - 2: يجري بوتيرة مقبولة ولكن ببعض التعثر أو التردد.
            - 3: يجري بثقة وتوازن جيدين لمعظم المسافة.
            - 4: يجري بسرعة جيدة وتوازن ممتاز.
            - 5: يجري بسرعة عالية وتناسق حركي ممتاز وواضح.
            """,
            "Ball_Feeling": """
            **معايير تقييم الإحساس بالكرة (5-8 سنوات):**
            - 0: يتجنب لمس الكرة أو يفقدها فورًا عند اللمس.
            - 1: يلمس الكرة بقدم واحدة فقط بشكل متردد، الكرة تبتعد كثيرًا.
            - 2: يحاول لمس الكرة بكلتا القدمين، لكن التحكم ضعيف.
            - 3: يظهر بعض التحكم الأساسي، يبقي الكرة قريبة أحيانًا.
            - 4: يظهر تحكمًا جيدًا، يلمس الكرة بباطن وظاهر القدم، يحافظ عليها قريبة نسبيًا.
            - 5: يظهر تحكمًا ممتازًا ولمسات واثقة ومتنوعة، يبقي الكرة قريبة جدًا أثناء الحركة البسيطة.
            """,
            "Focus_On_Task": """
            **معايير تقييم التركيز وتنفيذ المطلوب (5-8 سنوات):** (يُقيّم بناءً على السلوك المُلاحظ في الفيديو المتعلق بالمهمة الكروية الظاهرة)
            - 0: لا يُظهر أي اهتمام بالمهمة الكروية، يتشتت تمامًا.
            - 1: يبدأ المهمة لكن يتشتت بسرعة وبشكل متكرر.
            - 2: يحاول إكمال المهمة لكن يفتقر للتركيز المستمر، يتوقف أو ينظر حوله كثيرًا.
            - 3: يركز بشكل مقبول على المهمة، يكمل أجزاء منها بانتباه.
            - 4: يظهر تركيزًا جيدًا ومستمرًا على المهمة الكروية المعروضة في الفيديو.
            - 5: يظهر تركيزًا عاليًا وانغماسًا واضحًا في المهمة الكروية، يحاول بجدية وإصرار.
            """,
            "First_Touch_Simple": """
            **معايير تقييم اللمسة الأولى (استلام بسيط) (5-8 سنوات):**
            - 0: الكرة ترتد بعيدًا جدًا عن السيطرة عند أول لمسة.
            - 1: يوقف الكرة بصعوبة، تتطلب لمسات متعددة للسيطرة.
            - 2: يستلم الكرة بشكل مقبول لكنها تبتعد قليلاً، يتطلب خطوة إضافية للتحكم.
            - 3: استلام جيد، اللمسة الأولى تبقي الكرة ضمن نطاق قريب.
            - 4: استلام جيد جدًا، لمسة أولى نظيفة تهيئ الكرة أمامه مباشرة.
            - 5: استلام ممتاز، لمسة أولى ناعمة وواثقة، سيطرة فورية.
            """
        }
        specific_rubric = rubrics_5_8.get(skill_key_en, specific_rubric)

    # --- Rubrics for Age Group: 8 Years and Older ---
    elif age_group == AGE_GROUP_8_PLUS:
        skill_name_ar = SKILLS_LABELS_AGE_8_PLUS_AR.get(skill_key_en, skill_key_en)
        rubrics_8_plus = {
             "Jumping": """
             **معايير تقييم القفز بالكرة (تنطيط الركبة) (8+ سنوات):**
             - 0: لا توجد محاولات أو لمسات ناجحة بالركبة أثناء الطيران.
             - 1: لمسة واحدة ناجحة بالركبة أثناء الطيران، مع تحكم ضعيف.
             - 2: لمستان ناجحتان بالركبة أثناء الطيران، تحكم مقبول.
             - 3: ثلاث لمسات ناجحة بالركبة، تحكم جيد وثبات.
             - 4: أربع لمسات ناجحة، تحكم ممتاز وثبات هوائي جيد.
             - 5: خمس لمسات أو أكثر، تحكم استثنائي، إيقاع وثبات ممتازين.
             """,
             "Running_Control": """
             **معايير تقييم الجري بالكرة (التحكم) (8+ سنوات):**
             - 0: تحكم ضعيف جدًا، الكرة تبتعد كثيرًا عن القدم.
             - 1: تحكم ضعيف، الكرة تبتعد بشكل ملحوظ أحيانًا.
             - 2: تحكم مقبول، الكرة تبقى ضمن نطاق واسع حول اللاعب.
             - 3: تحكم جيد، الكرة تبقى قريبة بشكل عام أثناء الجري بسرعات مختلفة.
             - 4: تحكم جيد جدًا، الكرة قريبة باستمرار حتى مع تغيير السرعة والاتجاه البسيط.
             - 5: تحكم ممتاز، الكرة تبدو ملتصقة بالقدم، سيطرة كاملة حتى مع المناورات.
             """,
             "Passing": """
             **معايير تقييم التمرير (8+ سنوات):**
             - 0: تمريرة خاطئة تمامًا أو ضعيفة جدًا أو بدون دقة.
             - 1: تمريرة بدقة ضعيفة أو قوة غير مناسبة بشكل كبير.
             - 2: تمريرة مقبولة تصل للهدف ولكن بقوة أو دقة متوسطة.
             - 3: تمريرة جيدة ودقيقة بقوة مناسبة للمسافة والهدف.
             - 4: تمريرة دقيقة جدًا ومتقنة بقوة مثالية، تضع المستلم في وضع جيد.
             - 5: تمريرة استثنائية، دقة وقوة وتوقيت مثالي، تكسر الخطوط أو تضع المستلم في موقف ممتاز.
             """,
             "Receiving": """
             **معايير تقييم استقبال الكرة (8+ سنوات):**
             - 0: فشل في السيطرة على الكرة تمامًا عند الاستقبال.
             - 1: لمسة أولى سيئة، الكرة تبتعد كثيرًا أو تتطلب جهدًا للسيطرة عليها.
             - 2: استقبال مقبول، الكرة تحت السيطرة بعد لمستين أو بحركة إضافية.
             - 3: استقبال جيد، لمسة أولى نظيفة تبقي الكرة قريبة ومتاحة للعب.
             - 4: استقبال جيد جدًا، لمسة أولى ممتازة تهيئ الكرة للخطوة التالية بسهولة (تمرير، تسديد، مراوغة).
             - 5: استقبال استثنائي، لمسة أولى مثالية تحت الضغط، تحكم فوري وسلس، يسمح باللعب السريع.
             """,
             "Zigzag": """
             **معايير تقييم المراوغة (زجزاج) (8+ سنوات):**
             - 0: فقدان السيطرة على الكرة عند محاولة تغيير الاتجاه بين الأقماع.
             - 1: تغيير اتجاه بطيء مع ابتعاد الكرة عن القدم بشكل واضح.
             - 2: تغيير اتجاه مقبول مع الحفاظ على الكرة ضمن نطاق تحكم واسع، يلمس الأقماع أحيانًا.
             - 3: تغيير اتجاه جيد مع إبقاء الكرة قريبة نسبيًا، يتجنب الأقماع.
             - 4: تغيير اتجاه سريع وسلس مع إبقاء الكرة قريبة جدًا من القدم.
             - 5: تغيير اتجاه خاطف وسلس مع سيطرة تامة على الكرة (تبدو ملتصقة بالقدم)، وخفة حركة واضحة.
             """
        }
        specific_rubric = rubrics_8_plus.get(skill_key_en, specific_rubric)

    # --- Construct the Final Prompt ---
    prompt = f"""
    مهمتك هي تقييم مهارة كرة القدم '{skill_name_ar}' المعروضة في الفيديو للاعب ضمن الفئة العمرية '{age_group}'.
    استخدم المعايير التالية **حصراً** لتقييم الأداء وتحديد درجة رقمية من 0 إلى {MAX_SCORE_PER_SKILL}:

    {specific_rubric}

    شاهد الفيديو بعناية. بناءً على المعايير المذكورة أعلاه فقط، ما هي الدرجة التي تصف أداء اللاعب بشكل أفضل؟

    هام جدًا: قم بالرد بالدرجة الرقمية الصحيحة فقط (مثال: "3" أو "5"). لا تقم بتضمين أي شروحات أو أوصاف أو أي نص آخر أو رموز إضافية. فقط الرقم.
    """
    return prompt


# --- Prompt function for Biomechanics Analysis (Star Page) ---
def create_prompt_for_biomechanics():
    """Creates the prompt for the biomechanical analysis."""
    prompt = f"""
مهمتك هي إجراء تحليل بيوميكانيكي لحركة اللاعب في الفيديو المقدم، مع التركيز على مقاطع الجري أو الحركة الرياضية الواضحة.
استخرج المقاييس الـ 13 التالية وقدمها **كقائمة مرقمة ودقيقة**. لكل مقياس، قدم القيمة الرقمية المقدرة أو الفئة المطلوبة.

**هام جداً:**
*   إذا لم تتمكن من تقدير قيمة مقياس معين بشكل معقول بسبب جودة الفيديو أو عدم وضوح الحركة، اكتب بوضوح القيمة '{NOT_CLEAR_AR}' لهذا المقياس.
*   التزم بالتنسيق المطلوب بدقة: رقم، نقطة، مسافة، اسم المقياس بالعربي كما هو مذكور بالأسفل، نقطتان، مسافة، القيمة المقدرة أو '{NOT_CLEAR_AR}'.
*   لا تقم بتضمين أي نص إضافي أو تفسيرات أو مقدمات أو خواتيم خارج هذه القائمة المرقمة.

**المقاييس المطلوبة والمعايير المساعدة للتقييم:**

1.  متوسط زاوية الركبة اليمنى: (بالدرجات، أثناء مرحلة الدفع أو الوقوف إن أمكن)
    *   (معيار خطورة مساعد: > 145 أو < 110 درجة)
2.  متوسط زاوية الركبة اليسرى: (بالدرجات، أثناء مرحلة الدفع أو الوقوف إن أمكن)
    *   (معيار خطورة مساعد: > 145 أو < 110 درجة)
3.  متوسط عدم التماثل: (كنسبة مئوية %، تقدير الفرق بين الجانبين في زوايا الركبة أو طول الخطوة)
    *   (معيار خطورة مساعد: > 15% خطر، > 10% متوسط)
4.  متوسط زاوية التلامس: (زاوية القدم/الساق الأمامية مع الأرض عند أول تلامس، بالدرجات)
    *   (معيار خطورة مساعد: > 70 أو < 110 -> خطر [ملاحظة: هذا المعيار قد يكون غير دقيق، اعتمد على التقدير البصري للزاوية])
5.  أقصى تسارع: (تقدير نسبي لأعلى قيمة لتغير السرعة، رقم بدون وحدة)
    *   (معيار خطورة مساعد: > 500,000 خطر، > 250,000 متوسط [ملاحظة: هذه أرقام نسبية، قدر القيمة البصرية للتسارع])
6.  عدد الخطوات: (إجمالي عدد الخطوات الواضحة في المقطع الذي تم تحليله)
7.  تردد الخطوات: (متوسط عدد الخطوات في الثانية، رقم عشري)
    *   (معيار خطورة مساعد: < 1.5 أو > 3 خطر)
8.  متوسط ثني الورك: (متوسط زاوية مفصل الورك، بالدرجات، ركز على مرحلة التأرجح الأمامي إن أمكن)
    *   (معيار خطورة مساعد: > 35 درجة قد يشير لتحميل زائد أو حركة غير فعالة)
9.  متوسط ميل الجذع: (متوسط زاوية ميل الجذع للأمام بالنسبة للعمودي، بالدرجات)
    *   (معيار خطورة مساعد: > 15 درجة خطر)
10. متوسط إمالة الحوض: (بالدرجات، تقدير للإمالة الأمامية/الخلفية، إيجابي للأمامية)
11. متوسط دوران الصدر: (بالدرجات، تقدير لمتوسط دوران الجذع العلوي حول المحور العمودي)
12. مستوى الخطورة: (قم بتصنيف شامل بناءً على عدد ومدى تجاوز المعايير المساعدة أعلاه: 'منخفض'، 'متوسط'، 'مرتفع')
13. درجة الخطورة: (عين درجة رقمية تقديرية من 0 إلى 5 بناءً على التقييم الشامل للخطورة، حيث 0=لا خطورة واضحة، 5=خطورة عالية جداً)


**مثال للتنسيق المطلوب:**
1. متوسط زاوية الركبة اليمنى: 151.3
2. متوسط زاوية الركبة اليسرى: 151.0
3. متوسط عدم التماثل: 5.6%
4. متوسط زاوية التلامس: 24.2
5. أقصى تسارع: 473953
6. عدد الخطوات: 37
7. تردد الخطوات: 1.8
8. متوسط ثني الورك: {NOT_CLEAR_AR}
9. متوسط ميل الجذع: 15.4
10. متوسط إمالة الحوض: -1.8
11. متوسط دوران الصدر: -30.9
12. مستوى الخطورة: متوسط
13. درجة الخطورة: 3
"""
    return prompt


# --- Video Upload/Processing Function (Common) ---
def upload_and_wait_gemini(video_path, display_name="video_upload", status_placeholder=st.empty()):
    # --- (Code from previous step - no changes needed here) ---
    uploaded_file = None
    status_placeholder.info(f"⏳ جاري رفع الفيديو '{os.path.basename(display_name)}'...") # Use display name
    logging.info(f"Starting upload for {display_name}")
    try:
        safe_display_name = f"upload_{int(time.time())}_{os.path.basename(display_name)}"
        uploaded_file = genai.upload_file(path=video_path, display_name=safe_display_name)
        status_placeholder.info(f"📤 اكتمل الرفع لـ '{display_name}'. برجاء الانتظار للمعالجة بواسطة Google...")
        logging.info(f"Upload API call successful for {display_name}, file name: {uploaded_file.name}. Waiting for ACTIVE state.")

        timeout = 300
        start_time = time.time()
        while uploaded_file.state.name == "PROCESSING":
            if time.time() - start_time > timeout:
                logging.error(f"Timeout waiting for file processing for {uploaded_file.name} ({display_name})")
                raise TimeoutError(f"انتهت مهلة معالجة الفيديو '{display_name}'. حاول مرة أخرى أو استخدم فيديو أقصر.")
            time.sleep(15) # Check less frequently
            uploaded_file = genai.get_file(uploaded_file.name)
            logging.debug(f"File {uploaded_file.name} ({display_name}) state: {uploaded_file.state.name}")

        if uploaded_file.state.name == "FAILED":
            logging.error(f"File processing failed for {uploaded_file.name} ({display_name})")
            raise ValueError(f"فشلت معالجة الفيديو '{display_name}' من جانب Google.")
        elif uploaded_file.state.name != "ACTIVE":
             logging.error(f"Unexpected file state {uploaded_file.state.name} for {uploaded_file.name} ({display_name})")
             raise ValueError(f"حالة ملف فيديو غير متوقعة: {uploaded_file.state.name} لـ '{display_name}'.")

        status_placeholder.success(f"✅ الفيديو '{display_name}' جاهز للتحليل.")
        logging.info(f"File {uploaded_file.name} ({display_name}) is ACTIVE.")
        return uploaded_file

    except Exception as e:
        status_placeholder.error(f"❌ خطأ أثناء رفع/معالجة الفيديو لـ '{display_name}': {e}")
        logging.error(f"Upload/Wait failed for '{display_name}': {e}", exc_info=True)
        if uploaded_file and uploaded_file.state.name != "ACTIVE":
            try:
                logging.warning(f"Attempting to delete potentially failed/stuck file: {uploaded_file.name} ({display_name})")
                genai.delete_file(uploaded_file.name)
                logging.info(f"Cleaned up failed/stuck file: {uploaded_file.name}")
            except Exception as del_e:
                 logging.warning(f"Failed to delete file {uploaded_file.name} after upload error: {del_e}")
        return None


# --- Analysis function for Skill Evaluation (Legend Page) ---
def analyze_video_with_prompt(gemini_file_obj, skill_key_en, age_group, status_placeholder=st.empty()):
    # --- (Code from previous step - no changes needed here) ---
    score = 0 # Default score
    if age_group == AGE_GROUP_5_8:
        skill_name_ar = SKILLS_LABELS_AGE_5_8_AR.get(skill_key_en, skill_key_en)
    elif age_group == AGE_GROUP_8_PLUS:
        skill_name_ar = SKILLS_LABELS_AGE_8_PLUS_AR.get(skill_key_en, skill_key_en)
    else:
        skill_name_ar = skill_key_en # Fallback
    prompt = create_prompt_for_skill(skill_key_en, age_group)
    status_placeholder.info(f"🧠 Gemini يحلل الآن مهارة '{skill_name_ar}' للفئة العمرية '{age_group}'...")
    logging.info(f"Requesting analysis for skill '{skill_key_en}' (Age: {age_group}) using file {gemini_file_obj.name}")
    # logging.debug(f"Prompt for {skill_key_en} (Age: {age_group}):\n{prompt}") # Optional prompt logging

    try:
        # Make API call
        response = model.generate_content([prompt, gemini_file_obj], request_options={"timeout": 180}) # Increased timeout

        # --- Response Checking & Parsing (simplified for brevity, keep full checks from previous step) ---
        if not response.candidates:
             st.warning(f"⚠️ استجابة Gemini فارغة لـ '{skill_name_ar}'. النتيجة=0.")
             logging.warning(f"Response candidates list empty for {skill_key_en} (Age: {age_group}). File: {gemini_file_obj.name}")
             return 0 # Return default score

        try:
            raw_score_text = response.text.strip()
            match = re.search(r'\d+', raw_score_text)
            if match:
                parsed_score = int(match.group(0))
                score = max(0, min(MAX_SCORE_PER_SKILL, parsed_score)) # Clamp score
                status_placeholder.success(f"✅ اكتمل تحليل '{skill_name_ar}'. النتيجة: {score}")
                logging.info(f"Analysis for {skill_key_en} (Age: {age_group}) successful. Raw: '{raw_score_text}', Score: {score}. File: {gemini_file_obj.name}")
            else:
                 st.warning(f"⚠️ لم يتم العثور على رقم في استجابة Gemini لـ '{skill_name_ar}' ('{raw_score_text}'). النتيجة=0.")
                 logging.warning(f"Could not parse score (no digits) for {skill_key_en} (Age: {age_group}) from text: '{raw_score_text}'. File: {gemini_file_obj.name}")
                 score = 0
        except Exception as e_parse:
             st.warning(f"⚠️ لم نتمكن من تحليل النتيجة من استجابة Gemini لـ '{skill_name_ar}'. الخطأ: {e_parse}. النتيجة=0.")
             logging.warning(f"Score parsing error for {skill_key_en} (Age: {age_group}): {e_parse}. File: {gemini_file_obj.name}. Response text: {response.text[:100] if hasattr(response, 'text') else 'N/A'}")
             score = 0

    except Exception as e:
        # Handle API errors, timeouts, etc.
        st.error(f"❌ حدث خطأ أثناء تحليل Gemini لـ '{skill_name_ar}': {e}")
        logging.error(f"Gemini analysis failed for {skill_key_en} (Age: {age_group}): {e}. File: {gemini_file_obj.name}", exc_info=True)
        score = 0

    return score


# --- NEW Analysis function for Biomechanics (Star Page) ---
def analyze_biomechanics_video(gemini_file_obj, status_placeholder=st.empty()):
    """Analyzes video for biomechanics, parses the list output."""
    results = {key: NOT_CLEAR_AR for key in BIOMECHANICS_METRICS_EN} # Initialize with "Not Clear"

    prompt = create_prompt_for_biomechanics()
    status_placeholder.info(f"🧠 Gemini يحلل الآن الفيديو للبيوميكانيكا...")
    logging.info(f"Requesting biomechanics analysis using file {gemini_file_obj.name}")
    # logging.debug(f"Biomechanics Prompt:\n{prompt}") # Optional: log the full prompt

    try:
        # Make API call with longer timeout for potentially complex analysis
        response = model.generate_content([prompt, gemini_file_obj], request_options={"timeout": 300})

        # --- Optional DEBUG block ---
        # try:
        #     with st.expander("🐞 معلومات تصحيح للبيوميكانيكا (اضغط للتوسيع)", expanded=False):
        #         st.write("**Prompt Feedback:**", response.prompt_feedback)
        #         st.write("**Raw Text Response:**")
        #         st.text(response.text)
        #         logging.info(f"Full Gemini Response Object for Biomechanics: {response}")
        # except Exception as debug_e:
        #     logging.warning(f"Error displaying debug info in UI for biomechanics: {debug_e}")
        # --- End Optional DEBUG block ---

        if not response.candidates:
             status_placeholder.warning("⚠️ استجابة Gemini للبيوميكانيكا فارغة.")
             logging.warning(f"Response candidates list empty for biomechanics. File: {gemini_file_obj.name}")
             return results # Return default "Not Clear" results

        raw_text = response.text.strip()
        logging.info(f"Gemini Raw Response Text for Biomechanics:\n{raw_text}")

        # --- Parsing the numbered list ---
        parsed_count = 0
        # Create a mapping from the Arabic label in the prompt to the English key
        label_to_key_map = {label.split(':')[0].strip(): key for key, label in BIOMECHANICS_LABELS_AR.items()}
        # Add handling for labels potentially without units in the prompt/response mapping
        label_to_key_map_simple = {label.split('(')[0].strip(): key for key, label in BIOMECHANICS_LABELS_AR.items()}


        lines = raw_text.split('\n')
        for line in lines:
            line = line.strip()
            # Regex to capture: number, dot, space, LABEL NAME, colon, space, VALUE
            match = re.match(r"^\d+\.\s+(.+?):\s+(.+)$", line)
            if match:
                label_ar_from_response = match.group(1).strip()
                value = match.group(2).strip()

                # Try to find the corresponding English key
                metric_key_en = None
                if label_ar_from_response in label_to_key_map:
                    metric_key_en = label_to_key_map[label_ar_from_response]
                elif label_ar_from_response in label_to_key_map_simple: # Fallback without units
                     metric_key_en = label_to_key_map_simple[label_ar_from_response]


                if metric_key_en and metric_key_en in results:
                    # Clean up value slightly (remove potential extra quotes)
                    value = value.strip('\'"')
                    results[metric_key_en] = value
                    parsed_count += 1
                    logging.debug(f"Parsed Biomechanics: {metric_key_en} = {value}")
                else:
                    logging.warning(f"Unmatched/Unknown label in biomechanics response line: '{label_ar_from_response}' in line: '{line}'")
            # else:
            #     # Log lines that didn't match the expected format
            #     if line: # Avoid logging empty lines
            #         logging.debug(f"Skipping non-matching line in biomechanics response: '{line}'")


        if parsed_count > 0:
             status_placeholder.success(f"✅ اكتمل تحليل البيوميكانيكا. تم تحليل {parsed_count} مقياس.")
             logging.info(f"Biomechanics analysis successful. Parsed {parsed_count} metrics. File: {gemini_file_obj.name}")
             # Log if some metrics remained "Not Clear"
             not_clear_count = sum(1 for v in results.values() if v == NOT_CLEAR_AR)
             if not_clear_count > 0:
                  logging.warning(f"{not_clear_count} biomechanics metrics remained '{NOT_CLEAR_AR}'.")
                  status_placeholder.warning(f"⚠️ تم تحليل {parsed_count} مقياس، ولكن {not_clear_count} مقياس لم تكن واضحة في الفيديو.")

        else:
             status_placeholder.warning("⚠️ لم يتمكن النموذج من تحليل أي مقاييس بيوميكانيكية من الفيديو بالتنسيق المتوقع.")
             logging.warning(f"Failed to parse any biomechanics metrics from response. Raw text:\n{raw_text}")
             # Keep results as default "Not Clear"

    except Exception as e:
        status_placeholder.error(f"❌ حدث خطأ أثناء تحليل Gemini للبيوميكانيكا: {e}")
        logging.error(f"Gemini biomechanics analysis failed: {e}. File: {gemini_file_obj.name}", exc_info=True)
        # Keep results as default "Not Clear"

    return results


# --- File Deletion Function (Common) ---
def delete_gemini_file(gemini_file_obj, status_placeholder=st.empty()):
    # --- (Code from previous step - no changes needed here) ---
    if not gemini_file_obj: return
    try:
        display_name = gemini_file_obj.display_name # Should contain the unique upload name
        status_placeholder.info(f"🗑️ جاري حذف الملف المرفوع '{display_name}' من التخزين السحابي...")
        logging.info(f"Attempting to delete cloud file: {gemini_file_obj.name} (Display: {display_name})")
        genai.delete_file(gemini_file_obj.name)
        logging.info(f"Cloud file deleted successfully: {gemini_file_obj.name} (Display: {display_name})")
    except Exception as e:
        st.warning(f"⚠️ لم نتمكن من حذف الملف السحابي {gemini_file_obj.name} (Display: {display_name}): {e}")
        logging.warning(f"Could not delete cloud file {gemini_file_obj.name} (Display: {display_name}): {e}")


# =========== Grading and Plotting Functions =================

def evaluate_final_grade_from_individual_scores(scores_dict):
    # --- (Code from previous step - no changes needed here) ---
    if not scores_dict:
        return {"scores": {}, "total_score": 0, "grade": "N/A", "max_score": 0}
    total = sum(scores_dict.values())
    max_possible = len(scores_dict) * MAX_SCORE_PER_SKILL
    percentage = (total / max_possible) * 100 if max_possible > 0 else 0
    if percentage >= 90: grade = 'ممتاز (A)'
    elif percentage >= 75: grade = 'جيد جداً (B)'
    elif percentage >= 55: grade = 'جيد (C)'
    elif percentage >= 40: grade = 'مقبول (D)'
    else: grade = 'ضعيف (F)'
    return {"scores": scores_dict, "total_score": total, "grade": grade, "max_score": max_possible}

def plot_results(results, skills_labels_ar):
    # --- (Code from previous step - no changes needed here) ---
    if not results or 'scores' not in results or not results['scores']:
        logging.warning("Plotting attempted with invalid or empty results.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, get_display(arabic_reshaper.reshape("لا توجد بيانات لعرضها")),
                ha='center', va='center', color='white')
        fig.patch.set_alpha(0); ax.set_facecolor((0, 0, 0, 0)); ax.axis('off')
        return fig
    scores_dict = results['scores']
    valid_keys_en = [key for key in scores_dict.keys() if key in skills_labels_ar]
    if not valid_keys_en:
         logging.warning("No matching keys found between results and skills_labels_ar for plotting.")
         fig, ax = plt.subplots(); ax.text(0.5, 0.5, get_display(arabic_reshaper.reshape("خطأ: عدم تطابق بيانات الرسم")), ha='center', va='center', color='white'); fig.patch.set_alpha(0); ax.set_facecolor((0, 0, 0, 0)); ax.axis('off'); return fig
    try:
        reshaped_labels = [get_display(arabic_reshaper.reshape(skills_labels_ar[key_en])) for key_en in valid_keys_en]
        scores = [scores_dict[key_en] for key_en in valid_keys_en]
        grade_display = results.get('grade', 'N/A')
        if grade_display != 'N/A' and grade_display != 'غير مكتمل':
            plot_title_text = f"التقييم النهائي - التقدير: {grade_display} ({results.get('total_score', 0)}/{results.get('max_score', 0)})"
        else:
            plot_title_text = "نتيجة المهارة";  # Default or single skill
            if len(valid_keys_en) == 1: plot_title_text = f"نتيجة مهارة: {reshaped_labels[0]}"
        plot_title = get_display(arabic_reshaper.reshape(plot_title_text))
        y_axis_label = get_display(arabic_reshaper.reshape(f"الدرجة (من {MAX_SCORE_PER_SKILL})"))
    except Exception as e:
        st.warning(f"حدث خطأ أثناء تهيئة نص الرسم البياني العربي: {e}"); logging.warning(f"Arabic reshaping/label preparation failed for plot: {e}")
        reshaped_labels = valid_keys_en; scores = [scores_dict[key_en] for key_en in valid_keys_en]
        plot_title = f"Evaluation - Grade: {results.get('grade','N/A')} ({results.get('total_score',0)}/{results.get('max_score',0)})"; y_axis_label = f"Score (out of {MAX_SCORE_PER_SKILL})"
    fig, ax = plt.subplots(figsize=(max(6, len(scores)*1.5), 6)) # Dynamic width
    bars = ax.bar(reshaped_labels, scores)
    ax.set_ylim(0, MAX_SCORE_PER_SKILL + 0.5)
    ax.set_ylabel(y_axis_label, fontsize=12, fontweight='bold', color='white')
    ax.set_title(plot_title, fontsize=14, fontweight='bold', color='white')
    colors = ['#2ca02c' if s >= 4 else '#ff7f0e' if s >= 2.5 else '#d62728' for s in scores]
    for bar, color in zip(bars, colors): bar.set_color(color)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f'{yval}', ha='center', va='bottom', fontsize=11, color='white', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.6, color='gray')
    ax.tick_params(axis='x', labelsize=11, rotation=15, colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray'); ax.spines['bottom'].set_color('gray')
    fig.patch.set_alpha(0); ax.set_facecolor((0, 0, 0, 0))
    plt.tight_layout(); return fig


# =========== Streamlit App Layout (Arabic) ====================================

# Initialize session state variables
if 'page' not in st.session_state: st.session_state.page = PAGE_HOME
if 'evaluation_results' not in st.session_state: st.session_state.evaluation_results = None # For Legend page results
if 'biomechanics_results' not in st.session_state: st.session_state.biomechanics_results = None # For Star page results
if 'analysis_mode' not in st.session_state: st.session_state.analysis_mode = MODE_SINGLE_VIDEO_ALL_SKILLS_AR # Default for Legend
if 'selected_skill_key' not in st.session_state: st.session_state.selected_skill_key = None
if 'selected_age_group' not in st.session_state: st.session_state.selected_age_group = AGE_GROUP_8_PLUS # Default age for Legend
if 'uploaded_file_state' not in st.session_state: st.session_state.uploaded_file_state = None # Can hold file for any page temporarily
if 'gemini_file_object' not in st.session_state: st.session_state.gemini_file_object = None # Can hold processed file for any page

# --- Helper to clear state on page change ---
def clear_page_specific_state():
    st.session_state.evaluation_results = None
    st.session_state.biomechanics_results = None
    st.session_state.uploaded_file_state = None
    # Keep gemini_file_object? Maybe clear it too for simplicity, forcing re-upload on page switch
    # If we clear it, add cleanup here:
    if 'gemini_file_object' in st.session_state and st.session_state.gemini_file_object:
         logging.info(f"Clearing Gemini file object {st.session_state.gemini_file_object.name} due to page switch.")
         # delete_gemini_file(st.session_state.gemini_file_object, st.empty()) # Add deletion on page switch if desired
         st.session_state.gemini_file_object = None


# --- Page config, CSS, etc. is still above here ---

# 1) Top row: AI League logo on the left
# 1) Top row: AI League logo on the left (adjust widths as needed)
col_left, col_mid, col_right = st.columns([1,3,1])
with col_left:
    # Make sure the file "ai_league_logo.png" is in the same folder or fix path
    st.image("ai_league_logo.png", width=240)

# 2) Center the Scout Eye logo and titles
col_empty1, col_scout_eye, col_empty2 = st.columns([1,2,1])
with col_scout_eye:
    # Make sure "scout_eye_logo.png" is in the same folder or fix path
    st.image("scout_eye_logo.png", width=700)


# 3) Center the three main buttons
col_b1, col_b2, col_b3 = st.columns([1,2,1])
with col_b2:
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✔️ الشخص المناسب", key="btn_person"):
            st.session_state.page = PAGE_PERSON
    with c2:
        if st.button("⭐ نجم لا يغيب", key="btn_star"):
            st.session_state.page = PAGE_STAR
    with c3:
        if st.button("⚽ إسطورة الغد", key="btn_legend"):
            st.session_state.page = PAGE_LEGEND


# --- Conditional Page Content ---

# ==================================
# ==      إسطورة الغد Page       ==
# ==================================
if st.session_state.page == PAGE_LEGEND:
    st.markdown("---")
    st.markdown("## ⚽ إسطورة الغد - تحليل المهارات بواسطة Gemini ⚽")

    # --- Age Group Selection ---
    st.markdown("<h3 style='text-align: center;'>1. اختر الفئة العمرية للموهبة</h3>", unsafe_allow_html=True)
    age_options = [AGE_GROUP_5_8, AGE_GROUP_8_PLUS]
    st.session_state.selected_age_group = st.radio(
        "الفئة العمرية:", options=age_options,
        index=age_options.index(st.session_state.selected_age_group),
        key="age_group_radio", horizontal=True
    )

    # Determine current skill set based on selected age
    if st.session_state.selected_age_group == AGE_GROUP_5_8:
        current_skills_en = SKILLS_AGE_5_8_EN
        current_skills_labels_ar = SKILLS_LABELS_AGE_5_8_AR
    else: # AGE_GROUP_8_PLUS
        current_skills_en = SKILLS_AGE_8_PLUS_EN
        current_skills_labels_ar = SKILLS_LABELS_AGE_8_PLUS_AR

    if 'selected_skill_key' in st.session_state and st.session_state.selected_skill_key not in current_skills_en:
        st.session_state.selected_skill_key = current_skills_en[0] if current_skills_en else None

    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.3); margin-top: 0.5em; margin-bottom: 1.5em;'>", unsafe_allow_html=True)

    # --- Analysis Mode Selection ---
    st.markdown("<h3 style='text-align: center;'>2. اختر طريقة التحليل</h3>", unsafe_allow_html=True)
    analysis_options = [MODE_SINGLE_VIDEO_ALL_SKILLS_AR, MODE_SINGLE_VIDEO_ONE_SKILL_AR]
    st.session_state.analysis_mode = st.radio(
        "طريقة التحليل:", options=analysis_options,
        index=analysis_options.index(st.session_state.analysis_mode),
        key="analysis_mode_radio", horizontal=True
    )

    st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.3); margin-top: 0.5em; margin-bottom: 1.5em;'>", unsafe_allow_html=True)

    # --- File Upload UI ---
    st.markdown("<h3 style='text-align: center;'>3. ارفع ملف الفيديو</h3>", unsafe_allow_html=True)
    uploaded_file_legend = None
    skill_to_analyze_key_en = None

    if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
        st.markdown(f"<p style='text-align: center; font-size: 1.1em;'>لتقييم جميع مهارات فئة '{st.session_state.selected_age_group}' ({len(current_skills_en)} مهارات)</p>", unsafe_allow_html=True)
        uploaded_file_legend = st.file_uploader(
            "📂 ارفع فيديو شامل واحد:", type=["mp4", "avi", "mov", "mkv", "webm"],
            key="upload_legend_all" # Page specific key
            )

    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
        st.markdown("<p style='text-align: center; font-size: 1.1em;'>لتقييم مهارة واحدة محددة من فيديو</p>", unsafe_allow_html=True)
        col_select, col_upload = st.columns([1, 2])
        with col_select:
             # Ensure selected skill is valid for the current age group
             valid_skill_index = 0
             if st.session_state.selected_skill_key in current_skills_en:
                 valid_skill_index = current_skills_en.index(st.session_state.selected_skill_key)
             else: # If previous skill not valid, default to first skill
                  st.session_state.selected_skill_key = current_skills_en[0] if current_skills_en else None

             st.session_state.selected_skill_key = st.selectbox(
                 "اختر المهارة:", options=current_skills_en,
                 format_func=lambda key: current_skills_labels_ar.get(key, key),
                 index=valid_skill_index,
                 key="select_legend_skill" # Page specific key
             )
             skill_to_analyze_key_en = st.session_state.selected_skill_key
             skill_label_for_upload = current_skills_labels_ar.get(skill_to_analyze_key_en, "المحددة")

        with col_upload:
            uploaded_file_legend = st.file_uploader(
                f"📂 ارفع فيديو مهارة '{skill_label_for_upload}'", type=["mp4", "avi", "mov", "mkv", "webm"],
                key="upload_legend_one" # Page specific key
                )

    # Store the Streamlit uploaded file object in session state
    if uploaded_file_legend:
        st.session_state.uploaded_file_state = uploaded_file_legend
    # Don't automatically clear if None, user might just be switching modes

    # Determine if ready to analyze
    ready_to_analyze_legend = False
    if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
        ready_to_analyze_legend = st.session_state.uploaded_file_state is not None
    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
        ready_to_analyze_legend = st.session_state.uploaded_file_state is not None and skill_to_analyze_key_en is not None

    st.markdown("---")

    # --- Analysis Button ---
    st.markdown("<h3 style='text-align: center;'>4. ابدأ التحليل</h3>", unsafe_allow_html=True)
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
    with button_col2:
        if st.button("🚀 بدء تحليل المهارات", key="start_legend_eval", disabled=not ready_to_analyze_legend, use_container_width=True):
            st.session_state.evaluation_results = None # Clear previous skill results
            local_temp_file_path = None
            analysis_error = False
            gemini_file_to_use = None

            # --- Check/Upload Video ---
            # Check if existing Gemini file object corresponds to the current upload
            should_upload = True
            current_upload_name = st.session_state.uploaded_file_state.name if st.session_state.uploaded_file_state else None
            if st.session_state.gemini_file_object and current_upload_name and st.session_state.gemini_file_object.display_name.endswith(current_upload_name):
                 try:
                      status_check_placeholder = st.empty()
                      status_check_placeholder.info("🔄 التحقق من حالة الفيديو المرفوع سابقاً...")
                      check_file = genai.get_file(st.session_state.gemini_file_object.name)
                      if check_file.state.name == "ACTIVE":
                           status_check_placeholder.success("✅ الفيديو المرفوع سابقاً لا يزال جاهزاً.")
                           should_upload = False
                           gemini_file_to_use = st.session_state.gemini_file_object
                           logging.info(f"Reusing existing ACTIVE Gemini file for Legend: {gemini_file_to_use.name}")
                           time.sleep(1) # Brief pause for user to see message
                           status_check_placeholder.empty()
                      else:
                           status_check_placeholder.warning(f"⚠️ الفيديو المرفوع سابقاً لم يعد صالحاً (الحالة: {check_file.state.name}). سيتم إعادة الرفع.")
                           logging.warning(f"Previous Gemini file {st.session_state.gemini_file_object.name} no longer ACTIVE (State: {check_file.state.name}). Re-uploading.")
                           st.session_state.gemini_file_object = None # Clear invalid reference
                           time.sleep(2)
                           status_check_placeholder.empty()
                 except Exception as e_check:
                      status_check_placeholder.warning(f"⚠️ فشل التحقق من الفيديو السابق ({e_check}). سيتم إعادة الرفع.")
                      logging.warning(f"Failed to check status of previous Gemini file {st.session_state.gemini_file_object.name}: {e_check}. Re-uploading.")
                      st.session_state.gemini_file_object = None
                      time.sleep(2)
                      status_check_placeholder.empty()

            if should_upload and st.session_state.uploaded_file_state:
                st.session_state.gemini_file_object = None # Ensure old object is cleared
                status_placeholder_upload = st.empty()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_state.name)[1]) as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file_state.getvalue())
                        local_temp_file_path = tmp_file.name
                    gemini_file_to_use = upload_and_wait_gemini(
                        local_temp_file_path, st.session_state.uploaded_file_state.name, status_placeholder_upload
                    )
                    if gemini_file_to_use:
                        st.session_state.gemini_file_object = gemini_file_to_use # Store successfully uploaded file
                    else:
                        analysis_error = True
                except Exception as e_upload:
                    status_placeholder_upload.error(f"❌ حدث خطأ فادح أثناء تحضير الفيديو: {e_upload}")
                    logging.error(f"Fatal error during Legend video prep/upload: {e_upload}", exc_info=True)
                    analysis_error = True
                finally:
                     if local_temp_file_path and os.path.exists(local_temp_file_path):
                         try: os.remove(local_temp_file_path); logging.info(f"Deleted local temp file: {local_temp_file_path}")
                         except Exception as e_del: logging.warning(f"Could not delete local temp file {local_temp_file_path}: {e_del}")

            # --- Analyze Skills ---
            if not analysis_error and gemini_file_to_use:
                results_dict = {}
                with st.spinner("🧠 Gemini يحلل المهارات المطلوبة..."):
                    analysis_status_container = st.container()
                    skills_to_process_keys = []
                    if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
                        skills_to_process_keys = current_skills_en
                    elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
                        if skill_to_analyze_key_en: skills_to_process_keys = [skill_to_analyze_key_en]

                    if not skills_to_process_keys:
                         st.error("لم يتم تحديد مهارات للتحليل."); analysis_error = True
                    else:
                         st.info(f"سيتم تحليل {len(skills_to_process_keys)} مهارة...")
                         for skill_key in skills_to_process_keys:
                             status_skill_analysis = analysis_status_container.empty()
                             score = analyze_video_with_prompt(
                                 gemini_file_to_use, skill_key,
                                 st.session_state.selected_age_group, status_skill_analysis
                             )
                             results_dict[skill_key] = score
                             # Add small delay if needed for API rate limits or UI updates
                             # time.sleep(1)

                         # --- Calculate Final Grade ---
                         if st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ALL_SKILLS_AR:
                             if len(results_dict) == len(current_skills_en):
                                 st.session_state.evaluation_results = evaluate_final_grade_from_individual_scores(results_dict)
                                 st.success("🎉 تم حساب التقييم النهائي للمهارات بنجاح!")
                                 st.balloons()
                             else:
                                 st.warning(f"لم يتم تحليل جميع المهارات المتوقعة ({len(current_skills_en)}). النتائج قد تكون غير مكتملة.")
                                 st.session_state.evaluation_results = {"scores": results_dict, "grade": "غير مكتمل", "total_score": sum(results_dict.values()), "max_score": len(current_skills_en) * MAX_SCORE_PER_SKILL}
                         elif st.session_state.analysis_mode == MODE_SINGLE_VIDEO_ONE_SKILL_AR:
                             if results_dict:
                                  st.session_state.evaluation_results = {"scores": results_dict, "grade": "N/A", "total_score": sum(results_dict.values()), "max_score": MAX_SCORE_PER_SKILL}
                                  analyzed_skill_label = current_skills_labels_ar.get(list(results_dict.keys())[0], '')
                                  st.success(f"🎉 اكتمل تحليل مهارة '{analyzed_skill_label}'!")
                             else:
                                  st.error("فشل تحليل المهارة المحددة."); analysis_error = True

            # Note: Cleanup of Gemini file happens on next run check or page switch

    # --- Display Stored Skill Evaluation Results ---
    if st.session_state.evaluation_results:
        # --- (Display logic from previous step - no changes) ---
        results = st.session_state.evaluation_results
        st.markdown("---")
        st.markdown("### 🏆 نتائج تقييم المهارات 🏆")
        plot_labels_ar = current_skills_labels_ar
        if 'grade' in results and results['grade'] != "N/A" and results['grade'] != "غير مكتمل":
            res_col1, res_col2 = st.columns(2)
            with res_col1: st.metric("🎯 التقدير العام", results['grade'])
            with res_col2: st.metric("📊 مجموع النقاط", f"{results.get('total_score', '0')} / {results.get('max_score', '0')}")
            st.markdown("#### 📈 رسم بياني للدرجات:")
            try:
                plot_fig = plot_results(results, plot_labels_ar)
                st.pyplot(plot_fig); plt.close(plot_fig)
            except Exception as plot_err:
                 st.error(f"حدث خطأ أثناء إنشاء الرسم البياني: {plot_err}"); logging.error(f"Plotting failed: {plot_err}", exc_info=True)
                 with st.expander("عرض الدرجات الخام"):
                     for key, score in results.get('scores', {}).items(): st.write(f"- {plot_labels_ar.get(key, key)}: {score}/{MAX_SCORE_PER_SKILL}")
        elif 'scores' in results and results['scores']:
            if len(results['scores']) == 1:
                skill_key_analyzed = list(results['scores'].keys())[0]; skill_label_analyzed = plot_labels_ar.get(skill_key_analyzed, skill_key_analyzed)
                score_analyzed = results['scores'][skill_key_analyzed]
                st.metric(f"🏅 نتيجة مهارة '{skill_label_analyzed}'", f"{score_analyzed} / {MAX_SCORE_PER_SKILL}")
                st.markdown("#### 📈 رسم بياني للدرجة:")
                try: plot_fig = plot_results(results, plot_labels_ar); st.pyplot(plot_fig); plt.close(plot_fig)
                except Exception as plot_err: st.error(f"حدث خطأ أثناء إنشاء الرسم البياني للمهارة الواحدة: {plot_err}"); logging.error(f"Single skill plotting failed: {plot_err}", exc_info=True)
            else: # Incomplete results
                st.warning("النتائج غير مكتملة.")
                st.metric("📊 مجموع النقاط (غير مكتمل)", f"{results.get('total_score', '0')} / {results.get('max_score', '0')}")
                st.markdown("#### 📈 رسم بياني للدرجات المتوفرة:")
                try: plot_fig = plot_results(results, plot_labels_ar); st.pyplot(plot_fig); plt.close(plot_fig)
                except Exception as plot_err: st.error(f"حدث خطأ أثناء إنشاء الرسم البياني للنتائج غير المكتملة: {plot_err}"); logging.error(f"Incomplete results plotting failed: {plot_err}", exc_info=True)
                    # with st.expander("عرض الدرجات الخام"):
                    #     for key, score in results.get('scores', {}).items(): st.write(f"- {plot_labels_ar.get(key, key)}: {score}/{MAX_SCORE_PER_SKILL}")
        else: st.warning("لم يتم العثور على نتائج لعرضها.")

# ==================================
# ==      نجم لا يغيب Page       ==
# ==================================
elif st.session_state.page == PAGE_STAR:
    st.markdown("---")
    st.markdown("## ⭐ نجم لا يغيب - التحليل البيوميكانيكي بواسطة Gemini ⭐")
    st.markdown("<p style='text-align: center; font-size: 1.1em;'>تحليل حركة اللاعب لاستخراج المقاييس البيوميكانيكية الرئيسية ومستوى الخطورة المحتمل.</p>", unsafe_allow_html=True)

    # --- File Upload ---
    st.markdown("<h3 style='text-align: center;'>1. ارفع فيديو الحركة (يفضل الجري)</h3>", unsafe_allow_html=True)
    uploaded_file_star = st.file_uploader(
        "📂 ارفع فيديو واحد للتحليل البيوميكانيكي:", type=["mp4", "avi", "mov", "mkv", "webm"],
        key="upload_star_biomechanics" # Page specific key
    )

    if uploaded_file_star:
        st.session_state.uploaded_file_state = uploaded_file_star
    # Don't clear if None immediately

    ready_to_analyze_star = st.session_state.uploaded_file_state is not None

    st.markdown("---")

    # --- Analysis Button ---
    st.markdown("<h3 style='text-align: center;'>2. ابدأ التحليل البيوميكانيكي</h3>", unsafe_allow_html=True)
    button_col1_star, button_col2_star, button_col3_star = st.columns([1, 2, 1])
    with button_col2_star:
        if st.button("🔬 بدء تحليل البيوميكانيكا", key="start_star_eval", disabled=not ready_to_analyze_star, use_container_width=True):
            st.session_state.biomechanics_results = None # Clear previous biomechanics results
            local_temp_file_path = None
            analysis_error = False
            gemini_file_to_use = None

            # --- Check/Upload Video (Similar logic as Legend page) ---
            should_upload = True
            current_upload_name = st.session_state.uploaded_file_state.name if st.session_state.uploaded_file_state else None
            if st.session_state.gemini_file_object and current_upload_name and st.session_state.gemini_file_object.display_name.endswith(current_upload_name):
                 try:
                      status_check_placeholder = st.empty()
                      status_check_placeholder.info("🔄 التحقق من حالة الفيديو المرفوع سابقاً...")
                      check_file = genai.get_file(st.session_state.gemini_file_object.name)
                      if check_file.state.name == "ACTIVE":
                           status_check_placeholder.success("✅ الفيديو المرفوع سابقاً لا يزال جاهزاً.")
                           should_upload = False
                           gemini_file_to_use = st.session_state.gemini_file_object
                           logging.info(f"Reusing existing ACTIVE Gemini file for Star: {gemini_file_to_use.name}")
                           time.sleep(1); status_check_placeholder.empty()
                      else:
                           status_check_placeholder.warning(f"⚠️ الفيديو السابق لم يعد صالحاً ({check_file.state.name}). سيتم الرفع."); logging.warning(f"Prev file {st.session_state.gemini_file_object.name} invalid ({check_file.state.name}). Re-uploading."); st.session_state.gemini_file_object = None; time.sleep(2); status_check_placeholder.empty()
                 except Exception as e_check:
                      status_check_placeholder.warning(f"⚠️ فشل التحقق ({e_check}). سيتم الرفع."); logging.warning(f"Failed check prev file {st.session_state.gemini_file_object.name}: {e_check}. Re-uploading."); st.session_state.gemini_file_object = None; time.sleep(2); status_check_placeholder.empty()

            if should_upload and st.session_state.uploaded_file_state:
                st.session_state.gemini_file_object = None
                status_placeholder_upload = st.empty()
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_state.name)[1]) as tmp_file:
                        tmp_file.write(st.session_state.uploaded_file_state.getvalue()); local_temp_file_path = tmp_file.name
                    gemini_file_to_use = upload_and_wait_gemini(local_temp_file_path, st.session_state.uploaded_file_state.name, status_placeholder_upload)
                    if gemini_file_to_use: st.session_state.gemini_file_object = gemini_file_to_use
                    else: analysis_error = True
                except Exception as e_upload:
                    status_placeholder_upload.error(f"❌ خطأ فادح أثناء تحضير الفيديو: {e_upload}"); logging.error(f"Fatal error during Star video prep/upload: {e_upload}", exc_info=True); analysis_error = True
                finally:
                     if local_temp_file_path and os.path.exists(local_temp_file_path):
                         try: os.remove(local_temp_file_path); logging.info(f"Deleted local temp file: {local_temp_file_path}")
                         except Exception as e_del: logging.warning(f"Could not delete local temp file {local_temp_file_path}: {e_del}")

            # --- Analyze Biomechanics ---
            if not analysis_error and gemini_file_to_use:
                with st.spinner("🔬 Gemini يحلل المقاييس البيوميكانيكية..."):
                    analysis_status_placeholder = st.empty()
                    st.session_state.biomechanics_results = analyze_biomechanics_video(
                        gemini_file_to_use,
                        analysis_status_placeholder
                    )
                    if not st.session_state.biomechanics_results or all(v == NOT_CLEAR_AR for v in st.session_state.biomechanics_results.values()):
                         # If results are empty or all are "Not Clear", maybe indicate failure more strongly
                         analysis_status_placeholder.error("❌ فشل تحليل البيوميكانيكا أو لم يتم التعرف على أي مقاييس.")
                    # No balloons for this one? Or maybe if Risk is Low?

            # Note: Cleanup handled implicitly

    # --- Display Biomechanics Results ---
    # --- Display Biomechanics Results ---
    # --- Display Biomechanics Results ---
       # --- Display Biomechanics Results ---
    # --- Display Biomechanics Results ---
    # --- Display Biomechanics Results ---
    # --- Display Biomechanics Results (Arabic Headers, English Data) ---
    # --- Display Biomechanics Results (Arabic Headers, English Table - LTR Order) ---
       # --- Display Biomechanics Results (Arabic, Simple Markdown, Separate Lines) ---
       # --- Display Biomechanics Results (Arabic Headers, English Data) ---
    if st.session_state.biomechanics_results:
        results_bio = st.session_state.biomechanics_results
        st.markdown("---")
        # --- KEEP ARABIC HEADER ---
        st.markdown("### 📊 نتائج التحليل البيوميكانيكي 📊") # Fallback

        st.markdown("---") # Add a visual separator

        # --- Display metric data in ENGLISH using st.write ---
        for key_en in BIOMECHANICS_METRICS_EN: # Iterate in defined order

            # Get ENGLISH Label
            display_label_en = BIOMECHANICS_LABELS_EN.get(key_en, key_en) # Use English labels dict

            # Get raw value (potentially numeric or Arabic text like 'غير واضح', 'منخفض')
            value_raw = results_bio.get(key_en, NOT_CLEAR_AR) # Default to original Arabic constant if key missing
            value_str = str(value_raw).strip().strip('\'"') # Clean the raw value

            # --- Translate known Arabic text values to ENGLISH for display ---
            display_value_en = BIO_VALUE_MAP_AR_TO_EN.get(value_str, value_str)
            # If value_str wasn't in the map (e.g., it's a number or unexpected text),
            # display_value_en will remain as the original cleaned value_str.

            # --- Display using simple st.write (LTR formatting is default/fine for English) ---
            # Use markdown for bolding the label
            st.write(f"**{display_label_en}:** {display_value_en}")

        # --- Display Risk Level and Score (Optionally, using st.metric with English Labels) ---
        # st.markdown("---") # Optional separator
        # risk_level_raw = results_bio.get("Risk_Level", NOT_CLEAR_AR)
        # risk_level_str = str(risk_level_raw).strip().strip('\'"')
        # risk_level_display_en = BIO_VALUE_MAP_AR_TO_EN.get(risk_level_str, risk_level_str)

        # risk_score_raw = results_bio.get("Risk_Score", NOT_CLEAR_AR)
        # risk_score_display_en = str(risk_score_raw).strip().strip('\'"') # Usually a number

        # col_risk1, col_risk2 = st.columns(2)
        # with col_risk1:
        #     # Use English label for metric
        #     st.metric("⚠️ Risk Level", risk_level_display_en)
        # with col_risk2:
        #      # Use English label for metric
        #     st.metric("🔢 Risk Score", risk_score_display_en)

# ==================================
# ==    الشخص المناسب Page (Placeholder) ==
# ==================================
elif st.session_state.page == PAGE_PERSON:
    st.markdown("---")
    st.markdown("## ✔️ الشخص المناسب في المكان المناسب ✔️")
    st.info("سيتم استخدام Gemini API لتحليل مجموعة بيانات (سيتم تحديدها لاحقاً) في هذه الميزة (قيد التطوير).")


# --- Footer ---
st.markdown("---")
st.caption("AI League - Scout Eye v1.3 (Gemini Powered - عربي) | بدعم من Google Gemini API")



# Put a small checkbox at the bottom-left
col_left, col_spacer, col_right = st.columns([1,4,1])
with col_left:
    show_advanced = st.checkbox("Advanced Gemini Options")

if show_advanced:
    st.write("### Choose a Gemini Model")
    gemini_models = [
        "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-pro-exp-03-25",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b"
    ]
    # Default selected = st.session_state.model_name if it's in the list
    default_index = gemini_models.index(st.session_state.model_name) \
        if st.session_state.model_name in gemini_models else 0

    chosen_model = st.selectbox(
        "Select a Gemini Model:",
        gemini_models,
        index=default_index
    )

    # Optionally, a "Test" button if you want to test the currently loaded model first
    if st.button("Test Current Model"):
        test_gemini_connection()

    # Button to *switch* the entire app to the newly chosen model
    if st.button("Use This Model"):
        st.session_state.model_name = chosen_model
        st.experimental_rerun()  # force a reload so the new model is loaded
