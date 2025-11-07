import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import sys

# 忽略常见的无关警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 兼容某些环境里已弃用的 np.bool
if not hasattr(np, 'bool'):
    np.bool = bool

# 页面配置
st.set_page_config(
    page_title="Prediction Model (Random Forest)",
    page_icon="🩺",
    layout="wide"
)

# 训练模型使用的特征（顺序必须与训练一致）
FEATURES = [
    '从发血到输血时间','是否有原发性血液疾病','年龄','TT','血液储存时间','有无不良反应史','过敏史','科室'
]

# 英文显示标签（仅影响界面显示，不影响编码）
FEATURE_LABELS = {
    '从发血到输血时间':'The time from blood release to blood transfusion' ,
    '年龄':'Age',
    'TT':'TT',
    '是否有原发性血液疾病':'Is there any primary blood disease?',
    '血液储存时间':'Blood storage time',
    '有无不良反应史': 'Have any adverse reactions occurred?',
    '过敏史': 'allergic history',
    '科室': 'administrative or technical offices',
}

# 字段说明（侧栏说明文字）
FEATURE_DESC = {
}

# 选项集合与显示格式化函数
YES_NO_OPTIONS = [0, 1]
YES_NO_FMT = lambda x: "No" if x == 0 else "Yes"

LEVEL2_OPTIONS = [0, 1]  # 0=Low, 1=Medium, 2=High
LEVEL2_FMT = lambda x: {0: "Less than 30 minutes", 1: "More than 30 minutes"}[x]

LEVEL6_OPTIONS = [0, 1, 2, 3, 4, 5]
LEVEL6_FMT = lambda x: {0: "ICU", 1: "Surgery department", 2: "General internal medicine department",
                        3: "High-risk internal medicine", 4: "Emergency department", 5: "Others"}[x]

LEVEL4_OPTIONS = [0, 1, 2, 3]
LEVEL4_FMT = lambda x: {0: "Less than 2 weeks", 1: "More than 2 weeks", 2: "More than 2 days", 3: "Less than 2 days"}[x]


# 加载模型；为部分环境提供 numpy._core 兼容兜底
@st.cache_resource
def load_model():
    model_path = 'model.pkl'
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            import numpy as _np
            sys.modules['numpy._core'] = _np.core
            sys.modules['numpy._core._multiarray_umath'] = _np.core._multiarray_umath
            sys.modules['numpy._core.multiarray'] = _np.core.multiarray
            sys.modules['numpy._core.umath'] = _np.core.umath
            return joblib.load(model_path)
        raise


def main():
    st.sidebar.title("Prediction Model (Random Forest)")
    st.sidebar.markdown(
        "- Predicts risk of adverse reactions to blood transfusion occur after the transfusion using 8 features.\n"
        "- Binary classification model (Random Forest)."
    )

    # 侧栏：展开的“特征与说明”
    with st.sidebar.expander("Features & Notes"):
        for k in FEATURES:
            st.markdown(f"- {FEATURE_LABELS.get(k,k)}: {FEATURE_DESC.get(k,'')}")

    AGE_MIN = 1
    AGE_MAX = 95
    TT_MIN = 1
    TT_MAX = 70

    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    # 页面标题与说明
    st.title("Risk Prediction")
    st.markdown("Enter the inputs below and click Predict.")

    # 三列布局：分组输入控件
    col1, col2, col3 = st.columns(3)

    with col1:
        从发血到输血时间 = st.selectbox(
            FEATURE_LABELS['从发血到输血时间'], LEVEL2_OPTIONS, format_func=LEVEL2_FMT
        )
        是否有原发性血液疾病 = st.selectbox(
            FEATURE_LABELS['是否有原发性血液疾病'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        年龄 = st.slider(
            "Age",
            min_value= int(AGE_MAX),
            max_value=int(AGE_MAX),
            value=int(AGE_MIN),
            step=1
        )
        年龄_raw = 年龄


    with col2:
        TT = st.number_input(
            "TT",
            min_value = TT_MIN,
            max_value = TT_MAX
        )
        TT_raw = TT
        血液储存时间 = st.selectbox(
            FEATURE_LABELS['血液储存时间'], LEVEL4_OPTIONS, format_func=LEVEL4_FMT
        )
        有无不良反应史 = st.selectbox(
            FEATURE_LABELS['有无不良反应史'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )


    with col3:
        过敏史 = st.selectbox(
            FEATURE_LABELS['过敏史'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        科室 = st.selectbox(
            FEATURE_LABELS['科室'], LEVEL6_OPTIONS, format_func=LEVEL6_FMT
        )


    if st.button("Predict"):
        # 按训练顺序组装输入行
        row = [
            从发血到输血时间, 是否有原发性血液疾病, 年龄_raw, TT_raw, 血液储存时间, 有无不良反应史, 过敏史, 科室
        ]
        input_df = pd.DataFrame([row], columns=FEATURES)

        try:
            proba = model.predict_proba(input_df)[0]
            pred = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # 提示预测类别与概率
        st.subheader("Prediction Result")
        st.markdown(
                f"Based on feature values, predicted possibility of 'Whether adverse reactions to blood transfusion occur after the transfusion' is: <span style='color:red;'>{proba[1] * 100:.2f}%</span>  \n"
                "When using this model to evaluate the risk of 'Whether adverse reactions to blood transfusion occur after the transfusion', "
                "we recommend that the optimal threshold value be set at 30%.  \n"
                "Please note: This prediction is generated by a machine learning model to assist your decision-making. "
                "It should not replace your professional judgment in evaluating the patient.",
                unsafe_allow_html=True
        )

        # SHAP 可解释性
        st.write("---")
        st.subheader("Explainability (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_df)

            # 兼容不同 shap 版本的返回格式
            if isinstance(sv, list):
                shap_value = np.array(sv[1][0])  # class 1 contribution
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                shap_value = sv[0]
                expected_value = explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_value = sv[0, :, 1]
                expected_value = explainer.expected_value[1]
            else:
                raise RuntimeError("Unrecognized SHAP output format")

            # 力导向图（Force Plot）
            try:
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES],
                    matplotlib=True,
                    show=False,
                    figsize=(20, 3)
                )
                st.pyplot(force_plot)
            except Exception as e:
                st.error(f"Force plot failed: {e}")
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")




if __name__ == "__main__":
    main()
