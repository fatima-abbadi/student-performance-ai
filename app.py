import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from groq import Groq

# ═══════════════════════════════
# إعدادات الصفحة
# ═══════════════════════════════
st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance AI Analyzer")
st.markdown("Upload your student grades CSV and get AI-powered insights.")
st.divider()

# ═══════════════════════════════
# Sidebar — الإعدادات
# ═══════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input("Groq API Key", type="password", 
                              placeholder="gsk_...")
    st.caption("Get your free key at console.groq.com")
    st.divider()
    weak_threshold  = st.slider("Weak course threshold",   40, 70, 60)
    medium_threshold = st.slider("Medium course threshold", 60, 85, 70)
    st.divider()
    st.caption("Weak < threshold < Medium < Good")

# ═══════════════════════════════
# رفع الملف
# ═══════════════════════════════
uploaded_file = st.file_uploader("📂 Upload Student Grades CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # تحديد المواد تلقائياً
    exclude = ['StudentID','Rank','Semester_1_Grade',
               'Semester_2_Grade','Final_Grade']
    courses = [c for c in df.columns if c not in exclude 
               and df[c].dtype in ['float64','int64']]

    st.success(f"✅ Loaded {len(df)} students | {len(courses)} courses detected")

    # ═══════════════════════════════
    # تبويبات
    # ═══════════════════════════════
    tab1, tab2, tab3 = st.tabs(["📊 Analysis", "🤖 ML Model", "💡 AI Recommendations"])

    # ───────────────────────────────
    # تبويب 1 — التحليل
    # ───────────────────────────────
    with tab1:
        st.subheader("Course Performance Overview")

        averages = df[courses].mean().sort_values()

        # تلوين المواد
        colors = []
        for avg in averages:
            if avg < weak_threshold:
                colors.append('#E24B4A')
            elif avg < medium_threshold:
                colors.append('#EF9F27')
            else:
                colors.append('#1D9E75')

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(averages.index, averages.values, color=colors, height=0.6)

        overall_avg = averages.mean()
        ax.axvline(x=overall_avg, color='#378ADD', linestyle='--', 
                   linewidth=1.5, label=f'Overall Avg: {overall_avg:.1f}')

        for bar, val in zip(bars, averages.values):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontsize=9)

        ax.set_xlim(0, 110)
        ax.set_xlabel('Average Score')
        ax.set_title('Student Performance by Course')

        red   = mpatches.Patch(color='#E24B4A', label=f'Weak (< {weak_threshold})')
        amber = mpatches.Patch(color='#EF9F27', label=f'Medium ({weak_threshold}-{medium_threshold})')
        green = mpatches.Patch(color='#1D9E75', label=f'Good (> {medium_threshold})')
        ax.legend(handles=[red, amber, green, ax.lines[0]])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

        # إحصاءات سريعة
        weak_courses   = averages[averages < weak_threshold].index.tolist()
        medium_courses = averages[(averages >= weak_threshold) & 
                                  (averages < medium_threshold)].index.tolist()
        good_courses   = averages[averages >= medium_threshold].index.tolist()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔴 Weak Courses",   len(weak_courses))
        with col2:
            st.metric("🟡 Medium Courses", len(medium_courses))
        with col3:
            st.metric("🟢 Good Courses",   len(good_courses))

        if weak_courses:
            st.error(f"⚠️ Weak courses: {', '.join(weak_courses)}")

    # ───────────────────────────────
    # تبويب 2 — ML Model
    # ───────────────────────────────
    with tab2:
        st.subheader("ML Model — Student Level Prediction")

        if 'Final_Grade' in df.columns:
            X     = df[courses]
            y_raw = df['Final_Grade']

            X_train, X_test, y_train_raw, y_test_raw = train_test_split(
                X, y_raw, test_size=0.2, random_state=42
            )

            def classify(grade):
                if grade < 60:   return 'Weak'
                elif grade < 75: return 'Average'
                else:            return 'Excellent'

            y_train = y_train_raw.apply(classify)
            y_test  = y_test_raw.apply(classify)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred   = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.metric("🎯 Model Accuracy", f"{accuracy*100:.1f}%")

            # Feature importance
            importances = model.feature_importances_
            indices     = np.argsort(importances)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            imp_colors = ['#E24B4A' if importances[i] > 0.08 else
                          '#EF9F27' if importances[i] > 0.04 else
                          '#1D9E75' for i in indices]

            ax2.barh([courses[i] for i in indices],
                     importances[indices], color=imp_colors, height=0.6)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Which Courses Affect Student Level the Most?')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)

            # تنبؤ بطالب جديد
            st.divider()
            st.subheader("🔍 Predict a New Student")
            st.caption("Enter grades for a new student to predict their level")

            cols     = st.columns(3)
            new_data = {}
            for i, course in enumerate(courses):
                with cols[i % 3]:
                    new_data[course] = st.number_input(
                        course.replace('_',' '), 
                        min_value=0.0, max_value=100.0, 
                        value=70.0, step=0.5
                    )

            if st.button("🔮 Predict Student Level"):
                new_df     = pd.DataFrame([new_data])
                prediction = model.predict(new_df)[0]
                colors_map = {'Weak':'🔴','Average':'🟡','Excellent':'🟢'}
                st.success(f"Predicted Level: {colors_map[prediction]} **{prediction}**")

        else:
            st.warning("⚠️ No Final_Grade column found — ML model needs it.")

    # ───────────────────────────────
    # تبويب 3 — توصيات AI
    # ───────────────────────────────
    with tab3:
        st.subheader("💡 AI-Powered Recommendations")

        if not groq_key:
            st.warning("⚠️ Please enter your Groq API Key in the sidebar first.")
        else:
            if st.button("🚀 Generate AI Recommendations"):
                with st.spinner("Analyzing with AI... please wait"):
                    try:
                        client_groq = Groq(api_key=groq_key)

                        avg_dict = {c: round(float(averages[c]),1) 
                                    for c in courses if c in averages.index}

                        prompt = f"""
You are an expert academic advisor for a Computer Science department.

Student data: {len(df)} students, {len(courses)} courses.

COURSE AVERAGES:
{avg_dict}

WEAK COURSES (below {weak_threshold}): {weak_courses}
MEDIUM COURSES ({weak_threshold}-{medium_threshold}): {medium_courses}

Please provide:
1. Why are students struggling in the weak courses?
2. Specific teaching improvements for EACH weak course
3. Top 5 new courses to add for better job market readiness
4. Priority action plan (short, medium, long term)

Be specific, practical, and professional.
"""
                        response = client_groq.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role":"user","content":prompt}],
                            max_tokens=1500
                        )
                        result = response.choices[0].message.content
                        st.markdown(result)

                        # تنزيل التقرير
                        st.download_button(
                            label="📥 Download Report",
                            data=result,
                            file_name="AI_Academic_Report.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

else:
    st.info("👆 Please upload a CSV file to get started.")
