import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq

# ═══════════════════════════════════════
# إعداد الصفحة
# ═══════════════════════════════════════
st.set_page_config(
    page_title="Student Performance Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Analyzer")
st.markdown("Upload any student grades CSV and get AI-powered recommendations.")
st.divider()

# ═══════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Settings")
    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_..."
    )
    st.caption("Get your free key at console.groq.com")
    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload any CSV file")
    st.markdown("2. Select grade columns")
    st.markdown("3. Click Analyze")
    st.markdown("4. Get AI recommendations per course")

# ═══════════════════════════════════════
# رفع الملف
# ═══════════════════════════════════════
uploaded_file = st.file_uploader("📂 Upload Student Grades CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded: {len(df)} students, {len(df.columns)} columns")

    with st.expander("👀 Preview Data"):
        st.dataframe(df.head())

    st.divider()

    # ═══════════════════════════════════════
    # اختيار الأعمدة — تلقائي وفليكسبل
    # ═══════════════════════════════════════
    st.subheader("📋 Select Columns")

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols     = df.columns.tolist()

    selected_courses = st.multiselect(
        "Select course columns to analyze:",
        options=all_cols,
        default=numeric_cols
    )

    final_col = st.selectbox(
        "Select Final Grade column:",
        options=numeric_cols,
        index=len(numeric_cols) - 1
    )

    # حدود التصنيف
    st.subheader("🎯 Classification Thresholds")
    col1, col2 = st.columns(2)
    with col1:
        weak_threshold = st.slider(
            "Weak below:", 
            min_value=40, max_value=70, value=60
        )
    with col2:
        avg_threshold = st.slider(
            "Average below:", 
            min_value=60, max_value=90, value=75
        )

    # ═══════════════════════════════════════
    # زر التحليل
    # ═══════════════════════════════════════
    if st.button("🔍 Analyze & Get AI Recommendations", type="primary"):

        if not groq_key:
            st.error("⚠️ Please enter your Groq API Key in the sidebar!")
            st.stop()

        if len(selected_courses) < 2:
            st.error("⚠️ Please select at least 2 course columns!")
            st.stop()

        # ── حساب المتوسطات ──
        with st.spinner("Analyzing..."):
            averages = df[selected_courses].mean().sort_values()

            weak   = averages[averages < weak_threshold]
            medium = averages[
                (averages >= weak_threshold) & (averages < avg_threshold)
            ]
            good   = averages[averages >= avg_threshold]

            # تصنيف الطلاب
            def classify(grade):
                if grade < weak_threshold:   return 'Weak'
                elif grade < avg_threshold:  return 'Average'
                else:                        return 'Excellent'

            df['Level'] = df[final_col].apply(classify)
            counts = df['Level'].value_counts()

        # ═══════════════════════════════════════
        # النتائج
        # ═══════════════════════════════════════
        st.divider()
        st.subheader("📊 Results")

        # Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))
        c2.metric("🔴 Weak",
                  f"{counts.get('Weak',0)}",
                  f"{counts.get('Weak',0)/len(df)*100:.1f}%")
        c3.metric("🟡 Average",
                  f"{counts.get('Average',0)}",
                  f"{counts.get('Average',0)/len(df)*100:.1f}%")
        c4.metric("🟢 Excellent",
                  f"{counts.get('Excellent',0)}",
                  f"{counts.get('Excellent',0)/len(df)*100:.1f}%")

        st.divider()

        # ── الرسومات ──
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Course Performance**")
            colors = [
                '#E24B4A' if v < weak_threshold else
                '#EF9F27' if v < avg_threshold else
                '#1D9E75'
                for v in averages.values
            ]
            fig1, ax1 = plt.subplots(figsize=(8, max(5, len(selected_courses)*0.4)))
            bars = ax1.barh(averages.index, averages.values,
                            color=colors, height=0.6)
            for bar, val in zip(bars, averages.values):
                ax1.text(val + 0.5,
                         bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}', va='center', fontsize=9)
            ax1.axvline(x=averages.mean(), color='#378ADD',
                        linestyle='--', linewidth=1.5)
            ax1.set_xlim(0, 110)
            ax1.set_xlabel('Average Score')
            red   = mpatches.Patch(color='#E24B4A',
                                   label=f'Weak (<{weak_threshold})')
            amber = mpatches.Patch(color='#EF9F27',
                                   label=f'Medium ({weak_threshold}-{avg_threshold})')
            green = mpatches.Patch(color='#1D9E75',
                                   label=f'Good (>{avg_threshold})')
            ax1.legend(handles=[red, amber, green], fontsize=9)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig1)

        with col_r:
            st.markdown("**Student Distribution**")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            level_colors = {
                'Weak':      '#E24B4A',
                'Average':   '#EF9F27',
                'Excellent': '#1D9E75'
            }
            pie_colors = [level_colors.get(l, '#888') for l in counts.index]
            ax2.pie(
                counts.values,
                labels=counts.index,
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            plt.tight_layout()
            st.pyplot(fig2)

        # ── تفاصيل المواد ──
        st.divider()
        col_w, col_m, col_g = st.columns(3)

        with col_w:
            st.markdown("**🔴 Weak Courses**")
            if len(weak) > 0:
                for c, v in weak.items():
                    st.error(f"{c}: {v:.1f}")
            else:
                st.success("None!")

        with col_m:
            st.markdown("**🟡 Medium Courses**")
            if len(medium) > 0:
                for c, v in medium.items():
                    st.warning(f"{c}: {v:.1f}")
            else:
                st.success("None!")

        with col_g:
            st.markdown("**🟢 Good Courses**")
            for c, v in good.items():
                st.success(f"{c}: {v:.1f}")

        # ═══════════════════════════════════════
        # AI Recommendations — خاصة بكل مادة
        # ═══════════════════════════════════════
        st.divider()
        st.subheader("🤖 AI Recommendations per Course")

        with st.spinner("Generating personalized recommendations..."):
            try:
                client_groq = Groq(api_key=groq_key)

                # بناء قائمة تفصيلية لكل مادة
                course_details = []
                for course in selected_courses:
                    avg  = df[course].mean()
                    low  = (df[course] < weak_threshold).sum()
                    pct  = low / len(df) * 100
                    course_details.append(
                        f"- {course}: avg={avg:.1f}, "
                        f"students below {weak_threshold}: {low} ({pct:.1f}%)"
                    )

                prompt = f"""
You are an expert academic advisor analyzing real student performance data.

DATASET: {len(df)} students, {len(selected_courses)} courses

DETAILED COURSE ANALYSIS:
{chr(10).join(course_details)}

STUDENT LEVELS (Final Grade: {final_col}):
- Weak (below {weak_threshold}): {counts.get('Weak',0)} students ({counts.get('Weak',0)/len(df)*100:.1f}%)
- Average ({weak_threshold}-{avg_threshold}): {counts.get('Average',0)} students ({counts.get('Average',0)/len(df)*100:.1f}%)
- Excellent (above {avg_threshold}): {counts.get('Excellent',0)} students ({counts.get('Excellent',0)/len(df)*100:.1f}%)

WEAK COURSES IDENTIFIED: {list(weak.index)}
MEDIUM COURSES IDENTIFIED: {list(medium.index)}

For EACH weak course specifically, provide:
1. Likely reasons students struggle in THIS specific course
2. Concrete teaching methods to improve pass rate for THIS course
3. Specific tools, resources, or projects for THIS course

Then provide:
4. Top 3 new courses to add based on the weak areas identified
5. One overall priority action plan

Be specific to each course name. Do not give generic advice.
"""

                response = client_groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000
                )

                recommendations = response.choices[0].message.content
                st.markdown(recommendations)

                # ── تنزيل التقرير ──
                st.divider()
                report  = "STUDENT PERFORMANCE REPORT\n"
                report += "=" * 50 + "\n\n"
                report += f"Total Students: {len(df)}\n"
                report += f"Courses Analyzed: {len(selected_courses)}\n"
                report += f"Weak threshold: {weak_threshold} | "
                report += f"Average threshold: {avg_threshold}\n\n"
                report += "COURSE AVERAGES:\n"
                for c, v in averages.items():
                    report += f"  {c}: {v:.1f}\n"
                report += "\n" + "=" * 50 + "\n"
                report += "AI RECOMMENDATIONS\n"
                report += "=" * 50 + "\n\n"
                report += recommendations

                st.download_button(
                    label="📥 Download Full Report",
                    data=report,
                    file_name="student_performance_report.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"❌ Groq Error: {str(e)}")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📂 Upload any CSV file")
    with col2:
        st.info("🎯 Select your grade columns")
    with col3:
        st.info("🤖 Get AI recommendations")
```

بعد ما تنسخي اضغطي:
```
Commit changes → Commit directly to main
