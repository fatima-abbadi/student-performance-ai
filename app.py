import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq

st.set_page_config(
    page_title="CS Department Analyzer",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CS Department Analyzer")
st.markdown("Analyze student performance and job market trends to improve your curriculum.")
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
    st.markdown("**Tabs:**")
    st.markdown("📊 Tab 1 — Analyze student grades")
    st.markdown("💼 Tab 2 — Analyze job market & update curriculum")

# ═══════════════════════════════════════
# Tabs
# ═══════════════════════════════════════
tab1, tab2 = st.tabs(["📊 Student Performance", "💼 Job Market & Curriculum"])

# ╔═══════════════════════════════════════╗
# ║           TAB 1 — STUDENTS            ║
# ╚═══════════════════════════════════════╝
with tab1:
    st.subheader("📊 Student Performance Analysis")
    st.markdown("Upload student grades CSV to analyze performance and get AI recommendations.")

    uploaded_students = st.file_uploader(
        "📂 Upload Student Grades CSV",
        type="csv",
        key="students"
    )

    if uploaded_students:
        df = pd.read_csv(uploaded_students)
        st.success(f"✅ Loaded: {len(df)} students, {len(df.columns)} columns")

        with st.expander("👀 Preview Data"):
            st.dataframe(df.head())

        st.divider()

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols     = df.columns.tolist()

        selected_courses = st.multiselect(
            "Select course columns:",
            options=all_cols,
            default=numeric_cols,
            key="courses_select"
        )

        final_col = st.selectbox(
            "Select Final Grade column:",
            options=numeric_cols,
            index=len(numeric_cols) - 1,
            key="final_col"
        )

        col1, col2 = st.columns(2)
        with col1:
            weak_threshold = st.slider("Weak below:", 40, 70, 60, key="weak_t")
        with col2:
            avg_threshold  = st.slider("Average below:", 60, 90, 75, key="avg_t")

        if st.button("🔍 Analyze Students", type="primary", key="btn_students"):

            if not groq_key:
                st.error("⚠️ Enter your Groq API Key in the sidebar!")
                st.stop()
            if len(selected_courses) < 2:
                st.error("⚠️ Select at least 2 courses!")
                st.stop()

            with st.spinner("Analyzing..."):
                averages = df[selected_courses].mean().sort_values()
                weak     = averages[averages < weak_threshold]
                medium   = averages[(averages >= weak_threshold) & (averages < avg_threshold)]
                good     = averages[averages >= avg_threshold]

                def classify(g):
                    if g < weak_threshold:  return 'Weak'
                    elif g < avg_threshold: return 'Average'
                    else:                   return 'Excellent'

                df['Level'] = df[final_col].apply(classify)
                counts = df['Level'].value_counts()

            st.divider()
            st.subheader("📊 Results")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Students", len(df))
            c2.metric("🔴 Weak",      f"{counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.1f}%)")
            c3.metric("🟡 Average",   f"{counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.1f}%)")
            c4.metric("🟢 Excellent", f"{counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.1f}%)")

            st.divider()
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
                bars = ax1.barh(averages.index, averages.values, color=colors, height=0.6)
                for bar, val in zip(bars, averages.values):
                    ax1.text(val+0.5, bar.get_y()+bar.get_height()/2,
                             f'{val:.1f}', va='center', fontsize=9)
                ax1.axvline(x=averages.mean(), color='#378ADD',
                            linestyle='--', linewidth=1.5)
                ax1.set_xlim(0, 110)
                ax1.set_xlabel('Average Score')
                red   = mpatches.Patch(color='#E24B4A', label=f'Weak (<{weak_threshold})')
                amber = mpatches.Patch(color='#EF9F27', label=f'Medium ({weak_threshold}-{avg_threshold})')
                green = mpatches.Patch(color='#1D9E75', label=f'Good (>{avg_threshold})')
                ax1.legend(handles=[red, amber, green], fontsize=9)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig1)

            with col_r:
                st.markdown("**Student Distribution**")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                lc = {'Weak':'#E24B4A','Average':'#EF9F27','Excellent':'#1D9E75'}
                ax2.pie(counts.values,
                        labels=counts.index,
                        colors=[lc.get(l,'#888') for l in counts.index],
                        autopct='%1.1f%%', startangle=90,
                        wedgeprops={'edgecolor':'white','linewidth':2})
                plt.tight_layout()
                st.pyplot(fig2)

            st.divider()
            cw, cm, cg = st.columns(3)
            with cw:
                st.markdown("**🔴 Weak**")
                for c, v in weak.items():   st.error(f"{c}: {v:.1f}")
            with cm:
                st.markdown("**🟡 Medium**")
                for c, v in medium.items(): st.warning(f"{c}: {v:.1f}")
            with cg:
                st.markdown("**🟢 Good**")
                for c, v in good.items():   st.success(f"{c}: {v:.1f}")

            st.divider()
            st.subheader("🤖 AI Recommendations per Course")

            with st.spinner("Generating AI recommendations..."):
                try:
                    client_groq = Groq(api_key=groq_key)
                    details = [
                        f"- {c}: avg={df[c].mean():.1f}, "
                        f"students below {weak_threshold}: "
                        f"{(df[c]<weak_threshold).sum()} "
                        f"({(df[c]<weak_threshold).sum()/len(df)*100:.1f}%)"
                        for c in selected_courses
                    ]
                    prompt = f"""
You are an expert academic advisor for a CS department.
{len(df)} students analyzed.

COURSE DETAILS:
{chr(10).join(details)}

WEAK COURSES: {list(weak.index)}
MEDIUM COURSES: {list(medium.index)}

STUDENT LEVELS:
- Weak: {counts.get('Weak',0)} ({counts.get('Weak',0)/len(df)*100:.1f}%)
- Average: {counts.get('Average',0)} ({counts.get('Average',0)/len(df)*100:.1f}%)
- Excellent: {counts.get('Excellent',0)} ({counts.get('Excellent',0)/len(df)*100:.1f}%)

For EACH weak course specifically provide:
1. Why students struggle in THIS course
2. Concrete teaching improvements for THIS course
3. Specific tools or projects for THIS course

Be specific to each course name. No generic advice.
"""
                    resp = client_groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=2000
                    )
                    recommendations = resp.choices[0].message.content
                    st.markdown(recommendations)

                    st.divider()
                    report  = "STUDENT PERFORMANCE REPORT\n" + "="*50 + "\n\n"
                    report += f"Students: {len(df)} | Courses: {len(selected_courses)}\n\n"
                    report += "COURSE AVERAGES:\n"
                    for c, v in averages.items():
                        report += f"  {c}: {v:.1f}\n"
                    report += "\n" + "="*50 + "\nAI RECOMMENDATIONS\n" + "="*50 + "\n\n"
                    report += recommendations

                    st.download_button(
                        "📥 Download Report",
                        data=report,
                        file_name="student_report.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ {str(e)}")
    else:
        st.info("👆 Upload a student grades CSV to get started.")

# ╔═══════════════════════════════════════╗
# ║        TAB 2 — JOB MARKET            ║
# ╚═══════════════════════════════════════╝
with tab2:
    st.subheader("💼 Job Market Analysis & Curriculum Update")
    st.markdown("Upload job listings CSV to find skill gaps and get curriculum recommendations.")

    # الخطة الدراسية الحالية
    CURRENT_PLAN = {
        "Programming & Software": [
            "Computer Programming", "Object Oriented Programming",
            "Data Structures", "Algorithms", "Software Engineering",
            "Operating Systems", "Programming Languages Theory"
        ],
        "Networks & Security": [
            "Computer Networks", "Network Programming",
            "IT Security", "Internet Technologies"
        ],
        "Data & AI": [
            "Databases", "Information Systems Analysis",
            "Artificial Intelligence", "Big Data Management"
        ],
        "Hardware & Systems": [
            "Digital Electronics", "Computer Architecture",
            "Microprocessors", "Signals & Systems",
            "Control Systems"
        ],
        "Math & Science": [
            "Calculus 1 & 2", "Engineering Mathematics",
            "Discrete Mathematics", "Numerical Analysis",
            "Probability & Statistics", "Physics"
        ],
        "Electives": [
            "IoT Platforms", "Cloud Computing",
            "Python Programming", "Data Sensors"
        ]
    }

    # عرض الخطة الحالية
    with st.expander("📚 View Current Curriculum Plan"):
        for category, courses in CURRENT_PLAN.items():
            st.markdown(f"**{category}**")
            cols = st.columns(3)
            for i, course in enumerate(courses):
                cols[i % 3].info(course)

    st.divider()

    uploaded_jobs = st.file_uploader(
        "📂 Upload Job Listings CSV (e.g. Dice.com)",
        type="csv",
        key="jobs"
    )

    if uploaded_jobs:
        with st.spinner("Loading job data..."):
            jobs_df = pd.read_csv(uploaded_jobs, on_bad_lines='skip')
        st.success(f"✅ Loaded: {len(jobs_df)} job listings")

        with st.expander("👀 Preview Jobs Data"):
            st.dataframe(jobs_df.head(3))

        # اختيار عمود الوصف
        text_cols = jobs_df.select_dtypes(include='object').columns.tolist()
        desc_col  = st.selectbox(
            "Select job description column:",
            options=text_cols,
            index=text_cols.index('jobdescription') if 'jobdescription' in text_cols else 0
        )

        if st.button("🔍 Analyze Job Market & Update Curriculum", type="primary", key="btn_jobs"):

            if not groq_key:
                st.error("⚠️ Enter your Groq API Key in the sidebar!")
                st.stop()

            with st.spinner("Extracting skills from job listings..."):

                # قائمة الـ skills المطلوبة في سوق العمل
                SKILLS = {
                    "Languages":    ["python","java","javascript","c++","c#","kotlin","swift","go","rust","scala","php","ruby","typescript"],
                    "Web & Mobile": ["react","angular","vue","node","django","flask","spring","html","css","android","ios","flutter"],
                    "Data & AI":    ["machine learning","deep learning","tensorflow","pytorch","pandas","numpy","sql","nosql","mongodb","spark","hadoop","tableau","power bi","data science"],
                    "Cloud & DevOps":["aws","azure","google cloud","docker","kubernetes","jenkins","ci/cd","terraform","linux","git","devops"],
                    "Security":     ["cybersecurity","penetration testing","network security","encryption","iam","firewall","siem","ethical hacking"],
                    "Other":        ["agile","scrum","rest api","microservices","blockchain","iot","embedded systems"]
                }

                # حساب تكرار كل skill
                skill_counts = {}
                desc_text = jobs_df[desc_col].dropna().str.lower().str.cat(sep=' ')

                for category, skills in SKILLS.items():
                    for skill in skills:
                        count = desc_text.count(skill)
                        if count > 0:
                            skill_counts[skill] = {
                                'count': count,
                                'category': category
                            }

                # ترتيب من الأعلى للأدنى
                sorted_skills = sorted(
                    skill_counts.items(),
                    key=lambda x: x[1]['count'],
                    reverse=True
                )
                top_skills = sorted_skills[:30]

            st.divider()
            st.subheader("📊 Top Skills Demanded by Job Market")

            # رسم Top 20 Skills
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            skills_names  = [s[0] for s in top_skills[:20]]
            skills_counts = [s[1]['count'] for s in top_skills[:20]]
            skills_cats   = [s[1]['category'] for s in top_skills[:20]]

            cat_colors = {
                'Languages':     '#378ADD',
                'Web & Mobile':  '#1D9E75',
                'Data & AI':     '#7F77DD',
                'Cloud & DevOps':'#EF9F27',
                'Security':      '#E24B4A',
                'Other':         '#888780'
            }
            bar_colors = [cat_colors.get(c,'#888') for c in skills_cats]

            bars3 = ax3.barh(skills_names, skills_counts, color=bar_colors, height=0.6)
            for bar, val in zip(bars3, skills_counts):
                ax3.text(val + 0.5, bar.get_y()+bar.get_height()/2,
                         str(val), va='center', fontsize=9)

            ax3.set_xlabel('Mentions in Job Listings')
            ax3.set_title('Top 20 In-Demand Skills', fontsize=13, fontweight='bold')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)

            # Legend
            handles = [mpatches.Patch(color=v, label=k) for k,v in cat_colors.items()]
            ax3.legend(handles=handles, fontsize=9, loc='lower right')
            plt.tight_layout()
            st.pyplot(fig3)

            # إحصاءات سريعة
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Jobs",         len(jobs_df))
            col2.metric("Skills Found",       len(skill_counts))
            col3.metric("Top Skill",          top_skills[0][0] if top_skills else "-")
            col4.metric("Top Skill Mentions", top_skills[0][1]['count'] if top_skills else 0)

            # توزيع الـ skills بالفئات
            st.divider()
            st.markdown("**Skills by Category**")
            cat_totals = {}
            for _, data in skill_counts.items():
                cat = data['category']
                cat_totals[cat] = cat_totals.get(cat, 0) + data['count']

            cat_df = pd.DataFrame(
                list(cat_totals.items()),
                columns=['Category','Total Mentions']
            ).sort_values('Total Mentions', ascending=False)

            for _, row in cat_df.iterrows():
                pct = row['Total Mentions'] / sum(cat_totals.values()) * 100
                st.progress(
                    int(pct),
                    text=f"**{row['Category']}** — {row['Total Mentions']:,} mentions ({pct:.1f}%)"
                )

            # ═══════════════════════════════════════
            # AI — مقارنة مع الخطة وتوصيات
            # ═══════════════════════════════════════
            st.divider()
            st.subheader("🤖 AI Curriculum Recommendations")

            with st.spinner("Comparing with current curriculum..."):
                try:
                    client_groq = Groq(api_key=groq_key)

                    top20_str = "\n".join([
                        f"- {s[0]} ({s[1]['category']}): {s[1]['count']} mentions"
                        for s in top_skills[:20]
                    ])

                    current_courses_str = "\n".join([
                        f"{cat}: {', '.join(courses)}"
                        for cat, courses in CURRENT_PLAN.items()
                    ])

                    prompt = f"""
You are an expert CS curriculum designer.

CURRENT UNIVERSITY CURRICULUM:
{current_courses_str}

TOP 20 SKILLS DEMANDED BY JOB MARKET (from {len(jobs_df)} real job listings):
{top20_str}

Based on this real data, provide a detailed report with:

1. SKILL GAPS: Which high-demand skills are NOT covered in the current curriculum?

2. COURSE UPDATES: For each existing course, what specific skills should be added?
   Format: Course Name → Add: [skill1, skill2, skill3]

3. NEW COURSES TO ADD: What new courses should be added to the curriculum?
   For each new course provide:
   - Course name
   - Skills it covers
   - Why it's important (job market demand evidence)
   - Suggested credit hours

4. PRIORITY ORDER: List recommendations from most urgent to least urgent.

Be specific. Reference the actual job market numbers. No generic advice.
"""

                    resp = client_groq.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=2500
                    )
                    curr_recommendations = resp.choices[0].message.content
                    st.markdown(curr_recommendations)

                    # تنزيل التقرير
                    st.divider()
                    report2  = "JOB MARKET & CURRICULUM REPORT\n" + "="*50 + "\n\n"
                    report2 += f"Job Listings Analyzed: {len(jobs_df)}\n"
                    report2 += f"Skills Found: {len(skill_counts)}\n\n"
                    report2 += "TOP SKILLS:\n"
                    for s, d in top_skills[:20]:
                        report2 += f"  {s}: {d['count']} mentions\n"
                    report2 += "\n" + "="*50 + "\nAI RECOMMENDATIONS\n" + "="*50 + "\n\n"
                    report2 += curr_recommendations

                    st.download_button(
                        "📥 Download Curriculum Report",
                        data=report2,
                        file_name="curriculum_report.txt",
                        mime="text/plain"
                    )

                except Exception as e:
                    st.error(f"❌ {str(e)}")

    else:
        st.info("👆 Upload a job listings CSV to analyze market demands.")
