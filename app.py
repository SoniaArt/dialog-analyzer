import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Анализ диалогов", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2rem; text-align: center; margin-bottom: 1.5rem; }
    .problem-card { background: #ffebee; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .similar-card { background: #f5f5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Анализ диалогов</div>', unsafe_allow_html=True)

@st.cache_resource
def get_analyzer():
    from analyzer import DialogAnalyzer
    return DialogAnalyzer()

@st.cache_resource
def get_database():
    from database import DialogDatabase
    return DialogDatabase()

with st.sidebar:
    st.header("Загрузка")
    uploaded = st.file_uploader("CSV с колонкой 'dialog'", type=['csv'])
    
    if uploaded:
        try:
            content = uploaded.getvalue().decode('utf-8')
            df_preview = pd.read_csv(io.StringIO(content))

            if 'dialog' not in df_preview.columns:
                st.error("В CSV должна быть колонка 'dialog'")
                st.stop()
            if df_preview.empty:
                st.warning("Файл пустой. Добавьте строки с диалогами.")
                st.stop()

            df_preview['dialog'] = df_preview['dialog'].fillna('').astype(str)
            st.session_state.upload_content = content
            st.success(f"{len(df_preview)} диалогов")
            st.dataframe(df_preview, width="stretch", height=250)
            
            if st.button("Анализировать", type="primary", use_container_width=True):
                with st.spinner("Анализ..."):
                    analyzer = get_analyzer()
                    results = [analyzer.analyze_dialog(d) for d in df_preview['dialog']]
                    df_preview['тема'] = [r['topic'] for r in results]
                    df_preview['эмоция'] = [r['emotion'] for r in results]
                    df_preview['проблемный'] = [r['is_problem'] for r in results]
                    df_preview['тип_проблемы'] = [r['problem_type'] for r in results]
                    df_preview['критичность'] = [r['problem_severity'] for r in results]
                    df_preview['балл_проблемы'] = [r['problem_score'] for r in results]
                    df_preview['причина'] = [r['problem_reason'] for r in results]
                    st.session_state.df = df_preview
                    st.session_state.db_loaded = False
                    st.success("Готово!")
                    st.rerun()
        except UnicodeDecodeError:
            st.error("Не удалось прочитать файл в кодировке UTF-8. Сохраните CSV в UTF-8 и загрузите снова.")
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {e}")

if 'df' in st.session_state:
    df = st.session_state.df
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Всего", len(df))
    with c2: st.metric("Проблемных", df['проблемный'].sum())
    with c3: st.metric("Негативных", (df['эмоция'] == 'негативный').sum())
    with c4: st.metric("Позитивных", (df['эмоция'] == 'позитивный').sum())
    with c5: st.metric("Критичных", (df['критичность'] == 'критический').sum())

    export_buffer = io.StringIO()
    df.to_csv(export_buffer, index=False, encoding='utf-8-sig')
    st.download_button(
        "Скачать результаты CSV",
        data=export_buffer.getvalue(),
        file_name="dialog_analysis_results.csv",
        mime="text/csv",
        width="stretch",
    )
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["Диалоги", "Проблемные", "Статистика", "Поиск"])
    
    with tab1:
        display = df[['dialog', 'тема', 'эмоция', 'проблемный', 'тип_проблемы', 'критичность', 'балл_проблемы']].copy()
        display.index = range(1, len(display) + 1)
        st.dataframe(display, width="stretch", height=500)
    
    with tab2:
        problems = df[df['проблемный'] == True]
        if len(problems) > 0:
            for _, row in problems.iterrows():
                st.markdown(f"""
                <div class="problem-card">
                    <strong>{row['тип_проблемы']}</strong> | {row['критичность']} | балл: {row['балл_проблемы']}<br>
                    <b>Тема:</b> {row['тема']} | <b>Эмоция:</b> {row['эмоция']}<br>
                    <b>Причина:</b> {row['причина']}<br>
                    {row['dialog'][:200]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Нет проблемных")
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Темы")
            st.bar_chart(df['тема'].value_counts())
        with col2:
            st.subheader("Эмоции")
            st.bar_chart(df['эмоция'].value_counts())

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Критичность проблем")
            st.bar_chart(df['критичность'].value_counts())
        with col4:
            st.subheader("Типы проблем")
            problem_types = df[df['проблемный'] == True]['тип_проблемы'].value_counts()
            if len(problem_types) > 0:
                st.bar_chart(problem_types)
            else:
                st.info("Проблемных типов не найдено")

        st.subheader("Доля негатива по темам")
        negative_by_topic = (
            df.assign(is_negative=df['эмоция'] == 'негативный')
            .groupby('тема')['is_negative']
            .mean()
            .sort_values(ascending=False)
            * 100
        )
        st.bar_chart(negative_by_topic)
    
    with tab4:
        st.subheader("Поиск")
        query = st.text_input("Введите текст", placeholder="брак, доставка, возврат...")
        if query:
            with st.spinner("Поиск..."):
                db = get_database()
                if not st.session_state.get('db_loaded', False):
                    source = st.session_state.get('upload_content')
                    if source:
                        db.load_dialogs(io.StringIO(source))
                    else:
                        search_df = pd.DataFrame({'dialog': df['dialog'].astype(str)})
                        db.load_dialogs(io.StringIO(search_df.to_csv(index=False)))
                    st.session_state.db_loaded = True
                similar = db.find_similar(query, top_k=5)
                if similar:
                    for i, (text, score) in enumerate(similar, 1):
                        st.markdown(f"""
                        <div class="similar-card">
                            <strong>#{i} - {score}%</strong><br>
                            {text[:300]}...
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Ничего не найдено")

else:
    st.info("Загрузите CSV файл")