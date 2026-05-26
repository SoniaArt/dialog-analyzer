import streamlit as st
import pandas as pd
import io
from database import DialogDatabase
from analyzer import DialogAnalyzer
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

st.set_page_config(page_title="Анализ диалогов", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2rem; text-align: center; margin-bottom: 1.5rem; }
    .problem-card { background: #ffebee; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    .similar-card { background: #f5f5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; }
    div[data-testid="stMetricValue"] { color: #000000 !important; }

    /* Меняем текст кнопки Upload */
    div[data-testid="stFileUploader"] button div p {
        font-size: 0;
    }

    div[data-testid="stFileUploader"] button div p::after {
        content: "Загрузить";
        font-size: 16px;
    }

    .stats-box {
        background: #f8f9fa;
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }

    .stats-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
    }

    .stats-name {
        font-size: 16px;
        font-weight: 500;
        color: #1f2937;
    }

    .stats-right {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .stats-count {
        background: #2563eb;
        color: white;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
    }

    .stats-percent {
        color: #6b7280;
        font-size: 14px;
        min-width: 55px;
        text-align: right;
    }

    .similar-card {
        background: #f5f7fa;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 16px;
    }
    
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Анализ диалогов</div>', unsafe_allow_html=True)

if 'loaded_count' in st.session_state:
    st.success(
        f"В ChromaDB сохранено {st.session_state.loaded_count} диалогов."
    )

@st.cache_resource
def init():
    return DialogDatabase(), DialogAnalyzer()

db, analyzer = init()

with st.sidebar:
    st.header("Загрузка")
    uploaded = st.file_uploader("CSV с колонкой 'dialog'", type=['csv'])
    
    if uploaded:
        content = uploaded.getvalue().decode('utf-8')
        df_preview = pd.read_csv(io.StringIO(content))
        st.success(f"{len(df_preview)} диалогов")
        st.dataframe(df_preview, use_container_width=True, height=250)
        
        if st.button("Анализировать", type="primary", use_container_width=True):
            with st.spinner("Анализ..."):
                loaded_count = db.load_dialogs(io.StringIO(content))
                results = [analyzer.analyze_dialog(d) for d in df_preview['dialog']]
                df_preview['тема'] = [r['topic'] for r in results]
                df_preview['эмоция'] = [r['emotion'] for r in results]
                df_preview['проблемный'] = [r['is_problem'] for r in results]
                st.session_state.df = df_preview
                st.session_state.loaded_count = loaded_count
                st.rerun()

if 'df' in st.session_state:
    df = st.session_state.df
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Всего", len(df))
    with c2: st.metric("Проблемных", df['проблемный'].sum())
    with c3: st.metric("Негативных", (df['эмоция'] == 'негативный').sum())
    with c4: st.metric("Позитивных", (df['эмоция'] == 'позитивный').sum())
    
    # -----------------------------
# ЭКСПОРТ РЕЗУЛЬТАТОВ
# -----------------------------

    export_df = df.copy()

    export_df['проблемный'] = export_df['проблемный'].map({
        True: 'Да',
        False: 'Нет'
    })

    export_df = export_df.rename(columns={
        "dialog": "Диалог",
        "тема": "Тема",
        "эмоция": "Эмоция",
        "проблемный": "Проблемный"
    })

    # CSV
    csv_result = export_df.to_csv(
        index=False,
        sep=";",
        encoding="utf-8-sig"
    )

    # Excel
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_df.to_excel(
            writer,
            index=False,
            sheet_name="Результаты анализа"
        )

        worksheet = writer.sheets["Результаты анализа"]

        worksheet.column_dimensions["A"].width = 80
        worksheet.column_dimensions["B"].width = 22
        worksheet.column_dimensions["C"].width = 18
        worksheet.column_dimensions["D"].width = 15

        header_fill = PatternFill(
            start_color="D9EAF7",
            end_color="D9EAF7",
            fill_type="solid"
        )

        thin_border = Border(
            left=Side(style="thin", color="BFBFBF"),
            right=Side(style="thin", color="BFBFBF"),
            top=Side(style="thin", color="BFBFBF"),
            bottom=Side(style="thin", color="BFBFBF")
        )

        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(
                horizontal="center",
                vertical="center",
                wrap_text=True
            )

        for row in worksheet.iter_rows():
            for cell in row:
                cell.border = thin_border
                cell.alignment = Alignment(
                    wrap_text=True,
                    vertical="top"
                )

        worksheet.row_dimensions[1].height = 22

        for row in range(2, worksheet.max_row + 1):
            worksheet.row_dimensions[row].height = 28

    # Две кнопки рядом
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Скачать результаты CSV",
            data=csv_result.encode("utf-8-sig"),
            file_name="analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download_col2:
        st.download_button(
            label="Скачать результаты Excel",
            data=excel_buffer.getvalue(),
            file_name="analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    
    tab1, tab2, tab3, tab4 = st.tabs(["Диалоги", "Проблемные", "Статистика", "Поиск"])
    
    with tab1:
        display = df[['dialog', 'тема', 'эмоция']].copy()
        display.index = range(1, len(display) + 1)
        st.dataframe(display, use_container_width=True, height=500)
    
    with tab2:
        problems = df[df['проблемный'] == True]
        if len(problems) > 0:
            for _, row in problems.iterrows():
                st.markdown(f"""
                <div class="problem-card">
                    <strong>{row['тема']}</strong> | {row['эмоция']}<br>
                    {row['dialog'][:200]}...
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Нет проблемных")
    
    with tab3:

        topic_counts = df['тема'].value_counts()
        emotion_counts = df['эмоция'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Распределение по темам")
            st.bar_chart(topic_counts)

            st.markdown("#### Детализация по темам")
            total_topics = topic_counts.sum()

            for topic, count in topic_counts.items():
                percent = (count / total_topics) * 100
                st.markdown(f"""
                    <div class="stats-box">
                        <div class="stats-row">
                            <div class="stats-name">{topic}</div>
                            <div class="stats-right">
                                <div class="stats-count">{count}</div>
                                <div class="stats-percent">{percent:.1f}%</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Распределение по эмоциям")
            st.bar_chart(emotion_counts)

            st.markdown("#### Детализация по эмоциям")
            total_emotions = emotion_counts.sum()

            for emo, count in emotion_counts.items():
                percent = (count / total_emotions) * 100
                st.markdown(f"""
                    <div class="stats-box">
                        <div class="stats-row">
                            <div class="stats-name">{emo}</div>
                            <div class="stats-right">
                                <div class="stats-count">{count}</div>
                                <div class="stats-percent">{percent:.1f}%</div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Поиск похожих обращений")

        query = st.text_input(
            "Введите текст обращения",
            placeholder="Например: заказ не пришёл, товар сломался, деньги списали..."
        )

        if query:
            with st.spinner("Ищем похожие обращения..."):

                similar = db.find_similar(
                    query,
                    top_k=5
                )

                if similar:
                    for i, (text, score) in enumerate(similar, 1):
                        st.markdown(f"""
                        <div class="similar-card">
                            <strong>{i} — совпадение {score}%</strong><br>
                            {text}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Похожие обращения не найдены")

else:
    st.info("Загрузите CSV файл")
