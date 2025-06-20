# app.py

import streamlit as st
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
import traceback

# --------------------------------------------------------------------------
# 기존 코랩의 분석 로직을 함수로 정의 (입력: 데이터프레임, 출력: 결과 데이터프레임)
# --------------------------------------------------------------------------
def get_survival_analysis(df: pd.DataFrame):
    try:
        # --- 1. 데이터 타입 변환 및 검증 (웹 입력용 강화) ---
        df = df.dropna(subset=['소재명', '날짜']) # 소재명, 날짜 없는 행 제거
        df = df[df['소재명'] != ''] # 소재명이 비어있지 않은 행만 사용
        
        df['광고비'] = pd.to_numeric(df['광고비'], errors='coerce').fillna(0)
        df['전환수'] = pd.to_numeric(df['전환수'], errors='coerce').fillna(0).astype(int)
        df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
        df = df.dropna(subset=['날짜']) # 날짜 형식이 잘못된 행 제거

        df = df.sort_values(by=['소재명', '날짜']).reset_index(drop=True)

        # --- 2. 핵심 지표 계산 ---
        df['누적광고비'] = df.groupby('소재명')['광고비'].cumsum()
        df['누적전환'] = df.groupby('소재명')['전환수'].cumsum()
        df['전환효율'] = df['누적전환'] / df['누적광고비']
        df['전환효율'] = df['전환효율'].fillna(0)

        # --- 3. '사망' 이벤트 정의 ---
        # 3-1. 효율 감소 이벤트 (기준: 1만원)
        df['전환효율_이전'] = df.groupby('소재명')['전환효율'].shift(1)
        df['효율감소율'] = np.where(df['전환효율_이전'] > 0, (df['전환효율'] - df['전환효율_이전']) / df['전환효율_이전'], 0)
        df['효율감소율'] = df['효율감소율'].fillna(0)
        df['효율감소이벤트'] = np.where((df['효율감소율'] < -0.1) & (df['누적광고비'] >= 10000), 1, 0)

        # 3-2. 2일 연속 전환 없음 이벤트 (기준: 1만원)
        df['전환없음'] = np.where(df['전환수'] == 0, 1, 0)
        df['첫전환이후'] = np.where(df.groupby('소재명')['누적전환'].cumsum() > 0, True, False)
        df['전환없음_이전'] = df.groupby('소재명')['전환없음'].shift(1).fillna(0)
        df['2일연속전환없음'] = np.where((df['전환없음'] == 1) & (df['전환없음_이전'] == 1) & (df['첫전환이후'] == True) & (df['누적광고비'] >= 10000), 1, 0)
        
        df['전환여부'] = np.where((df['효율감소이벤트'] == 1) | (df['2일연속전환없음'] == 1), 1, 0)

        # --- 4. 생존 분석 실행 ---
        result_list = []
        for 소재, group in df.groupby('소재명'):
            kmf = KaplanMeierFitter(label=소재)
            durations = group['누적광고비']
            events = group['전환여부']
            kmf.fit(durations, event_observed=events)
            surv_df = kmf.survival_function_.reset_index()
            surv_df.columns = ['누적광고비', '생존율']
            surv_df['소재명'] = 소재
            surv_df['생존율'] = (surv_df['생존율'] * 100).round(1)
            result_list.append(surv_df)

        if not result_list:
            return pd.DataFrame() # 빈 데이터프레임 반환
            
        result_df = pd.concat(result_list)
        
        # --- 5. 무전환 소재 단계별 페널티 적용 (후처리) ---
        non_converting_ads = df.groupby('소재명')['누적전환'].max()
        non_converting_ads = non_converting_ads[non_converting_ads == 0].index.tolist()
        processed_results = []
        for name, group in result_df.groupby('소재명'):
            if name in non_converting_ads:
                new_survival_rates = []
                for spend in group['누적광고비']:
                    if spend < 30000:
                        new_survival_rates.append(100.0)
                    elif 30000 <= spend < 50000:
                        new_survival_rates.append(50.0)
                    else:
                        new_survival_rates.append(0.0)
                group['생존율'] = new_survival_rates
            processed_results.append(group)
        result_df = pd.concat(processed_results)
        
        # --- 6. 최종 결과 정리 및 진단 추가 ---
        result_df = result_df.sort_values(by=['소재명', '누적광고비'])
        conditions = [
            (result_df['생존율'] >= 70), (result_df['생존율'] < 70) & (result_df['생존율'] >= 50),
            (result_df['생존율'] < 50) & (result_df['생존율'] >= 30),
            (result_df['생존율'] < 30) & (result_df['생존율'] > 0), (result_df['생존율'] == 0)
        ]
        choices = ['🟢 지속', '🟠 주의', '🟡 중단 고려', '🔴 중단', '❗ 사망']
        result_df['진단'] = np.select(conditions, choices, default='-')
        result_df['누적광고비'] = result_df['누적광고비'].apply(lambda x: f"{int(x):,}")
        result_df = result_df[['소재명', '누적광고비', '생존율', '진단']]

        return result_df

    except Exception as e:
        # 웹 앱에서는 에러를 직접 반환하여 표시
        st.error(f"분석 중 에러 발생: {e}")
        st.error(traceback.format_exc())
        return None

# -------------------- Streamlit 웹페이지 UI 구성 --------------------

# 페이지 레이아웃을 넓게 설정
st.set_page_config(layout="wide")

# 제목과 설명
st.title('🚀 광고 소재 생존 분석 자동화 툴')
st.write("아래 표에 데이터를 직접 붙여넣거나 입력한 후, '분석 시작하기' 버튼을 누르세요.")
st.write("---")

# 데이터 입력을 위한 빈 데이터프레임 생성
# 사용자가 데이터를 붙여넣기 쉽도록 예시 데이터 포함
sample_data = {
    '소재명': ['A소재', 'A소재', 'B소재', 'B소재', 'B소재'],
    '날짜': ['2025-06-18', '2025-06-19', '2025-06-18', '2025-06-19', '2025-06-20'],
    '광고비': [10000, 12000, 20000, 15000, 30000],
    '전환수': [1, 0, 2, 1, 0]
}
empty_df = pd.DataFrame(sample_data)

st.subheader("📋 데이터 입력")
st.info("엑셀이나 구글 시트에서 데이터를 복사한 후, 아래 표의 첫 번째 칸을 클릭하고 붙여넣기(Ctrl+V) 하세요.")

# 사용자가 데이터를 편집할 수 있는 인터랙티브 표 (데이터 에디터)
edited_df = st.data_editor(
    empty_df,
    num_rows="dynamic", # 사용자가 행을 동적으로 추가/삭제 가능
    height=300 # 표의 높이 지정
)

# 분석 시작 버튼
if st.button('📊 분석 시작하기'):
    # 입력된 데이터가 있는지 확인
    if edited_df.empty or edited_df['소재명'].str.strip().eq('').all():
        st.warning("분석할 데이터를 입력해주세요.")
    else:
        # 로딩 스피너와 함께 분석 함수 실행
        with st.spinner('열심히 분석 중입니다... 잠시만 기다려주세요...'):
            result_data = get_survival_analysis(edited_df)
        
        # 결과가 성공적으로 생성되었는지 확인
        if result_data is not None and not result_data.empty:
            st.success('분석이 완료되었습니다! 🎉')
            st.write("---")
            
            # 결과 출력
            st.subheader('📊 생존 분석 결과')
            st.dataframe(result_data, height=500, use_container_width=True)
            
            # 다운로드 버튼 추가
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8-sig')

            csv = convert_df_to_csv(result_data)
            st.download_button(
                label="결과 다운로드 (CSV)",
                data=csv,
                file_name='survival_analysis_result.csv',
                mime='text/csv',
            )