import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 페이지 설정
st.set_page_config(
    page_title="고급 혜성 궤도 시뮬레이터",
    page_icon="☄️",
    layout="wide"
)

# 천문학적 상수
AU = 1.496e11  # 천문단위 (미터)
G = 6.67430e-11  # 중력상수 (m³/kg/s²)
M_sun = 1.989e30  # 태양질량 (kg)
YEAR = 365.25 * 24 * 3600  # 1년 (초)

class EnhancedCometSimulator:
    def __init__(self, star_mass, comet_mass, initial_position, initial_velocity, 
                 base_mass_loss_rate, jet_strength, jet_randomness):
        """
        향상된 혜성 궤도 시뮬레이터
        
        Parameters:
        - star_mass: 항성 질량 (태양질량 단위)
        - comet_mass: 혜성 초기 질량 (kg)
        - initial_position: 초기 위치 [x, y] (AU)
        - initial_velocity: 초기 속도 [vx, vy] (m/s)
        - base_mass_loss_rate: 기본 질량 소실률 (kg/s at 1 AU)
        - jet_strength: 제트 효과 강도 (m/s²)
        - jet_randomness: 제트 방향 무작위성 (0-1)
        """
        self.star_mass = star_mass * M_sun
        self.initial_comet_mass = comet_mass
        self.current_comet_mass = comet_mass
        self.base_mass_loss_rate = base_mass_loss_rate
        self.jet_strength = jet_strength
        self.jet_randomness = jet_randomness
        self.is_extinct = False
        
        # 초기 조건
        self.position = np.array(initial_position) * AU  # AU를 미터로 변환
        self.velocity = np.array(initial_velocity)
        
        # 궤도 추적을 위한 리스트
        self.positions = [self.position.copy()]
        self.velocities = [self.velocity.copy()]
        self.masses = [self.current_comet_mass]
        self.times = [0.0]
        self.jet_accelerations = [np.array([0.0, 0.0])]
        
        # 제트 효과를 위한 내부 상태
        self.jet_phase = np.random.random() * 2 * np.pi
        
    def distance_dependent_mass_loss_rate(self, distance_au):
        """거리에 의존하는 질량 소실률"""
        # 태양에 가까워질수록 질량 소실률이 급격히 증가 (역제곱 법칙)
        # 1 AU에서 base_mass_loss_rate, 거리 제곱에 반비례
        return self.base_mass_loss_rate / (distance_au ** 2)
    
    def calculate_jet_acceleration(self, distance, velocity):
        """제트 효과로 인한 비중력 가속도 계산"""
        if self.current_comet_mass <= 0 or self.is_extinct:
            return np.array([0.0, 0.0])
        
        distance_au = distance / AU
        
        # 거리에 따른 제트 강도 (태양에 가까울수록 강함)
        distance_factor = 1.0 / (distance_au ** 1.5)
        
        # 질량에 따른 제트 강도 (질량이 작을수록 제트 효과가 더 큰 영향)
        mass_factor = self.initial_comet_mass / max(self.current_comet_mass, 1e-10)
        
        # 기본 제트 강도
        jet_magnitude = self.jet_strength * distance_factor * mass_factor
        
        # 제트 방향 계산 (주로 태양 반대 방향 + 무작위성)
        position_norm = np.linalg.norm(self.position)
        if position_norm > 0:
            # 태양 반대 방향 (방사 방향)
            radial_direction = self.position / position_norm
            
            # 궤도 접선 방향
            velocity_norm = np.linalg.norm(velocity)
            if velocity_norm > 0:
                tangential_direction = velocity / velocity_norm
            else:
                tangential_direction = np.array([0.0, 0.0])
            
            # 제트 방향에 무작위성 추가
            self.jet_phase += 0.1  # 시간에 따른 위상 변화
            random_factor = self.jet_randomness
            
            # 주로 방사 방향 + 약간의 접선 성분 + 무작위성
            jet_direction = (
                (1 - random_factor) * radial_direction +
                0.3 * random_factor * tangential_direction +
                0.2 * random_factor * np.array([
                    np.cos(self.jet_phase),
                    np.sin(self.jet_phase)
                ])
            )
            
            # 방향 정규화
            jet_direction_norm = np.linalg.norm(jet_direction)
            if jet_direction_norm > 0:
                jet_direction = jet_direction / jet_direction_norm
        else:
            jet_direction = np.array([0.0, 0.0])
        
        return jet_magnitude * jet_direction
    
    def step(self, dt):
        """한 시간 스텝 진행"""
        if self.is_extinct:
            return False
        
        # 현재 거리 계산
        distance = np.linalg.norm(self.position)
        distance_au = distance / AU
        
        # 중력 가속도 계산
        gravity_acc = -G * self.star_mass * self.position / (distance ** 3)
        
        # 제트 가속도 계산
        jet_acc = self.calculate_jet_acceleration(distance, self.velocity)
        
        # 총 가속도
        total_acc = gravity_acc + jet_acc
        
        # 위치와 속도 업데이트 (Verlet 적분법)
        new_position = self.position + self.velocity * dt + 0.5 * total_acc * dt**2
        
        # 새로운 위치에서의 가속도 계산
        new_distance = np.linalg.norm(new_position)
        new_gravity_acc = -G * self.star_mass * new_position / (new_distance ** 3)
        new_jet_acc = self.calculate_jet_acceleration(new_distance, self.velocity)
        new_total_acc = new_gravity_acc + new_jet_acc
        
        # 속도 업데이트
        new_velocity = self.velocity + 0.5 * (total_acc + new_total_acc) * dt
        
        # 질량 업데이트 (거리 의존적)
        mass_loss_rate = self.distance_dependent_mass_loss_rate(distance_au)
        mass_loss = mass_loss_rate * dt
        
        if self.current_comet_mass - mass_loss <= 0:
            self.current_comet_mass = 0
            self.is_extinct = True
        else:
            self.current_comet_mass -= mass_loss
        
        # 상태 업데이트
        self.position = new_position
        self.velocity = new_velocity
        
        # 기록 저장
        self.positions.append(self.position.copy())
        self.velocities.append(self.velocity.copy())
        self.masses.append(self.current_comet_mass)
        self.times.append(self.times[-1] + dt)
        self.jet_accelerations.append(jet_acc.copy())
        
        return True
    
    def simulate(self, total_time, dt):
        """전체 시뮬레이션 실행"""
        steps = int(total_time / dt)
        
        for i in range(steps):
            if not self.step(dt):
                break
        
        return (np.array(self.times), 
                np.array(self.positions), 
                np.array(self.velocities),
                np.array(self.masses),
                np.array(self.jet_accelerations))

def calculate_orbital_parameters(position, velocity, star_mass):
    """궤도 요소 계산"""
    mu = G * star_mass
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    
    # 비에너지 (specific energy)
    energy = 0.5 * v**2 - mu / r
    
    # 각운동량
    h = np.cross(position, velocity)
    h_magnitude = abs(h) if np.isscalar(h) else np.linalg.norm(h)
    
    # 이심률
    if energy < 0:  # 타원 궤도
        a = -mu / (2 * energy)  # 반장축
        eccentricity = np.sqrt(1 + (2 * energy * h_magnitude**2) / (mu**2))
    else:  # 포물선 또는 쌍곡선
        eccentricity = np.sqrt(1 + (2 * energy * h_magnitude**2) / (mu**2))
        a = mu / (2 * abs(energy))  # 쌍곡선의 경우
    
    return energy, h_magnitude, eccentricity, a

def main():
    st.title("☄️ 고급 혜성 궤도 시뮬레이터")
    st.markdown("제트 효과와 거리 의존적 질량 소실을 포함한 현실적인 혜성 궤도 시뮬레이션")
    
    # 사이드바 매개변수
    st.sidebar.header("🔧 시뮬레이션 매개변수")
    
    # 기본 물리 매개변수
    st.sidebar.subheader("⭐ 시스템 매개변수")
    star_mass = st.sidebar.slider(
        "항성 질량 (태양질량 단위)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1
    )
    
    comet_mass_exp = st.sidebar.slider(
        "혜성 초기 질량 (10^x kg)",
        min_value=10, max_value=15, value=12, step=1
    )
    comet_mass = 10**comet_mass_exp
    
    # 초기 조건
    st.sidebar.subheader("🚀 초기 조건")
    
    # 미리 정의된 궤도 옵션
    orbit_preset = st.sidebar.selectbox(
        "궤도 프리셋",
        ["사용자 정의", "할리 혜성형", "단주기 혜성", "장주기 혜성", "항성간 천체"]
    )
    
    if orbit_preset == "할리 혜성형":
        init_x, init_y = 0.6, 0.0
        init_vx, init_vy = 0, 35000
    elif orbit_preset == "단주기 혜성":
        init_x, init_y = 1.0, 0.0
        init_vx, init_vy = 0, 25000
    elif orbit_preset == "장주기 혜성":
        init_x, init_y = 2.0, 0.0
        init_vx, init_vy = 0, 15000
    elif orbit_preset == "항성간 천체":
        init_x, init_y = 1.0, 0.0
        init_vx, init_vy = 0, 50000
    else:  # 사용자 정의
        init_x = st.sidebar.slider("초기 X 위치 (AU)", -5.0, 5.0, 2.0, 0.1)
        init_y = st.sidebar.slider("초기 Y 위치 (AU)", -5.0, 5.0, 0.0, 0.1)
        init_vx = st.sidebar.slider("초기 X 속도 (m/s)", -50000, 50000, 0, 1000)
        init_vy = st.sidebar.slider("초기 Y 속도 (m/s)", -50000, 50000, 20000, 1000)
    
    # 향상된 물리 매개변수
    st.sidebar.subheader("🌋 향상된 물리 효과")
    
    base_mass_loss_exp = st.sidebar.slider(
        "기본 질량 소실률 (10^x kg/s at 1 AU)",
        min_value=2, max_value=8, value=5, step=1
    )
    base_mass_loss_rate = 10**base_mass_loss_exp
    
    jet_strength = st.sidebar.slider(
        "제트 효과 강도 (m/s²)",
        min_value=0.0, max_value=1e-3, value=1e-4, step=1e-5, format="%.1e"
    )
    
    jet_randomness = st.sidebar.slider(
        "제트 방향 무작위성",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1
    )
    
    # 시뮬레이션 설정
    st.sidebar.subheader("⏱️ 시뮬레이션 설정")
    sim_years = st.sidebar.slider(
        "시뮬레이션 기간 (년)",
        min_value=1, max_value=200, value=50, step=1
    )
    
    time_resolution = st.sidebar.slider(
        "시간 해상도 (일)",
        min_value=0.1, max_value=10.0, value=1.0, step=0.1
    )
    
    # 예상 생존 시간 계산
    estimated_lifetime = comet_mass / base_mass_loss_rate / YEAR
    st.sidebar.info(f"💡 1 AU에서의 예상 생존시간: {estimated_lifetime:.1f}년")
    
    if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
        # 시뮬레이터 초기화
        simulator = EnhancedCometSimulator(
            star_mass=star_mass,
            comet_mass=comet_mass,
            initial_position=[init_x, init_y],
            initial_velocity=[init_vx, init_vy],
            base_mass_loss_rate=base_mass_loss_rate,
            jet_strength=jet_strength,
            jet_randomness=jet_randomness
        )
        
        # 시뮬레이션 실행
        total_time = sim_years * YEAR
        dt = time_resolution * 24 * 3600  # 일을 초로 변환
        
        with st.spinner("고급 물리 효과를 포함한 시뮬레이션 계산 중..."):
            times, positions, velocities, masses, jet_accs = simulator.simulate(total_time, dt)
        
        # 결과 분석
        if simulator.is_extinct:
            extinction_time = times[-1] / YEAR
            st.error(f"🔥 **혜성이 {extinction_time:.1f}년 후 완전히 소멸되었습니다!**")
        
        # 메인 표시 영역
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌍 향상된 궤도 애니메이션")
            
            # 궤도 그래프
            fig = go.Figure()
            
            # 항성
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=30, color='gold', symbol='star'),
                name='항성',
                hovertemplate=f'<b>항성</b><br>질량: {star_mass:.1f} 태양질량<extra></extra>'
            ))
            
            # 궤도 경로 (색상으로 시간 표현)
            x_pos = positions[:, 0] / AU
            y_pos = positions[:, 1] / AU
            
            # 궤도 경로를 시간에 따라 색상 변화로 표현
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines',
                line=dict(color=times/YEAR, colorscale='Viridis', width=2),
                name='궤도 경로',
                hovertemplate='시간: %{marker.color:.1f}년<extra></extra>'
            ))
            
            # 초기 위치
            fig.add_trace(go.Scatter(
                x=[x_pos[0]], y=[y_pos[0]],
                mode='markers',
                marker=dict(size=12, color='green', symbol='circle'),
                name='시작점',
                hovertemplate='<b>시작점</b><extra></extra>'
            ))
            
            # 최종 위치
            fig.add_trace(go.Scatter(
                x=[x_pos[-1]], y=[y_pos[-1]],
                mode='markers',
                marker=dict(size=12, color='red', symbol='x' if simulator.is_extinct else 'circle'),
                name='종료점',
                hovertemplate=f'<b>{"소멸" if simulator.is_extinct else "종료"}점</b><extra></extra>'
            ))
            
            # 레이아웃
            max_distance = max(np.max(np.abs(x_pos)), np.max(np.abs(y_pos))) * 1.2
            fig.update_layout(
                title="향상된 혜성 궤도 (제트 효과 + 거리 의존적 질량 소실)",
                xaxis_title="거리 (AU)",
                yaxis_title="거리 (AU)",
                showlegend=True,
                width=700, height=600,
                xaxis=dict(scaleanchor="y", scaleratio=1, range=[-max_distance, max_distance]),
                yaxis=dict(range=[-max_distance, max_distance]),
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 물리량 변화")
            
            # 질량 변화
            fig_mass = go.Figure()
            fig_mass.add_trace(go.Scatter(
                x=times/YEAR, y=masses,
                mode='lines',
                name='질량',
                line=dict(color='red', width=3)
            ))
            fig_mass.update_layout(
                title="혜성 질량 변화",
                xaxis_title="시간 (년)",
                yaxis_title="질량 (kg)",
                height=300
            )
            st.plotly_chart(fig_mass, use_container_width=True)
            
            # 거리 변화
            distances = np.linalg.norm(positions, axis=1) / AU
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Scatter(
                x=times/YEAR, y=distances,
                mode='lines',
                name='거리',
                line=dict(color='blue', width=2)
            ))
            fig_dist.update_layout(
                title="태양으로부터의 거리",
                xaxis_title="시간 (년)",
                yaxis_title="거리 (AU)",
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # 상세 분석
        st.subheader("🔬 궤도 분석")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 초기 궤도 요소
            initial_energy, initial_h, initial_e, initial_a = calculate_orbital_parameters(
                positions[0], velocities[0], simulator.star_mass
            )
            st.markdown("### 🚀 초기 궤도")
            st.write(f"**이심률:** {initial_e:.3f}")
            st.write(f"**반장축:** {initial_a/AU:.2f} AU")
            st.write(f"**각운동량:** {initial_h:.2e} m²/s")
        
        with col2:
            # 최종 궤도 요소
            final_energy, final_h, final_e, final_a = calculate_orbital_parameters(
                positions[-1], velocities[-1], simulator.star_mass
            )
            st.markdown("### 🏁 최종 궤도")
            st.write(f"**이심률:** {final_e:.3f}")
            st.write(f"**반장축:** {final_a/AU:.2f} AU")
            st.write(f"**각운동량:** {final_h:.2e} m²/s")
        
        with col3:
            # 변화량
            st.markdown("### 📈 궤도 변화")
            st.write(f"**이심률 변화:** {final_e - initial_e:.3f}")
            st.write(f"**반장축 변화:** {(final_a - initial_a)/AU:.2f} AU")
            st.write(f"**각운동량 변화:** {(final_h - initial_h)/initial_h*100:.1f}%")
        
        # 제트 효과 분석
        st.subheader("🌋 제트 효과 분석")
        
        # 제트 가속도 크기
        jet_magnitudes = np.linalg.norm(jet_accs, axis=1)
        
        fig_jet = make_subplots(
            rows=2, cols=1,
            subplot_titles=('제트 가속도 크기', '거리별 질량 소실률'),
            vertical_spacing=0.15
        )
        
        # 제트 가속도
        fig_jet.add_trace(
            go.Scatter(x=times/YEAR, y=jet_magnitudes, name='제트 가속도'),
            row=1, col=1
        )
        
        # 거리별 질량 소실률
        mass_loss_rates = [simulator.distance_dependent_mass_loss_rate(d) for d in distances]
        fig_jet.add_trace(
            go.Scatter(x=times/YEAR, y=mass_loss_rates, name='질량 소실률'),
            row=2, col=1
        )
        
        fig_jet.update_layout(height=500, showlegend=False)
        fig_jet.update_xaxes(title_text="시간 (년)", row=2, col=1)
        fig_jet.update_yaxes(title_text="가속도 (m/s²)", row=1, col=1)
        fig_jet.update_yaxes(title_text="소실률 (kg/s)", row=2, col=1)
        
        st.plotly_chart(fig_jet, use_container_width=True)
        
        # 결과 요약
        st.subheader("📋 시뮬레이션 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 시뮬레이션 시간",
                f"{times[-1]/YEAR:.1f} 년"
            )
        
        with col2:
            mass_loss_percent = (comet_mass - masses[-1]) / comet_mass * 100
            st.metric(
                "총 질량 소실",
                f"{mass_loss_percent:.1f}%"
            )
        
        with col3:
            max_distance = np.max(distances)
            min_distance = np.min(distances)
            st.metric(
                "근일점/원일점",
                f"{min_distance:.2f} / {max_distance:.2f} AU"
            )
        
        with col4:
            avg_jet_acc = np.mean(jet_magnitudes)
            st.metric(
                "평균 제트 가속도",
                f"{avg_jet_acc:.2e} m/s²"
            )
    
    # 도움말
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 새로운 기능")
    st.sidebar.markdown("""
    **🌋 제트 효과:**
    - 비등방성 질량 소실로 인한 추진력
    - 거리와 질량에 의존하는 제트 강도
    - 무작위적 방향 변화로 궤도 불안정성 구현
    
    **📏 거리 의존적 질량 소실:**
    - 태양 근처에서 급격한 질량 소실
    - 역제곱 법칙 적용 (1/r²)
    - 현실적인 혜성 활동 모델링
    
    **🎯 향상된 정확도:**
    - 수치적분으로 비선형 효과 포함
    - 궤도 요소의 실시간 변화 추적
    """)

if __name__ == "__main__":
    main()
