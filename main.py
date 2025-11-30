import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path

# ------------------------------
# Carregar CSS
# ------------------------------
css_path = Path("style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ------------------------------
# Layout da página
# ------------------------------
st.set_page_config(
    page_title="Food Truck Queue Simulator",
    page_icon="",
    layout="wide"
)

st.markdown("""
<div class="header-container">
    <h1 class="title"> Food Truck — Simulação de Fila</h1>
    <p class="subtitle">Modelo M/D/1 com chegadas Poisson e atendimento determinístico</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Controles principais (modo e refresh)
# ------------------------------
if "modo_estatico" not in st.session_state:
    st.session_state.modo_estatico = True

if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

colA, colB = st.columns([1, 1])

with colA:
    if st.button("Alternar modo de simulação"):
        st.session_state.modo_estatico = not st.session_state.modo_estatico

with colB:
    if st.button('Recalcular'):
        st.session_state.refresh_counter += 1

st.markdown(
    f"### Modo atual: **{'Estático (fixo)' if st.session_state.modo_estatico else 'Aleatório'}**"
)

# --------------------------------------
# Sidebar - Configurações
# --------------------------------------
with st.sidebar:
    st.markdown("### Configurações")

    N = st.number_input("Número de clientes", min_value=1, value=40, step=1)
    intervalo_medio = st.number_input("Intervalo médio entre chegadas (min)", min_value=0.1, value=3.0, step=0.1)
    atendimento = st.number_input("Tempo de atendimento (min)", min_value=0.1, value=4.0, step=0.1)

    st.markdown("---")
    st.markdown("**Modelo:** M/D/1 (FIFO)")
    st.caption("Chegadas ~ Poisson | Atendimento determinístico")

# ------------------------------
# Simulação (reativa ao refresh)
# ------------------------------
_ = st.session_state.refresh_counter

if st.session_state.modo_estatico:
    np.random.seed(42)
else:
    np.random.seed(None)

intervalos = np.random.exponential(intervalo_medio, N)
chegadas = np.cumsum(intervalos)
servico = np.full(N, atendimento)

# ------------------------------
# Função de simulação M/D/1
# ------------------------------
def simular_md1(chegadas, servico):
    n = len(chegadas)
    inicio = np.zeros(n)
    fim = np.zeros(n)
    espera = np.zeros(n)

    servidor_livre = 0

    for i in range(n):
        inicio[i] = max(chegadas[i], servidor_livre)
        fim[i] = inicio[i] + servico[i]
        espera[i] = inicio[i] - chegadas[i]
        servidor_livre = fim[i]

    return inicio, fim, espera

inicio, fim, espera = simular_md1(chegadas, servico)

# ------------------------------
# Métricas
# ------------------------------
IC_medio = np.mean(np.diff(chegadas)) if N > 1 else intervalo_medio
TA_medio = atendimento
TE_medio = np.mean(espera)
TS_medio = np.mean(fim - chegadas)
lambda_real = 1 / IC_medio
NF_estimado = lambda_real * TE_medio

# ------------------------------
# Métricas — Cards
# ------------------------------
st.markdown("## Métricas do Sistema")

col1, col2, col3, col4, col5 = st.columns(5)
col1.markdown(f"<div class='metric-card'><h3>{IC_medio:.2f} min</h3><p>Intervalo Médio</p></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h3>{TA_medio:.2f} min</h3><p>Atendimento Médio</p></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h3>{TE_medio:.2f} min</h3><p>Espera Média</p></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h3>{TS_medio:.2f} min</h3><p>Tempo no Sistema</p></div>", unsafe_allow_html=True)
col5.markdown(f"<div class='metric-card'><h3>{NF_estimado:.2f}</h3><p>Tam. Médio da Fila</p></div>", unsafe_allow_html=True)

# ------------------------------
# Gráfico — Evolução real da fila
# ------------------------------
st.markdown("## Evolução Real da Fila ao Longo do Tempo")

eventos = []
for i in range(N):
    eventos.append({"tempo": chegadas[i], "tipo": "Chegada", "cliente": i+1})
    eventos.append({"tempo": inicio[i], "tipo": "Início Atendimento", "cliente": i+1})

eventos.sort(key=lambda x: x["tempo"])

fila = 0
linha_tempo = []

for ev in eventos:
    if ev["tipo"] == "Chegada":
        if inicio[ev["cliente"] - 1] > ev["tempo"]:
            fila += 1
    else:
        if fila > 0:
            fila -= 1

    linha_tempo.append({
        "Tempo": ev["tempo"],
        "Fila": fila,
        "Evento": ev["tipo"]
    })

df_fila = pd.DataFrame(linha_tempo)

area = alt.Chart(df_fila).mark_area(
    interpolate="step-after",
    opacity=0.25,
    color="#FF7B00"
).encode(
    x="Tempo:Q",
    y="Fila:Q"
)

line = alt.Chart(df_fila).mark_line(
    interpolate="step-after",
    color="#FF7B00",
    strokeWidth=3
).encode(
    x="Tempo",
    y="Fila"
)

points = alt.Chart(df_fila).mark_circle(size=80).encode(
    x="Tempo",
    y="Fila",
    color=alt.Color("Evento:N", scale=alt.Scale(
        domain=["Chegada", "Início Atendimento"],
        range=["#FF6B6B", "#4ECDC4"]
    )),
    tooltip=[
        alt.Tooltip("Tempo:Q", title="Tempo (min)", format=".2f"),
        "Fila",
        "Evento"
    ]
)

grafico_final = (area + line + points).properties(
    height=350,
    title="Tamanho da fila ao longo do tempo (modelo real M/D/1)"
)

st.altair_chart(grafico_final, use_container_width=True)

# ------------------------------
# Tabela do sistema
# ------------------------------
st.markdown("## Tabela do Sistema")

df = pd.DataFrame({
    "Cliente": np.arange(1, N+1),
    "Chegada": chegadas,
    "Início": inicio,
    "Fim": fim,
    "Atendimento": servico,
    "Espera": espera
})

st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV", csv, "simulacao_foodtruck.csv", "text/csv")

st.markdown("---")
st.caption("Modelo didático — simulação de fila para um Food Truck.")
