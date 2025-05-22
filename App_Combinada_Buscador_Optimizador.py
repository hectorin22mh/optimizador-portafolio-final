
import streamlit as st
st.set_page_config(page_title="Optimizador de Portafolios", layout="wide")


tab1, tab2 = st.tabs(["🔍 Buscar Acción", "📊 Optimizar Portafolio"])

with tab1:
    import streamlit as st
    import yfinance as yf
    import plotly.graph_objects as go
    import google.generativeai as genai
    import requests
    from PIL import Image
    from io import BytesIO
    import matplotlib.pyplot as plt
    from bs4 import BeautifulSoup  # asegúrate de tener esta importación al inicio del archivo
    def get_finviz_data(ticker):
        data = {
            "etfs": [],
            "peers": [],
            "components": []
        }
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            # ETFs relacionados
            held_by_tag = soup.find("a", string=lambda text: text and "Held by" in text)
            if held_by_tag and "href" in held_by_tag.attrs:
                href = held_by_tag["href"]
                held_by_str = href.split("t=")[-1]
                data["etfs"] = held_by_str.split(",")[:5]

            # Peers
            peers_tag = soup.find("a", string=lambda text: text and "Peers" in text)
            if peers_tag and "href" in peers_tag.attrs:
                href = peers_tag["href"]
                tickers_str = href.split("t=")[-1]
                data["peers"] = tickers_str.split(",")

            # Componentes
            data["components"].append(ticker)
            if data["etfs"]:
                data["components"].append(data["etfs"][0])
                data["components"].append(data["etfs"][-1])
            else:
                data["components"].extend(["N/A", "N/A"])
            data["components"].extend(data["peers"])

            # Guardar en el estado de sesión de Streamlit
            st.session_state['finviz_data'] = data

        except:
            pass
        return data
    
    ################################################
    tokenAI = "AIzaSyDjFAIJkM_2TIlJOTG_rmj7mS6f8IVWG-s"
    serpapi_key = "f483226aaed19fb5ffb0cdb9da3a27103289c96e145590a53493f3c2343fcbe2"
    ################################################
    
    def translate_with_gemini(text):
        try:
            genai.configure(api_key=tokenAI)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"Traduce al español este texto sin encabezado ni introducción, solo la traducción directa:\n\n{text}"
            )
            return response.text.strip() if hasattr(response, "text") and response.text else text
        except Exception as e:
            return f"[Error de traducción: {str(e)}]\n{text}"
    
    def get_similar_tickers(ticker):
        return []  # Deshabilitado para evitar errores de tipo con Timestamp
    
    @st.cache_data(ttl=1800)
    def get_news_from_serpapi(company_name, date_range="qdr:w", language="es", force_refresh=False):
        try:
            # Set query depending on language
            if language == "es":
                query = f"{company_name} resultados financieros"
            else:
                query = f"{company_name} financial results"
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_news",
                "q": query,
                "tbs": date_range,
                "hl": language,
                "api_key": serpapi_key
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json().get("news_results", [])
            else:
                return []
        except Exception as e:
            return [{"title": "Error al obtener noticias", "link": "#", "snippet": str(e)}]
    
    def fetch_stock_data(symbol, period, full_history=False):
        try:
            stock = yf.Ticker(symbol)
            history = stock.history(period="2d")  # Obtener los últimos 2 días para comparación
            info = stock.info
            market_cap = info.get('marketCap', 'N/A')
            volume = info.get('volume', 'N/A')
    
            if len(history) > 1:
                last_price = history.iloc[-1]['Close']
                prev_close_price = history.iloc[-2]['Close']  # Tomar el cierre del día anterior
                percent_change = ((last_price - prev_close_price) / prev_close_price) * 100
            elif len(history) == 1:
                last_price = history.iloc[-1]['Close']
                open_price = history.iloc[-1]['Open']
                percent_change = ((last_price - open_price) / open_price) * 100
            else:
                last_price, percent_change = None, None
    
            day_high = history.iloc[-1]['High']  # Máximo del día
            day_low = history.iloc[-1]['Low']  # Mínimo del día
            return last_price, percent_change, market_cap, volume, day_high, day_low
        except Exception:
            return None, None, None, None, None, None
    
    def get_investment_recommendation(symbol, last_price, day_high, day_low, market_cap, volume):
        try:
            # Obtener datos históricos de 1 año
            stock = yf.Ticker(symbol)
            history = stock.history(period="1y")
    
            if history.empty:
                return "No hay datos históricos suficientes para generar una recomendación."
    
            # Cálculo de métricas clave
            annual_return = ((history['Close'][-1] - history['Close'][0]) / history['Close'][0]) * 100
            volatility = history['Close'].pct_change().std() * (252**0.5) * 100  # Volatilidad anualizada
            max_close = history['Close'].max()
            min_close = history['Close'].min()
            avg_volume = int(history['Volume'].mean())
    
            beta = stock.info.get("beta", None)
            risk_free_rate = 0.04  # Tasa libre de riesgo aproximada (bono a 10 años USA)
            market_return = 0.08   # Rentabilidad promedio del mercado (S&P500)
    
            capm_estimate = None
            if beta is not None:
                capm_estimate = risk_free_rate + beta * (market_return - risk_free_rate)
    
            # Construir resumen para el prompt
            metrics_summary = f"""
            Análisis histórico del activo {symbol} (últimos 12 meses):
            - Rentabilidad anual estimada: {annual_return:.2f}%
            - Volatilidad anual: {volatility:.2f}%
            - Precio máximo: {max_close:.2f} USD
            - Precio mínimo: {min_close:.2f} USD
            - Volumen promedio: {avg_volume:,}
            """
    
            if beta is not None and capm_estimate is not None:
                metrics_summary += f"""
                - Beta estimada: {beta:.2f}
                - CAPM estimado (Rentabilidad esperada): {capm_estimate:.2%}
                """
    
            # Prompt con base en datos reales
            prompt = f"""
            Eres un analista financiero experto en análisis de riesgos de inversión. Tu especialidad es evaluar el riesgo-retorno de activos financieros a nivel global. Redacta la respuesta completamente en español neutro, sin inglés.

            A continuación se presentan los datos de un activo financiero. Analiza y proporciona un informe claro y profesional completamente en español, sin incluir código Python:

            {metrics_summary}

            Información adicional actual:
            - Última cotización: {last_price} USD
            - Máximo del día: {day_high} USD
            - Mínimo del día: {day_low} USD
            - Capitalización de mercado: {market_cap}
            - Volumen del día: {volume}

            Tu análisis debe incluir:
            - Un resumen del nivel de riesgo del activo.
            - Un reporte claro con hallazgos clave.
            - Un análisis estimado del CAPM (Capital Asset Pricing Model) basado en el riesgo sistemático del activo.
            - Una recomendación diferenciada para tres perfiles de inversionista: conservador, moderado y agresivo.

            🚨 Restricciones:
            - No devuelvas código Python.
            - No expliques cómo se calculan las métricas.
            - Usa lenguaje técnico, pero comprensible para personas con conocimientos intermedios en finanzas.
            """
    
            genai.configure(api_key=tokenAI)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return "No se pudo generar una recomendación en este momento."
    
    
    def plot_stock_chart(stock, symbol, period):
        try:
            history = stock.history(period=period)
    
            if history.empty:
                return None  
    
            first_price = history.iloc[0]['Close']
            last_price = history.iloc[-1]['Close']
            line_color = "#28A745" if last_price > first_price else "#DC3545"
    
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history.index, y=history['Close'], 
                mode='lines', name='Precio de Cierre', 
                line=dict(color=line_color, width=2)
            ))
            fig.update_layout(
                title=f"Historial de Precios de {symbol} ({period})",
                xaxis_title="Fecha",
                yaxis_title="Precio (USD)",
                hovermode="x",
                template="plotly_dark"
            )
            return fig
        except Exception:
            return None
    
    def calculate_period_change(stock, symbol, period):
        try:
            history = stock.history(period=period)
    
            if history.empty or len(history) < 2:
                return None  
    
            first_price = history.iloc[0]['Close']
            last_price = history.iloc[-1]['Close']
            percent_change = ((last_price - first_price) / first_price) * 100
    
            return percent_change
        except Exception:
            return None
    
    st.title("📈 Información de Empresas en la Bolsa")
    
    query_params = st.query_params

    # Mantener ticker desde URL o sesión
    if "typed_ticker" in st.session_state:
        ticker = st.session_state["typed_ticker"]
    else:
        ticker = query_params["ticker"][0] if "ticker" in query_params else ""

    # Mostrar input de texto con el ticker actual
    ticker = st.text_input("Escribe el símbolo bursátil", value=ticker, placeholder="Ejemplo: AAPL")

    # Si se escribe manualmente, actualiza manual_input
    if ticker:
        st.session_state["typed_ticker"] = ticker
        st.session_state["manual_input"] = True

    # Mostrar botones de ETFs justo debajo de la barra de búsqueda
    if ticker:
        info = None
        stock = None
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            finviz_data = get_finviz_data(ticker)

            if not info or 'longName' not in info:
                raise ValueError("No se encontró información relevante del símbolo.")

        except Exception as e:
            error_message = str(e).lower()
            if "rate limited" in error_message or "too many requests" in error_message:
                st.warning("📛 Has realizado demasiadas búsquedas seguidas. Yahoo Finance ha activado un límite temporal. Intenta de nuevo en unos minutos.")
            elif "502" in error_message:
                st.warning("⚠️ Finviz no respondió correctamente (Error 502). Esto suele ser temporal. Intenta de nuevo en unos minutos.")
            else:
                st.warning(f"⚠️ No se pudo recuperar la información para '{ticker}'. Error: {str(e)}")
            
            # Mostrar ticker fallido pero mantenerlo visible para modificar
            st.stop()
    
    # Definir opciones de período ANTES de usarlas
    period_options = {
        "1 Semana": "5d",
        "1 Mes": "1mo",
        "3 Meses": "3mo",
        "6 Meses": "6mo",
        "1 Año": "1y",
        "2 Años": "2y",
        "5 Años": "5y",
        "10 Años": "10y",
        "Todo": "max"
    }
    
    if ticker:
        # get_finviz_data(ticker), stock, info ya se ejecutaron arriba
        if info is None:
            st.error("No se pudo obtener la información de la acción. Intenta con otro símbolo.")
            if "typed_ticker" in st.session_state:
                st.session_state.pop("typed_ticker")
            st.stop()
        last_price, percent_change, market_cap, volume, day_high, day_low = fetch_stock_data(ticker, "1d", full_history=True)
        if 'longName' in info:
            # Mostrar título de la empresa y botón "Agregar al portafolio" en la misma fila
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f"<div style='text-align: left; font-size: 32px; font-weight: bold;'>{info['longName']}</div>",
                    unsafe_allow_html=True
                )
            with col2:
                if "tickers_portafolio" not in st.session_state:
                    st.session_state["tickers_portafolio"] = []
                if ticker in st.session_state["tickers_portafolio"]:
                    st.success("✅ Agregado")
                else:
                    if st.button("➕ Agregar al portafolio"):
                        st.session_state["tickers_portafolio"].append(ticker)
                        st.success(f"{ticker} agregado al portafolio.")
    
            col1, col2, col3 = st.columns([3, 1, 1])
    
            # Reducir tamaño y alinear a la izquierda con CSS
            st.markdown(
                """
                <style>
                div[data-baseweb="select"] {
                    max-width: 200px !important;
                    text-align: left !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
            market_cap_str = f"${market_cap:,.0f}" if isinstance(market_cap, (int, float)) else "N/D"
            volume_str = f"{volume:,}" if isinstance(volume, (int, float)) else "N/D"
    
            st.markdown(
                f"""
                <div style='text-align: center; font-size: 18px; margin-bottom: 10px;'>
                    <strong>Último:</strong> {last_price:.2f} USD | 
                    <strong>Máximo:</strong> {day_high:.2f} USD | 
                    <strong>Mínimo:</strong> {day_low:.2f} USD | 
                    <strong>Market Cap:</strong> {market_cap_str} | 
                    <strong>Volumen:</strong> {volume_str}
                </div>
                """,
                unsafe_allow_html=True
            )
    
    
            # Mostrar gráfica de la acción
            selected_period = st.selectbox("Selecciona el período del gráfico", list(period_options.keys()), key="period_select")
            
            period_change = calculate_period_change(stock, ticker, period_options[selected_period])
            
            if period_change is not None:
                change_color = "green" if period_change > 0 else "red"
                change_symbol = "+" if period_change > 0 else ""
                st.markdown(f"""
                    <div style='display: flex; justify-content: space-between; font-size: 18px; font-weight: bold;'>
                        <div></div>
                        <div style='color: {change_color};'>{change_symbol}{period_change:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            chart = plot_stock_chart(stock, ticker, period_options[selected_period])
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.write("No se encontraron datos para graficar.")

            # Mostrar botones de peers (acciones similares) y ETFs en bloques separados y alineados justo debajo de la gráfica
            if "finviz_data" in st.session_state:
                peers = st.session_state["finviz_data"].get("peers", [])[:5]
                etfs = st.session_state["finviz_data"].get("etfs", [])[:5]

                if peers:
                    st.markdown("##### 🔄 Acciones similares:")
                    peer_cols = st.columns(len(peers))
                    for i, peer in enumerate(peers):
                        with peer_cols[i]:
                            if st.button(peer, key=f"peer_button_{i}"):
                                st.session_state["manual_input"] = True
                                st.session_state["typed_ticker"] = peer
                                st.rerun()

                if etfs:
                    st.markdown("##### 📦 ETFs que incluyen esta acción:")
                    etf_cols = st.columns(len(etfs))
                    for i, etf in enumerate(etfs):
                        with etf_cols[i]:
                            if st.button(etf, key=f"etf_button_{i}"):
                                st.session_state["manual_input"] = True
                                st.session_state["typed_ticker"] = etf
                                st.rerun()
    
    
            
            if 'longBusinessSummary' in info and info['longBusinessSummary']:
                try:
                    original_text = info['longBusinessSummary']
                    translated_text = translate_with_gemini(original_text)
                    st.markdown(f"<div class='summary'>{translated_text}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.warning("No se pudo traducir la descripción. Se mostrará en inglés.")
                    st.markdown(f"<div class='summary'>{info['longBusinessSummary']}</div>", unsafe_allow_html=True)
    
            # Mover el bloque del sitio web aquí
            if 'website' in info and info['website']:
                st.markdown(f"<div class='link'><a href='{info['website']}' target='_blank'>Visitar sitio web</a></div>", unsafe_allow_html=True)
    
            # Sección de recomendación de inversión
            st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; margin-top: 40px;'>Análisis de Riesgo y Evaluación del Activo</div>", unsafe_allow_html=True)
    
            recommendation = get_investment_recommendation(ticker, last_price, day_high, day_low, market_cap, volume)
    
            st.markdown(f"<div class='summary'>{recommendation}</div>", unsafe_allow_html=True)
    
            # Noticias relacionadas
            st.markdown("<div style='text-align: center; font-size: 22px; font-weight: bold; margin-top: 40px;'>📰 Noticias Recientes</div>", unsafe_allow_html=True)
            date_options = {
                "Últimas 24 horas": "qdr:d",
                "Última semana": "qdr:w",
                "Último mes": "qdr:m",
                "Último año": "qdr:y"
            }
            # Selección de idioma antes del rango de fechas
            idioma_opciones = {
                "Español 🇲🇽": "es",
                "Inglés 🇺🇸": "en"
            }
            idioma_seleccionado = st.radio("🌐 Idioma de las noticias", list(idioma_opciones.keys()), horizontal=True)
            selected_range = st.selectbox("🕒 Rango de tiempo para noticias", list(date_options.keys()), index=1)
            news_articles = get_news_from_serpapi(
                info['longName'],
                date_options[selected_range],
                language=idioma_opciones[idioma_seleccionado],
                force_refresh=st.button("🔄 Forzar actualización de noticias")
            )

            if news_articles:
                max_articles = 10
                show_all = st.session_state.get("show_all_news", False)

                if not show_all and len(news_articles) > max_articles:
                    articles_to_display = news_articles[:max_articles]
                else:
                    articles_to_display = news_articles

                for article in articles_to_display:
                    title = article.get('title', 'Sin título')
                    link = article.get('link', '#')
                    snippet = article.get('snippet', '')
                    source = article.get('source', {}).get('name', 'Fuente desconocida')
                    date = article.get('date', 'Sin fecha')
                    thumbnail = article.get('thumbnail', '')
                    fallback_image = "https://via.placeholder.com/120x80?text=Noticia"
                    image_html = f"<img src='{thumbnail}' alt='miniatura' width='120' height='80' style='border-radius: 8px;'>" if thumbnail else f"<img src='{fallback_image}' alt='sin imagen' width='120' height='80' style='border-radius: 8px;'>"

                    st.markdown(f"""
                    <div style="display: flex; gap: 15px; background-color: #1e1e1e; border-left: 5px solid #1E90FF; border-radius: 10px; padding: 15px; margin-bottom: 10px;">
                        <div style="flex-shrink: 0;">
                            {image_html}
                        </div>
                        <div>
                            <h4 style="margin-bottom: 5px;"><a href="{link}" target="_blank" style="text-decoration: none; color: #1E90FF;">📰 {title}</a></h4>
                            <p style="color: #ccc; font-size: 15px; margin-bottom: 5px;">{snippet}</p>
                            <p style="color: #888; font-size: 13px; margin: 0;">📰 {source} — 🕒 {date}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if not show_all and len(news_articles) > max_articles:
                    if st.button("Ver más noticias"):
                        st.session_state["show_all_news"] = True
                elif show_all:
                    if st.button("Ver menos"):
                        st.session_state["show_all_news"] = False
            else:
                st.info("No se encontraron noticias recientes.")

            # Bloque duplicado de "Agregar al portafolio" eliminado para evitar repetición.
    
        else:
            st.error("No se encontró información para el símbolo ingresado.")
            similar_tickers = get_similar_tickers(ticker)
            if similar_tickers:
                st.write("Tal vez quisiste decir:")
                for t in similar_tickers:
                    st.write(f"🔹 {t}")
            else:
                st.info("No se encontraron sugerencias de símbolos similares.")
    
    # Estilos personalizados
    st.markdown(
        """
        <style>
        .stTextInput > div > div > input {
            text-align: center;
            font-size: 24px !important;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
        }
        .summary {
            text-align: justify;
            font-size: 18px;
        }
        .link {
            text-align: center;
            font-size: 16px;
            margin-top: 20px;
        }
        .info-container {
            display: flex;
            justify-content: space-around;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        .market-volume-container {
            display: flex;
            justify-content: space-around;
            font-size: 18px;
            font-weight: bold;
            margin-top: 10px;
        }
        .price {
            color: #FFFFFF !important;
            font-size: 22px !important;
            font-weight: bold;
        }
        .change {
            text-align: center;
        }
        .marketcap, .volume {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with tab2:
    import streamlit as st
    import yfinance as yf
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import datetime as dt
    import google.generativeai as genai
    from math import ceil
    
    import os
    tokenAI = "AIzaSyDjFAIJkM_2TIlJOTG_rmj7mS6f8IVWG-s"
    
    def translate_with_gemini(text):
        try:
            genai.configure(api_key=tokenAI)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                f"Traduce y resume en español el siguiente texto de forma concisa y clara. Usa lenguaje natural y profesional. No agregues encabezados ni introducciones. Solo devuelve el texto sintetizado en español: {text}"
            )
            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            elif hasattr(response, 'candidates'):
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return text
        except Exception as e:
            return f"Error al traducir: {str(e)}"
    
    st.title("📈 Optimizador de Portafolios con Simulación Monte Carlo")
    
    tickers_input = st.text_input(
        "Introduce los tickers separados por comas (ej. AAPL, MSFT, TSLA):",
        value=", ".join(st.session_state.get("tickers_portafolio", []))
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    
    if tickers:
        if st.button("Generar Portafolio"):
            end_date = dt.date.today()
            start_date = end_date - dt.timedelta(days=5*365)
    
            data = {}
            for ticker in tickers:
                try:
                    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                    # st.write(f"Columnas descargadas para {ticker}: {df.columns.tolist()}")
                    # st.write(f"Primeras filas de {ticker}:")
                    # st.dataframe(df.head())
                    df_close = df["Close"]
                    if isinstance(df_close, pd.DataFrame):
                        df_close = df_close.iloc[:, 0]
                    if isinstance(df_close, pd.Series) and not df_close.empty:
                        data[ticker] = df_close
                    else:
                        st.warning(f"Datos inválidos o vacíos para {ticker}, se omitirá.")
                except Exception as e:
                    st.error(f"No se pudieron obtener datos para {ticker}: {e}")
    
            if not data:
                st.error("No se obtuvieron datos válidos para ninguno de los tickers. Por favor intenta con otros.")
                st.stop()
    
            # Mostrar datos fundamentales de todas las empresas válidas
            st.markdown("<h2 style='text-align: center;'>📂 Acciones en portafolio</h2>", unsafe_allow_html=True)
    
            # Determinar número óptimo de columnas
            num_tickers = len(data)
            if num_tickers <= 3:
                cols = st.columns(num_tickers)
            else:
                cols = st.columns(2)
    
            # Mostrar la información en las columnas
            for i, (tck, serie) in enumerate(data.items()):
                with cols[i % len(cols)]:
                    try:
                        stock = yf.Ticker(tck)
                        info = stock.info  # Obtener info una sola vez para el ticker
                        st.markdown(f"<h3 style='text-align: center; margin-top: 0.5em;'>{tck}</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: justify;'><b>Nombre:</b> {info.get('longName', 'No disponible')}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align: justify;'><b>Sector:</b> {info.get('sector', 'No disponible')}</div>", unsafe_allow_html=True)
                        descripcion = info.get('longBusinessSummary', 'No disponible')
                        descripcion_traducida = translate_with_gemini(descripcion)
                        st.markdown(f"<div style='text-align: justify;'><b>Descripción:</b> {descripcion_traducida}</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"No se pudo obtener la información fundamental de {tck}: {e}")
    
            st.markdown("<br><br>", unsafe_allow_html=True)
    
            # Mostrar CAGR y Volatilidad lado a lado
            col1, col2 = st.columns(2)
    
            cagr_data = []
            vol_data = []
    
            for ticker, series in data.items():
                try:
                    dias = (series.index[-1] - series.index[0]).days
                    años_totales = dias / 365
                    fila = {"Ticker": ticker}
    
                    for años in [1, 3, 5]:
                        target_date = series.index[-1] - pd.DateOffset(years=años)
                        nearest_index = series.index.get_indexer([target_date], method='nearest')[0]
                        if series.index[nearest_index] >= series.index[0]:
                            precio_inicio = series.iloc[nearest_index]
                            precio_final = series.iloc[-1]
                            cagr = (precio_final / precio_inicio) ** (1 / años) - 1
                            fila[f"{años} año(s)"] = f"{cagr:.2%}"
                        else:
                            fila[f"{años} año(s)"] = "No disponible"
                    cagr_data.append(fila)
                except Exception as e:
                    st.warning(f"Error al calcular CAGR para {ticker}: {e}")
    
            with col1:
                st.markdown("### Rendimientos Anualizados (CAGR)")
                st.markdown("Este cálculo se basa en el crecimiento compuesto del precio desde el inicio hasta el final de cada periodo. Se muestra el rendimiento anualizado para 1, 3 y 5 años (si hay datos suficientes).")
                if cagr_data:
                    st.dataframe(pd.DataFrame(cagr_data))
    
            for ticker, series in data.items():
                try:
                    daily_returns = series.pct_change().dropna()
                    std_diaria = np.std(daily_returns)
                    vol_anual = std_diaria * np.sqrt(252)
                    vol_data.append({"Ticker": ticker, "Volatilidad Anual": f"{vol_anual:.2%}"})
                except Exception as e:
                    st.warning(f"Error al calcular la volatilidad para {ticker}: {e}")
    
            with col2:
                st.markdown("###  Volatilidad Anualizada")
                st.markdown("La volatilidad anualizada representa el riesgo del activo, calculado como la desviación estándar de los rendimientos diarios multiplicada por la raíz cuadrada de 252 (días hábiles por año).")
    
                if vol_data:
                    st.dataframe(pd.DataFrame(vol_data))
    


            # Mostrar gráfica histórica del benchmark

            if data:
                benchmark_ticker = "SPY"
                benchmark_df = yf.download(benchmark_ticker, start=start_date, end=end_date, auto_adjust=False)
                benchmark_returns = pd.Series(
                    np.log(benchmark_df["Close"].values.flatten()[1:] / benchmark_df["Close"].values.flatten()[:-1]),
                    index=benchmark_df["Close"].index[1:]
                )
                benchmark_cum_values = (1 + benchmark_returns).cumprod().values.flatten()
                benchmark_cum_return = pd.Series(benchmark_cum_values, index=benchmark_returns.index)

            benchmark_returns = np.log(benchmark_df["Close"] / benchmark_df["Close"].shift(1)).dropna()
    
            df_prices = pd.DataFrame(data)
            log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
    
            num_assets = len(tickers)
            num_portfolios = 5000
            results = np.zeros((3, num_portfolios))
            weights_list = []
    
            risk_free_rate = 0.044  # Tasa libre de riesgo estimada de EE.UU.
            progress_bar = st.progress(0)
    
            for i in range(num_portfolios):
                weights = np.random.random(num_assets)
                weights /= np.sum(weights)
                weights_list.append(weights)
                annual_return = np.sum(weights * log_returns.mean()) * 250
                annual_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 250, weights)))
    
                results[0, i] = annual_return
                results[1, i] = annual_volatility
                results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]  # Sharpe Ratio ajustado
                
                progress_bar.progress((i + 1) / num_portfolios)
    
            progress_bar.empty()
    
            results_df = pd.DataFrame({
                'Return': results[0],
                'Volatility': results[1],
                'Sharpe Ratio': results[2]
            })

            # === BLOQUE ALTAR INTERACTIVO ===
            import altair as alt

            st.markdown("### 🌐 Visualización de Frontera eficiente")
            alt_chart = alt.Chart(results_df).mark_circle(size=60).encode(
                x=alt.X('Volatility', title='Volatilidad Esperada'),
                y=alt.Y('Return', title='Rendimiento Esperado'),
                color=alt.Color('Sharpe Ratio', scale=alt.Scale(scheme='turbo')),
                tooltip=['Return', 'Volatility', 'Sharpe Ratio']
            ).interactive().properties(width=700, height=400)

            st.altair_chart(alt_chart, use_container_width=True)
    
            # Comparar benchmark vs portafolio con mayor Sharpe Ratio
            max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
            optimal_return = results_df.loc[max_sharpe_idx, 'Return']
            optimal_volatility = results_df.loc[max_sharpe_idx, 'Volatility']
            optimal_weights = weights_list[max_sharpe_idx]
            
            # Obtener índices de los tres portafolios clave
            idx_max_sharpe = results_df['Sharpe Ratio'].idxmax()
            idx_min_vol = results_df['Volatility'].idxmin()
            idx_max_return = results_df['Return'].idxmax()
            
            # Extraer resultados
            port_keys = {
                "➕ Máximo Sharpe Ratio": idx_max_sharpe,
                "➖ Mínima Volatilidad": idx_min_vol,
                "🔝 Máximo Retorno": idx_max_return
            }

            # Inicializar portafolios_guardados en el estado de sesión si no existe
            if "portafolios_guardados" not in st.session_state:
                st.session_state["portafolios_guardados"] = []

            # Mostrar pestañas con cada portafolio
            st.markdown("<h2 style='text-align: center;'>🎯 Comparativa de Portafolios Óptimos</h2>", unsafe_allow_html=True)
            tabs = st.tabs(list(port_keys.keys()))

            for i, (nombre, idx) in enumerate(port_keys.items()):
                with tabs[i]:
                    st.markdown(f"<h3 style='text-align: center; margin-top: 0.5em;'>{nombre}</h3>", unsafe_allow_html=True)
                    port_return = results_df.loc[idx, 'Return']
                    port_vol = results_df.loc[idx, 'Volatility']
                    port_sharpe = results_df.loc[idx, 'Sharpe Ratio']
                    port_weights = weights_list[idx]
                    port_pesos_dict = {tickers[j]: f"{w:.2%}" for j, w in enumerate(port_weights)}
                    port_pesos_df = pd.DataFrame.from_dict(port_pesos_dict, orient='index', columns=['Peso'])

                    col1, col2 = st.columns([1.1, 1.5])

                    with col1:
                        st.markdown(f"<p style='font-size: 16px;'>• <b>Rendimiento Esperado:</b> {port_return:.2%}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 16px;'>• <b>Volatilidad Esperada:</b> {port_vol:.2%}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 16px;'>• <b>Sharpe Ratio:</b> {port_sharpe:.2f} ℹ️ <span title='Relación entre rendimiento y riesgo. Mientras mayor sea, mejor compensación por riesgo.' style='cursor: help;'>[?]</span></p>", unsafe_allow_html=True)

                    with col2:
                        st.markdown("#### Composición del Portafolio")
                        st.dataframe(port_pesos_df)
                        # Composición Visual
                        fig_pie = go.Figure(data=[go.Pie(labels=list(port_pesos_dict.keys()), values=[float(w.strip('%')) for w in port_pesos_dict.values()], hole=0.3)])
                        fig_pie.update_layout(title="Distribución Visual del Portafolio", height=400)
                        st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{i}")

                    # --- NUEVO BLOQUE CENTRADO ---
                    with st.container():
                        left, center, right = st.columns([1, 6, 1])
                        with center:
                            st.subheader("📊 Métricas del Portafolio")
                            port_returns_series = pd.Series(log_returns @ port_weights, index=log_returns.index)
                            benchmark_aligned = benchmark_returns.loc[log_returns.index.intersection(benchmark_returns.index)]
                            common_index = port_returns_series.index.intersection(benchmark_aligned.index)
                            port_aligned = port_returns_series.loc[common_index]
                            bench_aligned = benchmark_aligned.loc[common_index]

                            if len(port_aligned) > 1 and len(bench_aligned) > 1:
                                common_index = port_aligned.index.intersection(bench_aligned.index)
                                port_vals = port_aligned.loc[common_index].values.flatten()
                                bench_vals = bench_aligned.loc[common_index].values.flatten()
                                if len(port_vals) > 1 and len(bench_vals) > 1 and len(port_vals) == len(bench_vals):
                                    cov_matrix = np.cov(port_vals, bench_vals)
                                    beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
                                else:
                                    beta = np.nan
                            else:
                                beta = np.nan

                            benchmark_mean = float(benchmark_aligned.mean()) * 252
                            expected_return = risk_free_rate + beta * (benchmark_mean - risk_free_rate)
                            alpha = float(port_return - expected_return)

                            aligned_benchmark = benchmark_aligned.reindex(port_returns_series.index).dropna()
                            aligned_portfolio = port_returns_series.loc[aligned_benchmark.index]

                            if len(aligned_portfolio) > 1:
                                tracking_diff = aligned_portfolio.values - aligned_benchmark.values
                                tracking_error = float(np.std(tracking_diff) * np.sqrt(252))
                            else:
                                tracking_error = np.nan

                            correl_matrix = df_prices.pct_change().dropna().corr()
                            upper_triangle = correl_matrix.where(np.triu(np.ones(correl_matrix.shape), k=1).astype(bool))
                            correl_promedio = float(upper_triangle.stack().mean())

                            metricas = pd.DataFrame({
                                "Métrica": [
                                    "Beta ℹ️",
                                    "Alpha ℹ️",
                                    "Tracking Error ℹ️",
                                    "Correlación Promedio ℹ️"
                                ],
                                "Valor": [
                                    f"{beta:.2f}" if not np.isnan(beta) else "N/A",
                                    f"{alpha:.2%}" if not np.isnan(alpha) else "N/A",
                                    f"{tracking_error:.2%}" if not np.isnan(tracking_error) else "N/A",
                                    f"{correl_promedio:.2f}"
                                ],
                                "Descripción": [
                                    "Sensibilidad del portafolio frente al mercado (S&P 500).",
                                    "Rendimiento extra sobre lo esperado según su riesgo (modelo CAPM).",
                                    "Diferencia estándar frente al benchmark.",
                                    "Relación promedio entre los activos del portafolio."
                                ]
                            })
                            st.dataframe(metricas)

                            # Comparativa con Benchmark fijo (SPY)
                            benchmark_annual_return = benchmark_returns.mean() * 252
                            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                            benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility

                            comparativa = pd.DataFrame({
                                "Métrica": ["Rendimiento Anual", "Volatilidad Anual", "Sharpe Ratio"],
                                "Portafolio": [f"{port_return:.2%}", f"{port_vol:.2%}", f"{port_sharpe:.2f}"],
                                "Benchmark (SPY)": [
                                    f"{float(benchmark_annual_return):.2%}",
                                    f"{float(benchmark_volatility):.2%}",
                                    f"{float(benchmark_sharpe):.2f}"
                                ]
                            })
                            st.subheader("📊 Comparativa con el Benchmark (SPY)")
                            st.dataframe(comparativa)

                            # === Comparativa con Benchmark ===
                            # === Selección interactiva de Benchmark (mover aquí) ===
                            # Benchmark fijo: SPY
                            selected_benchmark = "S&P 500 (SPY)"

                            st.subheader("📈 Rendimiento Acumulado Comparado")
                            port_daily_returns = pd.Series(log_returns @ port_weights, index=log_returns.index)
                            port_cum_returns = (1 + port_daily_returns).cumprod()

                            if benchmark_cum_return is not None:
                                comparison_df = pd.DataFrame({
                                    'Portafolio': port_cum_returns,
                                    "S&P 500 (SPY)": benchmark_cum_return
                                }).dropna()
                                st.line_chart(comparison_df)

                            st.subheader("🧠 Conclusión del Portafolio")
                            try:
                                pesos_dict = {tickers[j]: f"{w:.2%}" for j, w in enumerate(port_weights)}
                                genai.configure(api_key=tokenAI)
                                model = genai.GenerativeModel('gemini-1.5-flash')
                                prompt_conclusion = f"""
Analiza el siguiente portafolio de inversión con base en estos datos clave:

- Composición: {pesos_dict}
- Rendimiento esperado anual: {port_return:.2%}
- Volatilidad anual estimada: {port_vol:.2%}
- Comparativa frente al benchmark (SPY): basada en rendimiento acumulado

Redacta una conclusión clara y profesional en español. Sé directo y técnico, enfocándote en si el portafolio tiene buena relación riesgo-rendimiento, si está diversificado, y si supera al benchmark. No uses saludos ni recomendaciones vagas. Presenta el análisis de forma precisa, sin adornos, usando un lenguaje conciso y fácil de leer.
"""
                                response = model.generate_content(prompt_conclusion)
                                conclusion_text = response.text.strip() if hasattr(response, 'text') and response.text else "No se pudo generar la conclusión."
                                st.markdown(f"<div style='text-align: justify; font-size: 18px;'>{conclusion_text}</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"No se pudo generar la conclusión con IA: {str(e)}")



    