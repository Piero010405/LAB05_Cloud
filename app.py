import streamlit as st
import joblib
import pickle
import numpy as np
import psycopg2
import pandas as pd

import psycopg2
# Fetch variables
# CADENA SUPABASE
# postgresql://postgres.vjazwlsapjqtgddzpzya:[YOUR-PASSWORD]@aws-1-us-east-2.pooler.supabase.com:6543/postgres

USER = "postgres.vjazwlsapjqtgddzpzya" #os.getenv("user")
PASSWORD = "USIL$2025#$%" # os.getenv("password")
HOST = "aws-1-us-east-2.pooler.supabase.com" #os.getenv("host")
PORT = "6543" #os.getenv("port")
DBNAME = "postgres" #os.getenv("dbname")

# Configuración de la página
st.set_page_config(page_title="Predictor de Iris", page_icon="🌸")

# Función para obtener una conexión a la base de datos
@st.cache_resource
def get_connection():
    """
    Establece y retorna una conexión a la base de datos de Supabase.
    Usa st.cache_resource para reutilizar la conexión entre re-ejecuciones de la app.
    """
    try:
        connection = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        return connection
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return None

# Función para cargar los modelos
@st.cache_resource
def load_models():
    try:
        model = joblib.load('components/iris_model.pkl')
        scaler = joblib.load('components/iris_scaler.pkl')
        with open('components/model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo.")
        return None, None, None


# Riesgo Crediticio
@st.cache_resource
def load_credit_model():
    try:
        with open('components/demo_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("No se encontró el archivo del modelo de crédito.")
        return None

# Título
st.title("🌸 Predictor de Especies de Iris")

tab1, tab2 = st.tabs(["Iris", "Riesgo Crediticio"])

# ------------------------------
# Tab 1: Iris
# ------------------------------
with tab1:
    # Cargar modelos
    model, scaler, model_info = load_models()

    # Obtener la conexión a la base de datos
    conn = get_connection()

    if conn is not None:
        try:
            # Prueba de conexión inicial y muestra la hora actual
            cursor = conn.cursor()
            cursor.execute("SELECT NOW();")
            db_time = cursor.fetchone()[0]
            st.success(f"Conexión a la base de datos exitosa. Hora del servidor: {db_time}")
            cursor.close()
        except Exception as e:
            st.error(f"Error en la consulta de prueba: {e}")
            conn = None

    if model is not None:
        # Inputs
        st.header("Ingresa las características de la flor:")
        
        sepal_length = st.number_input("Longitud del Sépalo (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        sepal_width = st.number_input("Ancho del Sépalo (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        petal_length = st.number_input("Longitud del Pétalo (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
        petal_width = st.number_input("Ancho del Pétalo (cm)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        # Botón de predicción
        if st.button("Predecir Especie"):
            # Preparar datos
            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            # Estandarizar
            features_scaled = scaler.transform(features)
            
            # Predecir
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Mostrar resultado
            target_names = model_info['target_names']
            predicted_species = target_names[prediction]
            
            st.success(f"Especie predicha: **{predicted_species}**")
            st.write(f"Confianza: **{max(probabilities):.1%}**")
            
            # Mostrar todas las probabilidades
            st.write("Probabilidades:")
            for species, prob in zip(target_names, probabilities):
                st.write(f"- {species}: {prob:.1%}")

            # --- Parte nueva para guardar la predicción ---
            if conn is not None:
                try:
                    cursor = conn.cursor()
                    # SQL INSERT con los nombres de columna de tu tabla
                    insert_query = """
                    INSERT INTO table_iris (ls, "as", lp, ap, prediction) 
                    VALUES (%s, %s, %s, %s, %s);
                    """
                    # Los valores deben coincidir con el orden de las columnas en la query
                    values_to_insert = (sepal_length, sepal_width, petal_length, petal_width, predicted_species)
                    cursor.execute(insert_query, values_to_insert)
                    conn.commit()
                    st.info("Predicción guardada en la base de datos.")
                except Exception as e:
                    st.error(f"Error al guardar la predicción: {e}")
                finally:
                    cursor.close()
            # --- Fin de la parte nueva ---

        # --- Parte nueva para mostrar el historial de predicciones ---
        st.markdown("---")
        st.header("Historial de Predicciones")
        
        if conn is not None:
            try:
                cursor = conn.cursor()
                # SQL SELECT para obtener todos los datos, ordenados por los más recientes
                cursor.execute("SELECT created_at, ls, \"as\", lp, ap, prediction FROM table_iris ORDER BY created_at DESC;")
                records = cursor.fetchall()
                
                if records:
                    # Nombres de las columnas para el DataFrame
                    column_names = ["Fecha y Hora", "Longitud Sépalo", "Ancho Sépalo", "Longitud Pétalo", "Ancho Pétalo", "Predicción"]
                    # Convertir los resultados a un DataFrame de pandas
                    df = pd.DataFrame(records, columns=column_names)
                    # Mostrar el DataFrame en Streamlit
                    st.dataframe(df)
                else:
                    st.info("Aún no hay predicciones en la base de datos.")
            except Exception as e:
                st.error(f"Error al cargar el historial de predicciones: {e}")
            finally:
                cursor.close()
        # --- Fin de la parte nueva ---
        
# ------------------------------
# Tab 2: Riesgo Crediticio
# ------------------------------
with tab2:
    credit_model = load_credit_model()
    if credit_model is not None:
        st.header("🏦 Predicción de Riesgo Crediticio")

        # Inputs
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        income = st.number_input("Ingreso Mensual Neto", min_value=0, value=2000)
        missed_pmnt = st.number_input("Número de pagos atrasados", min_value=0, value=0)
        active_tl = st.number_input("Líneas de crédito activas", min_value=0, value=1)
        marital_status = st.selectbox("Estado Civil", ["Soltero", "Casado"])
        gender = st.selectbox("Género", ["Femenino", "Masculino"])
        time_employed = st.number_input("Años en el empleo actual", min_value=0, value=5)

        # Convertir variables categóricas
        married_flag = 1 if marital_status == "Casado" else 0
        single_flag = 1 if marital_status == "Soltero" else 0
        female_flag = 1 if gender == "Femenino" else 0
        male_flag = 1 if gender == "Masculino" else 0

        # Calcular un Credit Score ficticio
        def fake_credit_score(age, income, missed_pmnt):
            base = 600
            score = base + (income/1000) - (missed_pmnt*20) + (age/2)
            return max(300, min(900, int(score)))

        credit_score = fake_credit_score(age, income, missed_pmnt)
        st.info(f"💳 Credit Score simulado: **{credit_score}**")

        if st.button("Predecir Aprobación"):
            features = np.array([[
                age, income, credit_score, active_tl, missed_pmnt,
                married_flag, single_flag, female_flag, male_flag,
                time_employed
            ]])

            prediction = credit_model.predict(features)[0]

            st.success(f"Resultado del modelo: **{prediction}**")