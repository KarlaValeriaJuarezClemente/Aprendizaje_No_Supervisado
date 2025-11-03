import os
import logging 
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from difflib import SequenceMatcher
from functools import lru_cache
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CREAR CARPETAS NECESARIAS AUTOMÁTICAMENTE
carpeta_datos = 'Datos'
carpeta_kpis = 'KPIs'
carpeta_ans = 'ANS'

# Crear todas las carpetas necesarias
os.makedirs(carpeta_datos, exist_ok=True)
os.makedirs(carpeta_kpis, exist_ok=True)
os.makedirs(carpeta_ans, exist_ok=True)

print(f"Carpetas creadas/verificadas:")
print(f"  - {carpeta_datos}")
print(f"  - {carpeta_kpis}")
print(f"  - {carpeta_ans}")

ruta_sueno = os.path.join(carpeta_datos, 'calidad_de_sueño.xlsx')
ruta_salud = os.path.join(carpeta_datos, 'salud_mental.xlsx')

try:
    if not os.path.exists(ruta_sueno):
        raise FileNotFoundError(f"Archivo no encontrado en la ruta: {ruta_sueno}")
    
    if ruta_sueno.endswith('.csv'):
        test_sueno = pd.read_csv(ruta_sueno, nrows=1)
    else:
        test_sueno = pd.read_excel(ruta_sueno, nrows=1)
    print("  Archivo de sueño es legible.")
except Exception as e:
    print(f"Error fatal leyendo archivo de sueño: {e}")
    raise

try:
    if not os.path.exists(ruta_salud):
        raise FileNotFoundError(f"Archivo no encontrado en la ruta: {ruta_salud}")

    if ruta_salud.endswith('.csv'):
        test_salud = pd.read_csv(ruta_salud, nrows=1)
    else:
        test_salud = pd.read_excel(ruta_salud, nrows=1)
    print("  Archivo de salud es legible.")
except Exception as e:
    print(f"Error fatal leyendo archivo de salud: {e}")
    raise

ruta_salida_xlsx = 'Datos/Datos_Limpios_Unificados.xlsx'
ruta_salida_csv = 'Datos/Datos_Limpios_Unificados.csv'

class Limpiador:
    def __init__(self):
        # mapeo plano para respuestas
        self.mapeo_si_no_flat = {
            'sí': 'Sí', 'si': 'Sí', 's': 'Sí', 'yes': 'Sí', 'y': 'Sí', '1': 'Sí','sip': 'Sí','sipi': 'Sí','afirmativo': 'Sí', 'correcto': 'Sí', 'claro': 'Sí', 'si tengo': 'Sí',
                   'cuento con': 'Sí', 'tengo': 'Sí', 'por supuesto': 'Sí', 'obvio': 'Sí', 'siempre': 'Sí', 'definitivamente': 'Sí',
                   'Si, la verdad.': 'Sí', 'si, la verdad': 'Sí', 'concierto que si': 'Sí', 'afirmativo': 'Sí','todo el maldito tiempo': 'Sí', 'mucho': 'Sí', 'cada día': 'Sí', 'diario': 'Sí', 'todo el tiempo': 'Sí',
                   'si la verdad': 'Sí',
            'no': 'No', 'n': 'No', '0': 'No', 'nope': 'No', 'negativo': 'No', 'para nada': 'No', 'ninguno': 'No', 'nada': 'No',
                  'jamás': 'No', 'no tengo': 'No', 'no cuento': 'No', 'concierto que no': 'No','considero que no': 'No', 'no según yo': 'No', 'no según lo': 'No'
        }
        # patrones más flexibles para detectar columnas por tipo
        self.patrones_horas = ['hora', 'horas', 'duerm', 'sueño', 'sueno', 'dormir', 'sleep', 'descanso', 'descansa']
        self.patrones_desvelo = ['desvel', 'trasnoch', 'noche', 'insomnio', 'vela', 'no dormir']
        self.patrones_estres = ['estres', 'estrés', 'stress', 'abrum', 'presión', 'presion', 'tensión', 'tension']
        self.patrones_ansiedad = ['ansiedad', 'ansios', 'nervios', 'preocup', 'inquiet', 'angustia']
        # posibles nombres ID
        self.patrones_id = ['numero', 'número', 'num', 'num_cuenta', 'cuenta', 'no_cuenta', 'matricula', 'matrícula', 'id', 'código', 'codigo', 'identificacion']

    @staticmethod
    @lru_cache(maxsize=10000)
    def similitud_texto(a: str, b: str) -> float:
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    def normalizar_si_no(self, valor):
        if pd.isna(valor):
            return np.nan
        s = str(valor).strip().lower()
        # quitar tildes simples y espacios
        s_clean = s.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        if s_clean in self.mapeo_si_no_flat:
            return self.mapeo_si_no_flat[s_clean]
        # heurísticos más robustos
        if any(k in s_clean for k in ['si', 'sí', 'yes', 'siempre', 'tengo', 'afirmativo', 'claro', 'correcto']):
            return 'Sí'
        if any(k in s_clean for k in ['no', 'nunca', 'ningun', 'ninguno', 'nada', 'negativo', 'nope']):
            return 'No'
        return valor  # deja el original si no se puede normalizar

    def extraer_numero(self, valor, rango: Tuple[float,float] = (0, 100)):
        if pd.isna(valor):
            return np.nan
        s = str(valor).lower().strip()
        # mapear números escritos
        mapping = {
            'cero':0,'uno':1,'dos':2,'tres':3,'cuatro':4,'cinco':5,'seis':6,'siete':7,'ocho':8,'nueve':9,'diez':10,
            'once':11,'doce':12,'trece':13,'catorce':14,'quince':15
        }
        for k,v in mapping.items():
            if re.search(rf'\b{k}\b', s):
                if rango[0] <= v <= rango[1]:
                    return float(v)
        # buscar número - patrones más flexibles
        m = re.search(r'(\d+[\.,]?\d*)', s)
        if m:
            try:
                num = float(m.group(1).replace(',', '.'))
                if rango[0] <= num <= rango[1]:
                    return num
            except:
                return np.nan
        # buscar rangos (ej: "6-7 horas")
        m_rango = re.search(r'(\d+)[\s\-]+(\d+)', s)
        if m_rango:
            try:
                num1 = float(m_rango.group(1))
                num2 = float(m_rango.group(2))
                promedio = (num1 + num2) / 2
                if rango[0] <= promedio <= rango[1]:
                    return promedio
            except:
                pass
        return np.nan

    def detectar_columna_id(self, df: pd.DataFrame) -> Optional[str]:
        # prioridad: columnas que contienen patrones exactos
        for c in df.columns:
            lc = c.lower()
            for p in self.patrones_id:
                if p in lc:
                    return c
        # fallback: buscar por similitud con 'numero cuenta'
        referencia = 'numero de cuenta'
        mejor = (None, 0.0)
        for c in df.columns:
            sim = self.similitud_texto(c, referencia)
            if sim > 0.6:  # Umbral más alto para evitar falsos positivos
                return c
        return None

    def detectar_columnas_por_tipo(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        cols = {'horas': [], 'desvelo': [], 'estres': [], 'ansiedad': []}
        for c in df.columns:
            lc = c.lower()
            if any(p in lc for p in self.patrones_horas):
                cols['horas'].append(c)
            if any(p in lc for p in self.patrones_desvelo):
                cols['desvelo'].append(c)
            if any(p in lc for p in self.patrones_estres):
                cols['estres'].append(c)
            if any(p in lc for p in self.patrones_ansiedad):
                cols['ansiedad'].append(c)
        return cols

    def aplicar_limpieza(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalizar strings y tratar valores vacíos
        for c in df.select_dtypes(include=['object', 'string']).columns:
            df[c] = df[c].astype('string').str.strip()
            df[c] = df[c].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'null': pd.NA})

        # detectar columnas por tipo
        tipos = self.detectar_columnas_por_tipo(df)
        print(f"Columnas detectadas por tipo: {tipos}")

        # para horas: crear columna estandarizada
        for c in tipos['horas']:
            nueva_col = f'{c}_estandarizada'
            df[nueva_col] = df[c].apply(lambda x: self.extraer_numero(x, (0,24)))
            print(f"Columna {c} -> {nueva_col}: {df[nueva_col].notna().sum()} valores extraídos")

        # para desvelos: 0-7 días por semana
        for c in tipos['desvelo']:
            nueva_col = f'{c}_estandarizada'
            df[nueva_col] = df[c].apply(lambda x: self.extraer_numero(x, (0,7)))
            print(f"Columna {c} -> {nueva_col}: {df[nueva_col].notna().sum()} valores extraídos")

        # normalizar si/no en estres y ansiedad
        for c in tipos['estres']:
            df[c] = df[c].apply(self.normalizar_si_no)
            print(f"Columna {c} normalizada: {df[c].value_counts().to_dict()}")

        for c in tipos['ansiedad']:
            df[c] = df[c].apply(self.normalizar_si_no)
            print(f"Columna {c} normalizada: {df[c].value_counts().to_dict()}")

        return df

limpiador = Limpiador()

print("\n Limpiador inicializado.\n")

try:
    if ruta_sueno.endswith('.csv'):
        df_sueno = pd.read_csv(ruta_sueno)
    else:
        df_sueno = pd.read_excel(ruta_sueno)
        
    if ruta_salud.endswith('.csv'):
        df_salud = pd.read_csv(ruta_salud)
    else:
        df_salud = pd.read_excel(ruta_salud)

    logging.info("Archivos cargados: sueno %s, salud %s", df_sueno.shape, df_salud.shape)
    print(f"Columnas en sueño: {list(df_sueno.columns)}")
    print(f"Columnas en salud: {list(df_salud.columns)}")
except Exception as e:
    logging.error("Error al leer archivos: %s", e)
    raise

# Aplicar limpieza
df_sueno_clean = limpiador.aplicar_limpieza(df_sueno)
df_salud_clean = limpiador.aplicar_limpieza(df_salud)

# Mostrar detecciones rápidas
id_sueno = limpiador.detectar_columna_id(df_sueno_clean)
id_salud = limpiador.detectar_columna_id(df_salud_clean)
print("ID detectado en Sueño:", id_sueno)
print("ID detectado en Salud:", id_salud)

"""#Homologación"""

def elegir_id_col(df1, df2, limpiador_obj: Limpiador) -> Tuple[Optional[str], Optional[str], str]:
    id1 = limpiador_obj.detectar_columna_id(df1)
    id2 = limpiador_obj.detectar_columna_id(df2)
    # estandarizar nombre final
    nombre_id_final = 'numero_cuenta'
    # renombrar si existen
    if id1:
        df1 = df1.rename(columns={id1: nombre_id_final})
    if id2:
        df2 = df2.rename(columns={id2: nombre_id_final})
    return df1, df2, nombre_id_final

df_sueno_clean, df_salud_clean, id_col = elegir_id_col(df_sueno_clean, df_salud_clean, limpiador)
print("ID usado para merge:", id_col)
# Asegurar tipo string y strip
for df in [df_sueno_clean, df_salud_clean]:
    if id_col in df.columns:
        df[id_col] = df[id_col].astype('string').str.strip()

"""#Unificación"""

def combinar_columnas_similares(df, base_name):
    candidates = [c for c in df.columns if base_name in c.lower()]
    if len(candidates) <= 1:
        return df
    # crear columna combinada priorizando columnas no nulas en orden
    df[base_name] = df[candidates].bfill(axis=1).iloc[:,0]
    return df

if id_col and id_col in df_sueno_clean.columns and id_col in df_salud_clean.columns:
    df_unificado = pd.merge(df_sueno_clean, df_salud_clean, on=id_col, how='outer', suffixes=('_sueno','_salud'))
    logging.info("Merge outer realizado por ID. Resultado: %s", df_unificado.shape)
else:
    print("No se pudo hacer merge por ID, concatenando por filas...")
    df_sueno_clean['_fuente'] = 'sueno'
    df_salud_clean['_fuente'] = 'salud'
    df_unificado = pd.concat([df_sueno_clean, df_salud_clean], ignore_index=True, sort=False)
    logging.info("Concatenacion por filas. Resultado: %s", df_unificado.shape)

# Combinar columnas similares
df_unificado = combinar_columnas_similares(df_unificado, 'estres')
df_unificado = combinar_columnas_similares(df_unificado, 'ansiedad')

# combina horas estandarizadas si hay más de una
horas_cols = [c for c in df_unificado.columns if c.endswith('_estandarizada') and any(p in c.lower() for p in ['hora','sueno','sueño','duerm','sleep'])]
if horas_cols:
    df_unificado['horas_sueno_estandarizadas'] = df_unificado[horas_cols].bfill(axis=1).iloc[:,0]
    print(f"Columnas de horas combinadas: {horas_cols}")

print("Columnas resultantes:", list(df_unificado.columns)[:15])
print("\nPrimeras filas del dataset unificado:")
print(df_unificado.head()) 

try:
    df_unificado.to_excel(ruta_salida_xlsx, index=False, engine='openpyxl')
    df_unificado.to_csv(ruta_salida_csv, index=False)
    logging.info("Archivo unificado guardado: %s  (y %s)", ruta_salida_xlsx, ruta_salida_csv)
except Exception as e:
    logging.error("Error guardando archivo final: %s", e)
    raise

def calcular_kpis(df: pd.DataFrame) -> Dict[str, float]:
    kpis = {}

    # Horas de sueño - buscar cualquier columna con horas estandarizadas
    horas_cols = [c for c in df.columns if 'hora' in c.lower() and 'estandarizada' in c.lower()]
    if not horas_cols:
        # Buscar columnas numéricas que podrían contener horas
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dropna().between(0, 24).any():
                horas_cols = [col]
                break

    if horas_cols:
        col_horas = horas_cols[0]
        s = df[col_horas].dropna()
        s = pd.to_numeric(s, errors='coerce').dropna()
        if len(s) > 0:
            kpis['horas_promedio'] = float(s.mean())
            kpis['horas_mediana'] = float(s.median())
            kpis['horas_std'] = float(s.std())
            kpis['pct_insuficientes'] = float((s < 6).mean() * 100)
            kpis['pct_adecuado_7_9'] = float(((s >= 7) & (s <= 9)).mean() * 100)
            print(f"KPIs de sueño calculados usando columna: {col_horas}")
            print(f"  Muestra: {len(s)} registros, Rango: {s.min():.1f}-{s.max():.1f}h")

    # Estres y ansiedad (buscar columnas que contenghan estos términos)
    for name in ['estres', 'ansiedad']:
        posibles_cols = [c for c in df.columns if name in c.lower()]
        for col in posibles_cols:
            ser = df[col].dropna()
            if len(ser) > 0:
                # Contar respuestas afirmativas
                si_count = ser.astype(str).str.lower().str.contains('sí|si|yes|1', na=False).sum()
                kpis[f'pct_{name}_si'] = float((si_count / len(ser)) * 100)
                print(f"KPI {name} calculado usando columna: {col}")
                print(f"  Muestra: {len(ser)}, Sí: {si_count}, No: {len(ser) - si_count}")
                break  # Usar la primera columna encontrada

    logging.info("KPIs calculados: %s", kpis)
    return kpis

kpis = calcular_kpis(df_unificado)
print("\nKPIs calculados:")
for k, v in kpis.items():
    print(f"  {k}: {v:.2f}")

"""Visualización KPIs"""

def crear_dashboard_kpis(kpis, df_unificado):
    # Determinar cuántos subplots necesitamos
    num_plots = 0
    tiene_sueno = 'horas_promedio' in kpis and not pd.isna(kpis['horas_promedio'])
    tiene_estres = 'pct_estres_si' in kpis and not pd.isna(kpis['pct_estres_si'])
    tiene_ansiedad = 'pct_ansiedad_si' in kpis and not pd.isna(kpis['pct_ansiedad_si'])

    if tiene_sueno:
        num_plots += 2
    if tiene_estres:
        num_plots += 1
    if tiene_ansiedad:
        num_plots += 1

    if num_plots == 0:
        print("No hay suficientes datos para crear el dashboard.")
        return

    # Configurar subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    plot_idx = 0

    # Distribución de horas de sueño
    if tiene_sueno:
        horas_cols = [c for c in df_unificado.columns if 'hora' in c.lower() and 'estandarizada' in c.lower()]
        if horas_cols:
            data_sueno = df_unificado[horas_cols[0]].dropna()
            data_sueno = pd.to_numeric(data_sueno, errors='coerce').dropna()

            if len(data_sueno) > 0:
                axes[plot_idx].hist(data_sueno, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                axes[plot_idx].axvline(kpis['horas_promedio'], color='red', linestyle='--',
                                     label=f'Promedio: {kpis["horas_promedio"]:.1f}h')
                axes[plot_idx].axvline(kpis['horas_mediana'], color='green', linestyle='--',
                                     label=f'Mediana: {kpis["horas_mediana"]:.1f}h')
                axes[plot_idx].set_title('Distribución de Horas de Sueño', fontweight='bold')
                axes[plot_idx].set_xlabel('Horas de sueño')
                axes[plot_idx].set_ylabel('Frecuencia')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1

    # KPIs de Sueño
    if tiene_sueno:
        kpis_sueno = {k: v for k, v in kpis.items() if any(x in k for x in ['horas', 'pct_insuficientes', 'pct_adecuado'])}
        kpis_sueno = {k: v for k, v in kpis_sueno.items() if not pd.isna(v)}

        if kpis_sueno:
            labels = [k.replace('_', ' ').title() for k in kpis_sueno.keys()]
            values = list(kpis_sueno.values())
            bars = axes[plot_idx].bar(labels, values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'][:len(values)])
            axes[plot_idx].set_title('KPIs de Calidad de Sueño', fontweight='bold')
            axes[plot_idx].tick_params(axis='x', rotation=45)
            for bar, value in zip(bars, values):
                axes[plot_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                  f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            plot_idx += 1

    # Estrés
    if tiene_estres:
        estres_si = kpis['pct_estres_si']
        estres_no = 100 - estres_si
        sizes_estres = [estres_si, estres_no]
        labels_pie = ['Sí', 'No']
        colors = ['#ff6b6b', '#c8d6e5']

        axes[plot_idx].pie(sizes_estres, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[plot_idx].set_title('Prevalencia de Estrés', fontweight='bold')
        plot_idx += 1

    # Ansiedad
    if tiene_ansiedad:
        ansiedad_si = kpis['pct_ansiedad_si']
        ansiedad_no = 100 - ansiedad_si
        sizes_ansiedad = [ansiedad_si, ansiedad_no]
        labels_pie = ['Sí', 'No']
        colors = ['#ff6b6b', '#c8d6e5']

        axes[plot_idx].pie(sizes_ansiedad, labels=labels_pie, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[plot_idx].set_title('Prevalencia de Ansiedad', fontweight='bold')
        plot_idx += 1

    # Ocultar ejes no utilizados
    for i in range(plot_idx, 4):
        axes[i].axis('off')

    plt.tight_layout()
    
    save_path = os.path.join(carpeta_kpis, 'General_KPIs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"KPIs guardado en: {save_path}")

print("KPIs Completo")
crear_dashboard_kpis(kpis, df_unificado)

"""
EDA Univariado
"""
num_cols = df_unificado.select_dtypes(include=[np.number]).columns.tolist()
print("Columnas numéricas detectadas:", num_cols)

if num_cols:
    # Filtrar columnas que realmente tienen datos
    num_cols_validas = []
    for col in num_cols:
        non_na_count = df_unificado[col].notna().sum()
        if non_na_count > 0:
            num_cols_validas.append(col)
            print(f"  {col}: {non_na_count} valores no nulos")

    if num_cols_validas:
        resumen = df_unificado[num_cols_validas].describe().T
        print(resumen) # cambiado display por print

        # Histogramas y boxplots solo para columnas con datos
        for c in num_cols_validas[:4]:  # Mostrar máximo 4 para no saturar
        
            nombre_limpio = c.replace('_estandarizada', '').replace('_', ' ').capitalize()

            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Histograma
            data_clean = df_unificado[c].dropna()
            if len(data_clean) > 0:
                # Convertir a numérico por si acaso
                data_clean = pd.to_numeric(data_clean, errors='coerce').dropna()

                if len(data_clean) > 0:
                    # Calcular estadísticas antes de graficar
                    mean_val = data_clean.mean()
                    median_val = data_clean.median()
                    std_val = data_clean.std()
                    min_val = data_clean.min()
                    max_val = data_clean.max()

                    # Crear histograma
                    n, bins, patches = ax1.hist(data_clean, bins=15, alpha=0.7, color='skyblue',
                                              edgecolor='black', density=False)

                    # Evitar superposición
                    max_freq = max(n)
                    bin_width = bins[1] - bins[0]

                    # Posicionar el texto en un área menos congestionada
                    text_x = min_val + (max_val - min_val) * 0.02  # 2% desde el inicio
                    text_y = max_freq * 0.85  # 85% de la altura máxima

                    stats_text = f'Media: {mean_val:.2f}\nMediana: {median_val:.2f}\nDesv: {std_val:.2f}\nMín: {min_val:.2f}\nMáx: {max_val:.2f}\nN: {len(data_clean)}'

                    # Añadir estadísticas en una posición fija que no interfiera
                    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
                            fontsize=9, family='monospace')

                    # Añadir líneas de media y mediana
                    ax1.axvline(mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Media: {mean_val:.2f}')
                    ax1.axvline(median_val, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Mediana: {median_val:.2f}')

                    ax1.set_title(f'Distribución: {nombre_limpio}', fontweight='bold', pad=20)
                    ax1.set_xlabel(nombre_limpio)
                    ax1.set_ylabel('Frecuencia')
                    ax1.legend(loc='upper right')
                    ax1.grid(True, alpha=0.3)

                    # Ajustar límites para mejor visualización
                    ax1.set_xlim(min_val - bin_width, max_val + bin_width)

            # Boxplot en el segundo subplot
            if len(data_clean) > 0:
                # Crear boxplot
                boxplot = ax2.boxplot(data_clean, vert=True, patch_artist=True,
                                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                                    medianprops=dict(color='darkred', linewidth=2),
                                    flierprops=dict(marker='o', markerfacecolor='red', markersize=4))

                # Añadir puntos de datos para mostrar distribución real
                y = data_clean.values
                x = np.random.normal(1, 0.04, size=len(y))
                ax2.scatter(x, y, alpha=0.4, color='blue', s=20)

                # Estadísticas para el boxplot
                q1 = data_clean.quantile(0.25)
                q3 = data_clean.quantile(0.75)
                iqr = q3 - q1

                box_stats = f'Q1: {q1:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}'
                ax2.text(0.95, 0.95, box_stats, transform=ax2.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=9, family='monospace')

                ax2.set_title(f'Boxplot: {nombre_limpio}', fontweight='bold', pad=20)
                ax2.set_ylabel(nombre_limpio)
                ax2.set_xlabel('Distribución')
                ax2.grid(True, alpha=0.3)

                # Ocultar el eje x ya que solo tenemos una variable
                ax2.set_xticks([])

            else:
                # Si no hay datos, mostrar mensaje
                ax1.text(0.5, 0.5, 'No hay datos\nnuméricos',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax1.transAxes, fontsize=12, color='red')
                ax1.set_title(f'Distribución: {nombre_limpio}', fontweight='bold')

                ax2.text(0.5, 0.5, 'No hay datos\nnuméricos',
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=12, color='red')
                ax2.set_title(f'Boxplot: {nombre_limpio}', fontweight='bold')

            plt.tight_layout()

            filename_c = nombre_limpio.replace(' ', '_').replace(':', '') \
                                      .replace('?', '').replace('¿', '') \
                                      .replace('.', '').replace('/', '') \
                                      .replace('\\', '')
            save_path = os.path.join(carpeta_kpis, f'{filename_c}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig) 
            print(f"Gráfico univariado guardado en: {save_path}")

            if len(data_clean) > 0:
                print(f"\nResumen estadístico para '{nombre_limpio}':")
                print(f"   Rango: {min_val:.2f} - {max_val:.2f}")
                print(f"   Media ± Desv: {mean_val:.2f} ± {std_val:.2f}")
                print(f"   Mediana: {median_val:.2f}")
                print(f"   Asimetría: {data_clean.skew():.2f}")
                print(f"   Curtosis: {data_clean.kurtosis():.2f}")
                print(f"   Valores únicos: {data_clean.nunique()}")
                print("-" * 50)

    else:
        print("No se detectaron columnas numéricas con datos para EDA univariado.")
else:
    print("No se detectaron columnas numéricas para EDA univariado.")

"""EDA Bivariado"""

if len(num_cols) >= 2:
    # Filtrar solo columnas verdaderamente numéricas
    numeric_cols_clean = []
    for col in num_cols:
        try:
            # Intentar convertir a numérico
            pd.to_numeric(df_unificado[col].dropna(), errors='raise')
            numeric_cols_clean.append(col)
        except:
            print(f"Advertencia: La columna '{col}' contiene valores no numéricos y será excluida del análisis de correlación")

    if len(numeric_cols_clean) >= 2:
        corr = df_unificado[numeric_cols_clean].corr()

        # Heatmap
    if len(num_cols) >= 2:
        corr = df_unificado[num_cols].corr()

        # Crea una lista de nombres limpios
        clean_names = [col.replace('_estandarizada', '').replace('_', ' ').capitalize() 
                       for col in corr.columns]
        # Asigna los nombres limpios al índice y columnas del DataFrame de correlación
        corr.columns = clean_names
        corr.index = clean_names

        fig_corr = plt.figure(figsize=(10,8))
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
        plt.title('Matriz de correlaciones (numéricas)')

        save_path = os.path.join(carpeta_kpis, 'Correlacion.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig_corr) 
        print(f"Heatmap de correlación guardado en: {save_path}")
        
        print("Top correlaciones absolutas (excluye 1.0):")
        corr_vals = corr.abs().unstack().sort_values(ascending=False).drop_duplicates()
        print(corr_vals[(corr_vals < 1)].head(10)) 
    else:
        print("No hay suficientes columnas numéricas para correlación bivariada.")

    # Si hay horas y desvelo, scatter plots con manejo de errores
    if 'horas_sueno_estandarizadas' in df_unificado.columns:
        posibles = [c for c in df_unificado.columns if 'desvelo' in c.lower() or 'desvel' in c.lower() or 'dias' in c.lower()]

    if posibles:
        for p in posibles:
            try:
                nombre_limpio_p = p.replace('_estandarizada', '').replace('_', ' ').capitalize()

                # Crear copia temporal y convertir a numérico
                tmp = df_unificado[['horas_sueno_estandarizadas', p]].copy()

                # Convertir ambas columnas a numérico, forzando errores a NaN
                tmp['horas_sueno_estandarizadas'] = pd.to_numeric(tmp['horas_sueno_estandarizadas'], errors='coerce')
                tmp[p] = pd.to_numeric(tmp[p], errors='coerce')

                # Eliminar filas con NaN
                tmp = tmp.dropna()

                if len(tmp) > 5:
                    # --- MODIFICACIÓN ---
                    # Asignar la figura a una variable
                    fig_scatter = plt.figure(figsize=(8, 6))
                    # --- FIN DE MODIFICACIÓN ---

                    # Scatter plot
                    scatter = sns.scatterplot(data=tmp, x=p, y='horas_sueno_estandarizadas',
                                             alpha=0.6, s=60)

                    # Calcular correlación
                    corr_val = tmp['horas_sueno_estandarizadas'].corr(tmp[p])

                    # Añadir línea de tendencia solo si hay suficientes puntos
                    if len(tmp) > 1:
                        try:
                            z = np.polyfit(tmp[p], tmp['horas_sueno_estandarizadas'], 1)
                            p_line = np.poly1d(z)
                            plt.plot(tmp[p], p_line(tmp[p]), "r--", alpha=0.8, label='Línea de tendencia')
                        except:
                            print(f"No se pudo calcular la línea de tendencia para {p}")

                    # Añadir correlación en recuadro separado
                    plt.text(0.05, 0.95, f'Correlación: {corr_val:.2f}',
                            transform=plt.gca().transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            verticalalignment='top', fontsize=12)

                    plt.title(f'Horas sueño vs {nombre_limpio_p}', fontweight='bold')
                    plt.xlabel(nombre_limpio_p)
                    plt.ylabel('Horas de sueño') # También limpiamos esta etiqueta

                    # Añadir leyenda si hay línea de tendencia
                    if len(tmp) > 1:
                        plt.legend()

                    plt.tight_layout()

                    filename_p = nombre_limpio_p.replace(' ', '_').replace(':', '') \
                                                .replace('?', '').replace('¿', '') \
                                                .replace('.', '').replace('/', '') \
                                                .replace('\\', '')
                    save_path = os.path.join(carpeta_kpis, 'Sueño_vs_Desvelo.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    plt.close(fig_scatter) 
                    print(f"Gráfico scatter guardado en: {save_path} ({len(tmp)} puntos válidos)")


                else:
                    print(f"No hay suficientes datos válidos para {p} (solo {len(tmp)} puntos)")

            except Exception as e:
                print(f"Error al procesar {p}: {str(e)}")
                continue

class KMeansOptimizer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_params = {}
        self.results = {}

    def preparar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Preparando datos para clustering...")
        
        # Primero buscar columnas estandarizadas que sabemos son relevantes
        columnas_estandarizadas = [col for col in df.columns if 'estandarizada' in col.lower()]
        
        # Si no hay estandarizadas, buscar columnas numéricas con patrones específicos
        if not columnas_estandarizadas:
            patrones_relevantes = [
                'hora', 'horas', 'sueño', 'sueno', 'duerm', 'sleep', 'descanso',
                'desvelo', 'desvela', 'trasnoch', 'noche', 'insomnio',
                'estres', 'estrés', 'stress', 
                'ansiedad', 'ansios', 'nervios',
                'puntuacion', 'puntaje', 'score'
            ]
            
            columnas_relevantes = []
            for col in df.select_dtypes(include=[np.number]).columns:
                col_lower = col.lower()
                if any(patron in col_lower for patron in patrones_relevantes):
                    columnas_relevantes.append(col)
            
            # Si aún no hay suficientes, agregar todas las numéricas
            if len(columnas_relevantes) < 3:
                todas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
                columnas_relevantes.extend([col for col in todas_numericas if col not in columnas_relevantes])
        else:
            columnas_relevantes = columnas_estandarizadas
        
        # Filtrar columnas con suficiente completitud (>60%)
        datos_clustering = df[columnas_relevantes].copy()
        completitud = datos_clustering.notna().mean()
        columnas_validas = completitud[completitud > 0.6].index.tolist()
        
        if not columnas_validas:
            print("Advertencia: No hay columnas con suficiente completitud, usando todas disponibles")
            columnas_validas = columnas_relevantes
        
        datos_clustering = datos_clustering[columnas_validas]
        
        # Imputar valores faltantes con la mediana
        datos_clustering = datos_clustering.fillna(datos_clustering.median())
        
        # Eliminar columnas con varianza cercana a cero
        varianzas = datos_clustering.var()
        columnas_varianza = varianzas[varianzas > 0.01].index.tolist()
        datos_clustering = datos_clustering[columnas_varianza]
        
        print(f"Columnas seleccionadas para clustering ({len(columnas_varianza)}): {list(datos_clustering.columns)}")
        print(f"Forma final de los datos: {datos_clustering.shape}")
        print(f"Completitud: {datos_clustering.notna().mean().mean():.1%}")
        
        return datos_clustering

    def escalar_datos(self, datos: pd.DataFrame) -> np.ndarray:
        print("Escalando datos...")
        datos_escalados = self.scaler.fit_transform(datos)
        print(f"Datos escalados: {datos_escalados.shape}")
        return datos_escalados

    def metodo_elbow_mejorado(self, datos_escalados: np.ndarray, max_k: int = 12) -> Dict:
        print("Aplicando metodo del codo mejorado...")
        inercia = []
        silhouette_scores = []
        calinski_scores = []
        davies_scores = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
                etiquetas = kmeans.fit_predict(datos_escalados)
                inercia.append(kmeans.inertia_)

                # Calcular metricas solo si hay al menos 2 clusters y suficientes muestras
                if len(np.unique(etiquetas)) > 1:
                    silhouette_avg = silhouette_score(datos_escalados, etiquetas)
                    calinski_avg = calinski_harabasz_score(datos_escalados, etiquetas)
                    davies_avg = davies_bouldin_score(datos_escalados, etiquetas)
                    
                    silhouette_scores.append(silhouette_avg)
                    calinski_scores.append(calinski_avg)
                    davies_scores.append(davies_avg)
                else:
                    silhouette_scores.append(-1)
                    calinski_scores.append(-1)
                    davies_scores.append(float('inf'))
                    
                print(f"   k={k}: Inercia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
                    
            except Exception as e:
                print(f"   Error con k={k}: {e}")
                inercia.append(np.nan)
                silhouette_scores.append(-1)
                calinski_scores.append(-1)
                davies_scores.append(float('inf'))

        # Encontrar punto de codo usando el metodo de la segunda derivada
        inercia_valida = [x for x in inercia if not np.isnan(x)]
        if len(inercia_valida) > 2:
            primera_derivada = np.diff(inercia_valida)
            segunda_derivada = np.diff(primera_derivada)
            # El codo esta donde la segunda derivada es maxima (mas negativa)
            elbow_point = np.argmin(segunda_derivada) + 3  # +3 por los diffs y empezar en k=2
            elbow_point = min(elbow_point, len(k_range) - 1)
        else:
            elbow_point = 2

        # Encontrar mejor k segun silhouette score
        silhouette_valores = [s for s in silhouette_scores if s > 0]
        if silhouette_valores:
            best_silhouette_idx = np.argmax(silhouette_valores)
            best_silhouette_k = k_range[best_silhouette_idx]
            best_silhouette_score = max(silhouette_valores)
        else:
            best_silhouette_k = 2
            best_silhouette_score = -1

        # Encontrar mejor k segun Calinski-Harabasz
        calinski_valores = [c for c in calinski_scores if c > 0]
        if calinski_valores:
            best_calinski_k = k_range[np.argmax(calinski_valores)]
            best_calinski_score = max(calinski_valores)
        else:
            best_calinski_k = 2
            best_calinski_score = -1

        # Decision final: priorizar silhouette score si es bueno, sino usar codo
        if best_silhouette_score > 0.5:
            k_final = best_silhouette_k
            metodo = "silhouette"
        elif best_silhouette_score > 0.3:
            k_final = best_silhouette_k
            metodo = "silhouette_moderado"
        else:
            k_final = elbow_point
            metodo = "codo"

        resultados = {
            'k_range': list(k_range),
            'inercia': inercia,
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores,
            'davies_scores': davies_scores,
            'elbow_k': elbow_point,
            'best_silhouette_k': best_silhouette_k,
            'best_silhouette_score': best_silhouette_score,
            'best_calinski_k': best_calinski_k,
            'best_calinski_score': best_calinski_score,
            'k_final': k_final,
            'metodo_seleccion': metodo
        }

        return resultados

    def grid_search_optimization(self, datos_escalados: np.ndarray, k_range: list = None) -> Dict:
        print("Optimizando con GridSearchCV...")
        
        if k_range is None:
            # Usar un rango mas inteligente basado en el tamano de los datos
            n_muestras = len(datos_escalados)
            max_k_reasonable = min(10, n_muestras // 10)  # Maximo 10 clusters o 10 muestras por cluster
            k_range = list(range(2, max(3, max_k_reasonable + 1)))
        
        print(f"   Rango de k probado: {k_range}")

        param_grid = {
            'n_clusters': k_range,
            'init': ['k-means++'],  # Solo k-means++ para mejor convergencia
            'n_init': [15, 20],
            'max_iter': [300, 500],
            'algorithm': ['lloyd', 'elkan'],
            'random_state': [42]
        }

        kmeans = KMeans()
        
        try:
            grid_search = GridSearchCV(
                estimator=kmeans,
                param_grid=param_grid,
                scoring='silhouette_score',
                cv=min(5, len(datos_escalados) // 10),  # CV adaptativo
                n_jobs=-1,
                verbose=0,
                error_score='raise'
            )

            grid_search.fit(datos_escalados)
            
            resultados = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'exitoso': True
            }
            
            print(f"   Mejores parametros GridSearch: {grid_search.best_params_}")
            print(f"   Mejor score GridSearch: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"   Error en GridSearchCV: {e}")
            print("   Usando configuracion por defecto...")
            # Configuracion por defecto robusta
            resultados = {
                'best_params': {'n_clusters': 3, 'init': 'k-means++', 'n_init': 20, 'max_iter': 300, 'random_state': 42},
                'best_score': -1,
                'best_estimator': None,
                'cv_results': {},
                'exitoso': False
            }

        return resultados

    def randomized_search_optimization(self, datos_escalados: np.ndarray, n_iter: int = 25) -> Dict:
        print("Optimizando con RandomizedSearchCV...")
        
        n_muestras = len(datos_escalados)
        max_k_reasonable = min(12, n_muestras // 5)

        param_dist = {
            'n_clusters': randint(2, max_k_reasonable + 1),
            'init': ['k-means++', 'random'],
            'n_init': randint(10, 25),
            'max_iter': randint(200, 500),
            'algorithm': ['lloyd', 'elkan'],
            'random_state': [42]
        }

        kmeans = KMeans()
        
        try:
            random_search = RandomizedSearchCV(
                estimator=kmeans,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring='silhouette_score',
                cv=min(5, len(datos_escalados) // 10),
                n_jobs=-1,
                random_state=42,
                verbose=0
            )

            random_search.fit(datos_escalados)
            
            resultados = {
                'best_params': random_search.best_params_,
                'best_score': random_search.best_score_,
                'best_estimator': random_search.best_estimator_,
                'cv_results': random_search.cv_results_,
                'exitoso': True
            }
            
            print(f"   Mejores parametros RandomizedSearch: {random_search.best_params_}")
            print(f"   Mejor score RandomizedSearch: {random_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"   Error en RandomizedSearchCV: {e}")
            print("   Usando configuracion por defecto...")
            resultados = {
                'best_params': {'n_clusters': 3, 'init': 'k-means++', 'n_init': 20, 'max_iter': 300, 'random_state': 42},
                'best_score': -1,
                'best_estimator': None,
                'cv_results': {},
                'exitoso': False
            }

        return resultados

    def entrenar_modelo_robusto(self, datos_escalados: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
        print(f"Entrenando modelo K-means con k={k}...")
        
        # Parametros robustos para mejor convergencia
        kmeans = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=25,  # Mas inicializaciones para estabilidad
            max_iter=400,
            tol=1e-6,   # Tolerancia mas estricta
            random_state=42,
            algorithm='lloyd'
        )
        
        etiquetas = kmeans.fit_predict(datos_escalados)
        print("   Modelo entrenado exitosamente")
        
        return kmeans, etiquetas

    def evaluar_modelo(self, modelo, datos_escalados: np.ndarray, etiquetas: np.ndarray) -> Dict:
        print("Evaluando modelo...")
        
        try:
            metricas = {
                'inercia': modelo.inertia_,
                'silhouette_score': silhouette_score(datos_escalados, etiquetas),
                'calinski_harabasz_score': calinski_harabasz_score(datos_escalados, etiquetas),
                'davies_bouldin_score': davies_bouldin_score(datos_escalados, etiquetas),
                'n_clusters': len(np.unique(etiquetas)),
                'tamaño_clusters': np.bincount(etiquetas),
                'porcentaje_clusters': (np.bincount(etiquetas) / len(etiquetas)) * 100,
                'balance_clusters': np.std(np.bincount(etiquetas)) / np.mean(np.bincount(etiquetas))  # Coeficiente de variacion
            }
        except Exception as e:
            print(f"   Error calculando algunas metricas: {e}")
            metricas = {
                'inercia': modelo.inertia_,
                'silhouette_score': -1,
                'calinski_harabasz_score': -1,
                'davies_bouldin_score': -1,
                'n_clusters': len(np.unique(etiquetas)),
                'tamaño_clusters': np.bincount(etiquetas),
                'porcentaje_clusters': (np.bincount(etiquetas) / len(etiquetas)) * 100,
                'balance_clusters': -1
            }
        
        return metricas

    def interpretar_calidad_clustering(self, metricas: Dict) -> str:
        silhouette = metricas['silhouette_score']
        
        if silhouette > 0.7:
            return "EXCELENTE"
        elif silhouette > 0.5:
            return "BUENA"
        elif silhouette > 0.3:
            return "MODERADA"
        elif silhouette > 0.1:
            return "DEBIL"
        else:
            return "POCO ESTRUCTURADO"

    def visualizar_resultados(self, datos_escalados: np.ndarray, etiquetas: np.ndarray,
                            datos_originales: pd.DataFrame, metodo_elbow: Dict):
        print("Generando visualizaciones...")

        # Crear figura principal
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Metodo del codo
        axes[0, 0].plot(metodo_elbow['k_range'], metodo_elbow['inercia'], 'bo-', linewidth=2, markersize=6)
        axes[0, 0].axvline(metodo_elbow['k_final'], color='red', linestyle='--', linewidth=2,
                          label=f'k seleccionado = {metodo_elbow["k_final"]}')
        axes[0, 0].set_xlabel('Numero de Clusters (k)')
        axes[0, 0].set_ylabel('Inercia')
        axes[0, 0].set_title('Metodo del Codo - Seleccion de k', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Silhouette scores
        axes[0, 1].plot(metodo_elbow['k_range'], metodo_elbow['silhouette_scores'], 'go-', linewidth=2, markersize=6)
        axes[0, 1].axvline(metodo_elbow['k_final'], color='red', linestyle='--', linewidth=2,
                          label=f'k seleccionado = {metodo_elbow["k_final"]}')
        axes[0, 1].set_xlabel('Numero de Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Silhouette Score vs Numero de Clusters', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Visualizacion PCA
        try:
            pca = PCA(n_components=2)
            datos_pca = pca.fit_transform(datos_escalados)

            scatter = axes[1, 0].scatter(datos_pca[:, 0], datos_pca[:, 1], c=etiquetas,
                                       cmap='viridis', alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
            axes[1, 0].set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
            axes[1, 0].set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
            axes[1, 0].set_title('Visualizacion 2D de Clusters (PCA)', fontweight='bold')
            plt.colorbar(scatter, ax=axes[1, 0])
            axes[1, 0].grid(True, alpha=0.3)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error en PCA: {e}', ha='center', va='center', 
                           transform=axes[1, 0].transAxes, fontsize=10)
            axes[1, 0].set_title('Visualizacion 2D de Clusters')

        # 4. Distribucion de clusters
        unique, counts = np.unique(etiquetas, return_counts=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
        bars = axes[1, 1].bar(unique, counts, color=colors, alpha=0.8, edgecolor='black')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Numero de Muestras')
        axes[1, 1].set_title('Distribucion de Clusters', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Anotar valores en las barras
        for bar, count in zip(bars, counts):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{count}\n({count/len(etiquetas)*100:.1f}%)', 
                           ha='center', va='bottom', fontweight='bold', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(carpeta_ans, 'KMeans_Analisis_Completo.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Heatmap de caracteristicas por cluster
        try:
            datos_con_clusters = datos_originales.copy()
            datos_con_clusters['Cluster'] = etiquetas
            
            promedios_cluster = datos_con_clusters.groupby('Cluster').mean()
            
            fig, ax = plt.subplots(figsize=(max(12, len(promedios_cluster.columns) * 0.8), 
                                           max(8, len(promedios_cluster) * 0.6)))
            
            sns.heatmap(promedios_cluster.T, 
                       annot=True, 
                       cmap='RdYlBu_r', 
                       fmt='.2f',
                       center=0,
                       cbar_kws={'label': 'Valor Promedio Estandarizado'},
                       ax=ax)
            
            ax.set_title('Caracteristicas Promedio por Cluster', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
            ax.set_ylabel('Variables', fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=0)
            plt.yticks(rotation=0)
            
            plt.tight_layout()
            save_path = os.path.join(carpeta_ans, 'Cluster_Caracteristicas.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar datos del heatmap
            promedios_cluster.T.to_csv(os.path.join(carpeta_ans, 'Cluster_Caracteristicas_Data.csv'))
            
        except Exception as e:
            print(f"Error generando heatmap: {str(e)}")

    def ejecutar_analisis_completo(self, df: pd.DataFrame) -> Dict:
        print("\n" + "-"*30)
        print("ANALISIS K-MEANS")
        print("-"*30)

        # 1. Preparar datos
        print("\n PREPARANDO DATOS...")
        datos_originales = self.preparar_datos(df)
        
        if datos_originales.empty:
            print("ERROR: No hay datos validos para clustering")
            return {}
            
        datos_escalados = self.escalar_datos(datos_originales)
        
        if datos_escalados.size == 0:
            print("ERROR: No hay datos escalados validos")
            return {}

        # 2. Determinar k optimo
        print("\n DETERMINANDO NUMERO OPTIMO DE CLUSTERS...")
        metodo_elbow = self.metodo_elbow_mejorado(datos_escalados)
        k_optimo = metodo_elbow['k_final']
        
        print(f"   k seleccionado: {k_optimo}")
        print(f"   Metodo de seleccion: {metodo_elbow['metodo_seleccion']}")
        print(f"   Silhouette score esperado: {metodo_elbow['best_silhouette_score']:.3f}")

        # 3. Optimizacion con GridSearchCV
        print("\n OPTIMIZACION CON GRIDSEARCHCV...")
        grid_results = self.grid_search_optimization(datos_escalados)

        # 4. Optimizacion con RandomizedSearchCV
        print("\n OPTIMIZACION CON RANDOMIZEDSEARCHCV...")
        random_results = self.randomized_search_optimization(datos_escalados)

        # 5. Seleccionar y entrenar mejor modelo
        print("\n SELECCIONANDO Y ENTRENANDO MEJOR MODELO...")
        
        # Estrategia de seleccion robusta
        grid_score = grid_results['best_score'] if grid_results['exitoso'] else -1
        random_score = random_results['best_score'] if random_results['exitoso'] else -1
        
        if grid_score > 0 and grid_score >= random_score:
            self.best_model = grid_results['best_estimator']
            self.best_params = grid_results['best_params']
            print("   Modelo seleccionado: GridSearchCV")
            etiquetas = self.best_model.labels_
        elif random_score > 0:
            self.best_model = random_results['best_estimator']
            self.best_params = random_results['best_params']
            print("   Modelo seleccionado: RandomizedSearchCV")
            etiquetas = self.best_model.labels_
        else:
            # Fallback: usar k del metodo del codo
            print("   Usando metodo del codo como fallback")
            self.best_model, etiquetas = self.entrenar_modelo_robusto(datos_escalados, k_optimo)
            self.best_params = {'n_clusters': k_optimo}

        # 6. Evaluar modelo final
        print("\nEVALUANDO MODELO FINAL...")
        metricas = self.evaluar_modelo(self.best_model, datos_escalados, etiquetas)
        calidad = self.interpretar_calidad_clustering(metricas)

        # Mostrar resultados finales
        print("\n" + "-"*44)
        print("RESULTADOS FINALES DEL CLUSTERING")
        print("-"*44)
        print(f"Numero de clusters: {metricas['n_clusters']}")
        print(f"Silhouette Score: {metricas['silhouette_score']:.4f}")
        print(f"Calinski-Harabasz Score: {metricas['calinski_harabasz_score']:.4f}")
        print(f"Davies-Bouldin Score: {metricas['davies_bouldin_score']:.4f}")
        print(f"Inercia: {metricas['inercia']:.4f}")
        print(f"Balance de clusters: {metricas['balance_clusters']:.3f}")
        print(f"Tamano de clusters: {metricas['tamaño_clusters']}")
        print(f"Distribucion (%): {metricas['porcentaje_clusters']}")
        print(f"CALIDAD DEL CLUSTERING: {calidad}")
        print("="*70)

        # Generar visualizaciones
        self.visualizar_resultados(datos_escalados, etiquetas, datos_originales, metodo_elbow)

        # Preparar resultados completos
        self.results = {
            'best_model': self.best_model,
            'best_params': self.best_params,
            'labels': etiquetas,
            'metrics': metricas,
            'calidad': calidad,
            'elbow_results': metodo_elbow,
            'grid_search_results': grid_results,
            'randomized_search_results': random_results,
            'scaled_data': datos_escalados,
            'original_data': datos_originales
        }

        # Guardar resultados
        self.guardar_resultados()

        return self.results

    def guardar_resultados(self):
        try:
            # Guardar metricas principales
            resultados_df = pd.DataFrame({
                'Metrica': [
                    'Numero de Clusters',
                    'Silhouette Score',
                    'Calinski-Harabasz Score',
                    'Davies-Bouldin Score',
                    'Inercia',
                    'Balance de Clusters',
                    'Calidad'
                ],
                'Valor': [
                    self.results['metrics']['n_clusters'],
                    f"{self.results['metrics']['silhouette_score']:.4f}",
                    f"{self.results['metrics']['calinski_harabasz_score']:.4f}",
                    f"{self.results['metrics']['davies_bouldin_score']:.4f}",
                    f"{self.results['metrics']['inercia']:.4f}",
                    f"{self.results['metrics']['balance_clusters']:.3f}",
                    self.results['calidad']
                ],
                'Interpretacion': [
                    'Numero de grupos identificados',
                    '-1 a 1 (>0.5 bueno, >0.7 excelente)',
                    'Valores mas altos = mejor separacion',
                    'Valores mas bajos = mejor separacion',
                    'Suma de distancias al cuadrado (menor es mejor)',
                    'Desviacion estandar relativa (menor es mejor)',
                    'Evaluacion general de calidad'
                ]
            })

            resultados_df.to_csv(os.path.join(carpeta_ans, 'KMeans_Resultados.csv'), index=False)

            # Guardar parametros y distribucion detallada
            with open(os.path.join(carpeta_ans, 'KMeans_Resumen_Detallado.txt'), 'w') as f:
                f.write("RESUMEN DEL ANALISIS K-MEANS\n")
                f.write("-" * 33 + "\n\n")
                
                f.write("PARAMETROS OPTIMOS DEL MODELO:\n")
                f.write("-" * 35 + "\n")
                for param, value in self.best_params.items():
                    f.write(f"{param}: {value}\n")
                
                f.write(f"\nMETODO DE SELECCION: {self.results['elbow_results']['metodo_seleccion']}\n")
                
                f.write("\nMETRICAS DE EVALUACION:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Silhouette Score: {self.results['metrics']['silhouette_score']:.4f}\n")
                f.write(f"Calinski-Harabasz: {self.results['metrics']['calinski_harabasz_score']:.4f}\n")
                f.write(f"Davies-Bouldin: {self.results['metrics']['davies_bouldin_score']:.4f}\n")
                f.write(f"Inercia: {self.results['metrics']['inercia']:.4f}\n")
                f.write(f"Balance: {self.results['metrics']['balance_clusters']:.3f}\n")
                f.write(f"CALIDAD: {self.results['calidad']}\n\n")
                
                f.write("DISTRIBUCION DE CLUSTERS:\n")
                f.write("-" * 25 + "\n")
                for i, (tam, pct) in enumerate(zip(self.results['metrics']['tamaño_clusters'], 
                                                 self.results['metrics']['porcentaje_clusters'])):
                    f.write(f"Cluster {i}: {tam} personas ({pct:.1f}%)\n")

            print("Resultados guardados en carpeta ANS/")

        except Exception as e:
            print(f"Error guardando resultados: {e}")

# EJECUTAR EL ANALISIS COMPLETO
print("\n" + ""*33)
print("INICIANDO ANALISIS K-MEANS")
print("-"*33)

# Inicializar y ejecutar el optimizador
kmeans_optimizer = KMeansOptimizer()
resultados_kmeans = kmeans_optimizer.ejecutar_analisis_completo(df_unificado)

# ANALISIS DE INTERPRETACION DE CLUSTERS
if resultados_kmeans:
    print("\n" + "-"*33)
    print("INTERPRETACION DE CLUSTERS")
    print("-"*33)

    datos_con_clusters = resultados_kmeans['original_data'].copy()
    datos_con_clusters['Cluster'] = resultados_kmeans['labels']

    # Calcular estadisticas por cluster
    estadisticas_clusters = datos_con_clusters.groupby('Cluster').agg(['mean', 'std', 'count']).round(3)

    print("Estadisticas descriptivas por Cluster:")
    print(estadisticas_clusters)

    # Analisis de perfil de clusters
    print("\nPERFIL DE CADA CLUSTER:")
    print("-" * 40)
    
    promedios_cluster = datos_con_clusters.groupby('Cluster').mean()
    
    for cluster in promedios_cluster.index:
        print(f"\nCLUSTER {cluster} (n={np.sum(resultados_kmeans['labels'] == cluster)}):")
        # Ordenar caracteristicas por valor mas distintivo
        caracteristicas = promedios_cluster.loc[cluster].sort_values(ascending=False)
        top_3 = caracteristicas.head(3)
        bottom_3 = caracteristicas.tail(3)
        
        print("   Caracteristicas mas altas:")
        for carac, valor in top_3.items():
            print(f"     - {carac}: {valor:.2f}")
            
        print("   Caracteristicas mas bajas:")
        for carac, valor in bottom_3.items():
            print(f"     - {carac}: {valor:.2f}")

    # Guardar datos con clusters
    datos_con_clusters.to_csv('Datos/Datos_Con_Clusters.csv', index=False)
    estadisticas_clusters.to_csv(os.path.join(carpeta_ans, 'Estadisticas_Clusters_Detalladas.csv'))
    
    print("\n" + "-"*33)
    print("ANALISIS COMPLETADO EXITOSAMENTE")
    print("-"*33)
    print("ARCHIVOS GENERADOS:")
    print(f"  - {carpeta_ans}/KMeans_Analisis_Completo.png")
    print(f"  - {carpeta_ans}/Cluster_Caracteristicas.png")
    print(f"  - {carpeta_ans}/KMeans_Resultados.csv")
    print(f"  - {carpeta_ans}/KMeans_Resumen_Detallado.txt")
    print(f"  - {carpeta_ans}/Estadisticas_Clusters_Detalladas.csv")
    print("  - Datos/Datos_Con_Clusters.csv")

else:
    print("\nERROR: No se pudieron obtener resultados del clustering")

"""Mean Shift"""
df_calidad_sueño = pd.read_excel('Datos/calidad_de_sueño.xlsx')

df_calidad_sueño.dropna(inplace=True) # Elimina filas con datos nulos
col_categoricas = ['1. ¿Cuántas horas duermes al día?',
                   '2. Cuando despiertas, ¿Cómo evalúas que hayas dormido? Explica tu respuesta y por qué crees que sea así.',
                   '3. ¿Tomas algún medicamento para dormir?',
                   '4. ¿Consideras que alguna bebida (alcohol, café, té, etc.) te afecta el sueño? ¿Por qué?',
                   '5. ¿Cuántos días te desvelas a la semana?',
                   '6. ¿Consideras que el estrés y los estudios afectan tu calidad de sueño? Explica tu respuesta.',
                   '7. ¿Usas algún dispositivo electrónico antes de dormir?, ¿Con qué frecuencia?',
                   '8. ¿Tienes alguna rutina antes de acostarte?',
                   '9. ¿Realizas algún deporte y/o actividad física?',
                   '10. ¿Cómo consideras tu alimentación?',
                   '11. ¿Crees que tu estado de ánimo influye en tu calidad de sueño?',
                   '12. ¿Te ha pasado que si duermes menos descansas más? ¿Cuál es tu teoría de este suceso?',
                   '13. ¿Tienes algún trastorno del sueño?',
                   '14. ¿Compartes habitación con alguien más? ¿Consideras que eso afecte tu calidad de sueño?',
                   '15. ¿Utilizas algún aparato para ayudar a dormir?',
                   '16. ¿Los fines de semana descansas más?',
                   '17. ¿Te irritas fácilmente si no duermes bien?',
                   '18. ¿Qué tanto afecta tus actividades diarias cuando no duermes bien?']
for col in col_categoricas:
    if col in df_calidad_sueño.columns:
      print(f'Columna {col}: \n {df_calidad_sueño[col].nunique()} subniveles')
    else:
      print(f'Columna {col} not found in DataFrame.')

# --- Imprime los valores únicos de cada columna ---
print("--- Valores Únicos ANTES de estandarizar ---")
for col in col_categoricas:
    if col in df_calidad_sueño.columns:
        print(f"\n--- Columna: {col} ---")
        print(df_calidad_sueño[col].unique()[:20])

#--- Estandariza y mapea los datos ---#

for columna in df_calidad_sueño.columns:
    if columna in col_categoricas:
        df_calidad_sueño[columna] = df_calidad_sueño[columna].str.lower()

# Mapeo para binarios
map_si_no = {
    r'^(s(i|í)|ejercicio|ocupo|calistenia|gimnasio|mas o menos|wsi|correr|a veces|duermo|bastante|mucho|totalmente|demasiados|considero que si).*': 'sí',
    r'.*(\b(posiblemente|creo|depende|no lo se|mmm)\b).*': 'no lo se',
    r'^(n(o|ó)|ninguno|n).*': 'no'
}

# Mapeo para calidad
map_calidad = {
    r'.*(\b(buena|adecuada|variada)\b).*': 'buena',
    r'.*(\b(regu|regular|maso menos|balanceada|normal)\b).*': 'regular',
    r'.*(\b(mala|no)\b).*': 'mala'
}

# Mapeo para valores numericos
map_horas = {
    r'.*(\b(4|cuatro)\b|menos).*': '4 o menos horas',
    r'.*(\b(5|cinco|5hrs)\b).*': '5 horas',
    r'.*(\b(6|seis)\b).*': '6 horas',
    r'.*(\b(7|siete)\b).*': '7 horas',
    r'.*(\b(8|ocho)\b).*': '8 horas',
    r'.*(\b(9|nueve)\b).*': '9 horas',
    r'.*(\b(10|diez)\b|mas).*': '10 o más horas'

}

# Mapeo de frecuencia
map_frecuencia_desvelo = {
    r'.*(\b(0|cero)\b|ninguno|nunca|no).*': '0 días',
    r'.*(\b(1|uno)\b|un dia|pocas veces).*': '1 día',
    r'.*(\b(2|dos)\b).*': '2 días',
    r'.*(\b(3|tres)\b|moderada).*': '3 días',
    r'.*(\b(4|cuatro)\b|bastante|mucha).*': '4 días',
    r'.*(\b(5|cinco)\b).*': '5 días',
    r'.*(\b(6|seis)\b|casi siempre).*': '6 días',
    r'.*(\b(7|siete)\b|todos|diario|siempre|diariamente).*': '7 días'

}

# --- Mapeamos las columnas --- #

# Columnas del tipo SÍ/NO
cols_si_no = [
    '3. ¿Tomas algún medicamento para dormir?',
    '9. ¿Realizas algún deporte y/o actividad física?',
    '11. ¿Crees que tu estado de ánimo influye en tu calidad de sueño?',
    '13. ¿Tienes algún trastorno del sueño?',
    '15. ¿Utilizas algún aparato para ayudar a dormir?',
    '16. ¿Los fines de semana descansas más?',
    '17. ¿Te irritas fácilmente si no duermes bien?'
]

# Columna del tipo calidad
cols_calidad = [
    '10. ¿Cómo consideras tu alimentación?'
]

# Columna de HORAS
cols_horas = ['1. ¿Cuántas horas duermes al día?']

# Columna de FRECUENCIA
cols_frecuencia = [
    '5. ¿Cuántos días te desvelas a la semana?',
    '7. ¿Usas algún dispositivo electrónico antes de dormir?, ¿Con qué frecuencia?'
]


# Remplaza los registros meidante los diccionarios
for col in cols_si_no: # --- Si/No --- #
    if col in df_calidad_sueño.columns:
        df_calidad_sueño[col] = df_calidad_sueño[col].replace(map_si_no, regex=True)

for col in cols_calidad: # --- Calidad --- #
     if col in df_calidad_sueño.columns:
        df_calidad_sueño[col] = df_calidad_sueño[col].replace(map_calidad, regex=True)

for col in cols_horas: # --- Horas --- #
     if col in df_calidad_sueño.columns:
        df_calidad_sueño[col] = df_calidad_sueño[col].replace(map_horas, regex=True)

for col in cols_frecuencia: # --- Horas --- #
     if col in df_calidad_sueño.columns:
        df_calidad_sueño[col] = df_calidad_sueño[col].replace(map_frecuencia_desvelo, regex=True)

# --- Graficas de los datos estandarizados --- #

# Columnas que no se grafican
col_cualitativas = [
    '2. Cuando despiertas, ¿Cómo evalúas que hayas dormido? Explica tu respuesta y por qué crees que sea así.',
    '4. ¿Consideras que alguna bebida (alcohol, café, té, etc.) te afecta el sueño? ¿Por qué?',
    '6. ¿Consideras que el estrés y los estudios afectan tu calidad de sueño? Explica tu respuesta.',
    '8. ¿Tienes alguna rutina antes de acostarte?',
    '12. ¿Te ha pasado que si duermes menos descansas más? ¿Cuál es tu teoría de este suceso?',
    '14. ¿Compartes habitación con alguien más? ¿Consideras que eso afecte tu calidad de sueño?',
    '18. ¿Qué tanto afecta tus actividades diarias cuando no duermes bien?'
]

# --- Graficacion --- #
col_para_graficar = [col for col in col_categoricas if col not in col_cualitativas and col in df_calidad_sueño.columns]
nrows_dinamico = len(col_para_graficar)
'''
fig, ax = plt.subplots(nrows=nrows_dinamico, ncols=1, figsize=(10, nrows_dinamico * 5))
fig.subplots_adjust(hspace=1.5)

for i, col in enumerate(col_para_graficar):
    current_ax = ax[i] if nrows_dinamico > 1 else ax

    sns.countplot(x=col, data=df_calidad_sueño, ax=current_ax, order = df_calidad_sueño[col].value_counts().index)
    current_ax.set_title(col)
    current_ax.set_xticklabels(current_ax.get_xticklabels(), rotation=30)'''

# --- Clustering --- #

df_cluster = df_calidad_sueño[col_para_graficar].copy()

map_calidad_num = {
    'mala': 0,
    'regular': 1,
    'buena': 2
}

map_si_no_num = {
    'no lo se': 0,
    'no': 0,
    'sí': 1
}

map_horas_num = {
    '4 o menos horas': 0,
    '5 horas': 1,
    '6 horas': 2,
    '7 horas': 3,
    '8 horas': 4,
    '9 horas': 5,
    '10 o más horas': 6
}
map_frecuencia_num = {
    '0 días': 0,
    '1 día': 1,
    '2 días': 2,
    '3 días': 3,
    '4 días': 4,
    '5 días': 5,
    '6 días': 6,
    '7 días': 7
}

df_cluster['1. ¿Cuántas horas duermes al día?'] = df_cluster['1. ¿Cuántas horas duermes al día?'].map(map_horas_num)
df_cluster['3. ¿Tomas algún medicamento para dormir?'] = df_cluster['3. ¿Tomas algún medicamento para dormir?'].map(map_si_no_num)
df_cluster['5. ¿Cuántos días te desvelas a la semana?'] = df_cluster['5. ¿Cuántos días te desvelas a la semana?'].map(map_frecuencia_num)
df_cluster['7. ¿Usas algún dispositivo electrónico antes de dormir?, ¿Con qué frecuencia?'] = df_cluster['7. ¿Usas algún dispositivo electrónico antes de dormir?, ¿Con qué frecuencia?'].map(map_frecuencia_num)
df_cluster['9. ¿Realizas algún deporte y/o actividad física?'] = df_cluster['9. ¿Realizas algún deporte y/o actividad física?'].map(map_si_no_num)
df_cluster['10. ¿Cómo consideras tu alimentación?'] = df_cluster['10. ¿Cómo consideras tu alimentación?'].map(map_calidad_num)
df_cluster['11. ¿Crees que tu estado de ánimo influye en tu calidad de sueño?'] = df_cluster['11. ¿Crees que tu estado de ánimo influye en tu calidad de sueño?'].map(map_si_no_num)
df_cluster['13. ¿Tienes algún trastorno del sueño?'] = df_cluster['13. ¿Tienes algún trastorno del sueño?'].map(map_si_no_num)
df_cluster['15. ¿Utilizas algún aparato para ayudar a dormir?'] = df_cluster['15. ¿Utilizas algún aparato para ayudar a dormir?'].map(map_si_no_num)
df_cluster['16. ¿Los fines de semana descansas más?'] = df_cluster['16. ¿Los fines de semana descansas más?'].map(map_si_no_num)
df_cluster['17. ¿Te irritas fácilmente si no duermes bien?'] = df_cluster['17. ¿Te irritas fácilmente si no duermes bien?'].map(map_si_no_num)

df_processed = df_cluster.copy()
df_processed.fillna(0, inplace=True)
print(df_processed.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

print(f"\nDatos listos: {X_scaled.shape[0]} encuestados, {X_scaled.shape[1]} características (columnas numéricas)")

#--- Aplicacion del método --- #
X = X_scaled
bandwidth = estimate_bandwidth(X, quantile=0.25, n_samples=500, random_state=42)

if bandwidth == 0:
    print("El ancho de banda es 0, no se pueden formar clusters. Prueba un 'quantile' más alto (ej. 0.3).")
else:
    print(f"Ancho de Banda estimado (bandwidth): {bandwidth:.2f}")

    # Entrenamiento del modelo
    ms_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms_model.fit(X)
    labels = ms_model.labels_
    cluster_centers = ms_model.cluster_centers_
    n_clusters_ = len(np.unique(labels))

    print(f"Número de clusters estimados automáticamente: {n_clusters_}")

    # --- Grafica --- #

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    centers_pca = pca.transform(cluster_centers)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.7, label='Encuestados')
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1],s=200, c='red', marker='X', linewidths=2, edgecolors='black', label='Centroides (Perfiles)')

    plt.title(f'Clustering Mean Shift de Calidad de Sueño (Clusters: {n_clusters_})')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True, linestyle='-', alpha=0.6)
    
    save_path = os.path.join(carpeta_ans, 'MeanShift_Clustering.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

    df_calidad_sueño['cluster'] = labels

    print("\n--- PERFIL DE LOS CLUSTERS (Moda) ---")
cluster_profiles = df_calidad_sueño.groupby('cluster')[col_para_graficar].agg(lambda x: x.value_counts().index[0] if not x.value_counts().empty else 'Sin Datos')
print(cluster_profiles)


"""DBSCAN"""

# Prepare 'estres' and 'ansiedad' columns for numerical use
mapeo_binario = {'Sí': 1, 'No': 0}

# Create numeric columns
df_unificado['estres_numerico'] = df_unificado['¿Con qué frecuencia te sientes estresado o abrumado? '].map(mapeo_binario)
df_unificado['ansiedad_numerico'] = df_unificado['¿Has sentido ansiedad o preocupación sin una razón clara últimamente? '].map(mapeo_binario)

# Define features for clustering
caracteristicas = [
    '1. ¿Cuántas horas duermes al día?',
    '5. ¿Cuántos días te desvelas a la semana?',
    'estres_numerico',
    'ansiedad_numerico'
]

# Aplicar limpieza a las columnas de texto antes de convertirlas a numéricas
df_clean = df_unificado.copy()

# Mapeo para convertir texto a números (igual que en Mean Shift)
map_horas_num = {
    '4 o menos horas': 0, '5 horas': 1, '6 horas': 2, '7 horas': 3,
    '8 horas': 4, '9 horas': 5, '10 o más horas': 6
}

map_frecuencia_num = {
    '0 días': 0, '1 día': 1, '2 días': 2, '3 días': 3, '4 días': 4,
    '5 días': 5, '6 días': 6, '7 días': 7
}

# Aplicar mapeo si las columnas existen y contienen texto
if '1. ¿Cuántas horas duermes al día?' in df_clean.columns:
    df_clean['1. ¿Cuántas horas duermes al día?'] = df_clean['1. ¿Cuántas horas duermes al día?'].map(map_horas_num)

if '5. ¿Cuántos días te desvelas a la semana?' in df_clean.columns:
    df_clean['5. ¿Cuántos días te desvelas a la semana?'] = df_clean['5. ¿Cuántos días te desvelas a la semana?'].map(map_frecuencia_num)

# Select features and drop rows with missing values
X = df_clean[caracteristicas].dropna()

# Convertir a float para asegurar que sean numéricos
X = X.astype(float)

# Check if there's enough data
if len(X) == 0:
    print("No hay suficientes datos limpios para realizar el clustering DBSCAN.")
else:
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN (recuerda ajustar eps y min_samples)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(X_scaled)

    # --- LÓGICA DE ASIGNACIÓN SIMPLIFICADA ---
    # 1. Inicializa la nueva columna en df_unificado con Nulos (pd.NA)
    df_unificado['cluster_dbscan'] = pd.NA

    # 2. Asigna los clusters SÓLO a las filas que se usaron (las del índice de X)
    df_unificado.loc[X.index, 'cluster_dbscan'] = clusters
    # --- FIN DE LA LÓGICA SIMPLIFICADA ---

    # Print results
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    print(f"DBSCAN clustering performed with {len(X)} samples.")
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")

    print("\nPrimeras filas con clusters DBSCAN (incluyendo filas con NA si no se usaron):")
    print(df_unificado.head())



"""Hierarchical clustering"""

# Seleccionar las columnas numéricas relevantes
# Excluir columnas con muchos NaN o que no sean relevantes para el clustering inicial
clustering_cols = [
    '1. ¿Cuántas horas duermes al día?', # Usando el nombre de columna original que tiene datos
    '5. ¿Cuántos días te desvelas a la semana?', # Usando el nombre de columna original que tiene datos
    # Añadir otras columnas numéricas relevantes si es necesario y tienen suficientes datos
    #'Puntuación' # Incluir si la puntuación es relevante para el clustering
    'estres_numerico', # Incluir columnas numéricas creadas
    'ansiedad_numerico' # Incluir columnas numéricas creadas
]

# Filtrar el DataFrame para incluir solo las columnas seleccionadas
df_clustering = df_unificado[clustering_cols].copy()

print(f"DataFrame original para clustering shape: {df_clustering.shape}")
print("Tipos de datos antes de la conversión:")
print(df_clustering.dtypes)

# Aplicar limpieza y conversión a numérico para las columnas de texto
# Mapeo para convertir texto a números (igual que en Mean Shift)
map_horas_num = {
    '4 o menos horas': 4, '5 horas': 5, '6 horas': 6, '7 horas': 7,
    '8 horas': 8, '9 horas': 9, '10 o más horas': 10,
    '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10
}

map_frecuencia_num = {
    '0 días': 0, '1 día': 1, '2 días': 2, '3 días': 3, '4 días': 4,
    '5 días': 5, '6 días': 6, '7 días': 7,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7
}

# Función para convertir valores a numérico de forma segura
def convertir_a_numerico(valor, mapeo=None):
    if pd.isna(valor):
        return np.nan
    try:
        # Si ya es numérico, devolver como float
        return float(valor)
    except (ValueError, TypeError):
        # Si es texto, aplicar mapeo si existe
        if mapeo and str(valor).strip() in mapeo:
            return mapeo[str(valor).strip()]
        else:
            # Intentar extraer números del texto
            import re
            numeros = re.findall(r'\d+', str(valor))
            if numeros:
                return float(numeros[0])
            else:
                return np.nan

# Aplicar conversión a las columnas de horas y desvelos
if '1. ¿Cuántas horas duermes al día?' in df_clustering.columns:
    df_clustering['1. ¿Cuántas horas duermes al día?'] = df_clustering['1. ¿Cuántas horas duermes al día?'].apply(
        lambda x: convertir_a_numerico(x, map_horas_num)
    )

if '5. ¿Cuántos días te desvelas a la semana?' in df_clustering.columns:
    df_clustering['5. ¿Cuántos días te desvelas a la semana?'] = df_clustering['5. ¿Cuántos días te desvelas a la semana?'].apply(
        lambda x: convertir_a_numerico(x, map_frecuencia_num)
    )

# Asegurar que las columnas de estrés y ansiedad sean numéricas
if 'estres_numerico' in df_clustering.columns:
    df_clustering['estres_numerico'] = pd.to_numeric(df_clustering['estres_numerico'], errors='coerce')

if 'ansiedad_numerico' in df_clustering.columns:
    df_clustering['ansiedad_numerico'] = pd.to_numeric(df_clustering['ansiedad_numerico'], errors='coerce')

print(f"\nDataFrame después de la conversión shape: {df_clustering.shape}")
print("Tipos de datos después de la conversión:")
print(df_clustering.dtypes)
print("\nValores nulos por columna:")
print(df_clustering.isnull().sum())

# Eliminar filas con NaN después de la conversión
df_clustering_clean = df_clustering.dropna()
print(f"\nDataFrame limpio para clustering shape: {df_clustering_clean.shape}")

if df_clustering_clean.shape[0] < 2:
    print("No hay suficientes filas con datos completos para clustering.")
else:
    # Verificar que todos los datos sean numéricos
    print("\nVerificación final - todos los datos deben ser numéricos:")
    print(df_clustering_clean.head())
    
    # Realizar el clustering jerárquico
    # Usamos 'ward' linkage que minimiza la varianza dentro de cada clúster
    try:
        linked = linkage(df_clustering_clean, 'ward')

        # Visualizar el dendrograma
        plt.figure(figsize=(15, 7))
        dendrogram(linked,
                   orientation='top',
                   labels=df_clustering_clean.index.tolist(),  # Usar el índice como etiqueta
                   distance_sort='descending',
                   show_leaf_counts=True) # Mostrar el número de puntos en cada hoja

        plt.title('Dendrograma de Clustering Jerárquico', fontweight='bold')
        plt.xlabel('Índice de Observación (o número de puntos en la hoja)')
        plt.ylabel('Distancia (Ward)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_path = os.path.join(carpeta_ans, 'Hierarchical_Clustering.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 
        print("\nDendrograma generado exitosamente.")      
    
    except Exception as e:
        print(f"Error en el clustering jerárquico: {e}")