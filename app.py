"""Aplikasi Streamlit: Deteksi Gambar Asli vs Buatan AI (deepfake) versi 2.

Memakai tiga model ONNX (CNN dari Nol, MobileNetV3, EfficientNet) yang dilatih di notebook.
Hanya butuh: streamlit, numpy, onnxruntime, pillow.
"""
import json
from pathlib import Path

import altair as alt
import numpy as np
import onnxruntime as ort
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

BASE = Path(__file__).parent
BATAS_RENDAH = 0.30   # di bawah ini: kemungkinan foto asli
BATAS_TINGGI = 0.60   # di atas ini: kemungkinan deepfake

WARNA_ZONA = {"Rendah": "#1b9e77", "Sedang": "#e6a117", "Tinggi": "#d1495b"}
WARNA_MODEL = ["#8c8c8c", "#4c78a8", "#7b4ea3"]

st.set_page_config(page_title="Deteksi Gambar Asli vs Buatan AI", page_icon="🕵️", layout="wide")


@st.cache_resource
def muat_semua():
    with open(BASE / "metrik_model.json", encoding="utf-8") as f:
        metrik = json.load(f)
    sesi = {nama: ort.InferenceSession(str(BASE / info["file"]), providers=["CPUExecutionProvider"])
            for nama, info in metrik["model"].items()}
    return metrik, sesi


def siapkan_gambar(sumber, ukuran):
    """Langkah yang SAMA dengan saat pelatihan: putar sesuai EXIF, RGB, potong persegi dari tengah, kecilkan, skala 0-1."""
    gambar = ImageOps.exif_transpose(Image.open(sumber)).convert("RGB")
    kecil = ImageOps.fit(gambar, (ukuran, ukuran), method=Image.Resampling.BILINEAR)
    x = np.asarray(kecil, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]          # bentuk (1, 3, tinggi, lebar)
    return gambar, kecil, x


def peluang_deepfake(sesi, x):
    hasil = sesi.run(None, {sesi.get_inputs()[0].name: x})[0]
    return float(hasil[0][1])


def zona(p):
    if p < BATAS_RENDAH:
        return "Rendah"
    if p < BATAS_TINGGI:
        return "Sedang"
    return "Tinggi"


def grafik_peluang(df):
    dasar = alt.Chart(df).encode(x=alt.X("Model:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)))
    batang = dasar.mark_bar().encode(
        y=alt.Y("Peluang:Q", scale=alt.Scale(domain=[0, 100]), title="Peluang buatan AI (%)"),
        color=alt.Color("Zona:N", scale=alt.Scale(domain=list(WARNA_ZONA), range=list(WARNA_ZONA.values())),
                        legend=alt.Legend(title="Zona")),
        tooltip=["Model", alt.Tooltip("Peluang:Q", format=".1f"), "Zona"],
    )
    angka = dasar.mark_text(dy=-8, fontSize=13).encode(y="Peluang:Q", text=alt.Text("Peluang:Q", format=".1f"))
    garis = alt.Chart(pd.DataFrame({"y": [BATAS_RENDAH * 100, BATAS_TINGGI * 100]})).mark_rule(
        strokeDash=[5, 4], color="#666666").encode(y="y:Q")
    return (batang + angka + garis).properties(height=320, width="container")


def grafik_metrik(df):
    urutan = list(dict.fromkeys(df["Model"]))
    dasar = alt.Chart(df).encode(
        x=alt.X("Metrik:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Model:N", sort=urutan),
    )
    batang = dasar.mark_bar().encode(
        y=alt.Y("Nilai:Q", scale=alt.Scale(domain=[0, 1]), title="Nilai (0 sampai 1)"),
        color=alt.Color("Model:N", sort=urutan, scale=alt.Scale(domain=urutan, range=WARNA_MODEL[:len(urutan)]),
                        legend=alt.Legend(title="Model")),
        tooltip=["Model", "Metrik", alt.Tooltip("Nilai:Q", format=".3f")],
    )
    angka = dasar.mark_text(dy=-7, fontSize=11).encode(y="Nilai:Q", text=alt.Text("Nilai:Q", format=".2f"))
    return (batang + angka).properties(height=340, width="container")


try:
    metrik, sesi_model = muat_semua()
except Exception as galat:   # file model/json belum diunggah ke GitHub
    st.error("File model belum lengkap. Pastikan `metrik_model.json` dan semua file `.onnx` "
             "ada di folder yang sama dengan `app.py` di GitHub.")
    st.caption(f"Detail teknis: {galat}")
    st.stop()

UKURAN = int(metrik["ukuran_gambar"])
NAMA_MODEL = list(metrik["model"])
TERBAIK = metrik["model_terbaik"]

# ---------------------------------------------------------------- tampilan
st.title("🕵️ Deteksi Gambar: Asli atau Buatan AI?")
st.caption("Mini project Machine Learning. Tiga model jaringan saraf membandingkan penilaian mereka "
           "terhadap foto atau gambar yang kamu unggah.")

with st.sidebar:
    st.header("Tentang aplikasi")
    data = metrik.get("dataset", {})
    st.write(f"**Data latih:** {data.get('nama', '-')}")
    if data.get("jumlah_gambar"):
        st.write(f"**Jumlah gambar dipakai:** {data['jumlah_gambar']:,}".replace(",", "."))
    st.write(f"**Ukuran masukan model:** {UKURAN} x {UKURAN} piksel")
    st.write(f"**Model keputusan utama:** {TERBAIK}")
    st.divider()
    st.markdown(
        f"**Zona keputusan** (peluang buatan AI)\n"
        f"- di bawah {BATAS_RENDAH:.0%}: kemungkinan foto asli\n"
        f"- {BATAS_RENDAH:.0%} sampai {BATAS_TINGGI:.0%}: tidak pasti, perlu dicek manual\n"
        f"- {BATAS_TINGGI:.0%} ke atas: kemungkinan buatan AI"
    )
    st.divider()
    st.caption("Batasan: model hanya belajar dari gambar asli dan gambar buatan beberapa generator AI di data latih. "
               "Generator lain, gambar yang sudah diedit, dikompres ulang, atau dipotong bisa dinilai keliru. "
               "Hasilnya perkiraan untuk pembelajaran, bukan alat bukti.")

kiri, kanan = st.columns([1, 2], gap="large")

with kiri:
    st.subheader("1. Pilih foto")
    unggahan = st.file_uploader("Unggah foto atau gambar (JPG, PNG, atau WEBP)", type=["jpg", "jpeg", "png", "webp"])
    sumber = unggahan
    nama_sumber = unggahan.name if unggahan is not None else ""

hasil = None
if sumber is not None:
    try:
        asli, kecil, x = siapkan_gambar(sumber, UKURAN)
    except Exception:
        with kiri:
            st.error("File ini tidak bisa dibaca sebagai gambar. Coba file JPG atau PNG lain.")
        st.stop()
    with kiri:
        st.image(asli, caption=f"{nama_sumber} ({asli.width} x {asli.height} piksel)", width=280)
        with st.expander("Gambar yang sebenarnya dilihat model"):
            st.image(kecil, caption=f"Bagian tengah foto, dikecilkan jadi {UKURAN} x {UKURAN} piksel. "
                                    "Bagian pinggir yang terpotong tidak ikut dinilai.")
        if min(asli.size) < 128:
            st.warning("Foto ini kecil (di bawah 128 piksel di salah satu sisi). Hasil kurang bisa dipercaya.")
    hasil = {nama: peluang_deepfake(sesi_model[nama], x) for nama in NAMA_MODEL}

with kanan:
    st.subheader("2. Hasil penilaian")
    if hasil is None:
        st.info("Unggah sebuah foto atau gambar untuk melihat hasilnya.")
    else:
        p_utama = hasil[TERBAIK]
        z = zona(p_utama)
        st.metric(f"Peluang buatan AI menurut {TERBAIK} (model keputusan utama)", f"{p_utama * 100:.1f}%")
        if z == "Rendah":
            st.success("Zona RENDAH: kemungkinan besar foto asli.")
        elif z == "Sedang":
            st.warning("Zona SEDANG: model tidak yakin. Periksa secara manual atau coba foto lain.")
        else:
            st.error("Zona TINGGI: kemungkinan besar gambar buatan AI (deepfake).")

        jumlah_fake = sum(p >= 0.5 for p in hasil.values())
        if jumlah_fake in (0, len(hasil)):
            st.caption(f"Ketiga model sepakat: {'buatan AI' if jumlah_fake else 'asli'}.")
        else:
            st.caption(f"Model tidak sepakat: {jumlah_fake} dari {len(hasil)} model menilai gambar ini buatan AI.")

        df_hasil = pd.DataFrame({
            "Model": NAMA_MODEL,
            "Peluang": [round(hasil[n] * 100, 1) for n in NAMA_MODEL],
            "Zona": [zona(hasil[n]) for n in NAMA_MODEL],
        })
        st.markdown("**Perbandingan penilaian ketiga model untuk foto ini**")
        st.altair_chart(grafik_peluang(df_hasil))
        st.caption(f"Garis putus-putus menandai batas zona ({BATAS_RENDAH:.0%} dan {BATAS_TINGGI:.0%}).")

st.divider()
st.subheader("3. Perbandingan performa ketiga model (data uji)")
st.write("Angka di bawah dihitung di notebook pada data uji yang tidak dipakai saat melatih. "
         "Gambar buatan AI dianggap kelas positif.")

nama_metrik = {"akurasi": "Akurasi", "presisi": "Presisi", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
baris = [{"Model": n, "Metrik": label, "Nilai": float(metrik["model"][n][kunci])}
         for n in NAMA_MODEL for kunci, label in nama_metrik.items()]
st.altair_chart(grafik_metrik(pd.DataFrame(baris)))

tabel = pd.DataFrame([{
    "Model": ("⭐ " if n == TERBAIK else "") + n,
    "Akurasi": metrik["model"][n]["akurasi"], "Presisi": metrik["model"][n]["presisi"],
    "Recall": metrik["model"][n]["recall"], "F1": metrik["model"][n]["f1"],
    "ROC-AUC": metrik["model"][n]["roc_auc"], "PR-AUC": metrik["model"][n]["pr_auc"],
    "Pra-latih": "Ya" if metrik["model"][n].get("pralatih") else "Tidak",
    "Parameter": metrik["model"][n]["parameter"],
    "Waktu latih (detik)": metrik["model"][n]["waktu_latih_detik"],
    "Ukuran file (MB)": metrik["model"][n]["ukuran_mb"],
} for n in NAMA_MODEL])
st.dataframe(tabel, hide_index=True)
st.caption("⭐ = model keputusan utama (F1 tertinggi di data validasi).")
