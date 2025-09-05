import os
from pytube import YouTube

def normalizar_url(url):
    # Si es un short, convertir a formato watch?v=
    if "youtube.com/shorts/" in url:
        video_id = url.split("/")[-1]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def descargar_video(url, carpeta_destino="descargas_video"):
    # Normalizar URL
    url = normalizar_url(url)

    # Crear carpeta si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    try:
        print("🔗 Obteniendo información del video...")
        yt = YouTube(url)

        print(f"🎬 Título: {yt.title}")

        # Seleccionar la mejor resolución con audio incluido
        stream = yt.streams.filter(progressive=True, file_extension="mp4").order_by("resolution").desc().first()

        print(f"⬇️ Descargando en {stream.resolution}...")
        ruta_video = stream.download(output_path=carpeta_destino)

        print(f"✅ Video guardado en: {ruta_video}")

    except Exception as e:
        print("❌ Error:", e)


if __name__ == "__main__":
    url = input("👉 Pega la URL de YouTube: ")
    descargar_video(url)
