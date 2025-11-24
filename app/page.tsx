"use client";

import { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";

export default function Home() {
  const [model, setModel] = useState(null);
  const [mensaje, setMensaje] = useState("");
  const [numA, setNumA] = useState("");
  const [numB, setNumB] = useState("");

  // 🚀 Cargar ZIP desde /public automático
  async function cargarModeloDesdePublic() {
    try {
      setMensaje("Cargando modelo desde carpeta raíz...");

      const response = await fetch("/modelo_resta.zip");
      const blob = await response.blob();

      const data = await blob.arrayBuffer();
      const zip = await JSZip.loadAsync(data);

      let modelJsonFile = null;
      let weightsBinFile = null;

      for (const filename of Object.keys(zip.files)) {
        if (filename.endsWith("model.json")) {
          const jsonStr = await zip.files[filename].async("string");
          modelJsonFile = new File([jsonStr], "model.json");
        }
        if (filename.endsWith(".bin")) {
          const binBuf = await zip.files[filename].async("arraybuffer");
          weightsBinFile = new File([binBuf], "group1-shard1of1.bin");
        }
      }

      if (!modelJsonFile || !weightsBinFile) {
        setMensaje("❌ El ZIP no contiene un modelo TensorFlow.js válido");
        return;
      }

      const modelo = await tf.loadLayersModel(
        tf.io.browserFiles([modelJsonFile, weightsBinFile])
      );

      setModel(modelo);
      setMensaje("Modelo cargado correctamente desde /public 🎉");
    } catch (error) {
      console.error(error);
      setMensaje("❌ Error al cargar el modelo");
    }
  }

  async function predecir() {
    if (!model) {
      setMensaje("Primero carga el modelo desde la carpeta raíz");
      return;
    }

    if (numA === "" || numB === "") {
      setMensaje("Ingresa ambos números");
      return;
    }

    const a = parseFloat(numA);
    const b = parseFloat(numB);

    const entrada = tf.tensor2d([[a, b]]);
    const salida = model.predict(entrada).dataSync();

    setMensaje("Predicción: " + salida[0]);
  }

  return (
    <div style={{ padding: 40, maxWidth: 500 }}>
      <h1>Calculadora IA – Modelo desde carpeta raíz</h1>

      <button
        onClick={cargarModeloDesdePublic}
        style={{
          padding: 10,
          width: "100%",
          background: "#444",
          color: "white",
          marginBottom: 20,
        }}
      >
        Cargar modelo desde /public
      </button>

      <hr />

      <h2>Ingresa los números</h2>

      <input
        type="number"
        placeholder="Número A"
        value={numA}
        onChange={(e) => setNumA(e.target.value)}
        style={{ width: "100%", padding: 8, marginBottom: 10 }}
      />

      <input
        type="number"
        placeholder="Número B"
        value={numB}
        onChange={(e) => setNumB(e.target.value)}
        style={{ width: "100%", padding: 8 }}
      />

      <button
        onClick={predecir}
        style={{
          marginTop: 20,
          padding: 10,
          width: "100%",
          background: "green",
          color: "white",
          fontSize: 18,
        }}
      >
        Ejecutar predicción
      </button>

      <p style={{ marginTop: 20, fontSize: 20 }}>{mensaje}</p>
    </div>
  );
}
