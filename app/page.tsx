"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";

export default function Home() {
  // El modelo debe ser LayersModel | null
  const [model, setModel] = useState<tf.LayersModel | null>(null);

  const [mensaje, setMensaje] = useState<string>("");
  const [numA, setNumA] = useState<string>("");
  const [numB, setNumB] = useState<string>("");

  // 🚀 Cargar ZIP desde /public automáticamente
  async function cargarModeloDesdePublic() {
    try {
      setMensaje("Cargando modelo desde /public...");

      const response = await fetch("/modelo_resta.zip");
      const blob = await response.blob();
      const buffer = await blob.arrayBuffer();

      const zip = await JSZip.loadAsync(buffer);

      let modelJsonFile: File | null = null;
      let weightsBinFile: File | null = null;

      for (const filename of Object.keys(zip.files)) {
        if (filename.endsWith("model.json")) {
          const jsonStr = await zip.files[filename].async("string");
          modelJsonFile = new File([jsonStr], "model.json", {
            type: "application/json",
          });
        }

        if (filename.endsWith(".bin")) {
          const binBuf = await zip.files[filename].async("arraybuffer");
          weightsBinFile = new File([binBuf], "group1-shard1of1.bin", {
            type: "application/octet-stream",
          });
        }
      }

      if (!modelJsonFile || !weightsBinFile) {
        setMensaje("❌ El ZIP no contiene un modelo TensorFlow válido");
        return;
      }

      const loadedModel = await tf.loadLayersModel(
        tf.io.browserFiles([modelJsonFile, weightsBinFile])
      );

      setModel(loadedModel);
      setMensaje("Modelo cargado correctamente 🎉");
    } catch (error) {
      console.error(error);
      setMensaje("❌ Error al cargar el modelo");
    }
  }

  async function predecir() {
    if (!model) {
      setMensaje("Primero carga el modelo");
      return;
    }

    if (numA === "" || numB === "") {
      setMensaje("Ingresa ambos números");
      return;
    }

    const a = parseFloat(numA);
    const b = parseFloat(numB);

    // Entrada correctamente tipada
    const entrada: tf.Tensor2D = tf.tensor2d([[a, b]]);

    // Predict devuelve Tensor | Tensor[]
    const pred = model.predict(entrada);

    if (!pred || Array.isArray(pred)) {
      setMensaje("Error en la predicción");
      return;
    }

    const salida = pred.dataSync()[0];

    setMensaje("Predicción: " + salida);
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
