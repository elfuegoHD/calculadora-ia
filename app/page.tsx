"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";

export default function Home() {
  // Modelos independientes
  const [modelSuma, setModelSuma] = useState<tf.LayersModel | null>(null);
  const [modelResta, setModelResta] = useState<tf.LayersModel | null>(null);

  // UI
  const [operacion, setOperacion] = useState<"suma" | "resta">("suma");
  const [mensaje, setMensaje] = useState<string>("");
  const [numA, setNumA] = useState<string>("");
  const [numB, setNumB] = useState<string>("");

  // ============================================
  // 🔥 Cargar ambos modelos desde el ZIP
  // ============================================
  async function cargarModelosZIP() {
    try {
      setMensaje("Cargando modelos desde ZIP...");

      // Tu ZIP debe estar en /public
      const response = await fetch("/modelos.zip");
      const blob = await response.blob();
      const buffer = await blob.arrayBuffer();

      const zip = await JSZip.loadAsync(buffer);

      // Función auxiliar para cargar un modelo dentro de una carpeta
      async function cargarModeloDesdeCarpeta(carpeta: string) {
        let modelJsonFile: File | null = null;
        let weightsFile: File | null = null;

        for (const filename of Object.keys(zip.files)) {
          if (filename.startsWith(carpeta) && filename.endsWith("model.json")) {
            const jsonStr = await zip.files[filename].async("string");
            modelJsonFile = new File([jsonStr], "model.json", {
              type: "application/json",
            });
          }

          if (filename.startsWith(carpeta) && filename.endsWith(".bin")) {
            const binBuf = await zip.files[filename].async("arraybuffer");
            weightsFile = new File([binBuf], "weights.bin", {
              type: "application/octet-stream",
            });
          }
        }

        if (!modelJsonFile || !weightsFile) {
          throw new Error("Modelo incompleto dentro del ZIP: " + carpeta);
        }

        return await tf.loadLayersModel(
          tf.io.browserFiles([modelJsonFile, weightsFile])
        );
      }

      // Cargar suma
      const suma = await cargarModeloDesdeCarpeta("modelo_suma");
      setModelSuma(suma);

      // Cargar resta
      const resta = await cargarModeloDesdeCarpeta("modelo_resta");
      setModelResta(resta);

      setMensaje("Modelos cargados correctamente 🎉");
    } catch (error) {
      console.error(error);
      setMensaje("❌ Error al cargar los modelos");
    }
  }

  // ============================================
  // ✨ Ejecutar predicción
  // ============================================
  async function predecir() {
    const modelo =
      operacion === "suma" ? modelSuma : modelResta;

    if (!modelo) {
      setMensaje("Primero carga los modelos");
      return;
    }

    if (numA === "" || numB === "") {
      setMensaje("Ingresa ambos números");
      return;
    }

    const a = parseFloat(numA);
    const b = parseFloat(numB);

    const entrada: tf.Tensor2D = tf.tensor2d([[a, b]]);
    const pred = modelo.predict(entrada);

    if (!pred || Array.isArray(pred)) {
      setMensaje("Error en la predicción");
      return;
    }

    const salida = pred.dataSync()[0];
    setMensaje(`Resultado de la ${operacion}: ${salida}`);
  }

  return (
    <div style={{ padding: 40, maxWidth: 500 }}>
      <h1>Calculadora IA – Suma y Resta</h1>

      {/* BOTÓN CARGAR ZIP */}
      <button
        onClick={cargarModelosZIP}
        style={{
          padding: 10,
          width: "100%",
          background: "#444",
          color: "white",
          marginBottom: 20,
        }}
      >
        Cargar modelos desde ZIP
      </button>

      {/* SELECCIÓN DE OPERACIÓN */}
      <select
        value={operacion}
        onChange={(e) => setOperacion(e.target.value as "suma" | "resta")}
        style={{ padding: 8, width: "100%", marginBottom: 20 }}
      >
        <option value="suma">Suma</option>
        <option value="resta">Resta</option>
      </select>

      {/* INPUTS */}
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

      {/* BOTÓN DE PREDICCIÓN */}
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
