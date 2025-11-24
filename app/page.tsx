"use client";

import { useState } from "react";
import * as tf from "@tensorflow/tfjs";
import JSZip from "jszip";

export default function Home() {
  const [modeloSuma, setModeloSuma] = useState<tf.LayersModel | null>(null);
  const [modeloResta, setModeloResta] = useState<tf.LayersModel | null>(null);

  const [operacion, setOperacion] = useState<"suma" | "resta">("suma");
  const [mensaje, setMensaje] = useState<string>("");

  const [numA, setNumA] = useState<string>("");
  const [numB, setNumB] = useState<string>("");

  // --------------------------
  // 🚀 Cargar ZIP desde /public
  // --------------------------
  async function cargarModelos() {
    try {
      setMensaje("Cargando modelos desde modelo_resta.zip ...");

      const response = await fetch("/modelo_resta.zip");
      const blob = await response.blob();
      const buf = await blob.arrayBuffer();

      const zip = await JSZip.loadAsync(buf);

      // MODELOS
      let sumaJson: File | null = null;
      let sumaBin: File | null = null;

      let restaJson: File | null = null;
      let restaBin: File | null = null;

      for (const filename of Object.keys(zip.files)) {
        const file = zip.files[filename];

        // SUMA ----------------------
        if (filename.includes("modelo_suma") && filename.endsWith("model.json")) {
          const jsonStr = await file.async("string");
          sumaJson = new File([jsonStr], "model.json");
        }

        if (filename.includes("modelo_suma") && filename.endsWith(".bin")) {
          const bin = await file.async("arraybuffer");
          sumaBin = new File([bin], "group1-shard1of1.bin");
        }

        // RESTA ----------------------
        if (filename.includes("modelo_resta") && filename.endsWith("model.json")) {
          const jsonStr = await file.async("string");
          restaJson = new File([jsonStr], "model.json");
        }

        if (filename.includes("modelo_resta") && filename.endsWith(".bin")) {
          const bin = await file.async("arraybuffer");
          restaBin = new File([bin], "group1-shard1of1.bin");
        }
      }

      if (!sumaJson || !sumaBin || !restaJson || !restaBin) {
        setMensaje("❌ El ZIP no contiene ambos modelos.");
        return;
      }

      // Cargar ambos
      const modelSuma = await tf.loadLayersModel(tf.io.browserFiles([sumaJson, sumaBin]));
      const modelResta = await tf.loadLayersModel(tf.io.browserFiles([restaJson, restaBin]));

      setModeloSuma(modelSuma);
      setModeloResta(modelResta);

      setMensaje("Modelos cargados correctamente 🎉");
    } catch (err) {
      console.error(err);
      setMensaje("❌ Error cargando los modelos");
    }
  }

  // --------------------------
  // 🚀 Predecir
  // --------------------------
  function predecir() {
    if (!modeloSuma || !modeloResta) {
      setMensaje("Carga los modelos primero");
      return;
    }

    if (numA === "" || numB === "") {
      setMensaje("Ingresa ambos números");
      return;
    }

    const a = parseFloat(numA);
    const b = parseFloat(numB);

    const entrada = tf.tensor2d([[a, b]]);

    let pred: tf.Tensor;

    if (operacion === "suma") {
      pred = modeloSuma.predict(entrada) as tf.Tensor;
    } else {
      pred = modeloResta.predict(entrada) as tf.Tensor;
    }

    const salida = pred.dataSync()[0];

    setMensaje(`Resultado de la ${operacion}: ${salida}`);
  }

  return (
    <div style={{ padding: 40, maxWidth: 500 }}>
      <h1>Calculadora IA – Suma y Resta desde ZIP</h1>

      <button
        onClick={cargarModelos}
        style={{ padding: 10, width: "100%", marginBottom: 20 }}
      >
        Cargar modelos desde ZIP
      </button>

      <h3>Selecciona operación</h3>

      <select
        value={operacion}
        onChange={(e) => setOperacion(e.target.value as any)}
        style={{ padding: 10, width: "100%", marginBottom: 20 }}
      >
        <option value="suma">Suma (+)</option>
        <option value="resta">Resta (-)</option>
      </select>

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
        }}
      >
        Ejecutar predicción
      </button>

      <p style={{ marginTop: 20 }}>{mensaje}</p>
    </div>
  );
}
