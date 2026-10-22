/**
 * MemCentric-Sim Interactive Dashboard Controller
 */

// Embedded benchmark dataset for instant responsive visualization
const benchmarkData = {
  "Heterogeneous Pipeline": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 10.23,
      dataSaved: 94.8,
      edpGain: 93.3,
      hostTasks: 1,
      pimTasks: 3,
      busMovedKB: 32.8,
      tasks: [
        { name: "Pipeline_Stage1_LogFilter", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "Low Operational Intensity (0.12 FLOP/B) & High Reduction Ratio (99.9% saved)" },
        { name: "Pipeline_Stage2_EntityLookup", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "Irregular random access, direct bank parallelism avoids CXL serialization" },
        { name: "Pipeline_Stage3_MLInference", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.00 KB", overhead: "35c", reason: "High Operational Intensity (10.6 FLOP/B) & cache reuse favors 4-core AVX host" },
        { name: "Pipeline_Stage4_GlobalReduction", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "Sequential vector stream reduced near memory into scalar sum" }
      ],
      desc: "End-to-End Analytics & ML Pipeline dynamically alternating between In-Memory Filtering/Lookup and Host Vector ML Inference."
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 4,
      pimTasks: 0,
      busMovedKB: 638.4,
      tasks: [
        { name: "Pipeline_Stage1_LogFilter", category: "STREAMING", target: "HOST", cycles: 56960, dataMoved: "312.5 KB", overhead: "0c", reason: "Static Policy: Baseline Host" },
        { name: "Pipeline_Stage2_EntityLookup", category: "RANDOM_ACCESS", target: "HOST", cycles: 61080, dataMoved: "128.0 KB", overhead: "0c", reason: "Static Policy: Baseline Host" },
        { name: "Pipeline_Stage3_MLInference", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "0c", reason: "Static Policy: Baseline Host" },
        { name: "Pipeline_Stage4_GlobalReduction", category: "STREAMING", target: "HOST", cycles: 24900, dataMoved: "128.0 KB", overhead: "0c", reason: "Static Policy: Baseline Host" }
      ],
      desc: "Baseline execution forcing all raw data to traverse external CXL bus to CPU caches."
    },
    "STATIC_PIM": {
      speedup: 6.08,
      dataSaved: 97.4,
      edpGain: 85.5,
      hostTasks: 0,
      pimTasks: 4,
      busMovedKB: 16.5,
      tasks: [
        { name: "Pipeline_Stage1_LogFilter", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "0c", reason: "Static Policy: Always PIM" },
        { name: "Pipeline_Stage2_EntityLookup", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "0c", reason: "Static Policy: Always PIM" },
        { name: "Pipeline_Stage3_MLInference", category: "HIGH_REUSE", target: "PIM", cycles: 18450, dataMoved: "16.0 KB", overhead: "0c", reason: "Static Policy: Always PIM (Suffers due to low PIM clock & ALUs)" },
        { name: "Pipeline_Stage4_GlobalReduction", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "0c", reason: "Static Policy: Always PIM" }
      ],
      desc: "Static offload forcing even compute-intensive GEMM into low-frequency PIM cores."
    },
    "ROOFLINE_OI": {
      speedup: 10.23,
      dataSaved: 94.8,
      edpGain: 93.3,
      hostTasks: 1,
      pimTasks: 3,
      busMovedKB: 32.8,
      tasks: [
        { name: "Pipeline_Stage1_LogFilter", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "Operational Intensity (0.12 FLOP/B) < 2.0 Threshold" },
        { name: "Pipeline_Stage2_EntityLookup", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "Operational Intensity (0.25 FLOP/B) < 2.0 Threshold" },
        { name: "Pipeline_Stage3_MLInference", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "35c", reason: "Operational Intensity (10.6 FLOP/B) >= 2.0 Threshold" },
        { name: "Pipeline_Stage4_GlobalReduction", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "Operational Intensity (0.25 FLOP/B) < 2.0 Threshold" }
      ],
      desc: "Threshold rule partitioning by Operational Intensity (FLOP/Byte)."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 10.23,
      dataSaved: 94.8,
      edpGain: 93.3,
      hostTasks: 1,
      pimTasks: 3,
      busMovedKB: 32.8,
      tasks: [
        { name: "Pipeline_Stage1_LogFilter", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "Congestion-aware & low reduction ratio offloaded" },
        { name: "Pipeline_Stage2_EntityLookup", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "Congestion-aware pointer chasing offloaded" },
        { name: "Pipeline_Stage3_MLInference", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "35c", reason: "Host compute favored" },
        { name: "Pipeline_Stage4_GlobalReduction", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "Congestion-aware reduction offloaded" }
      ],
      desc: "Monitors CXL bus queue depth and prioritizes PIM offloading for bandwidth hogs."
    }
  },
  "High-Reuse (GEMM)": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 32.0,
      tasks: [
        { name: "GEMM_64x64", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "35c", reason: "Operational Intensity (10.6 FLOP/B) & Cache reuse makes 4-core Host CPU 1.96x faster than PIM" }
      ],
      desc: "Dense Matrix Multiplication (64x64). High temporal reuse across inner loop tiles."
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 32.0,
      tasks: [
        { name: "GEMM_64x64", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "0c", reason: "Baseline Host" }
      ],
      desc: "Baseline Host CPU."
    },
    "STATIC_PIM": {
      speedup: 0.51,
      dataSaved: 49.2,
      edpGain: -172.1,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 16.0,
      tasks: [
        { name: "GEMM_64x64", category: "HIGH_REUSE", target: "PIM", cycles: 8250, dataMoved: "16.0 KB", overhead: "0c", reason: "Always PIM (Severe compute bottleneck in PIM)" }
      ],
      desc: "Naive PIM offload suffers 49% slowdown due to low ALU throughput in memory logic die."
    },
    "ROOFLINE_OI": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 32.0,
      tasks: [
        { name: "GEMM_64x64", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "35c", reason: "OI (10.6) >= 2.0 -> Kept on Host" }
      ],
      desc: "Roofline threshold."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 32.0,
      tasks: [
        { name: "GEMM_64x64", category: "HIGH_REUSE", target: "HOST", cycles: 4210, dataMoved: "32.0 KB", overhead: "35c", reason: "Host compute favored" }
      ],
      desc: "Adaptive congestion."
    }
  },
  "Streaming (Vector Sum)": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 27.99,
      dataSaved: 100.0,
      edpGain: 97.9,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "Streaming_Vector_Reduction_32768", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "Pure streaming vector reduced to 8-byte scalar sum, sparing 128KB bus traffic" }
      ],
      desc: "Linear streaming vector reduction. Zero temporal locality, memory bandwidth bound."
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 128.0,
      tasks: [
        { name: "Streaming_Vector_Reduction_32768", category: "STREAMING", target: "HOST", cycles: 24900, dataMoved: "128.0 KB", overhead: "0c", reason: "Baseline Host" }
      ],
      desc: "Baseline Host CPU."
    },
    "STATIC_PIM": {
      speedup: 27.99,
      dataSaved: 100.0,
      edpGain: 97.9,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "Streaming_Vector_Reduction_32768", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "0c", reason: "Always PIM" }
      ],
      desc: "Always PIM."
    },
    "ROOFLINE_OI": {
      speedup: 27.99,
      dataSaved: 100.0,
      edpGain: 97.9,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "Streaming_Vector_Reduction_32768", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "OI (0.25) < 2.0 -> Offloaded to PIM" }
      ],
      desc: "Roofline threshold."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 27.99,
      dataSaved: 100.0,
      edpGain: 97.9,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "Streaming_Vector_Reduction_32768", category: "STREAMING", target: "PIM", cycles: 890, dataMoved: "0.01 KB", overhead: "35c", reason: "PIM offload" }
      ],
      desc: "Adaptive congestion."
    }
  },
  "Streaming (DB Scan)": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 45.57,
      dataSaved: 100.0,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "DB_Scan_Filter_50000", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "Database predicate scan filtered inside DRAM, sending only matching scalar sum" }
      ],
      desc: "Database Column Filter & Aggregate Scan (50,000 records). 45.6x faster than host!"
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 390.6,
      tasks: [
        { name: "DB_Scan_Filter_50000", category: "STREAMING", target: "HOST", cycles: 56960, dataMoved: "390.6 KB", overhead: "0c", reason: "Baseline Host" }
      ],
      desc: "Baseline Host CPU."
    },
    "STATIC_PIM": {
      speedup: 45.57,
      dataSaved: 100.0,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "DB_Scan_Filter_50000", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "0c", reason: "Always PIM" }
      ],
      desc: "Always PIM."
    },
    "ROOFLINE_OI": {
      speedup: 45.57,
      dataSaved: 100.0,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "DB_Scan_Filter_50000", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "OI (0.12) < 2.0" }
      ],
      desc: "Roofline threshold."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 45.57,
      dataSaved: 100.0,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.01,
      tasks: [
        { name: "DB_Scan_Filter_50000", category: "STREAMING", target: "PIM", cycles: 1250, dataMoved: "0.01 KB", overhead: "35c", reason: "PIM offload" }
      ],
      desc: "Adaptive congestion."
    }
  },
  "Random Access (Graph BFS)": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 31.55,
      dataSaved: 98.9,
      edpGain: 98.8,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 1.6,
      tasks: [
        { name: "Graph_BFS_8000_Nodes", category: "RANDOM_ACCESS", target: "PIM", cycles: 2890, dataMoved: "1.6 KB", overhead: "35c", reason: "Pointer-chasing graph traversal exploits DRAM Bank-Level Parallelism without cache thrashing" }
      ],
      desc: "Graph Breadth-First Search Frontier Expansion. Pointer chasing with irregular memory access."
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 145.2,
      tasks: [
        { name: "Graph_BFS_8000_Nodes", category: "RANDOM_ACCESS", target: "HOST", cycles: 91200, dataMoved: "145.2 KB", overhead: "0c", reason: "Baseline Host" }
      ],
      desc: "Baseline Host CPU."
    },
    "STATIC_PIM": {
      speedup: 31.55,
      dataSaved: 98.9,
      edpGain: 98.8,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 1.6,
      tasks: [
        { name: "Graph_BFS_8000_Nodes", category: "RANDOM_ACCESS", target: "PIM", cycles: 2890, dataMoved: "1.6 KB", overhead: "0c", reason: "Always PIM" }
      ],
      desc: "Always PIM."
    },
    "ROOFLINE_OI": {
      speedup: 31.55,
      dataSaved: 98.9,
      edpGain: 98.8,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 1.6,
      tasks: [
        { name: "Graph_BFS_8000_Nodes", category: "RANDOM_ACCESS", target: "PIM", cycles: 2890, dataMoved: "1.6 KB", overhead: "35c", reason: "Random access OI < 2.0" }
      ],
      desc: "Roofline threshold."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 31.55,
      dataSaved: 98.9,
      edpGain: 98.8,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 1.6,
      tasks: [
        { name: "Graph_BFS_8000_Nodes", category: "RANDOM_ACCESS", target: "PIM", cycles: 2890, dataMoved: "1.6 KB", overhead: "35c", reason: "PIM offload" }
      ],
      desc: "Adaptive congestion."
    }
  },
  "Random Access (Embedding)": {
    "DYNAMIC_ANALYTICAL": {
      speedup: 24.63,
      dataSaved: 99.9,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.06,
      tasks: [
        { name: "Sparse_Embedding_Lookup_1000", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "DLRM sparse embedding pooling aggregated in memory, saving 64KB random vector transfers" }
      ],
      desc: "Sparse Recommendation Embedding Table Gathering (1000 lookups, pooled into single vector)."
    },
    "STATIC_HOST": {
      speedup: 1.00,
      dataSaved: 0.0,
      edpGain: 0.0,
      hostTasks: 1,
      pimTasks: 0,
      busMovedKB: 64.0,
      tasks: [
        { name: "Sparse_Embedding_Lookup_1000", category: "RANDOM_ACCESS", target: "HOST", cycles: 61080, dataMoved: "64.0 KB", overhead: "0c", reason: "Baseline Host" }
      ],
      desc: "Baseline Host CPU."
    },
    "STATIC_PIM": {
      speedup: 24.63,
      dataSaved: 99.9,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.06,
      tasks: [
        { name: "Sparse_Embedding_Lookup_1000", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "0c", reason: "Always PIM" }
      ],
      desc: "Always PIM."
    },
    "ROOFLINE_OI": {
      speedup: 24.63,
      dataSaved: 99.9,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.06,
      tasks: [
        { name: "Sparse_Embedding_Lookup_1000", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "Sparse lookup OI < 2.0" }
      ],
      desc: "Roofline threshold."
    },
    "ADAPTIVE_CONGESTION": {
      speedup: 24.63,
      dataSaved: 99.9,
      edpGain: 99.2,
      hostTasks: 0,
      pimTasks: 1,
      busMovedKB: 0.06,
      tasks: [
        { name: "Sparse_Embedding_Lookup_1000", category: "RANDOM_ACCESS", target: "PIM", cycles: 2480, dataMoved: "0.06 KB", overhead: "35c", reason: "PIM offload" }
      ],
      desc: "Adaptive congestion."
    }
  }
};

let rooflineChartInstance = null;
let policyChartInstance = null;

function initCharts() {
  // 1. Roofline Chart
  const ctxRoofline = document.getElementById("rooflineChart").getContext("2d");
  
  // Generate logarithmic OI points: 0.01 to 100
  const oiValues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0];
  const hostBusBW = 64.0;       // GB/s
  const hostPeakFlops = 204.8;  // GFLOPs
  const pimInternalBW = 1024.0; // GB/s
  const pimPeakFlops = 16.0;    // GFLOPs

  const hostCurve = oiValues.map(oi => ({ x: oi, y: Math.min(hostPeakFlops, oi * hostBusBW) }));
  const pimCurve = oiValues.map(oi => ({ x: oi, y: Math.min(pimPeakFlops, oi * pimInternalBW) }));

  rooflineChartInstance = new Chart(ctxRoofline, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Host CPU Roofline (64 GB/s CXL, 204.8 GFLOPs)",
          data: hostCurve,
          borderColor: "#3b82f6",
          borderWidth: 2.5,
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          pointRadius: 0,
          tension: 0.1
        },
        {
          label: "Near-Memory PIM Roofline (1024 GB/s TSV, 16.0 GFLOPs)",
          data: pimCurve,
          borderColor: "#f59e0b",
          borderWidth: 2.5,
          borderDash: [5, 5],
          backgroundColor: "rgba(245, 158, 11, 0.05)",
          pointRadius: 0,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Operational Intensity (FLOPs / Byte)", color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" }
        },
        y: {
          type: "logarithmic",
          title: { display: true, text: "Attainable Performance (GFLOPs/s)", color: "#94a3b8", font: { size: 10 } },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" }
        }
      },
      plugins: {
        legend: {
          labels: { color: "#cbd5e1", font: { size: 10 } }
        }
      }
    }
  });

  // 2. Policy Performance Bar Chart
  const ctxPolicy = document.getElementById("policyChart").getContext("2d");
  policyChartInstance = new Chart(ctxPolicy, {
    type: "bar",
    data: {
      labels: ["STATIC_HOST", "STATIC_PIM", "ROOFLINE_OI", "DYNAMIC_ANALYTICAL"],
      datasets: [{
        label: "Speedup vs Baseline",
        data: [1.0, 6.08, 10.23, 10.23],
        backgroundColor: ["#475569", "#f59e0b", "#3b82f6", "#10b981"],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: { display: true, text: "Speedup (x)", color: "#94a3b8" },
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#64748b" }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#cbd5e1", font: { size: 10 } }
        }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });
}

function updateDashboard() {
  const workload = document.getElementById("workload-select").value;
  const policy = document.getElementById("policy-select").value;

  const wlData = benchmarkData[workload];
  if (!wlData) return;

  const currentRun = wlData[policy] || wlData["DYNAMIC_ANALYTICAL"];

  // Update KPIs
  document.getElementById("kpi-speedup").textContent = `${currentRun.speedup.toFixed(2)}x`;
  document.getElementById("kpi-data-saved").textContent = `${currentRun.dataSaved.toFixed(1)}%`;
  document.getElementById("kpi-edp").textContent = `${currentRun.edpGain.toFixed(1)}%`;
  document.getElementById("kpi-placement").textContent = `H: ${currentRun.hostTasks} | P: ${currentRun.pimTasks}`;
  document.getElementById("workload-desc").innerHTML = `<strong>Workload Profile:</strong> ${currentRun.desc}`;
  document.getElementById("active-policy-badge").textContent = `Active Policy: ${policy}`;

  // Update Diagram stats
  document.getElementById("stat-host-tasks").textContent = `Active Tasks: ${currentRun.hostTasks}`;
  document.getElementById("stat-pim-tasks").textContent = `Active Tasks: ${currentRun.pimTasks}`;
  document.getElementById("stat-bus-moved").textContent = `Data Transferred: ${currentRun.busMovedKB} KB (${currentRun.dataSaved}% Saved)`;

  // Update Policy Chart
  if (policyChartInstance) {
    const policies = ["STATIC_HOST", "STATIC_PIM", "ROOFLINE_OI", "DYNAMIC_ANALYTICAL"];
    const speedups = policies.map(p => (wlData[p] ? wlData[p].speedup : 1.0));
    policyChartInstance.data.datasets[0].data = speedups;
    policyChartInstance.update();
  }

  // Update Table
  const tbody = document.getElementById("trace-table-body");
  tbody.innerHTML = "";
  currentRun.tasks.forEach(t => {
    const tr = document.createElement("tr");
    const targetBadgeClass = t.target === "PIM" ? "target-pim" : "target-host";
    tr.innerHTML = `
      <td><strong>${t.name}</strong></td>
      <td><span class="badge badge-subtle">${t.category}</span></td>
      <td><span class="target-badge ${targetBadgeClass}">${t.target}</span></td>
      <td>${t.cycles.toLocaleString()} c</td>
      <td>${t.dataMoved}</td>
      <td>${t.overhead}</td>
      <td style="font-family: var(--font-main); color: var(--text-secondary);">${t.reason}</td>
    `;
    tbody.appendChild(tr);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  updateDashboard();

  document.getElementById("workload-select").addEventListener("change", updateDashboard);
  document.getElementById("policy-select").addEventListener("change", updateDashboard);
  document.getElementById("run-sim-btn").addEventListener("click", () => {
    const btn = document.getElementById("run-sim-btn");
    btn.innerHTML = "<span>🔄 Re-evaluating...</span>";
    setTimeout(() => {
      btn.innerHTML = "<span>⚡ Re-evaluate Architecture</span>";
      updateDashboard();
    }, 300);
  });
});
