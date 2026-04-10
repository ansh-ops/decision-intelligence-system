const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://127.0.0.1:8000";

export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/runs/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed: ${text}`);
  }

  return res.json();
}

export async function startRun(
  runId: string,
  filePath: string,
  targetOverride?: string,
  mode = "deterministic"
) {
  const params = new URLSearchParams({ file_path: filePath, mode });
  if (targetOverride) {
    params.set("target_override", targetOverride);
  }

  const res = await fetch(`${API_BASE}/runs/${runId}/start?${params.toString()}`, {
    method: "POST",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Run start failed: ${text}`);
  }

  return res.json();
}

export async function getRun(runId: string) {
    if (!runId || runId === "undefined") {
      throw new Error("Invalid runId");
    }
  
    const res = await fetch(`${API_BASE}/runs/${runId}`, {
      method: "GET",
      cache: "no-store",
    });
  
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Run fetch failed: ${text}`);
    }
  
    return res.json();
  }
