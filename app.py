from __future__ import annotations

import json
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import scanpy as sc
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "ddls-week5-s2-novel-or-known-dataset.zip"
EXTRACTED = ROOT / "data" / "pbmc3k.h5ad"


def ensure_dataset() -> Path:
    if EXTRACTED.exists():
        return EXTRACTED
    with zipfile.ZipFile(ARCHIVE) as archive:
        member = next(name for name in archive.namelist() if name.endswith("pbmc3k.h5ad"))
        with archive.open(member) as source, EXTRACTED.open("wb") as target:
            target.write(source.read())
    return EXTRACTED


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = ensure_dataset()
    app.state.adata = sc.read_h5ad(path)
    app.state.adata.obs["leiden"] = app.state.adata.obs["leiden"].astype(str)
    yield
    del app.state.adata


app = FastAPI(title="PBMC Explorer", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


def adata_from_request() -> Any:
    return app.state.adata


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@app.get("/api/umap")
def umap(cluster: str | None = None):
    adata = adata_from_request()
    coords = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs["leiden"].astype(str).to_numpy()
    keep = np.ones(len(labels), dtype=bool) if cluster in (None, "all") else labels == cluster
    if cluster not in (None, "all") and not keep.any():
        raise HTTPException(404, "Unknown cluster")
    rows = [
        {"cell": str(adata.obs_names[i]), "x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": labels[i],
         "n_genes": jsonable(adata.obs["n_genes"].iloc[i]), "pct_mito": jsonable(adata.obs["pct_mito"].iloc[i])}
        for i in np.flatnonzero(keep)
    ]
    return {"data": rows}


@app.get("/api/qc/{metric}")
def qc_expression(metric: str):
    if metric not in {"pct_mito", "total_counts"}:
        raise HTTPException(404, "Metric must be pct_mito or total_counts")
    adata = adata_from_request()
    coords = np.asarray(adata.obsm["X_umap"])
    labels = adata.obs["leiden"].astype(str).to_numpy()
    values = adata.obs[metric].to_numpy()
    return {"metric": metric, "data": [{"cell": str(adata.obs_names[i]), "x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": labels[i], "value": float(values[i])} for i in range(adata.n_obs)]}


@app.get("/api/gene/{gene}")
def gene_expression(gene: str):
    adata = adata_from_request()
    matches = np.flatnonzero(adata.var_names.astype(str).to_numpy() == gene)
    if not len(matches):
        raise HTTPException(404, f"Gene not found: {gene}")
    values = adata[:, matches[0]].X
    values = values.toarray().ravel() if hasattr(values, "toarray") else np.asarray(values).ravel()
    labels = adata.obs["leiden"].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["X_umap"])
    return {"gene": gene, "data": [{"cell": str(adata.obs_names[i]), "x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": labels[i], "expression": float(values[i])} for i in range(adata.n_obs)]}


@app.get("/api/clusters")
def cluster_quality(top: int = 10):
    if not 1 <= top <= 100:
        raise HTTPException(422, "top must be between 1 and 100")
    adata = adata_from_request()
    labels = adata.obs["leiden"].astype(str).to_numpy()
    clusters = sorted(np.unique(labels), key=lambda x: int(x))
    # Rank each cluster against all other cells. Ranking a single-cluster
    # subset leaves no reference group and produces invalid/repeated markers.
    ranked = sc.tl.rank_genes_groups(
        adata,
        "leiden",
        groups=clusters,
        reference="rest",
        n_genes=top,
        method="wilcoxon",
        copy=True,
    )
    results = ranked.uns["rank_genes_groups"]
    output = {}
    for cluster in clusters:
        mask = labels == cluster
        names = results["names"][cluster]
        scores = results["scores"][cluster]
        output[cluster] = {"cells": int(mask.sum()), "mean_n_genes": float(np.mean(adata.obs["n_genes"].to_numpy()[mask])), "mean_pct_mito": float(np.mean(adata.obs["pct_mito"].to_numpy()[mask])), "markers": [{"gene": str(g), "score": float(s)} for g, s in zip(names, scores)]}
    return output


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PBMC Explorer</title>
<script src="https://cdn.tailwindcss.com"></script><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body class="bg-slate-50 text-slate-900"><main class="mx-auto max-w-7xl p-4 sm:p-6"><header class="mb-5"><h1 class="text-2xl font-bold">PBMC single-cell explorer</h1><p class="text-sm text-slate-600">Explore the supplied UMAP, gene expression, cluster markers, and quality metrics.</p></header>
<section class="grid grid-cols-1 gap-4 lg:grid-cols-[18rem_1fr]"><aside class="rounded-xl bg-white p-4 shadow-sm"><label class="mb-1 block text-sm font-medium">Cluster</label><select id="cluster" class="mb-4 w-full rounded border p-2"><option value="all">All clusters</option></select><label class="mb-1 block text-sm font-medium">Colour by</label><select id="colourBy" class="w-full rounded border p-2"><option value="cluster">Cluster</option><option value="pct_mito">% mitochondrial genes</option><option value="total_counts">Total counts</option><option value="gene">Gene expression</option></select><input id="gene" class="mt-2 hidden w-full rounded border p-2" placeholder="e.g. CD3D"><button id="plotGene" class="mt-2 hidden w-full rounded bg-indigo-600 px-3 py-2 text-white">Plot expression</button><button id="reset" class="mt-2 w-full rounded border px-3 py-2">Reset clusters</button><div id="status" class="mt-4 text-xs text-slate-500"></div></aside><div class="min-w-0 rounded-xl bg-white p-2 shadow-sm sm:p-4"><div id="plot" class="h-[75vh] min-h-[34rem]"></div></div></section><section class="mt-5 rounded-xl bg-white p-4 shadow-sm"><h2 class="mb-2 font-semibold">Cluster quality and top markers</h2><div class="overflow-x-auto"><table class="w-full text-left text-sm"><thead><tr class="border-b"><th class="p-2">Cluster</th><th class="p-2">Cells</th><th class="p-2">Mean genes</th><th class="p-2">Mean % mito</th><th class="p-2">Top markers</th></tr></thead><tbody id="table"></tbody></table></div></section><section class="mt-5 rounded-xl bg-white p-4 shadow-sm"><h2 class="mb-2 font-semibold">Tentative cluster cell-type identities</h2><p class="mb-3 text-sm text-slate-600">Interpretations based on the interview’s reference markers and the supplied top marker rankings. These are hypotheses, not confirmed annotations.</p><div class="overflow-x-auto"><table class="w-full text-left text-sm"><thead><tr class="border-b"><th class="p-2">Cluster</th><th class="p-2">Tentative identity</th><th class="p-2">Reasoning</th></tr></thead><tbody><tr class="border-b"><td class="p-2">0</td><td class="p-2">T cells (uncertain)</td><td class="p-2">Top listed genes are mostly ribosomal plus LDHB, which are not cell-type-specific. Confirm with CD3D, CD3E, TRBC1/2, IL7R, and LTB.</td></tr><tr class="border-b"><td class="p-2">1</td><td class="p-2">Classical/inflammatory monocytes</td><td class="p-2">LYZ, S100A8, and S100A9 match the interview’s monocyte markers.</td></tr><tr class="border-b"><td class="p-2">2</td><td class="p-2">NK or cytotoxic lymphocytes</td><td class="p-2">NKG7, GZMA, CST7, and CTSW indicate a cytotoxic program. Distinguish NK from cytotoxic T cells using NK markers (GNLY, KLRD1, PRF1) and T-cell markers (CD3D/E, TRBC1/2).</td></tr><tr class="border-b"><td class="p-2">3</td><td class="p-2">B cells</td><td class="p-2">CD79A, CD79B, and CD74 support B-cell identity; HLA-DRA and HLA-DPB1 are consistent with antigen presentation.</td></tr><tr class="border-b"><td class="p-2">4</td><td class="p-2">FCGR3A-positive monocytes (non-classical/intermediate-like)</td><td class="p-2">LST1 and FCGR3A match the interview’s markers for a circulating monocyte subset. This is a plausible known-population explanation, not evidence of novelty.</td></tr><tr class="border-b"><td class="p-2">5</td><td class="p-2">Antigen-presenting cells (uncertain)</td><td class="p-2">HLA-DPA1, HLA-DPB1, HLA-DRA, and CD74 indicate antigen presentation but are not specific to dendritic cells. Check FCER1A, CD1C, CLEC10A, and CST3 before assigning DC identity.</td></tr><tr class="border-b"><td class="p-2">6</td><td class="p-2">Platelet-associated cells</td><td class="p-2">PF4 and PPBP suggest a platelet/megakaryocyte-derived signature, outside the interview’s main immune-cell categories.</td></tr><tr class="border-b"><td class="p-2">7</td><td class="p-2">Unresolved</td><td class="p-2">ACTG1, CFL1, GAPDH, and SLC25A5 are not distinctive for the interview’s cell types. With only 10 cells, inspect broader expression and quality before interpreting.</td></tr></tbody></table></div></section></main>
<script>
const $=id=>document.getElementById(id); let markerData;
async function loadUMAP(url='/api/umap'){const r=await fetch(url);if(!r.ok)throw Error(await r.text());const j=await r.json();const rows=j.data;const clusters=[...new Set(rows.map(d=>d.cluster))].sort((a,b)=>Number(a)-Number(b));const palette=['#2563eb','#dc2626','#16a34a','#9333ea','#ea580c','#0891b2','#db2777','#65a30d'];const focus=$('cluster').value;const traces=clusters.map(cluster=>{const group=rows.filter(d=>d.cluster===cluster);const color=palette[Number(cluster)%palette.length];return{x:group.map(d=>d.x),y:group.map(d=>d.y),text:group.map(d=>`${d.cell}<br>Cluster ${d.cluster}<br>Genes: ${d.n_genes}<br>Mito: ${Number(d.pct_mito).toFixed(2)}%`),name:`Cluster ${cluster}`,mode:'markers',type:'scattergl',marker:{size:focus==='all'?6:7,color,opacity:focus==='all'?0.65:1,line:{width:0.5,color:'#fff'}},hovertemplate:'%{text}<extra></extra>'}});Plotly.react('plot',traces,{title:'UMAP',xaxis:{title:'UMAP 1',zeroline:false},yaxis:{title:'UMAP 2',zeroline:false},showlegend:true,legend:{title:{text:'Cluster'}},margin:{t:40,r:10,b:45,l:50},paper_bgcolor:'white',plot_bgcolor:'white'} ,{responsive:true});$('status').textContent=`Showing ${rows.length.toLocaleString()} cells`}
async function init(){const r=await fetch('/api/clusters');markerData=await r.json();Object.keys(markerData).forEach(c=>$('cluster').insertAdjacentHTML('beforeend',`<option>${c}</option>`));$('table').innerHTML=Object.entries(markerData).map(([c,d])=>`<tr class="border-b"><td class="p-2">${c}</td><td class="p-2">${d.cells}</td><td class="p-2">${d.mean_n_genes.toFixed(1)}</td><td class="p-2">${d.mean_pct_mito.toFixed(2)}</td><td class="p-2">${d.markers.map(x=>x.gene).join(', ')}</td></tr>`).join('');await loadUMAP()}
$('cluster').onchange=()=>loadUMAP(`/api/umap?cluster=${$('cluster').value}`);$('reset').onclick=()=>{$('cluster').value='all';loadUMAP()};$('colourBy').onchange=()=>{const isGene=$('colourBy').value==='gene';$('gene').classList.toggle('hidden',!isGene);$('plotGene').classList.toggle('hidden',!isGene);if(!isGene)plotMetric($('colourBy').value)};async function plotMetric(metric){if(metric==='cluster'){loadUMAP();return}try{const r=await fetch(`/api/qc/${metric}`);if(!r.ok)throw Error(await r.text());const j=await r.json();const rows=j.data;const label=metric==='pct_mito'?'% mitochondrial genes':'Total counts';Plotly.react('plot',[{x:rows.map(d=>d.x),y:rows.map(d=>d.y),mode:'markers',type:'scattergl',marker:{size:7,color:rows.map(d=>d.value),colorscale:'Viridis',showscale:true,colorbar:{title:label}},text:rows.map(d=>`${d.cell}<br>${label}: ${d.value}`),hovertemplate:'%{text}<extra></extra>'}],{title:`${label} on UMAP`,xaxis:{title:'UMAP 1',zeroline:false},yaxis:{title:'UMAP 2',zeroline:false},margin:{t:40,r:10,b:45,l:50}},{responsive:true})}catch(e){$('status').textContent=e.message}}$('plotGene').onclick=async()=>{const gene=$('gene').value.trim();if(!gene)return;try{const r=await fetch(`/api/gene/${encodeURIComponent(gene)}`);if(!r.ok)throw Error(await r.text());const j=await r.json();const rows=j.data;Plotly.react('plot',[{x:rows.map(d=>d.x),y:rows.map(d=>d.y),mode:'markers',type:'scattergl',marker:{size:7,color:rows.map(d=>d.expression),colorscale:'Viridis',showscale:true,colorbar:{title:gene}},text:rows.map(d=>`${d.cell}<br>${gene}: ${d.expression}`),hovertemplate:'%{text}<extra></extra>'}],{title:`${gene} expression on UMAP`,xaxis:{title:'UMAP 1',zeroline:false},yaxis:{title:'UMAP 2',zeroline:false},margin:{t:40,r:10,b:45,l:50}},{responsive:true})}catch(e){$('status').textContent=e.message}};init().catch(e=>$('status').textContent=e.message);
</script></body></html>'''
