"""Post-process verified 2D snapshots into quantitative POD/DMD results and figures.

This script requires the exact snapshot archive made by run_2d_workflow.py. It uses
no synthetic field and treats its pressure-gradient centroid as a geometry screen,
not a shock curve or a registered moving coordinate.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from foam2d_io import read_labels
from rom2d_methods import choose_rank, exact_dmd, metrics, pod, reconstruct_dmd, require_uniform_times, standardize_and_stack
from shock2d import displacement_screen, neighbour_graph, shock_indicator

CASE = Path("/mnt/f/Jahidul_Islam/DetFOAM_2DWedge_NonReact_For_ROM")
OUT = CASE / "ROM_DMD_2D_Journal_Work"
DPI = 220

def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def md(path: Path, value: str) -> None:
    path.write_text(value.strip()+"\n", encoding="utf-8")

def save(name: str) -> None:
    for ext in ("png","pdf"):
        plt.savefig(OUT/"09_figures"/f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close()

def draw_field(ax, xy, value, title, unit, cmap="viridis", limits=None):
    if limits is None:
        item=ax.scatter(xy[:,0],xy[:,1],c=value,s=1.0,marker="s",cmap=cmap,linewidths=0,rasterized=True)
    else:
        item=ax.scatter(xy[:,0],xy[:,1],c=value,s=1.0,marker="s",cmap=cmap,vmin=limits[0],vmax=limits[1],linewidths=0,rasterized=True)
    ax.set(aspect="equal", xlabel="x [m]", ylabel="y [m]", title=title)
    return item

def selected_rank(s, target):
    return min(choose_rank(s, target), len(s))

def runs(field, time_s, target, ranks):
    dt=require_uniform_times(time_s); probe=exact_dmd(field,dt,1); mx=min(field.shape[1]-1,len(probe.singular_values))
    sel=selected_rank(probe.singular_values[:mx],target)
    outputs=[]
    for rank in sorted(set([r for r in ranks if r<=mx]+[sel,mx])):
        model=exact_dmd(field,dt,rank)
        rec=reconstruct_dmd(model,len(time_s))
        p_rec,_,_=pod(field,rank)
        outputs.extend([
            {"method":"standard_dmd","rank":model.rank,"selected_rank":sel,**metrics(field,rec)},
            {"method":"pod","rank":rank,"selected_rank":sel,**metrics(field,p_rec)}])
    selected=exact_dmd(field,dt,sel)
    return probe,selected,reconstruct_dmd(selected,len(time_s)),pod(field,sel)[0],outputs

def main():
    config=json.loads((OUT/"config"/"analysis_config.json").read_text())
    data=np.load(OUT/"02_data_extraction"/"snapshots_2d_p_rho_U.npz")
    xy,time_s=data["xy_m"],data["time_s"]
    dt=require_uniform_times(time_s)
    fields={"p":data["p"],"rho":data["rho"],"Ux":data["Ux"],"Uy":data["Uy"]}
    labels={"p":"pressure [Pa]","rho":"density [kg m$^{-3}$]","Ux":"x velocity [m s$^{-1}$]","Uy":"y velocity [m s$^{-1}$]"}
    ranks=config["rom"]["ranks"]; target=config["rom"]["energy_target"]
    quality=[]; all_errors=[]; singular=[]; eigens=[]; recon={}; models={}
    for name, q in fields.items():
        for j,t in enumerate(time_s):
            quality.append({"field":name,"time_s":float(t),"n_cells":int(q.shape[0]),"finite":bool(np.isfinite(q[:,j]).all()),
                            "minimum":float(q[:,j].min()),"maximum":float(q[:,j].max())})
        probe,model,dmd_rec,pod_rec,errors=runs(q,time_s,target,ranks)
        models[name]=model; recon[name]={"dmd":dmd_rec,"pod":pod_rec}
        for r in errors:
            r["field"]=name; all_errors.append(r)
        cumulative=np.cumsum(probe.singular_values**2)/np.sum(probe.singular_values**2)
        for i,s in enumerate(probe.singular_values,1):
            singular.append({"field":name,"rank":i,"singular_value":float(s),
                             "energy_fraction":float(s*s/np.sum(probe.singular_values**2)),
                             "cumulative_energy":float(cumulative[i-1])})
        omega=np.log(model.eigenvalues)/dt
        for i,lam in enumerate(model.eigenvalues,1):
            eigens.append({"field":name,"mode":i,"eigenvalue_real":float(lam.real),"eigenvalue_imag":float(lam.imag),
                           "modulus":float(abs(lam)),"growth_rate_per_s":float(omega[i-1].real),
                           "frequency_hz":float(omega[i-1].imag/(2*np.pi)),
                           "amplitude_modulus":float(abs(model.amplitudes[i-1]))})
    csv_write(OUT/"03_preprocessing"/"data_quality_checks.csv",quality)
    csv_write(OUT/"10_tables"/"reconstruction_errors.csv",all_errors)
    csv_write(OUT/"10_tables"/"singular_values.csv",singular)
    csv_write(OUT/"10_tables"/"dmd_eigenvalues.csv",eigens)
    # Combined p/rho/Ux/Uy state, with explicitly recorded standardisation.
    combined,meta=standardize_and_stack(fields)
    c_probe,c_model,c_dmd,c_pod,c_rows=runs(combined,time_s,target,ranks)
    for r in c_rows:r["field"]="combined_scaled_p_rho_Ux_Uy"
    csv_write(OUT/"08_comparison"/"combined_state_reconstruction_errors.csv",c_rows)
    (OUT/"03_preprocessing"/"combined_state_metadata.json").write_text(json.dumps(
        {k:{"slice":v["slice"],"scale":float(v["scale"])} for k,v in meta.items()},indent=2)+"\n")
    # 2D global gradient geometry screen. It is deliberately not shock tracking.
    owner=read_labels(CASE/"constant"/"polyMesh"/"owner")
    neighbour=read_labels(CASE/"constant"/"polyMesh"/"neighbour")
    graph=neighbour_graph(xy.shape[0],owner,neighbour)
    indicators=[]; geometry=[]
    for j,t in enumerate(time_s):
        screen=shock_indicator(xy,fields["p"][:,j],graph)
        indicators.append(screen["indicator"])
        mask=screen["candidate_mask"]
        geometry.append({"time_s":float(t),"gradient_threshold_pa_per_m":screen["threshold"],
                         "candidate_cell_count":int(mask.sum()),
                         "candidate_x_mean_m":float(xy[mask,0].mean()),"candidate_y_mean_m":float(xy[mask,1].mean())})
    indicators=np.column_stack(indicators)
    displacement=displacement_screen(indicators,xy)
    for row,centre,distance in zip(geometry,displacement["weighted_centroid_xy"],displacement["centroid_displacement_m"]):
        row.update({"global_gradient_weighted_x_m":float(centre[0]),"global_gradient_weighted_y_m":float(centre[1]),
                    "global_gradient_centroid_displacement_m":float(distance)})
    csv_write(OUT/"06_shock_geometry"/"pressure_gradient_geometry_screen.csv",geometry)
    # Train/test uses initial two thirds of verified snapshot sequence, visible in output.
    n_train=int(np.floor(2*len(time_s)/3)); train=np.arange(len(time_s))<n_train; test=~train
    train_rows=[]
    for name,q in fields.items():
        train_model=exact_dmd(q[:,train],dt,min(models[name].rank,n_train-1))
        forecast=reconstruct_dmd(train_model,len(time_s))
        for scope,mask in (("train",train),("held_out",test)):
            train_rows.append({"field":name,"method":"standard_dmd","scope":scope,"rank":train_model.rank,
                               "n_snapshots":int(mask.sum()),"time_start_s":float(time_s[mask][0]),
                               "time_end_s":float(time_s[mask][-1]),**metrics(q[:,mask],forecast[:,mask])})
    csv_write(OUT/"10_tables"/"training_testing_performance.csv",train_rows)
    # Figures: fields and reconstruction comparisons at actual selected times.
    reps=[0,len(time_s)//2,len(time_s)-1]
    for name,q in fields.items():
        limits=(float(q.min()),float(q.max()))
        fig,axes=plt.subplots(3,3,figsize=(11,9),sharex=True,sharey=True)
        for col,j in enumerate(reps):
            for row,(key,title) in enumerate((("fom","FOM"),("pod","POD"),("dmd","DMD"))):
                values=q[:,j] if key=="fom" else recon[name][key][:,j]
                m=draw_field(axes[row,col],xy,values,f"{title}, t={time_s[j]*1e6:.0f} us",labels[name],limits=limits)
        fig.colorbar(m,ax=axes.ravel().tolist(),shrink=.75,label=labels[name])
        fig.suptitle(f"{name}: full-order and selected-rank ROM reconstruction")
        save(f"fom_pod_dmd_{name}_representative_times")
    # singular spectra and rank-error comparison
    fig,axes=plt.subplots(2,2,figsize=(9,6.5))
    for ax,(name,q) in zip(axes.ravel(),fields.items()):
        d=[r for r in singular if r["field"]==name]
        ax.semilogy([r["rank"] for r in d],[r["singular_value"] for r in d],"o-")
        ax.set(title=name,xlabel="rank",ylabel="singular value");ax.grid(alpha=.25)
    fig.suptitle("Centered snapshot singular-value spectra");save("singular_value_spectra")
    fig,axes=plt.subplots(2,2,figsize=(9,6.5))
    for ax,(name,q) in zip(axes.ravel(),fields.items()):
        for method,style in (("pod","o-"),("standard_dmd","s--")):
            d=[r for r in all_errors if r["field"]==name and r["method"]==method]
            ax.semilogy([r["rank"] for r in d],[r["relative_frobenius"] for r in d],style,label=method)
        ax.set(title=name,xlabel="rank",ylabel="relative Frobenius error");ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("In-sample reconstruction error");save("reconstruction_error_vs_rank")
    # Pressure DMD eigenspectrum and mode shapes
    pm=models["p"]; theta=np.linspace(0,2*np.pi,401)
    plt.figure(figsize=(5,4.5));plt.plot(np.cos(theta),np.sin(theta),"k:",lw=.8);plt.scatter(pm.eigenvalues.real,pm.eigenvalues.imag,c=abs(pm.amplitudes),s=40,cmap="viridis")
    plt.gca().set_aspect("equal");plt.xlabel("Re(lambda)");plt.ylabel("Im(lambda)");plt.title("Selected-rank pressure DMD spectrum");save("pressure_dmd_eigenspectrum")
    order=np.argsort(abs(pm.amplitudes))[::-1][:min(6,pm.rank)]
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True)
    for ax,k in zip(axes.ravel(),order):
        values=pm.modes[:,k].real; scale=max(abs(values).max(),np.finfo(float).eps)
        m=draw_field(ax,xy,values/scale,f"mode {k+1}; |b|={abs(pm.amplitudes[k]):.2e}","normalised real mode",cmap="coolwarm",limits=(-1,1))
    fig.colorbar(m,ax=axes.ravel().tolist(),shrink=.75,label="normalised real mode")
    fig.suptitle("Amplitude-ranked pressure DMD modes");save("dominant_pressure_dmd_modes")
    # Geometry screen
    j=-1;screen=shock_indicator(xy,fields["p"][:,j],graph);fig,ax=plt.subplots(figsize=(7,4.8))
    m=draw_field(ax,xy,np.log10(np.maximum(screen["indicator"],np.finfo(float).tiny)),f"log10 |grad p| at {time_s[j]*1e6:.0f} us","log10(Pa/m)",cmap="magma")
    ax.scatter(xy[screen["candidate_mask"],0],xy[screen["candidate_mask"],1],s=3,c="cyan",label="top 1% global gradient cells")
    ax.legend(fontsize=8);fig.colorbar(m,ax=ax,label="log10(Pa/m)");fig.suptitle("Global pressure-gradient geometry screen, not a shock curve");save("pressure_gradient_geometry_screen")
    plt.figure(figsize=(7,3.5));plt.plot(time_s*1e6,displacement["weighted_centroid_xy"][:,0]*1e3,label="weighted x");plt.plot(time_s*1e6,displacement["weighted_centroid_xy"][:,1]*1e3,label="weighted y")
    plt.xlabel("time [us]");plt.ylabel("global gradient centroid [mm]");plt.legend();plt.grid(alpha=.25);plt.title("Global gradient-centroid diagnostic, not shock tracking");save("gradient_centroid_diagnostic")
    # Held out pressure final map and all-field time errors
    q=fields["p"]; split=int(n_train); tm=exact_dmd(q[:,train],dt,min(models["p"].rank,n_train-1));forecast=reconstruct_dmd(tm,len(time_s))
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True);limits=(float(q.min()),float(q.max()))
    for ax,v,title in zip(axes,(q[:,-1],forecast[:,-1],abs(q[:,-1]-forecast[:,-1])),("FOM at held-out final time","DMD forecast","absolute error")):
        m=draw_field(ax,xy,v,title,"Pa",cmap="magma" if "error" in title else "viridis",limits=None if "error" in title else limits)
        fig.colorbar(m,ax=ax,shrink=.7)
    fig.suptitle(f"Training through {time_s[n_train-1]*1e6:.0f} us; held-out final {time_s[-1]*1e6:.0f} us")
    save("held_out_pressure_forecast")
    # Results docs derive solely from just-computed tables.
    sel_rows=[r for r in all_errors if r["rank"]==r["selected_rank"]]
    summary=["# 2D results summary","","All values below are computed from 18 verified snapshots at 2 to 36 us. The first output at 0 us is excluded because rho is absent.","",
             "## Selected-rank in-sample errors","", "| Field | Model | rank | relative Frobenius error |", "|---|---|---:|---:|"]
    for r in sel_rows:summary.append(f'| {r["field"]} | {r["method"]} | {r["rank"]} | {r["relative_frobenius"]:.6g} |')
    summary += ["","## Interpretation boundary","",
                "POD is an in-sample energy-optimal projection. Exact DMD reconstruction uses a first-snapshot amplitude and its error is therefore not required to decrease monotonically with rank. The global pressure-gradient centroid is a geometry-screening diagnostic, not a shock curve or an error reference.",
                "","## Held-out evaluation","",
                f"Training uses the first {n_train} snapshots ({time_s[0]*1e6:.0f} to {time_s[n_train-1]*1e6:.0f} us); the remaining {len(time_s)-n_train} snapshots ({time_s[n_train]*1e6:.0f} to {time_s[-1]*1e6:.0f} us) are held out. Metrics are in 10_tables/training_testing_performance.csv."]
    md(OUT/"11_results"/"results_summary.md","\n".join(summary))
    md(OUT/"FINAL_RESEARCH_SUMMARY.md",f"""# Final 2D research summary

## Verified dataset
The ROM dataset contains 18 complete parallel snapshots from 2 to 36 microseconds at a uniform 2 microsecond interval, reconstructed through cellProcAddressing onto 50,625 global cells. The directory contains no saved snapshots after 36 microseconds, even though the supplied solver log has later records.

## Methods
Mean-centred POD and exact DMD are evaluated for p, rho, Ux and Uy. Combined-state scaling uses each variable's global space-time standard deviation. A global pressure-gradient screen is included, but no moving-frame model is claimed because a shock curve/ROI has not been physically defined.

## Quantitative results
See 10_tables/reconstruction_errors.csv, 10_tables/singular_values.csv, 10_tables/dmd_eigenvalues.csv and 10_tables/training_testing_performance.csv. The selected rank criterion is 99.9 percent centered-snapshot singular-value energy.

## Limitations
The global gradient centroid is not a shock trajectory. Training/testing evaluates standard Eulerian DMD only. No shock-attached conclusion is made until a robust shock-curve extraction and registration model are validated from these data.

## Next scientific decision
Inspect pressure-gradient maps and dominant modes to specify a physically defensible shock curve and ROI. Then compare Eulerian ROM against that explicitly documented geometry-aligned model.
""")
    print(json.dumps({"status":"complete","snapshot_count":len(time_s),"time_start_s":float(time_s[0]),"time_end_s":float(time_s[-1]),"train_count":n_train,"held_out_count":len(time_s)-n_train}))

if __name__=="__main__":
    main()
