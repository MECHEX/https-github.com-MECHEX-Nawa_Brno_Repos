#Only to store example of the configuration file.

PLOT_JOBS: List[Dict] = [
    {
        "parts": ["part4"],
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"]
    },
    {
        "parts":       ["part3"], 
        "fluids":      ["Fluid1"],
        "plots":       ["mean"],          # overlay opcjonalnie
        "metrics":     ["h", "f"],           

        "mean_ma_fluids": ["Fluid1"],
        "mean_ma_windows": [4, 8, 12, 16], #uśrednianie ruchome
        "mean_ma_edges": "strict",
        "mean_ma_center": True,
    },
    {
        "parts": ["part1", "part2"], 
        "fluids": ["Fluid1"], 
        "plots": ["mean"], 
        "metrics": ["h", "f"],

        "mean_ma_fluids": ["Fluid1"],
        "mean_ma_windows": [4, 8, 12, 16], #uśrednianie ruchome
        "mean_ma_edges": "strict",
        "mean_ma_center": True,
    },
    {
        "parts": ["part1", "part2"], 
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"]
    },
    {
        "parts": ["part3"],          
        "fluids": ["Fluid1", "Fluid2"], 
        "plots": ["overlay", "mean"], 
        "metrics": ["h", "f"],
    },
    {
        "parts": ["part1"],          
        "fluids": ["Fluid2"], 
        "plots": ["overlay"], 
        "metrics": ["h"],
        "t0_s": 0.0, 
        "n_steps": 15*20,           #figs for not full time, start from t0_s and n_steps
    },
    {
        "parts": ["part1", "part2"],          
        "fluids": ["Fluid2"], 
        "plots": ["overlay"], 
        "metrics": ["h"],
        "t0_s": 2.0,                #figs for not full time, start from t0_s and duration_s
        "duration_s": 8.0, 
    },

]